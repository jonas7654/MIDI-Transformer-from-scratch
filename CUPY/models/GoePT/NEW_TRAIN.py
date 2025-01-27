





import sys
import os
import datetime
import argparse
from functools import partial
from collections import deque
from types import NoneType
import json
from miditok import REMI, Structured
import wandb

import cupy as cp
import numpy as np

from pathlib import Path
from sklearn.metrics import root_mean_squared_error, accuracy_score
from rich.progress import Progress
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from icecream import ic

sys.path.append('.')
import config
from model import GoePT
from layers import Softmax

ic.configureOutput(includeContext=True)
ic.disable()


# Imports for PyTorch DataLoader and Dataset
from torch.utils.data import DataLoader, Dataset
from miditok.pytorch_data import DataCollator  # DataCollator for padding
import torch

# Custom MemmapDataset class
class MemmapDataset(Dataset):
    def __init__(self, bin_file_path, seq_length):
        self.bin_file_path = bin_file_path
        self.seq_length = seq_length
        self.data = np.memmap(bin_file_path, dtype=np.uint16, mode="r")
        self.num_samples = len(self.data) - seq_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length].astype(np.int64)
        y = self.data[idx + 1:idx + 1 + self.seq_length].astype(np.int64)
        return torch.tensor(x), torch.tensor(y)

# Function to create DataLoader
def create_dataloader(split, data_dir, context_length, batch_size):
    bin_file_path = os.path.join(
        data_dir,
        f'{config.vo_size}_{split}_{config.tokenizer_name_str}_seq_len_{config.context_length}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'
    )
    dataset = MemmapDataset(bin_file_path, context_length)
    collator = DataCollator(pad_token_id=0)  # Replace 0 with the correct pad_token_id
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), collate_fn=collator)
    return dataloader




def compute_token_accuracy(logits, targets):
    softmax = Softmax(axis = 0)
    
    pred_tokens = softmax.forward(logits)
    # axis -1 uses the last axis which is the vocabulary
    pred_tokens = cp.argmax(pred_tokens, axis = -1)    
    
    return accuracy_score(targets.flatten(), pred_tokens.flatten())

def evaluate_test_data(model, data_length, test_batch_size):
    get_batch = partial(read_datasets,
                            data_dir=config.data_dir,
                            context_length=config.context_length,
                            batch_size=test_batch_size,
                            rng=np.random.default_rng(config.seed))
    
    loop_range = data_length // test_batch_size
    all_losses = []
    accuracy = 0
    
    for i in range(loop_range):
        print(f"batch: {i}")
        X, Y = get_batch('test')
        
        logits, loss = model.forward(X, Y)  # Forward pass
        all_losses.append(loss.item())
        
        batch_accuracy = compute_token_accuracy(logits, Y)
        accuracy += batch_accuracy
        
        wandb.log({"batch" : i,
                   "batch_accuracy": batch_accuracy,
                   "cumulative_accuracy": accuracy })
        

    test_loss = np.mean(all_losses)
    wandb.log({"test_loss": test_loss})



def read_datasets(split, data_dir, context_length, batch_size, rng):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_train_{config.tokenizer_name_str}_seq_len_{config.context_length}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    elif split == 'test':
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_test_{config.tokenizer_name_str}_seq_len_{config.context_length}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_val_{config.tokenizer_name_str}_seq_len_{config.context_length}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    
    ix = rng.integers(len(data) - context_length, size=(batch_size,))
    
    
    x = np.stack([(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = np.stack([(data[i+1:i+1+context_length].astype(np.int64)) for i in ix])
    
    # Load batches directly to GPU memory
    x = cp.asarray(x)
    y = cp.asarray(y)

    return x, y



def compute_gradient(target, prediction, one_hot_lookup):

    target = cp.stack([one_hot_lookup[token] for token in target])

    return (prediction - target), target


def get_log_output_table(log_output_buffer: deque) -> Table:

    table = Table()

    table.add_column('Time', style='cyan', no_wrap=True)
    table.add_column('Epoch', style='cyan')
    table.add_column('Train loss', style='green')


    for timestamp, epoch, loss in log_output_buffer:
        table.add_row(f'{timestamp}', f'{epoch}', f'{loss:.5e}')

    return table

def main():
    # Argument parsing and setup
    parser = argparse.ArgumentParser(description='NanoGPT from scratch')
    # (Arguments remain unchanged)

    args = parser.parse_args()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer = config.tokenizer_name(params=config.vocab_file)

    
     
    midi_paths = list(Path("/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/dataset_train").glob("**/*.mid"))
    dataset = DatasetMIDI(
        files_paths=midi_paths,
        tokenizer=tokenizer,
        max_seq_len=config.context_length,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
    )
    collator = DataCollator(tokenizer.pad_token_id)
    data_loader = DataLoader(dataset=dataset, collate_fn=collator)


    
    # Initialize Weights & Biases (wandb)
    wandb.init(
        project="MIDI-Transformer",
        config={
            # W&B Configurations (unchanged)
        }
    )

    # Initialize model
    model = GoePT(
        context_length=config.context_length,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        dropout=config.dropout_rate,
        batch_size=config.batch_size,
        lr=config.learning_rate,
        vocab_size=tokenizer.vocab_size,
        n_heads=config.n_heads,
        regularization=config.regularization,
        reg_alpha=config.reg_alpha
    )

    # Create DataLoaders
    train_loader = create_dataloader('train', config.data_dir, config.context_length, config.batch_size)
    val_loader = create_dataloader('val', config.data_dir, config.context_length, config.batch_size)
    test_loader = create_dataloader('test', config.data_dir, config.context_length, config.batch_size)

    # Pre-generate one-hot vectors using the vocab size
    one_hot_lookup = cp.eye(tokenizer.vocab_size)
    iter_num = 0
    best_val_loss = 1e9

    print("[INFO] Starting training...")
    for epoch in range(config.epochs):
        print(f"[INFO] Epoch {epoch + 1}/{config.epochs} started.")
        epoch_loss = 0

        for batch in data_loader:
            print(batch)
            logits, loss = model.forward(X, Y)

            # Scale the loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps

            # Compute gradient
            raw_grad, target = compute_gradient(Y, logits, one_hot_lookup)
            grad = loss * raw_grad
            model.backward(grad)

            # Accumulate epoch loss
            epoch_loss += loss.item()

            # Log loss to W&B
            wandb.log({"loss": loss.item()})

            iter_num += 1

            # Print progress every `log_interval`
            if iter_num % config.log_interval == 0:
                print(f"[INFO] Iteration {iter_num}: Loss = {loss.item()}")

        # Update model after all micro-steps
        model.update()
        print(f"[INFO] Epoch {epoch + 1} training completed. Avg Loss: {epoch_loss / len(train_loader):.4f}")

        # Evaluation on Validation Set
        if epoch % config.eval_interval == 0:
            val_losses = []
            for X, Y in val_loader:
                logits, loss = model.forward(X, Y)
                val_losses.append(loss.item())

            val_loss_mean = np.mean(val_losses)
            print(f"[INFO] Validation Loss after Epoch {epoch + 1}: {val_loss_mean:.4f}")

            # Save model checkpoint if validation loss improves
            if val_loss_mean < best_val_loss:
                checkpoint_path = os.path.join(config.checkpoint_dir, f'{wandb.run.name}_{epoch}.json')
                with open(checkpoint_path, mode='w', encoding='utf-8') as out_file:
                    json.dump(model.state_dict(), out_file)

                print(f"[INFO] New best validation loss: {val_loss_mean:.4f}. Checkpoint saved at {checkpoint_path}")
                best_val_loss = val_loss_mean

            wandb.log({"val_loss": val_loss_mean, "epoch": epoch})

    print("[INFO] Training complete!")
    # Finish W&B logging
    wandb.finish()
if __name__ == '__main__':
    main()