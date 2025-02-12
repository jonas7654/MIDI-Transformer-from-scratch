import sys
import os
import datetime
import argparse
from functools import partial
from collections import deque
import json
from miditok import REMI, Structured
import wandb
import cupy as cp
import numpy as np
from sklearn.metrics import root_mean_squared_error
from rich.progress import Progress
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from icecream import ic


sys.path.append('.')

from model import GoePT
os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"

def read_datasets(split, data_dir, context_length, batch_size, rng, config):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_train_{config.tokenizer_name}_seq_len_{config.context_length}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    elif split == 'test':
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_test_{config.tokenizer_name}_seq_len_{config.context_length}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_val_{config.tokenizer_name}_seq_len_{config.context_length}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    
    ix = rng.integers(len(data) - context_length, size=(batch_size,))
    
    
    x = np.stack([(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = np.stack([(data[i+1:i+1+context_length].astype(np.int64)) for i in ix])
    
    # Load batches directly to GPU memory
    x = cp.asarray(x)
    y = cp.asarray(y)

    return x, y




def compute_gradient(target, prediction, one_hot_lookup, reg = False, alpha = 0, batch_size = None, padding_token_idx = 0):
    target = cp.stack([one_hot_lookup[token] for token in target])
    cross_entropy_grad = prediction - target
    if (reg):
        # Compute gradient for cross-entropy loss
        

        # Compute gradient for padding penalty regularization
        batch_size = prediction.shape[0]
        padding_grad = cp.zeros_like(prediction)
        padding_grad[:, padding_token_idx] = alpha / batch_size

        # Combine gradients
        total_grad = cross_entropy_grad + padding_grad
        return total_grad, target
    

    return cross_entropy_grad, target

def get_log_output_table(log_output_buffer: deque) -> Table:
    table = Table()
    table.add_column('Time', style='cyan', no_wrap=True)
    table.add_column('Epoch', style='cyan')
    table.add_column('Train loss', style='green')
    for timestamp, epoch, loss in log_output_buffer:
        table.add_row(f'{timestamp}', f'{epoch}', f'{loss:.5e}')
    return table


def generate_vocab_file_path(tokenizer_name_str, vo_size, base_dir="/path/to/vocab/files"):

    return os.path.join(base_dir, f"tokenizer_{tokenizer_name_str}_{vo_size}.json")


def train(config=None):
    # Initialize Weights & Biases (wandb)
    wandb.init(
        project="MIDI-Transformer-parameter-search",
        config=config
    )
    config = wandb.config  # Access the dynamically passed config
    
    # Dynamically generate the vocab file path
    vocab_file = generate_vocab_file_path(
        tokenizer_name_str=config.tokenizer_name,
        vo_size=config.vo_size,
        base_dir="/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers"
    )
    
    # Update config with the generated vocab_file
    wandb.config.update({"vocab_file": vocab_file}, allow_val_change=True)

    tokenizer_classes = {"REMI": REMI, "Structured": Structured}
    # Initialize the tokenizer using the dynamically chosen vocab file
    tokenizer_name_string = config.tokenizer_name
    tokenizer_class = tokenizer_classes[tokenizer_name_string]    
    
    tokenizer = tokenizer_class(params = vocab_file)

    model = GoePT(context_length=config.context_length,
                  n_layer=config.n_layer,
                  n_embd=config.n_embd,
                  dropout=config.dropout_rate,
                  batch_size=config.batch_size,
                  lr=config.lr,
                  vocab_size = tokenizer.vocab_size,
                  n_heads = config.n_heads,
                  regularization = config.regularization,
                  reg_alpha = config.reg_alpha,
                  relative_attention = config.relative_attention)

    rng = np.random.default_rng(config.seed)
    cp.random.seed(config.seed)

    get_batch = partial(
        read_datasets,
        data_dir=config.data_dir,
        context_length=config.context_length,
        batch_size=config.batch_size,
        rng=rng,
        config=config
    )

    one_hot_lookup = cp.eye(tokenizer.vocab_size)

    iter_num = 0
    best_val_loss = 1e9

    status_console = Console()
    status = status_console.status('[bold green]Starting training...', spinner='runner')
    progress_step = Progress(transient=True)
    header_panel = Panel(Group(status, progress_step))

    log_output_buffer = deque([], maxlen=16)
    table_update_func = partial(get_log_output_table, log_output_buffer=log_output_buffer)

    with Live(header_panel):
        while True:
            status.update(f'[bold green]Training epoch {iter_num + 1} ...')

            task_id = progress_step.add_task('Training')

            for micro_step in progress_step.track(
                range(config.gradient_accumulation_steps),
                total=config.gradient_accumulation_steps,
                task_id=task_id
            ):
                X, Y = get_batch('train')
                logits, loss = model.forward(X, Y)
                loss = loss / config.gradient_accumulation_steps

                #with open('train_losses.csv', 'a') as f:
                #    f.write(f'{iter_num}\t{loss:.8f}\n')

                raw_grad, target = compute_gradient(Y, logits, one_hot_lookup, reg = config.regularization,
                                                    alpha = config.reg_alpha,
                                                    batch_size = model.batch_size,
                                                    padding_token_idx=0)
                grad = loss * raw_grad
                model.backward(grad)

                log_output_buffer.append(
                    (datetime.datetime.now().isoformat(), iter_num + 1, loss.item() * config.gradient_accumulation_steps)
                )

                progress_step.console.clear()
                progress_step.console.print(table_update_func())
                progress_step.advance(task_id)

                wandb.log({"loss": loss.item()})

            progress_step.remove_task(task_id)
            task_id = progress_step.add_task('Updating model')
            model.update()
            progress_step.remove_task(task_id)

            if iter_num % config.eval_interval == 0:
                losses_val = cp.zeros(config.eval_iters)
                task_id = progress_step.add_task(f'Val loss evaluation')

                for k in progress_step.track(range(config.eval_iters), total=config.eval_iters, task_id=task_id):
                    X, Y = get_batch('val')
                    logits, loss = model.forward(X, Y)
                    losses_val[k] = loss.item()

                progress_step.remove_task(task_id)
                loss_val_mean = losses_val.mean()

                if loss_val_mean < best_val_loss:
                    checkpoint_path = os.path.join(config.checkpoint_dir, f'{wandb.run.name}_{iter_num}.json')
                    state_dict = model.state_dict()
                    with open(checkpoint_path, mode='w', encoding='utf-8') as out_file:
                        json.dump(state_dict, out_file)

                    best_val_loss = loss_val_mean

                wandb.log({
                    "train_loss": float(loss),
                    "val_loss": float(loss_val_mean),
                    "iteration_number": iter_num
                })

            iter_num += 1
            if iter_num > config.epochs:
                break

    wandb.finish()


if __name__ == '__main__':
    sweep_config = {
        "method": "bayes",  # Options: bayes, grid, random
        "metric": {
            "name": "val_loss",  # Metric to optimize
            "goal": "minimize"   # Minimize validation loss
        },
        "parameters": {
            "lr": {"values": [0.1, 0.01, 0.0001, 0.0005, 0.00001]},  # Learning rate options
            "batch_size": {"values": [32]},  # Batch size options
            "n_layer": {"values": [4, 6, 8, 10, 12]},  # Number of layers
            "n_heads": {"values": [4, 8, 16]},  # Number of attention heads
            "n_embd": {"values": [256, 512, 1024]},  # Embedding size
            "dropout_rate": {"values": [0, 0.1, 0.2, 0.3, 0.4]},  # Dropout
            "epochs": {"value": 150},  # Fixed number of epochs
            "gradient_accumulation_steps": {"value": 32},  # Fixed value
            "context_length": {"values": [32, 42]},  # Fixed value
            "seed": {"value": 1},  # Random seed
            "data_dir": {"value":  "/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized/"},  # Fixed data dir
            "checkpoint_dir": {"value": "/csghome/hpdc04/Transformer_Code/checkpoints/"},  # Fixed checkpoint dir
            "vo_size": {"values": [2048]},  # Vocabulary size for tokenizer
            "tokenizer_name": {"values": ["REMI"]},  # Tokenizer class name
            "manually_set_sos_eos_trunc": {"values": [True]},
            "eval_interval": {"value": 5},
            "eval_iters" : {"value": 200},
            "log_interval" : {"value" : 5},
            "regularization" : {"values": [True, False]},
            "reg_alpha": {"values": [0, 0.05, 0.1, 0.2, 0.3, 1, 2, 3 ,4]},
            "relative_attention": {"values": [True, False]}
        }
    }
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="MIDI-Transformer-parameter-search")

    # Run the sweep
    wandb.agent(sweep_id, function=train)
