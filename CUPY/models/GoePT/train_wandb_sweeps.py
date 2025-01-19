import sys
import os
import datetime
import argparse
from functools import partial
from collections import deque
import json
from miditok import REMI
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

ic.configureOutput(includeContext=True)
ic.disable()
WANDB_AGENT_DISABLE_FLAPPING=True
# Define WandB sweep configuration
sweep_config = {
    'method': 'bayes',  # Can also be 'grid' or 'random'
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'  
    },
    'parameters': {
        'context_length': {'values': [512]},
        'batch_size': {'values': [10, 16, 32]},
        'n_layer': {'values': [4, 6, 8, 10]},  # Hyperparameter search for the number of layers
        'n_embd': {'values': [256, 384, 512]},  # Hyperparameter search for embedding size
        'n_heads': {'values': [4, 6, 8]},  # Hyperparameter search for attention heads
        'dropout': {'values': [0.1, 0.2, 0.3, 0.4]},  # Hyperparameter search for dropout
        'lr': {'distribution': 'log_uniform_values', 'min': 0.1, 'max': 0.9},  # Learning rate search
        'epochs': {'value': 14},  # Fixed value for training duration
        'gradient_accumulation_steps': {'value': 32},  # Fixed value
        'eval_iters': {'value': 200},  # Fixed value
        'seed': {'value': 1},  # Fixed random seed
        'vocab_file': {'value': '/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_4096.json'},
        'data_dir': {'value': '/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized'},
        'checkpoint_dir': {'value': '/csghome/hpdc04/Transformer_Code/checkpoints'},
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,  # Minimum number of epochs before considering early stopping
        'eta': 2,      # Halve the number of runs at each bracket
        's': 1         # Number of brackets
    }
}


def sweep_train():
    wandb.init()
    config = wandb.config

    tokenizer = REMI(params=config.vocab_file)

    model = GoePT(
        context_length=config.context_length,
        n_layer=config.n_layer,
        n_embd=config.n_embd,
        dropout=config.dropout,
        batch_size=config.batch_size,
        lr=config.lr,
        vocab_size=tokenizer.vocab_size,
        n_heads=config.n_heads
    )

    rng = np.random.default_rng(config.seed)
    cp.random.seed(config.seed)

    get_batch = partial(
        read_datasets,
        data_dir=config.data_dir,
        context_length=config.context_length,
        batch_size=config.batch_size,
        rng=rng
    )

    one_hot_lookup = cp.eye(tokenizer.vocab_size)

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        for step in range(config.gradient_accumulation_steps):
            X, Y = get_batch('train')
            logits, loss = model.forward(X, Y)
            loss /= config.gradient_accumulation_steps

            raw_grad, _ = compute_gradient(Y, logits, one_hot_lookup)
            grad = loss * raw_grad
            model.backward(grad)

            wandb.log({"train_loss": loss.item()})

        # Validation loop
        val_losses = []
        for _ in range(config.eval_iters):
            X, Y = get_batch('val')
            _, loss = model.forward(X, Y)
            val_losses.append(loss.item())

        val_loss_mean = np.mean(val_losses)
        wandb.log({"val_loss": val_loss_mean})

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            wandb.log({"best_val_loss": best_val_loss})

            checkpoint_path = os.path.join(config.checkpoint_dir, f'model_{epoch}.json')
            with open(checkpoint_path, 'w') as f:
                json.dump(model.state_dict(), f)

    wandb.finish()


def read_datasets(split, data_dir, context_length, batch_size, rng):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = rng.integers(len(data) - context_length, size=(batch_size,))

    x = np.stack([data[i:i + context_length].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1:i + 1 + context_length].astype(np.int64) for i in ix])

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


if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project="MIDI-Transformer-parameter-search")
    wandb.agent(sweep_id, sweep_train)

