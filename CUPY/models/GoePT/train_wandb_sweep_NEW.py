import sys
import os
import datetime
import argparse
from functools import partial
from collections import deque
from types import NoneType
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
import config
from model import GoePT

ic.configureOutput(includeContext=True)
ic.disable()

def read_datasets(split, data_dir, context_length, batch_size, rng):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    ix = rng.integers(len(data) - context_length, size=(batch_size,))
    x = np.stack([(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = np.stack([(data[i+1:i+1+context_length].astype(np.int64)) for i in ix])
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

def train():
    wandb.init(project="MIDI-Transformer-parameter-search")
    config = wandb.config  # Load hyperparameters dynamically from the sweep config

    os.makedirs(config.checkpoint_dir, exist_ok=True)
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

    get_batch = partial(read_datasets,
                        data_dir=config.data_dir,
                        context_length=config.context_length,
                        batch_size=config.batch_size,
                        rng=rng)

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

            for micro_step in progress_step.track(range(config.gradient_accumulation_steps),
                                                  total=config.gradient_accumulation_steps,
                                                  task_id=task_id):
                X, Y = get_batch('train')
                logits, loss = model.forward(X, Y)
                loss = loss / config.gradient_accumulation_steps

                raw_grad, target = compute_gradient(Y, logits, one_hot_lookup)
                grad = loss * raw_grad
                model.backward(grad)

                log_output_buffer.append((datetime.datetime.now().isoformat(), iter_num + 1, loss.item() * config.gradient_accumulation_steps))
                wandb.log({"loss": loss.item()})
            
            progress_step.remove_task(task_id)
            model.update()

            if iter_num % config.eval_interval == 0:
                losses_val = cp.zeros(config.eval_iters)
                task_id = progress_step.add_task(f'Val loss evaluation')

                for k in progress_step.track(range(config.eval_iters),
                                             total=config.eval_iters,
                                             task_id=task_id):
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
                wandb.log({"val_loss": loss_val_mean})
            
            iter_num += 1
            if iter_num > config.epochs:
                break

    wandb.finish()

if __name__ == '__main__':
    sweep_config = {
        "method": "grid",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
        'context_length': {'values': [1024]},
        'batch_size': {'values': [6]},
        'n_layer': {'values': [8]},  # Hyperparameter search for the number of layers
        'n_embd': {'values': [256]},  # Hyperparameter search for embedding size
        'n_heads': {'values': [4]},  # Hyperparameter search for attention heads
        'dropout': {'values': [0.05]},  # Hyperparameter search for dropout
        'lr': {'values': [0.01]},  # Learning rate search
        'epochs': {'value': 100},  # Fixed value for training duration
        'gradient_accumulation_steps': {'value': 32},  # Fixed value
        'eval_iters': {'value': 200},  # Fixed value
        'seed': {'value': 1},  # Fixed random seed
        'vocab_file': {'value': '/csghome/hpdc04/Transformer_Code/CUPY/models/tokenizers/tokenizer_512.json'},
        'data_dir': {'value': '/csghome/hpdc04/Transformer_Code/CUPY/models/datasets/tokenized'},
        'checkpoint_dir': {'value': '/csghome/hpdc04/Transformer_Code/checkpoints'},
        'manually_set_sos_eos_trunc' : {'value': config.manually_set_sos_eos_trunc}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="MIDI-Transformer-parameter-search")
    wandb.agent(sweep_id, function=train)
