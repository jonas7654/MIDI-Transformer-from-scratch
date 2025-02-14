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
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_train_{config.tokenizer_name_str}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    elif split == 'test':
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_test_{config.tokenizer_name_str}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, f'{config.vo_size}_val_{config.tokenizer_name_str}_manual_tokens_{config.manually_set_sos_eos_trunc}.bin'), dtype=np.uint16, mode='r')
    
    ix = rng.integers(len(data) - context_length, size=(batch_size,))
    
    
    x = np.stack([(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = np.stack([(data[i+1:i+1+context_length].astype(np.int64)) for i in ix])
    
    # Load batches directly to GPU memory
    x = cp.asarray(x)
    y = cp.asarray(y)

    return x, y


def compute_gradient(target, prediction, one_hot_lookup, alpha = config.reg_alpha, batch_size = config.batch_size, padding_token_idx = 0):
    target = cp.stack([one_hot_lookup[token] for token in target])
    cross_entropy_grad = prediction - target
    if (config.regularization):
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


def main():
        # Training settings
    parser = argparse.ArgumentParser(description='NanoGPT from scratch')
    parser.add_argument('--data-dir', type=str,
                            default='/datasets/tokenized',
                            help='Dataset directory')
    parser.add_argument('--checkpoint-dir', type=str,
                                default='checkpoints/',
                                help='Checkpoint directory')
    parser.add_argument('--vocab-file', type=str,
                                default='tokenizer.json',
                                help='Vocabulary file')

    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--context-length', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=32, metavar='N')
    parser.add_argument('--eval-iters', type=int, default=200, metavar='N')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--eval-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-heads', type=int, default =6, metavar='N',
                       help = "number of attention heads in Multiheadattention")
    parser.add_argument('--dropout', type=float,  default=0.2, metavar='N',
                        help='Specify dropout rate')
    parser.add_argument('--n-layer', type=int, default=6, metavar='N')
    parser.add_argument('--n-embd', type=int, default=384, metavar='N')

    args = parser.parse_args()

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    tokenizer = config.tokenizer_name(params = config.vocab_file)

    # Initialize Weights & Biases (wandb)
    wandb.init(
        project=f"MIDI-Transformer", 
        config={
            "data_dir": config.data_dir,
            "checkpoint_dir": config.checkpoint_dir,
            "vocab_file": config.vocab_file,
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "epochs": config.epochs,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "eval_iters": config.eval_iters,
            "lr": config.learning_rate,
            "seed": config.seed,
            "log_interval": config.log_interval,
            "eval_interval": config.eval_interval,
            "dropout rate": config.dropout_rate,
            "vocab_size": tokenizer.vocab_size,
            "n_layer" : config.n_layer,
            "n_embd" : config.n_embd,
            "n_heads" : config.n_heads,
            "manually_set_sos_eos_trunc": config.manually_set_sos_eos_trunc,
            "tokenizer": config.tokenizer_name_str,
            "regularization" : config.regularization,
            "reg_alpha" : config.reg_alpha,
            "relative_attention": config.relative_attention,
            "use_lr_decay" : config.use_decay,
            "decay_rate" : config.decay_rate,
            "decay_intervals" : config.decay_interval
        }
    )
    
    
    model = GoePT(context_length=config.context_length,
                  n_layer=config.n_layer,
                  n_embd=config.n_embd,
                  dropout=config.dropout_rate,
                  batch_size=config.batch_size,
                  lr=config.learning_rate,
                  vocab_size = tokenizer.vocab_size,
                  n_heads = config.n_heads,
                  regularization = config.regularization,
                  reg_alpha = config.reg_alpha,
                  relative_attention = config.relative_attention)

    # state_dict = model.state_dict()
    # with open(os.path.join(args.checkpoint_dir, 'test_checkpoint.json'), mode='w', encoding='utf-8') as out_file:
    #     json.dump(state_dict, out_file)
    # with open(os.path.join(args.checkpoint_dir, 'test_checkpoint.json'), mode='r', encoding='utf-8') as in_file:
    #     state_dict = json.load(in_file)
    # model_loaded = GoePT.from_state_dict(state_dict)
    # ic(model_loaded)
    # exit()

    # training loop

    rng = np.random.default_rng(config.seed)
    cp.random.seed(config.seed)

    get_batch = partial(read_datasets,
                            data_dir=config.data_dir,
                            context_length=config.context_length,
                            batch_size=config.batch_size,
                            rng=rng)

    # Pre-generate one-hot vectors using the vocab size
    # for gradient computation
    one_hot_lookup = cp.eye(tokenizer.vocab_size)

    iter_num = 0

    best_val_loss = 1e9

    status_console = Console()
    status = status_console.status('[bold green]Starting training...', spinner='runner')
    progress_step = Progress(transient=True)
    header_panel = Panel(Group(status, progress_step))

    log_output_buffer = deque([], maxlen=16)

    table_update_func = partial(get_log_output_table,
                                    log_output_buffer=log_output_buffer)


    # Initialize learning rate decay parameters
    learning_rate = config.learning_rate
    decay_rate = config.decay_rate  # Example exponential decay
    decay_interval = config.decay_interval  # Decay every epoch
    min_lr = 0.0001
    # with status_console.screen():
    with Live(header_panel):

        while True:
            # progress_step.console.print(f'Starting epoch: {iter_num + 1}')
            status.update(f'[bold green]Training epoch {iter_num + 1} ...')

            task_id = progress_step.add_task('Training')

            for micro_step in progress_step.track(range(config.gradient_accumulation_steps),
                                                total=config.gradient_accumulation_steps,
                                                task_id=task_id):

                X, Y = get_batch('train')

                logits, loss = model.forward(X, Y)

                # Scale the loss to account for gradient accumulation
                loss = loss/config.gradient_accumulation_steps

                with open('train_losses.csv', 'a') as f:
                    f.write(f'{iter_num}\t{loss:.8f}\n')

                # Get raw gradient
                raw_grad, target = compute_gradient(Y, logits, one_hot_lookup)

                # Continue backward
                grad = loss*raw_grad

                model.backward(grad)

                log_output_buffer.append((datetime.datetime.now().isoformat(), iter_num + 1, loss.item()*config.gradient_accumulation_steps,
                                          ))

                progress_step.console.clear()
                progress_step.console.print(table_update_func())
                progress_step.advance(task_id)
                
                wandb.log({"loss": loss.item()})

            progress_step.remove_task(task_id)

            task_id = progress_step.add_task('Updating model')

            model.update()
            
            # Apply learning rate decay
            if config.use_decay:
                current_step = iter_num % decay_interval  # For cyclic restarts
                progress = cp.array(current_step / decay_interval, dtype=cp.float32)  # Convert to CuPy array
                new_lr = min_lr + 0.5 * (learning_rate - min_lr) * (1 + cp.cos(cp.pi * progress))
                new_lr = float(new_lr)  
                model.setLR(new_lr)
                wandb.log({"learning_rate": new_lr})

            progress_step.remove_task(task_id)

            # Evaluate the loss on train/val sets and write checkpoints

            if iter_num % config.eval_interval == 0:

                losses_val = cp.zeros(config.eval_iters)

                task_id = progress_step.add_task(f'Val loss evaluation')

                for k in progress_step.track(range(config.eval_iters),
                                            total=config.eval_iters,
                                            task_id=task_id):

                    X, Y = get_batch('val')

                    logits, loss = model.forward(X, Y)

                    losses_val[k] = loss.item()

                    progress_step.advance(task_id)

                progress_step.remove_task(task_id)

                loss_val_mean = losses_val.mean()

                if loss_val_mean < best_val_loss:

                    status_update_string = f'Val loss decreased from {best_val_loss:.4f} to {loss_val_mean:.4f}'

                    status_update_string += '. Saving checkpoint...'

                    status.update(status_update_string)

                    checkpoint_path = os.path.join(config.checkpoint_dir, f'{wandb.run.name}_{iter_num}.json')

                    state_dict = model.state_dict()

                    with open(checkpoint_path, mode='w', encoding='utf-8') as out_file:
                        json.dump(state_dict, out_file)

                    status.update(f'Saved checkpoint under {checkpoint_path}')

                    best_val_loss = loss_val_mean
                # Log evaluation metrics to W&B
                wandb.log({
                            "train_loss": float(loss),
                            "val_loss": float(loss_val_mean),
                            "iteration_number": iter_num
                          })
                
            iter_num += 1

            # termination conditions
            if iter_num > config.epochs:
                break
 
 # Finish W&B logging
wandb.finish()
if __name__ == '__main__':
    main()