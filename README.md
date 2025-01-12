# Transformer Model from Scratch

## Repository Contents
- **Core Model Code**: Python files implementing the Transformer model.
- **Shell Scripts**:
  - `run_directly_in_console.sh`: Script for running the code directly in the console.
  - - Note when directly running in console you need to to do `conda activate eml`
  - `run_script_GPU.sh`: Script for submitting the job via `sbatch`.
  - - After you can use `watch squeue -u $USER` in order to track the running time.
    - Within the file `model_py_GPU.log` are the console outputs
  * In order to train the tokenizer we cann do `./generate_tokenizer`
  * To create tokenized train, validation and test datasets with that tokenizer we can do `./generate_train_val_test_set`

## Wandb
- Currently logging is processed here: https://wandb.ai/me322-university-of-heidelberg/GoePT-Training?nw=nwuserme322

## Lakh Dataset
- http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz