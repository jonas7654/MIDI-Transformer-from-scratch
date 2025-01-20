# Transformer Model from Scratch

## Repository Contents
- **Core Model Code**: Python files implementing the Transformer model.
- **Shell Scripts**:
  - `train_model.sh`: Script for submitting the job via `sbatch`.
  - - Use `parameters.sh` in order to set the model parameters.
  - - After submitting you can use `watch squeue -u $USER` in order to track the running time.
    - Within the file `logs/model_py_GPU.log` are the console outputs
  * In order to train the tokenizer we can do `./generate_tokenizer`
  * To create tokenized train, validation and test datasets with that tokenizer we can do `./generate_train_val_test_set`
  - - IMPORTANT: in `CUPY/models/GoePT` there is a `config.py` file where we have to specify if we want to manually add SOS, EOS and a truncate token to the data.

## Wandb
- Currently logging is processed here: https://wandb.ai/me322-university-of-heidelberg/GoePT-Training?nw=nwuserme322

## Lakh Dataset
- http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz