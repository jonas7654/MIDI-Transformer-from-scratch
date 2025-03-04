# MIDI-Transformer from Scratch
Note: all functions are specified to run on the `hpdc04` user on the cluster.
## How to use the model
- In order to execute this repository you have to download the clean_midi data first and save it under `CUPY/models/datasets/`.
- To pre process the data you need to run `run_scripts/transpose_midi.sh`
- Model configuration is specified within `CUPY/models/GoePT/config.py`.
- After specifying the configuration setting up the training- validation and testdataset can be done by executing `run_scripts/generate_train_val_test.sh`.
- To finally run the training process you can either run `run_scripts/run_train_in_console.sh`to start training in the current bash or `run_scripts/train_model.sh` which let slurm allocate ressources for the training process. The Training process is logged via `wandb`. 

- Model checkpoints will be saved to `checkpoints/`.
- To proceed with inference given a pre-trained model you can use the inference script in `run_scripts/inference.sh`. The following parameters can be specified:
    - weights: Path to the json-checkpoint
    - vocab-file: Path to the json-vocabulary corresponding to the pre-trained model
    - input: Path to the input midi files (default: `Inference/input/`)
    - b: maximum amount of tokens to be generated
    - save-dir: Path to a directory where the generated sequences will be saved (default: `Inference/output/`)
    - manually-set-sos-eos-trunc: default: true (This should not be changed)



Once the parameters are set, you can execute the script by running the following command in the terminal:  

```bash
./inference.sh -p -T
```
Example for $p = 0.15$ and $T = 0.7$:
```bash
./inference.sh 0.15 0.7
```
## Lakh Dataset
- http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz
