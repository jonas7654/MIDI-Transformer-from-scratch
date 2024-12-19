# NN-transformer-from-scratch


Transformer Model from Scratch

This repository contains the implementation of a Transformer model written in Python, built from scratch. It is designed to be executed on clusters for efficient computation, with support for SLURM workload manager.

Repository Contents

Core Model Code: Python files implementing the Transformer model.

Shell Scripts:

run_directly_in_console.sh: Script for running the code directly in the console.

run_script_GPU.sh: Script for submitting jobs to a SLURM-managed cluster using sbatch.

Prerequisites

Ensure the following prerequisites are met:

Python 3.8 or later

Required Python packages (see requirements.txt)

Access to a compute cluster with SLURM (for run_script_GPU.sh)

Install dependencies:

pip install -r requirements.txt

Running the Code

Running Locally

To run the Transformer model directly in your console, use the run_directly_in_console.sh script:

bash run_directly_in_console.sh

This script is designed for local execution on a single machine.

Running on a SLURM Cluster

To submit the job to a SLURM-managed cluster, use the run_script_GPU.sh script. This script is optimized for GPU-based execution:

sbatch run_script_GPU.sh

Script Details

run_directly_in_console.sh: Directly executes the Python code locally.

run_script_GPU.sh: Submits a job to the SLURM workload manager. Ensure the SLURM configuration in the script matches your cluster’s requirements.

Customization

You can customize the execution scripts based on your specific requirements:

Modify the model parameters in the Python code.

Update SLURM-specific options (e.g., partition, GPUs, memory) in run_script_GPU.sh.

Example Output

After successful execution, the model outputs training logs and saves model checkpoints to the specified directory. Check the SLURM logs (slurm-<jobid>.out) for job details when using SLURM.

Troubleshooting

Ensure all required dependencies are installed.

Check SLURM configuration if jobs do not execute as expected.

Review error logs for Python exceptions or SLURM submission issues.

Contributing

Contributions are welcome! Please submit a pull request or open an issue if you encounter any problems.

License

This project is licensed under the MIT License. See the LICENSE file for details.

For additional help, please contact the repository maintainers.

