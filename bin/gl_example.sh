#!/bin/bash

#SBATCH --account=cse598s012w25_class

# TODO: Change this if needed
#SBATCH --time=8:00:00

#SBATCH --partition=gpu
#SBATCH --gpus=1

#SBATCH --mem=50g

# TODO: change these
#SBATCH --job-name=example_job
#SBATCH --output=example_output.out

# TODO: change this
#SBATCH --mail-user=UNIQNAME@umich.edu

#SBATCH --mail-type=BEGIN,END

module load python3.11-anaconda/2024.02
conda env create -f environment.yml
conda activate dynamic_penalty
pip install -r requirements.txt
pip install --force-reinstall --no-deps --no-cache-dir git+https://github.com/unslothai/unsloth-zoo.git

# TODO: change this
python train.py --project_name TODO_CHANGE_THIS --model_name TODO_CHANGE_THIS \
    --run_name TODO_CHANGE_THIS --reward_type TODO_CHANGE_THIS
