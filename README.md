# Dynamic Reward: Stabilizing Long Chain-of-Thought Reinforcement Learning

## Run the Project

### Initialize the Environment

If running on `Great Lakes`, make sure you have loaded `Python` version 3.11. For example,
```Shell
module load python3.11-anaconda/2024.02
```

Then, activate conda:
```Shell
conda env create -f environment.yml
conda activate dynamic_penalty
```

Use `pip` in conda to install additional dependencies:
```Shell
pip install -r requirements.txt
```

There's some bug in the stable version of unsloth-zoo, making it fail to work on V100. Please install the newest version from source to fix this issue:
```Shell
pip install --force-reinstall --no-deps --no-cache-dir git+https://github.com/unslothai/unsloth-zoo.git
```

## Documentations
Google doc: https://docs.google.com/document/d/1artwHjWP9XIchUnj2lUcb-06TveEp_3qNH-j8gaT_fc/edit?tab=t.0
