# Dynamic Reward: Stabilizing Long Chain-of-Thought Reinforcement Learning

Paper: [Link](https://www.zichenz.me/project/dynamic_reward/dynamic_reward.pdf)

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

For better evaluation, also install `symeval`:
```Shell
pip install "git+https://github.com/tongyx361/symeval.git"
```
