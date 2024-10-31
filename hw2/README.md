# CS 8803-DRL Fall 2024 Homework 2

## Setup
To run on your machine, refer to the following instructions. To run on Colab, the instructions are included in the colab versions of the ipynb files.

Build and use the `cs8803drl` environment as follows. For this homework, we only support installation via conda environments

**NOTE:** Install the GPU version of PyTorch if preferred. If you already have the GPU version of PyTorch but would like the assignment code to use CPU, then set the `ONLY_CPU` flag to `True` under `src/utils.py`. Having a GPU may not provide a substantial performance boost for `hw2_offline.ipynb` but will be very helpful for MPC in `hw2_mbrl.ipynb`.


### MuJoCo installation
Installation instructions for Mac and Linux can be found [here](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco). You only need to do stuff in the section named 'Install MuJoCo'. 

For Windows users, we recommend either running with Google Colab. MuJoCo has been supported on Windows in the past, but that is deprecated and the TAs do not have the capacity to assist with getting this running.

### Conda environment
Install conda on your system and then run
```bash
conda env create --name cs8803drl_hw2 --file=environment.yaml
conda activate cs8803drl_hw2
```
