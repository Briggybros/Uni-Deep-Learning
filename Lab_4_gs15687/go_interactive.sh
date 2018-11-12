#!/bin/bash
module add libs/tensorflow/1.2
srun -p gpu_veryshort --gres=gpu:1 -A comsm0018 --mem=4G --pty bash