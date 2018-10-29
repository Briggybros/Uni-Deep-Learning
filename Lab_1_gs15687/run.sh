#!/bin/bash

module add libs/tensorflow/1.2

let ipnport=($UID-6025)%65274

ipnip=$(hostname -i)

echo "Copy this as the parameter from tensorboard.sh locally"
echo "$ipnip:$ipnport"

read -n 1 -s -r -p "Press any key to start"

python $1 & tensorboard --logdir=logs/ --port=$ipnport