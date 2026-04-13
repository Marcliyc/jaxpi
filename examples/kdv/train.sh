#!/bin/bash

# CUDA_VISIBLE_DEVICES=3 nohup python grid_sweep.py --config=configs/grid_sweep_soap.py > grid_sweep_soap.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python main.py --config=configs/soap.py > soap.log 2>&1 &

#CUDA_VISIBLE_DEVICES=3 nohup python main.py --config=configs/grid_128_3_48_256_2.py > 128_3_48_256_2.log 2>&1 &

export CUDA_VISIBLE_DEVICES=0

#python main.py --config=configs/gaussian2.py > gaussian2.log 2>&1

python main.py --config=configs/gaussian2_soap.py > gaussian2_soap.log 2>&1
