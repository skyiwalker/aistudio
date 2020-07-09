#!/bin/sh
horovodrun -np 1 python /home/sky/dev/aistudio/workspace/ws-1/jobs/job-1/train.py --epochs 5 --batch-size 32 --test-batch-size 64 --lr 0.01 --momentum 0.9 --seed 42 --log-interval 10 --nprocs 1 --loss cross_entropy --optimizer SGD --debug --validation --net-name CIFAR10-CNN --dataset-loader CIFAR10           
