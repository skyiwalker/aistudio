#!/bin/sh
horovodrun -np 1 python /home/sky/dev/aistudio/workspace/ws-1/jobs/job-1/train.py --epochs 5 --batch-size 64 --test-batch-size 128 --lr 0.01 --momentum 0.5 --seed 42 --log-interval 10 --nprocs 1 --loss nll_loss --optimizer SGD --debug --net-name hymenoptera --dataset-loader hymenoptera           
