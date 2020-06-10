#!/bin/sh
horovodrun -np 1 python /home/sky/dev/aistudio/jobs/job-1/train.py --epochs 5 --batch-size 64 --test-batch-size 128 --lr 0.01 --momentum 0.5 --seed 42 --log-interval 10 --no-cuda False --nprocs 1 --loss cross_entropy --optimizer SGD --debug True --model-path /home/sky/dev/aistudio/models/model-1           
