#!/bin/bash
srun --gpus=1\
 --mem-per-gpu=12G\
 --cpus-per-gpu=8\
 --qos=eaton-high\
 --time=12:10:00\
 --partition=eaton-compute \
 --container-mounts=/scratch:/scratch,/home/kvedder/code/flownet3d.pytorch:/project,/Datasets:/Datasets\
 --container-image=docker-registry.grasp.cluster#flownet3d \
bash -c "./flownet3d_train.py models/model_l1_length.pth"
