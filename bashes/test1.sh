#!/bin/bash
source /home/qiyijie/anaconda3/etc/profile.d/conda.sh;
conda activate mindspore;

cd /home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2;
python finish_cloud.py --redis_port=23458; 
cd /home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2/fl_ckpt;
rm MLP_recovery_iteration_*;
cp /home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2/fl_ckpt/albert/MLP_recovery_iteration_0_220726_164030.ckpt /home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2/fl_ckpt/MLP_recovery_iteration_0_220726_164030.ckpt;

redis-server --port 23458 --save "";


