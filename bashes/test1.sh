#!/bin/bash
source /home/qiyijie/anaconda3/etc/profile.d/conda.sh;
conda activate mindspore;

cd ../;
rm fl_ckpt/AutoEncoder_recovery_iteration_*;

python finish_cloud.py --redis_port=23459;

redis-server --port 23459 --save "";


