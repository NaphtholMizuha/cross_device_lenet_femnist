#!/bin/bash
source /home/qiyijie/anaconda3/etc/profile.d/conda.sh;
conda activate mindspore;

cd ../;
python run_sched.py --yaml_config='/home/qiyijie/git_projects/mindspore/federated/example/cross_device_user/yamls/albert/albert.yaml' --scheduler_manage_address="192.168.199.162:18023";
python run_server.py --yaml_config='/home/qiyijie/git_projects/mindspore/federated/example/cross_device_user/yamls/albert/albert.yaml' --tcp_server_ip='192.168.199.162' checkpoint_dir='/home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2/fl_ckpt' --local_server_num=1 --http_server_address='192.168.199.162:9023';

echo "########## finished ##########"



