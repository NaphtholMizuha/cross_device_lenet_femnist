#!/bin/bash
gnome-terminal --tab --title=close.sh -- bash -c "cd /home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2/bashes && ./close.sh >close.log;exec bash"
gnome-terminal --tab --title=redis.sh -- bash -c "cd /home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2/bashes && ./redis.sh >redis.log;exec bash"
gnome-terminal --tab --title=restart.sh -- bash -c "cd /home/qiyijie/git_projects/mindspore/federated/example/cross_device_albert2/bashes && ./restart.sh >restart.log;exec bash"
