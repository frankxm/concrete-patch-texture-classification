#!/bin/bash

INTERVAL=1  # 秒数，可以根据需要修改

while true; do
  clear
  echo "当前时间: $(date)"
  echo "你的任务队列状态："
  squeue --me --cluster=all
  free -h 
  sleep $INTERVAL
done

