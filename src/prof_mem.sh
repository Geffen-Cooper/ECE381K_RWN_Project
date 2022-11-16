#!/bin/bash

echo $(date +%s.%N)
python train_constrained.py --gnn=GCN --k=50 --dataset=cora --compression_rate=big &#> /dev/null 2>&1 &
jobid=$(echo $!)
while [[ -n $(jobs -r) ]]; do pmap $jobid | tail -n 1 | awk '/[0-9]K/{print $2}'; sleep 0.2; done
echo $(date +%s.%N)