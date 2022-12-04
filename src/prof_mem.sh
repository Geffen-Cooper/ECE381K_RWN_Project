#!/bin/bash

echo $(date +%s.%N)
python train_no_student.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big &#> /dev/null 2>&1 &
# python train.py --gnn=GCN --k=1 --dataset=arxiv --compression_rate=big &

# python init_parallel.py --dataset arxiv --k 10
jobid=$(echo $!)
while [[ -n $(jobs -r) ]]; do top -bn1 -p $jobid | awk 'NR>6 {printf "%6s\n",$6}'; echo $(date +%s.%N); done

# taskset -c 0 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 1 &

# taskset -c 1 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 2 &

# taskset -c 2 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 3 &
# jobid=$(echo $!)
# taskset -c 3 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 4 &
# taskset -c 4 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 5 &
# taskset -c 5 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 6 &
# taskset -c 6 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 7 &
# taskset -c 7 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 8 &
# taskset -c 8 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 9 &
# taskset -c 9 python train_parallel.py --gnn=GCN --k=10 --dataset=arxiv --compression_rate=big --i 10 &

# while [[ -n $(jobs -r) ]]; do top -bn1 -p $jobid | awk 'NR>6 {printf "%6s\n",$6}'; echo $(date +%s.%N); done
# wait
echo $(date +%s.%N)