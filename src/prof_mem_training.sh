#!/bin/bash
gnn=$1
k=$2
dataset=$3
ts=$4
cr=$5
parallel=$6
student_only=$7

# create the log file
if [ "$ts" = "s" ]; then
    cr=$5
    log_file="logs/logmem_train_${gnn}_${dataset}_${ts}_${cr}_${k}.txt"
else
    cr="Big"
    log_file="logs/logmem_${gnn}_${dataset}_${ts}_${k}.txt"
fi

echo $log_file

# start time
echo $(date +%s.%N) > $log_file

# run training
if [ "$parallel" = "y" ]; then
    echo parallel
    python generate_partitions.py --dataset $dataset --k $k &
    jobid=$(echo $!)
    while [[ -n $(jobs -r) ]]; do top -bn1 -p $jobid | awk 'NR>6 {printf "%6s\n",$6}'>> $log_file; echo $(date +%s.%N) >> $log_file; done
    echo $(date +%s.%N) >> $log_file
    for (( c=0; c<$k; c++  ));
    do
        taskset -c $c python train_parallel.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --i=$(($c+1)) --student_only=$student_only >> $log_file &
    done
    jobid=$(echo $!)
    while [[ -n $(jobs -r) ]]; do top -bn1 -p $jobid | awk 'NR>6 {printf "%6s\n",$6}'>> $log_file; echo $(date +%s.%N) >> $log_file; done
    wait
    echo $(date +%s.%N) >> $log_file
else
    python train.py --gnn=$gnn --k=$k --dataset=$dataset --compression_rate=$cr --student_only=$student_only >> $log_file &
    jobid=$(echo $!)
    while [[ -n $(jobs -r) ]]; do top -bn1 -p $jobid | awk 'NR>6 {printf "%6s\n",$6}' >> $log_file; echo $(date +%s.%N) >> $log_file; done
fi

# end time
echo $(date +%s.%N) >> $log_file