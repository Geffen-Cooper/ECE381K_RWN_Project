#!/bin/bash
gnn=$1
k=$2
dataset=$3
ts=$4
cr=$5
parallel=$6

# create the log file
if [ "$ts" = "s" ]; then
    cr=$5
    log_file="logs/log_mem_inf_${gnn}_${dataset}_${ts}_${cr}_${k}.txt"
else
    cr="Big"
    log_file="logs/log_mem_inf_${gnn}_${dataset}_${ts}_${k}.txt"
fi

# if want to profile a single process
if [ "$parallel" = "y" ]; then
    # check if partitions already created
    FILE=graph_partitions/${dataset}_p1_k${k}.bin
    if test -f "$FILE"; then
        echo "partitions exists."
    else
        python generate_partitions.py --dataset $dataset --k $k
    fi
fi

echo $log_file

# start time
echo $(date +%s.%N) > $log_file

# run inference
if [ "$parallel" = "y" ]; then
    echo parallel
    taskset -c 0 python inference.py --gnn=$gnn --k=$k --ts=$ts --dataset=$dataset --compression_rate=$cr --parallel=$parallel --i=1 >> $log_file &
else
    python inference.py --gnn=$gnn --k=$k --ts=$ts --dataset=$dataset --compression_rate=$cr --parallel=$parallel >> $log_file &
fi

# log script
jobid=$(echo $!)
while [[ -n $(jobs -r) ]]; do top -bn1 -p $jobid | awk 'NR>6 {printf "%6s\n",$6}' >> $log_file; echo $(date +%s.%N) >> $log_file; done

# end time
echo $(date +%s.%N) >> $log_file

python viz_mem_inference.py --gnn=$gnn --k=$k --ts=$ts --dataset=$dataset --compression_rate=$cr --parallel=$parallel