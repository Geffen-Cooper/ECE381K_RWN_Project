#!/bin/bash

echo $(date +%s.%N)
python useless.py & #> /dev/null 2>&1 &
jobid=$(echo $!)
# while [[ -n $(jobs -r) ]]; do pmap $jobid | tail -n 1 | awk '/[0-9]K/{print $2}'; sleep 0.2; done
while [[ -n $(jobs -r) ]]; do top -bn1 -p $jobid | awk 'NR>6 {printf "%6s\n",$6}'; done
echo $(date +%s.%N)