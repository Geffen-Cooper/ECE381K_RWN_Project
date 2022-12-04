'''
    This file parses the memory profiling log and
    creates a plot to visualize the usage over time
'''

import matplotlib.pyplot as plt
import matplotlib
import argparse
import numpy as np
matplotlib.rcParams.update({'font.size': 22})

def visualize_mem(args):
    # memory checkpoints
    num_flags = 7
    flag_points = ["start","end","pre-import","import","load-data","load-model","forward"]
    cs = ['k','r','c','g','b','m','y']
    flag_times = []
    t = [0]

    # open the log file
    if args.ts == "s":
        log_file = "logs/logmem_"+args.gnn+"_"+args.dataset+"_"+args.ts+"_"+args.compression_rate+"_"+str(args.k)+".txt"
    else:
        log_file = "logs/logmem_"+args.gnn+"_"+args.dataset+"_"+args.ts+"_"+str(args.k)+".txt"
    with open(log_file) as file:
        lines = [line.rstrip() for line in file]

    # parse the start and end times
    flag_times.append(float(lines[0]))
    lines.remove(lines[0])
    flag_times.append(float(lines[-1]))
    lines.remove(lines[-1])

    # parse the checkpoints
    for flag in flag_points[2:num_flags]:
        flag_times.append(float(lines[lines.index(flag)+1].strip()))
        lines.remove(lines[lines.index(flag)+1])

    # parse the memory usage
    mems = []
    for idx,line in enumerate(lines):
        if line.strip() == "RES" and idx+2 < len(lines):
            m = lines[idx+1].strip()
            try:
                t.append(float(lines[idx+2].strip())-flag_times[0])
            except:
                t.append(float(lines[idx+3].strip())-flag_times[0])
            if m[-1].isdigit():
                mems.append(float(m)/1000000)
            elif m[-1] == "m":
                mems.append(float(m[:-1])/1000)
            elif m[-1] == "g":
                mems.append(float(m[:-1]))

    plt.plot(t[:-1],mems)
    plt.xlabel("sec")
    plt.ylabel("GB")

    for idx, f in enumerate(flag_points[:num_flags]):
        if f == "start":
            plt.axvline(x = flag_times[idx]-flag_times[0], color = cs[idx], label = f)
        elif f == "end":
            plt.axvline(x = flag_times[idx]-flag_times[0], color = cs[idx], label = f)
        elif f == "pre-import":
            plt.axvspan(flag_times[idx-2]-flag_times[0], flag_times[idx]-flag_times[0], alpha=0.3, color=cs[idx],label=f)
        else:
            plt.axvspan(flag_times[idx-1]-flag_times[0], flag_times[idx]-flag_times[0], alpha=0.3, color=cs[idx],label=f)
    plt.legend()
    plt.yticks(np.arange(0, max(mems), 0.05))
    plt.grid()

    if args.compression_rate == "big":
        size = "small"
    if args.compression_rate == "small":
        size = "small"

    # full graph vs parallel
    if args.parallel == "y":
        if args.ts == "s":
            plt.title("Memory usage during inference on "+args.dataset+" ("+size +" student, 1 out of "+ str(args.k)+ " partitions)")
        else:
            plt.title("Memory usage during inference on "+args.dataset+" (teacher, 1 out of "+ str(args.k)+ " partitions)")
    else:
        if args.ts == "s":
            plt.title("Memory usage during inference on "+args.dataset+" ("+size +" student, Full Graph)")
        else:
            plt.title("Memory usage during inference on "+args.dataset+" (teacher, Full Graph)")
    plt.show()



# ===================================== Command Line Arguments =====================================
def parse_args():
    parser = argparse.ArgumentParser(description="Training and Evaluation")

    # logging details
    parser.add_argument("--gnn",help="GNN architecture (GCN, GAT, GSAGE)",type=str, default="GCN")
    parser.add_argument("--k",help="how many partitions to split the input graph into",type=int, default=2)
    parser.add_argument("--dataset",help="name of the dataset (cora, citeseeor,arxiv)",type=str, default="cora")
    parser.add_argument("--compression_rate", help="compression rate for KD (big, medium, small)", type=str, default="medium")
    parser.add_argument("--ts",help="is the model teacher or student (t or s)",type=str, default="t")
    parser.add_argument("--parallel", help="if doing inference on a single core/partition (y,n)",type=str,default="n")

    args = parser.parse_args()
    print(args)
    return args


# ===================================== Main =====================================
if __name__ == "__main__":
    args = parse_args()
    visualize_mem(args)
