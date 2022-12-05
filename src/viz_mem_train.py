import matplotlib.pyplot as plt
import matplotlib
import argparse
matplotlib.rcParams.update({'font.size': 22})

def visualize_mem(args):
    t = []

    # open the log file
    if args.ts == "s":
        log_file = "logs/log_mem_train_"+args.gnn+"_"+args.dataset+"_"+args.ts+"_"+args.compression_rate+"_"+str(args.k)+".txt"
    else:
        log_file = "logs/log_mem_train_"+args.gnn+"_"+args.dataset+"_"+args.ts+"_"+str(args.k)+".txt"
    with open(log_file) as file:
        lines = [line.rstrip() for line in file]


    t.append(float(lines[0]))
    lines.remove(lines[0])

    mems = []
    for idx,line in enumerate(lines):
        if line.strip() == "RES" and idx+2 < len(lines):
            if lines[idx+1].strip()[0:4]=="1670":
                continue
            m = lines[idx+1].strip()
            try:
                t.append(float(lines[idx+2].strip())-t[0])
            except:
                continue
            if m[-1].isdigit():
                mems.append(float(m)/1000000)
            elif m[-1] == "m":
                mems.append(float(m[:-1])/1000)
            elif m[-1] == "g":
                mems.append(float(m[:-1]))
    t.append(float(lines[-1])-t[0])
    lines.remove(lines[-1])
    # print(t)
    t[0] = 0


    plt.plot(t[:-3],mems[:-1])
    plt.xlabel("sec")
    plt.ylabel("GB")

    if args.compression_rate == "big":
        size = "small"
    if args.compression_rate == "small":
        size = "small"

    if args.parallel == "y":
        if args.ts == "s":
            plt.title("Memory usage during training on "+args.dataset+" ("+size +" student, 1 out of "+ str(args.k)+ " partitions)")
        else:
            plt.title("Memory usage during training on "+args.dataset+" (teacher, 1 out of "+ str(args.k)+ " partitions)")
    else:
        if args.ts == "s":
            plt.title("Memory usage during training on "+args.dataset+" ("+size +" student, Full Graph)")
        else:
            plt.title("Memory usage during training on "+args.dataset+" (teacher, Full Graph)")
    # print("mems",mems)
    # print(t)



    # t = []

    # with open("logs/log_k1_ts.txt") as file:
    #     lines = [line.rstrip() for line in file]


    # t.append(float(lines[0]))
    # lines.remove(lines[0])


    # mems = []
    # for idx,line in enumerate(lines):
    #     if line.strip() == "RES" and idx+2 < len(lines):
    #         m = lines[idx+1].strip()
    #         t.append(float(lines[idx+2].strip())-t[0])
    #         if m[-1].isdigit():
    #             mems.append(float(m)/1000000)
    #         elif m[-1] == "m":
    #             mems.append(float(m[:-1])/1000)
    #         elif m[-1] == "g":
    #             mems.append(float(m[:-1]))
    # t.append(float(lines[-1])-t[0])
    # lines.remove(lines[-1])
    # # print(t)
    # t[0] = 0




    # plt.plot(t[:-3],mems[:-1],label="Teacher and Student, Full Graph")
    # plt.xlabel("sec")
    # plt.ylabel("GB")
    # # plt.xticks([0,50,100,150,200,250,300,350,400])
    # # plt.title("Memory usage when training ogbn-arxiv on CPU (teacher only)")
    # print("mems",mems)
    # print(t)


    # t = []

    # with open("logs/log_k10_t2.txt") as file:
    #     lines = [line.rstrip() for line in file]


    # t.append(float(lines[0]))
    # lines.remove(lines[0])


    # mems = []
    # for idx,line in enumerate(lines):
    #     if line.strip() == "RES" and idx+2 < len(lines):
    #         m = lines[idx+1].strip()
    #         t.append(float(lines[idx+2].strip())-t[0])
    #         if m[-1].isdigit():
    #             mems.append(float(m)/1000000)
    #         elif m[-1] == "m":
    #             mems.append(float(m[:-1])/1000)
    #         elif m[-1] == "g":
    #             mems.append(float(m[:-1]))
    # t.append(float(lines[-1])-t[0])
    # lines.remove(lines[-1])
    # # print(t)
    # t[0] = 0




    # plt.plot(t[:-3],mems[:-1],label="Teacher, 1 out of 10 partitions/processes")
    # plt.xlabel("sec")
    # plt.ylabel("GB")
    # # plt.xticks([0,50,100,150,200,250,300,350,400])
    # # plt.title("Memory usage when training ogbn-arxiv")
    # print("mems",mems)
    # print(t)

    # t = []

    # with open("logs/log_k10_ts2.txt") as file:
    #     lines = [line.rstrip() for line in file]


    # t.append(float(lines[0]))
    # lines.remove(lines[0])


    # mems = []
    # for idx,line in enumerate(lines):
    #     if line.strip() == "RES" and idx+2 < len(lines):
    #         m = lines[idx+1].strip()
    #         t.append(float(lines[idx+2].strip())-t[0])
    #         if m[-1].isdigit():
    #             mems.append(float(m)/1000000)
    #         elif m[-1] == "m":
    #             mems.append(float(m[:-1])/1000)
    #         elif m[-1] == "g":
    #             mems.append(float(m[:-1]))
    # t.append(float(lines[-1])-t[0])
    # lines.remove(lines[-1])
    # # print(t)
    # t[0] = 0




    # plt.plot(t[:-3],mems[:-1],label="Teacher and Student, 1 out of 10 partitions/processes")
    # plt.xlabel("sec")
    # plt.ylabel("GB")
    # # plt.xticks([0,50,100,150,200,250,300,350,400])
    # # plt.title("Memory usage when training ogbn-arxiv on CPU (teacher only)")
    # print("mems",mems)
    # print(t)

    # # for idx, f in enumerate(flag_points[:num_flags]):
    # #     if f == "proc-start" or f == "end":
    # #         plt.axvline(x = flag_times[idx]-flag_times[1], color = 'k', label = f)
    # #     else:
    # #         plt.axvspan(flag_times[idx-1]-flag_times[1], flag_times[idx]-flag_times[1], alpha=0.3, color=cs[idx],label=f)
    # # plt.legend()
    plt.grid()
    plt.legend()
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

