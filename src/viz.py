import matplotlib.pyplot as plt

num_flags = 9
flag_points = ["end","proc-start","pre-import","start-prog","load-data","load-model","forward","loss-val","backwards"]
cs = ['k','k','c','g','b','m','y','g','r']
flag_times = []

with open("log_k1.txt") as file:
    lines = [line.rstrip() for line in file]


flag_times.append(float(lines[-1]))
lines.remove(lines[-1])
flag_times.append(float(lines[0]))
lines.remove(lines[0])


for flag in flag_points[2:num_flags]:
    flag_times.append(float(lines[lines.index(flag)+1][1:]))

mems = []
for idx,line in enumerate(lines):
    if line[0:2].isdigit():
        mems.append(int(line[:-1])/1000000)

t = [0]
for idx,line in enumerate(mems):
    t.append(t[idx]+0.2)
plt.plot(t[:-1],mems)
plt.xlabel("sec")
plt.ylabel("GB")


for idx, f in enumerate(flag_points[:num_flags]):
    if f == "proc-start" or f == "end":
        plt.axvline(x = flag_times[idx]-flag_times[1], color = 'k', label = f)
    else:
        plt.axvspan(flag_times[idx-1]-flag_times[1], flag_times[idx]-flag_times[1], alpha=0.3, color=cs[idx],label=f)
plt.legend()
plt.show()