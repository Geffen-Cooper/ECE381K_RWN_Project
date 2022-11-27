import matplotlib.pyplot as plt

num_flags = 9
flag_points = ["end","proc-start","pre-import","start-prog","load-data","load-model","forward","loss-val","backwards"]
cs = ['k','k','c','g','b','m','y','g','r']
flag_times = []
t = [0]

with open("logs/log_k1.txt") as file:
    lines = [line.rstrip() for line in file]


flag_times.append(float(lines[-1]))
lines.remove(lines[-1])
flag_times.append(float(lines[0]))
lines.remove(lines[0])


for flag in flag_points[2:num_flags]:
    flag_times.append(float(lines[lines.index(flag)+1].strip()))
    lines.remove(lines[lines.index(flag)+1])

mems = []
for idx,line in enumerate(lines):
    if line.strip() == "RES" and idx+2 < len(lines):
        m = lines[idx+1].strip()
        t.append(float(lines[idx+2].strip())-flag_times[1])
        if m[-1].isdigit():
            mems.append(float(m)/1000000)
        elif m[-1] == "m":
            mems.append(float(m[:-1])/1000)
        elif m[-1] == "g":
            mems.append(float(m[:-1]))

print(t)
plt.plot(t[:-1],mems)
plt.xlabel("sec")
plt.ylabel("GB")
print("mems",mems)
print("flag times",flag_times)

for idx, f in enumerate(flag_points[:num_flags]):
    if f == "proc-start" or f == "end":
        plt.axvline(x = flag_times[idx]-flag_times[1], color = 'k', label = f)
    else:
        plt.axvspan(flag_times[idx-1]-flag_times[1], flag_times[idx]-flag_times[1], alpha=0.3, color=cs[idx],label=f)
plt.legend()
plt.grid()
plt.show()