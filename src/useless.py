import time 
time.sleep(1)
print("pre-import\n",time.time(),flush=True)
time.sleep(1)

import torch

print("start-prog\n", time.time())
time.sleep(1)

a = torch.rand((20,20), requires_grad=True)
b = torch.rand((20,20), requires_grad=True)
y = torch.tensor([1])
print("load-data\n", time.time(),flush=True)
time.sleep(1)

print("load-model\n", time.time(),flush=True)
time.sleep(1)

y_hat = torch.mean(a + b)
print("forward\n", time.time(),flush=True)
time.sleep(1)

L = y-y_hat
print("loss-val\n", time.time(),flush=True)
time.sleep(1)

# L.backward()
print("backwards\n", time.time(),flush=True)
time.sleep(1)

# import torch
# import torchvision.models as models
# from torch.profiler import profile, record_function, ProfilerActivity

# a = torch.tensor([1.], requires_grad=True)
# b = torch.tensor([2.], requires_grad=True)
# y = torch.tensor([1])


# with profile(activities=[ProfilerActivity.CPU],
#         profile_memory=True, record_shapes=True) as prof:
#         y_hat = a + b
#         L = y-y_hat
#         L.backward()

# print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))