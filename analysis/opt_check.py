import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import argparse
sns.set()
sns.set_style('ticks')


params = {'legend.fontsize': 50,
        'figure.figsize': (54, 27),
        'axes.labelsize': 60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':36}


plt.rcParams.update(params)

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

fig, axs = plt.subplots(2,3)

colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6"]

init = []
with open(f'./../src/mrl_te_optimization/outputs/init_mrl_TE.txt') as f:
    scores = f.readlines()
    init = [float(score.replace('\n','')) for score in scores]

opt = []
with open(f'./../src/mrl_te_optimization/outputs/opt_mrl_TE.txt') as f:
    scores = f.readlines()
    opt = [float(score.replace('\n','')) for score in scores]

init = np.array(init)
opt = np.array(opt)

init = np.power(10,init)
opt = np.power(10,opt)

SORT_INIT = False

if SORT_INIT:

    min_init_indices = np.argsort(init)
    init = init[min_init_indices[:min(int(len(init)),100)]]
    opt = opt[min_init_indices[:min(int(len(opt)),100)]]
else:
    init = init[:min(int(len(init)),100)]
    opt = opt[:min(int(len(opt)),100)]

diffs = [(opt[i]-init[i]) for i in range(len(init))]

count = 0
for i in range(len(diffs)):
    if diffs[i] < 0:
        count += 1

print(count)

print(np.average(init))
print(np.average(opt))
print(np.max(diffs/init))
print(np.average(diffs/init))

indices_sorted = np.argsort(diffs)[::-1]

diffs = np.sort(diffs)[::-1]

new_inits = []

for i in range(len(diffs)):
    new_inits.append(init[indices_sorted[i]])

N = min(100, len(diffs))

step = 1.0/N

new_n = [i * step for i in range(N)]

width = step

plt.rcParams.update({'font.size': 12})


print(len(diffs))

axs[0,1].bar(x=new_n, bottom=0, width=width, height=diffs,color=colors[0])
axs[0,1].set_xticks([])
axs[0,1].set_ylabel('TE Change')
axs[0,1].set_title('B',weight='bold',fontsize=60,loc='left')

axs[1,1].bar(x=new_n, bottom=0, width=width, height=new_inits,color=colors[3])
axs[1,1].set_xticks([])
axs[1,1].set_xlabel('UTR Samples')
axs[1,1].set_ylabel('Initial TE')
axs[1,1].set_title('E',weight='bold',fontsize=60,loc='left')

init = []
with open('./../src/exp_optimization/outputs/mul_init_exps.txt') as f:
    scores = f.readlines()
    init = [float(score.replace('\n','')) for score in scores]

opt = []
with open('./../src/exp_optimization/outputs/mul_opt_exps.txt') as f:
    scores = f.readlines()
    opt = [float(score.replace('\n','')) for score in scores]

init = np.array(init)
opt = np.array(opt)

init = np.power(10,init)
opt = np.power(10,opt)

print(np.average(init))
print(np.average(opt))

SORT_INIT = True

if SORT_INIT:

    min_init_indices = np.argsort(init)
    init = init[min_init_indices[:min(int(len(init)),100)]]
    opt = opt[min_init_indices[:min(int(len(opt)),100)]]
else:
    init = init[:min(int(len(init)),100)]
    opt = opt[:min(int(len(opt)),100)]

diffs = [(opt[i]-init[i]) for i in range(len(init))]

count = 0
for i in range(len(diffs)):
    if diffs[i] < 0:
        count += 1

print(count)

print(np.mean(diffs/init))
# print(diffs)

indices_sorted = np.argsort(diffs)[::-1]

print(f"Average Opt: {np.average(opt)}")
print(f"Average Init: {np.average(init)}")
print(f"Max Opt: {np.max(opt)}")
print(f"Max Init: {np.max(init)}")
print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
print(f"Max Increase (wrt Natural) : {np.max(opt/np.power(10,-0.63))}")

diffs = (opt - init)/init
print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")

print(np.max(diffs/init))

new_inits = []

for i in range(len(diffs)):
    new_inits.append(init[indices_sorted[i]])

N = min(100, len(diffs))

step = 1.0/N

new_n = [i * step for i in range(N)]

width = step

axs[0,0].bar(x=new_n, bottom=0, width=width, height=diffs,color=colors[0])
axs[0,0].set_xticks([])
# axs[0,0].set_xlabel('UTR Samples')
axs[0,0].set_ylabel('Log TPM Expression Change')
axs[0,0].set_title('A',weight='bold',fontsize=60,loc='left')

axs[1,0].bar(x=new_n, bottom=0, width=width, height=init,color=colors[3])
axs[1,0].set_xticks([])
axs[1,0].set_xlabel('UTR Samples')
axs[1,0].set_ylabel('Initial TPM Expression')
axs[1,0].set_title('D',weight='bold',fontsize=60,loc='left')

######## MRL

init = []
with open(f'/home/sina/UTR/optimization/mrl/init_mrl_FMRL.txt') as f:
    scores = f.readlines()
    init = [float(score.replace('\n','')) for score in scores]

opt = []
with open(f'/home/sina/UTR/optimization/mrl/opt_mrl_FMRL.txt') as f:
    scores = f.readlines()
    opt = [float(score.replace('\n','')) for score in scores]

init = np.array(init)
opt = np.array(opt)


SORT_INIT = False

if SORT_INIT:

    min_init_indices = np.argsort(init)
    init = init[min_init_indices[:min(int(len(init)),100)]]
    opt = opt[min_init_indices[:min(int(len(opt)),100)]]
else:
    init = init[:min(int(len(init)),100)]
    opt = opt[:min(int(len(opt)),100)]

diffs = [(opt[i]-init[i]) for i in range(len(init))]

count = 0
for i in range(len(diffs)):
    if diffs[i] < 0:
        count += 1

print(count)

print(np.average(init))
print(np.average(opt))
print(np.max(diffs/init))
print(np.average(diffs/init))

indices_sorted = np.argsort(diffs)[::-1]

diffs = np.sort(diffs)[::-1]

new_inits = []

for i in range(len(diffs)):
    new_inits.append(init[indices_sorted[i]])

N = min(100, len(diffs))

step = 1.0/N

new_n = [i * step for i in range(N)]

width = step

plt.rcParams.update({'font.size': 12})


print(len(diffs))

axs[0,2].bar(x=new_n, bottom=0, width=width, height=diffs,color=colors[0])
axs[0,2].set_xticks([])
axs[0,2].set_ylabel('MRL Change')
axs[0,2].set_title('C',weight='bold',fontsize=60,loc='left')

axs[1,2].bar(x=new_n, bottom=0, width=width, height=new_inits,color=colors[3])
axs[1,2].set_xticks([])
axs[1,2].set_xlabel('UTR Samples')
axs[1,2].set_ylabel('Initial MRL')
axs[1,2].set_title('F',weight='bold',fontsize=60,loc='left')

#############


fig.tight_layout()

plt.savefig('./plots/opt_init_comparison.png')