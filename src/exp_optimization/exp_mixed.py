import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import seaborn as sns
import os
import torch
import argparse
from util import *
sns.set()
sns.set_style('ticks')

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

params = {'legend.fontsize': 32,
        'figure.figsize': (32, 20),
        'axes.labelsize': 34,
        'axes.titlesize':34,
        'xtick.labelsize':34,
        'ytick.labelsize':24}

#POSTER
params = {'legend.fontsize': 50,
        'figure.figsize': (54, 54),
        'axes.labelsize':60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':40}

plt.rcParams.update(params)

colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6", '#451120']

np.random.seed(25)

fig, axs = plt.subplots(4,2, gridspec_kw={'width_ratios': [3, 1]})

def prepare_mttrans(seqs):
    seqs_init = torch.tensor(np.array(one_hot_all_motif(seqs),dtype=np.float32))

    seqs_init = torch.transpose(seqs_init, 1, 2)
    seqs_init = torch.tensor(seqs_init,dtype=torch.float32).to('cuda')
    return seqs_init

DISPLAY_DIFF = True

root_path = '/home/sina/UTR/optimization/exp/outputs/'
tpath = '/home/sina/UTR/MTtrans/checkpoint/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth'
mpath = '/home/sina/UTR/MTtrans/checkpoint/RL_hard_share_MTL/3M/schedule_lr-model_best_cv1.pth'
K = 64

PREFIX = 'outputs/'

# MIXED, REGULAR, GC_CONTROLED

TYPE = 'MIXED'

DISPLAY_DIFF = True

if TYPE == 'REGULAR':
    PREFIX = 'outputs/'
elif TYPE == 'MIXED':
    PREFIX = 'outputs_mixed/'
elif TYPE == 'GC_CONTROLED':
    PREFIX = 'outputs/gc_'

# if DISPLAY_DIFF:
gene_name = 'IFNG'

init = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init = [float(score.replace('\n','')) for score in scores]

opt = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt = [float(score.replace('\n','')) for score in scores]

init_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init_seqs = [score.replace('\n','') for score in scores]

opt_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'best_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt_seqs = [score.replace('\n','') for score in scores]

te_model = torch.load(tpath,map_location=torch.device('cuda'))['state_dict']  
te_model.train().to('cuda')

mrl_model = torch.load(mpath,map_location=torch.device('cuda'))['state_dict']  
mrl_model.train().to('cuda')

te_seqs_init = prepare_mttrans(init_seqs)
te_seqs_opt = prepare_mttrans(opt_seqs)

te_preds_init = np.reshape(te_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
te_preds_opt = np.reshape(te_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

mrl_preds_init = np.reshape(mrl_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
mrl_preds_opt = np.reshape(mrl_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

init = np.power(10,init)
opt = np.power(10,opt)

selected = random.choices([i for i in range(len(init))],k=64)
init = init[selected]
opt = opt[selected]

print(gene_name)
print(f"Average Opt: {np.average(opt)}")
print(f"Average Init: {np.average(init)}")
print(f"Max Opt: {np.max(opt)}")
print(f"Max Init: {np.max(init)}")
print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
print(f"Max Increase (wrt Natural) : {np.max(opt/np.power(10,-0.63))}")

diffs = (opt - init)/init
print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")
print(f"Max TE after opt: {np.max(te_preds_opt)}")

indices = np.argsort(opt)[::-1]
# indices = indices[:100]

init_large = []
init_small = []
opt_large = []
opt_small = []

for i in range(len(indices)):
    if diffs[indices[i]] >= 0:
        init_small.append(init[indices[i]])
        init_large.append(0)
        opt_small.append(0)
        opt_large.append(opt[indices[i]])
    else:
        init_large.append(init[indices[i]])
        init_small.append(0)
        opt_large.append(0)
        opt_small.append(opt[indices[i]])

width = 1.0/(len(indices))
bins = [(i+1) * width for i in range(len(indices))]

ns = [i * width for i in range(len(indices))]



# plt.rcParams.update({'font.size': 12})

# sns.barplot(x = 'total', y = 'abbrev', data = crashes,label = 'Total', color = 'b', edgecolor = 'w')

axs[0,0].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
axs[0,0].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
axs[0,0].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
axs[0,0].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")
axs[0,0].axhline(y = np.power(10,-1.09), color = colors[4], linestyle = '-', linewidth = 5)
# sns.barplot(x=ns,width=width,y=opt_large,color='r',ax=axs[0,0])
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=opt_small,color='tab:blue',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_large,color='tab:orange',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_small,color='tab:orange',edgecolor='white')

axs[0,0].set_title(gene_name,loc='left',style='italic')
# axs[0,1].set_title(gene_name,loc='left',style='italic')
axs[0,0].set_xticks([])

real_x = ['Optimized' for i in range(len(init))]
gen_x = ['Initial' for i in range(len(opt))]

# Expression

x = np.concatenate((gen_x,real_x))
y = np.concatenate((init,opt))

df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[0,1])

# MRL

x = np.concatenate((gen_x,real_x))
y = np.concatenate((mrl_preds_init,mrl_preds_opt))

# print(len(x))
# print(len(y))

# print(x)
# print(y)

df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[1,0])

# TE

x = np.concatenate((gen_x,real_x))
y = np.concatenate((te_preds_init,te_preds_opt))

df = pd.DataFrame({'x':x,'y':y})

sns.boxplot(x=df['x'],y=df['y'],ax=axs[0,1],palette={'Initial':colors[3],'Optimized':colors[0]})

orange_patch = mpatches.Patch(color='tab:orange', label='Initial Expression')
blue_patch = mpatches.Patch(color='tab:blue', label='Optimized Expression')
# fig.legend(handles=[orange_patch,blue_patch],loc='upper left')

# axs[0,0].set_ylabel('TPM Expression')
# axs[1,0].set_ylabel('TPM Expression')
# axs[0,0].get_yaxis().set_label_coords(-0.07,0.5)
# axs[1,0].get_yaxis().set_label_coords(-0.07,0.5)


# axs[1,0].set_xlabel('UTR Samples')
# axs[1,1].set_xlabel('UTR Samples')

gene_name = 'TLR6'

init = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init = [float(score.replace('\n','')) for score in scores]

opt = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt = [float(score.replace('\n','')) for score in scores]

init_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init_seqs = [score.replace('\n','') for score in scores]

opt_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'best_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt_seqs = [score.replace('\n','') for score in scores]

te_model = torch.load(tpath,map_location=torch.device('cuda'))['state_dict']  
te_model.train().to('cuda')

mrl_model = torch.load(mpath,map_location=torch.device('cuda'))['state_dict']  
mrl_model.train().to('cuda')

te_seqs_init = prepare_mttrans(init_seqs)
te_seqs_opt = prepare_mttrans(opt_seqs)

te_preds_init = np.reshape(te_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
te_preds_opt = np.reshape(te_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

mrl_preds_init = np.reshape(mrl_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
mrl_preds_opt = np.reshape(mrl_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

init = np.power(10,init)
opt = np.power(10,opt)

selected = random.choices([i for i in range(len(init))],k=64)
init = init[selected]
opt = opt[selected]

print(gene_name)
print(f"Average Opt: {np.average(opt)}")
print(f"Average Init: {np.average(init)}")
print(f"Max Opt: {np.max(opt)}")
print(f"Max Init: {np.max(init)}")
print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
print(f"Max Increase (wrt Natural) : {np.max(opt/np.power(10,-0.63))}")

diffs = (opt - init)/init
print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")
print(f"Max TE after opt: {np.max(te_preds_opt)}")

indices = np.argsort(opt)[::-1]
# indices = indices[:100]

init_large = []
init_small = []
opt_large = []
opt_small = []

for i in range(len(indices)):
    if diffs[indices[i]] >= 0:
        init_small.append(init[indices[i]])
        init_large.append(0)
        opt_small.append(0)
        opt_large.append(opt[indices[i]])
    else:
        init_large.append(init[indices[i]])
        init_small.append(0)
        opt_large.append(0)
        opt_small.append(opt[indices[i]])

width = 1.0/(len(indices))
bins = [(i+1) * width for i in range(len(indices))]

ns = [i * width for i in range(len(indices))]

# plt.rcParams.update({'font.size': 12})

# sns.barplot(x = 'total', y = 'abbrev', data = crashes,label = 'Total', color = 'b', edgecolor = 'w')

axs[1,0].bar(x=ns, bottom=0, width=width, height=opt_large, color= colors[0], edgecolor="white")
axs[1,0].bar(x=ns, bottom=0, width=width, height=opt_small, color= colors[0], edgecolor="white")
axs[1,0].bar(x=ns, bottom=0, width=width, height=init_small, color= colors[3], edgecolor="white")
axs[1,0].bar(x=ns, bottom=0, width=width, height=init_large, color= colors[3], edgecolor="white")
axs[1,0].axhline(y = np.power(10,-0.37), color = colors[4], linestyle = '-', linewidth = 5)
# sns.barplot(x=ns,width=width,y=opt_large,color='r',ax=axs[0,0])
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=opt_small,color='tab:blue',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_large,color='tab:orange',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_small,color='tab:orange',edgecolor='white')

axs[1,0].set_title(gene_name,loc='left',style='italic')
# axs[1,1].set_title(gene_name,loc='left',style='italic')
axs[1,0].set_xticks([])

real_x = ['Optimized' for i in range(len(init))]
gen_x = ['Initial' for i in range(len(opt))]

# Expression

x = np.concatenate((gen_x,real_x))
y = np.concatenate((init,opt))

df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[1,1])

# MRL

x = np.concatenate((gen_x,real_x))
y = np.concatenate((mrl_preds_init,mrl_preds_opt))

# print(len(x))
# print(len(y))

# print(x)
# print(y)

df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[1,0])

# TE

x = np.concatenate((gen_x,real_x))
y = np.concatenate((te_preds_init,te_preds_opt))

df = pd.DataFrame({'x':x,'y':y})

sns.boxplot(x=df['x'],y=df['y'],ax=axs[1,1],palette={'Initial':colors[3],'Optimized':colors[0]})

orange_patch = mpatches.Patch(color='tab:orange', label='Initial Expression')
blue_patch = mpatches.Patch(color='tab:blue', label='Optimized Expression')
# fig.legend(handles=[orange_patch,blue_patch],loc='upper left')

# axs[,0].set_ylabel('TPM Expression')
# axs[1,0].set_ylabel('TPM Expression')
# axs[0,0].get_yaxis().set_label_coords(-0.07,0.5)
# axs[1,0].get_yaxis().set_label_coords(-0.07,0.5)


# axs[1,0].set_xlabel('UTR Samples')
# axs[1,1].set_xlabel('UTR Samples')

gene_name = 'TNF'

init = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init = [float(score.replace('\n','')) for score in scores]

opt = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt = [float(score.replace('\n','')) for score in scores]

init_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init_seqs = [score.replace('\n','') for score in scores]

opt_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'best_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt_seqs = [score.replace('\n','') for score in scores]

te_model = torch.load(tpath,map_location=torch.device('cuda'))['state_dict']  
te_model.train().to('cuda')

mrl_model = torch.load(mpath,map_location=torch.device('cuda'))['state_dict']  
mrl_model.train().to('cuda')

te_seqs_init = prepare_mttrans(init_seqs)
te_seqs_opt = prepare_mttrans(opt_seqs)

te_preds_init = np.reshape(te_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
te_preds_opt = np.reshape(te_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

mrl_preds_init = np.reshape(mrl_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
mrl_preds_opt = np.reshape(mrl_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

init = np.power(10,init)
opt = np.power(10,opt)

selected = random.choices([i for i in range(len(init))],k=64)
init = init[selected]
opt = opt[selected]

print(gene_name)
print(f"Average Opt: {np.average(opt)}")
print(f"Average Init: {np.average(init)}")
print(f"Max Opt: {np.max(opt)}")
print(f"Max Init: {np.max(init)}")
print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
print(f"Max Increase (wrt Natural) : {np.max(opt/np.power(10,-0.63))}")

diffs = (opt - init)/init
print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")
print(f"Max TE after opt: {np.max(te_preds_opt)}")

indices = np.argsort(opt)[::-1]
# indices = indices[:100]

init_large = []
init_small = []
opt_large = []
opt_small = []

for i in range(len(indices)):
    if diffs[indices[i]] >= 0:
        init_small.append(init[indices[i]])
        init_large.append(0)
        opt_small.append(0)
        opt_large.append(opt[indices[i]])
    else:
        init_large.append(init[indices[i]])
        init_small.append(0)
        opt_large.append(0)
        opt_small.append(opt[indices[i]])

width = 1.0/(len(indices))
bins = [(i+1) * width for i in range(len(indices))]

ns = [i * width for i in range(len(indices))]


# plt.rcParams.update({'font.size': 12})

# sns.barplot(x = 'total', y = 'abbrev', data = crashes,label = 'Total', color = 'b', edgecolor = 'w')

axs[2,0].bar(x=ns, bottom=0, width=width, height=opt_large, color= colors[0], edgecolor="white")
axs[2,0].bar(x=ns, bottom=0, width=width, height=opt_small, color= colors[0], edgecolor="white")
axs[2,0].bar(x=ns, bottom=0, width=width, height=init_small, color= colors[3], edgecolor="white")
axs[2,0].bar(x=ns, bottom=0, width=width, height=init_large, color= colors[3], edgecolor="white")
axs[2,0].axhline(y = np.power(10,-0.91), color = colors[4], linestyle = '-', linewidth = 5)

# sns.barplot(x=ns,width=width,y=opt_large,color='r',ax=axs[0,0])
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=opt_small,color='tab:blue',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_large,color='tab:orange',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_small,color='tab:orange',edgecolor='white')

axs[2,0].set_title(gene_name,loc='left',style='italic')
# axs[2,1].set_title(gene_name,loc='left',style='italic')
axs[2,0].set_xticks([])

real_x = ['Optimized' for i in range(len(init))]
gen_x = ['Initial' for i in range(len(opt))]

# Expression

x = np.concatenate((gen_x,real_x))
y = np.concatenate((init,opt))

df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[2,1])

# MRL

x = np.concatenate((gen_x,real_x))
y = np.concatenate((mrl_preds_init,mrl_preds_opt))

# print(len(x))
# print(len(y))

# print(x)
# print(y)

# df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[1,0])

# TE

x = np.concatenate((gen_x,real_x))
y = np.concatenate((te_preds_init,te_preds_opt))

df = pd.DataFrame({'x':x,'y':y})

sns.boxplot(x=df['x'],y=df['y'],ax=axs[2,1],palette={'Initial':colors[3],'Optimized':colors[0]})

orange_patch = mpatches.Patch(color='tab:orange', label='Initial Expression')
blue_patch = mpatches.Patch(color='tab:blue', label='Optimized Expression')
# fig.legend(handles=[orange_patch,blue_patch],loc='upper left')

# axs[0,0].set_ylabel('TPM Expression')
# axs[1,0].set_ylabel('TPM Expression')
# axs[0,0].get_yaxis().set_label_coords(-0.07,0.5)
# axs[1,0].get_yaxis().set_label_coords(-0.07,0.5)


# axs[1,0].set_xlabel('UTR Samples')
# axs[1,1].set_xlabel('UTR Samples')

gene_name = 'TP53'

init = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init = [float(score.replace('\n','')) for score in scores]

opt = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt = [float(score.replace('\n','')) for score in scores]

init_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'init_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    init_seqs = [score.replace('\n','') for score in scores]

opt_seqs = []
with open('/home/sina/UTR/optimization/exp/'+PREFIX+'best_seqs_'+gene_name+'.txt') as f:
    scores = f.readlines()
    opt_seqs = [score.replace('\n','') for score in scores]

te_model = torch.load(tpath,map_location=torch.device('cuda'))['state_dict']  
te_model.train().to('cuda')

mrl_model = torch.load(mpath,map_location=torch.device('cuda'))['state_dict']  
mrl_model.train().to('cuda')

te_seqs_init = prepare_mttrans(init_seqs)
te_seqs_opt = prepare_mttrans(opt_seqs)

te_preds_init = np.reshape(te_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
te_preds_opt = np.reshape(te_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

mrl_preds_init = np.reshape(mrl_model.forward(te_seqs_init).cpu().data.numpy(),(-1))
mrl_preds_opt = np.reshape(mrl_model.forward(te_seqs_opt).cpu().data.numpy(),(-1))

init = np.power(10,init)
opt = np.power(10,opt)

selected = random.choices([i for i in range(len(init))],k=64)
init = init[selected]
opt = opt[selected]

print(gene_name)
print(f"Average Opt: {np.average(opt)}")
print(f"Average Init: {np.average(init)}")
print(f"Max Opt: {np.max(opt)}")
print(f"Max Init: {np.max(init)}")
print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
print(f"Max Increase (wrt Natural) : {np.max(opt/np.power(10,-0.63))}")

diffs = (opt - init)/init
print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")
print(f"Max TE after opt: {np.max(te_preds_opt)}")

indices = np.argsort(opt)[::-1]
# indices = indices[:100]

init_large = []
init_small = []
opt_large = []
opt_small = []

for i in range(len(indices)):
    if diffs[indices[i]] >= 0:
        init_small.append(init[indices[i]])
        init_large.append(0)
        opt_small.append(0)
        opt_large.append(opt[indices[i]])
    else:
        init_large.append(init[indices[i]])
        init_small.append(0)
        opt_large.append(0)
        opt_small.append(opt[indices[i]])

width = 1.0/(len(indices))
bins = [(i+1) * width for i in range(len(indices))]

ns = [i * width for i in range(len(indices))]

# plt.rcParams.update({'font.size': 12})

# sns.barplot(x = 'total', y = 'abbrev', data = crashes,label = 'Total', color = 'b', edgecolor = 'w')

axs[3,0].bar(x=ns, bottom=0, width=width, height=opt_large,  color=colors[0], edgecolor="white")
axs[3,0].bar(x=ns, bottom=0, width=width, height=opt_small,  color=colors[0], edgecolor="white")
axs[3,0].bar(x=ns, bottom=0, width=width, height=init_small,  color=colors[3], edgecolor="white")
axs[3,0].bar(x=ns, bottom=0, width=width, height=init_large,  color=colors[3], edgecolor="white")
axs[3,0].axhline(y = np.power(10,-0.63), color = colors[4], linestyle = '-', linewidth = 5)# sns.barplot(x=ns,width=width,y=opt_large,color='r',ax=axs[0,0])
# axs[3,0].text(x=0.3,y=np.power(10,-0.63)+0.1,s='AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')# sns.barplot(x=ns,width=width,y=opt_large,color='r',ax=axs[0,0])
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=opt_small,color='tab:blue',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_large,color='tab:orange',edgecolor='white')
# sns.barplot(ax=axs[0,0],x=ns,width=width,y=init_small,color='tab:orange',edgecolor='white')

axs[3,0].set_title(gene_name,loc='left',style='italic')
# axs[3,1].set_title(gene_name,loc='left',style='italic')
axs[3,0].set_xticks([])

real_x = ['Optimized' for i in range(len(init))]
gen_x = ['Initial' for i in range(len(opt))]

# Expression

x = np.concatenate((gen_x,real_x))
y = np.concatenate((init,opt))

df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[0,1])

# MRL

x = np.concatenate((gen_x,real_x))
y = np.concatenate((mrl_preds_init,mrl_preds_opt))

print(len(x))
print(len(y))

print(x)
print(y)

df = pd.DataFrame({'x':x,'y':y})

# sns.boxplot(x=df['x'],y=df['y'],ax=axs[1,0])

# TE

x = np.concatenate((gen_x,real_x))
y = np.concatenate((te_preds_init,te_preds_opt))

df = pd.DataFrame({'x':x,'y':y})

sns.boxplot(x=df['x'],y=df['y'],ax=axs[3,1],palette={'Initial':colors[3],'Optimized':colors[0]})

orange_patch = mpatches.Patch(color=colors[3], label='Initial Expression')
blue_patch = mpatches.Patch(color=colors[0], label='Optimized Expression')
fig.legend(handles=[orange_patch,blue_patch],loc=(0.52,0.95))

axs[0,0].set_ylabel('TPM Expression')
axs[1,0].set_ylabel('TPM Expression')
axs[2,0].set_ylabel('TPM Expression')
axs[3,0].set_ylabel('TPM Expression')
axs[3,0].set_xlabel('UTR Samples')
# axs[0,0].get_yaxis().set_label_coords(-0.07,0.5)
# axs[1,0].get_yaxis().set_label_coords(-0.07,0.5)
# axs[2,0].get_yaxis().set_label_coords(-0.07,0.5)
# axs[3,0].get_yaxis().set_label_coords(-0.07,0.5)

axs[0,1].set_xlabel('')
axs[1,1].set_xlabel('')
axs[2,1].set_xlabel('')
axs[3,1].set_xlabel('')
axs[0,1].set_ylabel('Log Translation Efficiency')
axs[1,1].set_ylabel('Log Translation Efficiency')
axs[2,1].set_ylabel('Log Translation Efficiency')
axs[3,1].set_ylabel('Log Translation Efficiency')
axs[0,1].yaxis.tick_right()
axs[1,1].yaxis.tick_right()
axs[2,1].yaxis.tick_right()
axs[3,1].yaxis.tick_right()
axs[0,1].yaxis.set_label_position("right")
axs[1,1].yaxis.set_label_position("right")
axs[2,1].yaxis.set_label_position("right")
axs[3,1].yaxis.set_label_position("right")


# axs[1,0].set_xlabel('UTR Samples')
# axs[1,1].set_xlabel('UTR Samples')

fig.tight_layout()
# plt.gcf().subplots_adjust(left=0.06)

plt.savefig(f'mixed_all.png')




