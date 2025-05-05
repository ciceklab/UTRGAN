import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import seaborn as sns
import os
import argparse
sns.set()
sns.set_style('ticks')

colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6"]


#POSTER
params = {'legend.fontsize': 50,
        'figure.figsize': (54, 38),
        'axes.labelsize': 60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':60}

plt.rcParams.update(params)


np.random.seed(25)

DISPLAY_DIFF = True

root_path = './../src/exp_optimization/'

PREFIX = 'outputs/'

# MIXED, REGULAR, GC_CONTROLED, MULT

TYPE = 'GC_CONTROLED'

DISPLAY_DIFF = True

if TYPE == 'REGULAR':
    PREFIX = 'outputs/'
elif TYPE == 'MIXED':
    PREFIX = 'outputs_joint/'
elif TYPE == 'GC_CONTROLED':
    PREFIX = 'outputs/gc_'
elif TYPE == 'K562':
    PREFIX = 'outputs/K562_'
elif TYPE == 'GM12878':
    PREFIX = 'outputs/GM12878_'

if DISPLAY_DIFF:
    parser = argparse.ArgumentParser(description="Gene Expression Optimization Visualization")

    # Add arguments
    parser.add_argument("-g", help="a list of gene names separated by comma")

    # Parse the arguments
    args = parser.parse_args()
    gene_names = args.g.split(',')

    gene_name = gene_names[0]

    init = []
    with open(root_path+PREFIX+'init_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open(root_path+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]


    init = np.power(10,init)
    opt = np.power(10,opt)
    diffs = (opt - init)/init



    print("####################################################################")
    print(f"{gene_name} results:")
    print(f"Max Opt: {np.max(opt):.2f}")
    print(f"Max Init: {np.max(init):.2f}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)*100:.2f}")
    print(f"Max Percent Increase (wrt Init): {np.max(diffs)*100:.2f}")

    indices = np.argsort(opt)[::-1]

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
    fig, axs = plt.subplots(2,2)


    axs[0,0].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[0,0].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[0,0].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[0,0].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")

    axs[0,0].set_title(gene_name,loc='left',style='italic',fontsize=64)
    axs[0,0].set_xticks([])
    
    gene_name = gene_names[1]

    init = []
    with open(root_path+PREFIX+'init_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open(root_path+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]

    init = np.power(10,init)
    opt = np.power(10,opt)
    diffs = (opt - init)/init

    print("####################################################################")
    print(f"{gene_name} results:")
    print(f"Max Opt: {np.max(opt):.2f}")
    print(f"Max Init: {np.max(init):.2f}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)*100:.2f}")
    print(f"Max Percent Increase (wrt Init): {np.max(diffs)*100:.2f}")

    indices = np.argsort(opt)[::-1]

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

    axs[0,1].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[0,1].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[0,1].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[0,1].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")
    axs[0,1].set_title(gene_name,loc='left',style='italic',fontsize=64)
    axs[0,1].set_xticks([])

    gene_name = gene_names[2]

    init = []
    with open(root_path+PREFIX+'init_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open(root_path+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]


    init = np.power(10,init)
    opt = np.power(10,opt)
    diffs = (opt - init)/init
    
    print("####################################################################")
    print(f"{gene_name} results:")
    print(f"Max Opt: {np.max(opt):.2f}")
    print(f"Max Init: {np.max(init):.2f}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)*100:.2f}")
    print(f"Max Percent Increase (wrt Init): {np.max(diffs)*100:.2f}")

    indices = np.argsort(opt)[::-1]

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

    axs[1,0].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[1,0].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[1,0].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[1,0].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")
    axs[1,0].set_title(gene_name,loc='left',style='italic',fontsize=64)
    axs[1,0].set_xticks([])

    gene_name = gene_names[3]

    init = []
    with open(root_path+PREFIX+'init_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open(root_path+PREFIX+'opt_exps_'+gene_name+'.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]

    init = np.power(10,init)
    opt = np.power(10,opt)
    diffs = (opt - init)/init

    print("####################################################################")
    print(f"{gene_name} results:")
    print(f"Max Opt: {np.max(opt):.2f}")
    print(f"Max Init: {np.max(init):.2f}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)*100:.2f}")
    print(f"Max Percent Increase (wrt Init): {np.max(diffs)*100:.2f}")
    print("####################################################################")


    indices = np.argsort(opt)[::-1]
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

    axs[1,1].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[1,1].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[1,1].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[1,1].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")
    axs[1,1].set_title(gene_name,loc='left',style='italic',fontsize=64)
    axs[1,1].set_xticks([])

    orange_patch = mpatches.Patch(color=colors[3], label='Initial Expression')
    blue_patch = mpatches.Patch(color=colors[0], label='Optimized Expression')
    fig.legend(handles=[orange_patch,blue_patch],loc='upper right')

    axs[0,0].set_ylabel('TPM Expression')
    axs[1,0].set_ylabel('TPM Expression')

    axs[1,0].set_xlabel('UTR Samples')
    axs[1,1].set_xlabel('UTR Samples')


    fig.tight_layout()
    plt.gcf().subplots_adjust(left=0.06)

    os.makedirs('./plots/',exist_ok=True)

    plt.savefig(f'./plots/exp_opt_all_{TYPE}_{gene_names}.png')




