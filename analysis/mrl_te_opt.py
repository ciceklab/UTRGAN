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
# parser = argparse.ArgumentParser()

# parser.add_argument('-g', type=str, required=True)

# args = parser.parse_args()

colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6"]

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

params = {'legend.fontsize': 32,
        'figure.figsize': (32, 10),
        'axes.labelsize': 34,
        'axes.titlesize':34,
        'xtick.labelsize':34,
        'ytick.labelsize':24}

#POSTER
params = {'legend.fontsize': 50,
        'figure.figsize': (54, 18),
        'axes.labelsize': 60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':36}

plt.rcParams.update(params)


np.random.seed(25)

DISPLAY_DIFF = True

root_path = '/home/sina/UTR/optimization/exp/outputs/'

K = 100

PREFIX = ''

# MRL, TE

TYPE = 'MRL'

DISPLAY_DIFF = True

if TYPE == 'REGULAR':
    PREFIX = 'outputs/'
elif TYPE == 'MIXED':
    PREFIX = 'outputs_mixed/'
elif TYPE == 'GC_CONTROLED':
    PREFIX = 'outputs/gc_'

if DISPLAY_DIFF:
    TYPE = 'MMRL'
    TITLE = "A"

    init = []
    with open('/home/sina/UTR/optimization/mrl/init_mrl_FMRL_MMRL.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open('/home/sina/UTR/optimization/mrl/opt_mrl_FMRL_MMRL.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]

    # TLR6_seqs = np.load(root_path+'best_seqs_TLR6.npy',allow_pickle=True)
    # best_TLR6 = TLR6_seqs[np.argmax(opt)]


    

    # init = np.power(10,init)
    init = np.array(init)
    # opt = np.power(10,opt)
    opt = np.array(opt)

    # selected = random.choices([i for i in range(len(init))],k=len(init))
    # init = init[selected]
    # opt = opt[selected]
    
    diffs = (opt - init)/init
    print(np.average(diffs))

    print(f"Average Opt: {np.average(opt)}")
    print(f"Average Init: {np.average(init)}")
    print(f"Max Opt: {np.max(opt)}")
    print(f"Max Init: {np.max(init)}")
    print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
    print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")

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
    fig, axs = plt.subplots(1,2)


    # plt.rcParams.update({'font.size': 12})

    # sns.barplot(x = 'total', y = 'abbrev', data = crashes,label = 'Total', color = 'b', edgecolor = 'w')

    axs[0].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[0].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[0].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[0].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")

    axs[0].set_title(TITLE,loc='left',weight='bold',fontsize=64)
    axs[0].set_xticks([])
    
    TYPE = "FMRL"
    TITLE = "B"

    init = []
    # with open('/home/sina/UTR/optimization/mrl/init_mrl_TE_MMRL.txt') as f:
    with open('/home/sina/UTR/optimization/mrl/init_mrl_TE_MMRL.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open('/home/sina/UTR/optimization/mrl/opt_mrl_TE_MMRL.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]

    # TP53_seqs = np.load(root_path+'best_seqs_TP53.npy',allow_pickle=True)
    # best_TP53 = TP53_seqs[np.argmax(opt)]

    init = np.power(10,init)
    init = np.array(init)
    opt = np.power(10,opt)
    opt = np.array(opt)

    # selected = random.choices([i for i in range(len(init))],k=len(init))
    # init = init[selected]
    # opt = opt[selected]
    
    diffs = (opt - init)/init
    print(np.average(diffs))

    print(f"Average Opt: {np.average(opt)}")
    print(f"Average Init: {np.average(init)}")
    print(f"Max Opt: {np.max(opt)}")
    print(f"Max Init: {np.max(init)}")
    print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
    print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
    print(f"Max Increase (wrt Natural) : {np.max(opt/np.power(10,-0.63))}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")

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

    axs[1].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[1].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[1].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[1].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")
    axs[1].set_title(TITLE,loc='left',weight='bold',fontsize=64)
    axs[1].set_xticks([])
    # plt.hist(x=init_large, bins= bins, bottom=0, width=width, color='tab:orange')
    # plt.hist(x=opt_large,  bins= bins, bottom=0, width=width, color='tab:blue')
    # plt.hist(x=opt_small,  bins= bins, bottom=0, width=width, color='tab:blue')
    # plt   .hist(x=init_small, bins= bins, bottom=0, width=width, color='tab:orange')

    orange_patch = mpatches.Patch(color=colors[3], label='Initial')
    blue_patch = mpatches.Patch(color=colors[0], label='Optimized')
    fig.legend(handles=[orange_patch,blue_patch],loc='upper right')
    # plt.xticks([])ncols=2

    # for ax in axs.flat:
    #     ax.set(xlabel='UTR Samples', ylabel='TPM Expression')

    axs[0].set_ylabel('Predicted MRL')
    axs[1].set_ylabel('Predicted TE')
    # axs[0].get_yaxis().set_label_coords(-0.07,0.5)
    

    axs[0].set_xlabel('UTR Samples')
    axs[1].set_xlabel('UTR Samples')

    # for ax in axs.flat:
    #     ax.label_outer()
    # formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    # plt.gca().yaxis.set_major_formatter(formatter)


    # plt.xlabel('UTR Samples')
    # # plt.title(args.g + ' Gene')
    # plt.ylabel('TPM Expression')
    # params = {'legend.fontsize': 'x-large',
    #       'figure.figsize': (15, 5),
    #      'axes.labelsize': 'x-large',
    #      'axes.titlesize':'x-large',
    #      'xtick.labelsize':'x-large',
    #      'ytick.labelsize':'x-large'}
    # plt.rcParams.update(params)


    # plt.rcParams.update({'font.size': 8})
    fig.tight_layout()
    # plt.gcf().subplots_adjust(left=0.06)

    plt.savefig(f'mrl_te_all.png')

    # with open('TNF.txt','w') as f:
    #     f.write(best_TNF.replace('U','T'))
    # with open('IFNG.txt','w') as f:
    #     f.write(best_INFG.replace('U','T'))
    # with open('TLR6.txt','w') as f:
    #     f.write(best_TLR6.replace('U','T'))
    # with open('TP53.txt','w') as f:
    #     f.write(best_TP53.replace('U','T'))

# else:
#     SORT_INIT = False

#     if SORT_INIT:
#         min_init_indices = np.argsort(init)
#         init = init[min_init_indices[:min(int(len(init)/2),100)]]
#         opt = opt[min_init_indices[:min(int(len(opt)/2),100)]]
#     else:
#         init = init[:min(int(len(init)/2),100)]
#         opt = opt[:min(int(len(opt)/2),100)]



#     diffs = [(opt[i]-init[i]) for i in range(len(init))]

#     diffs = np.sort(diffs)[::-1]

#     N = min(100, len(diffs))

#     step = 1.0/N

#     new_n = [i * step for i in range(N)]

#     width = step


#     plt.rcParams.update({'font.size': 12})

#     plt.bar(x=new_n, bottom=0, width=width, height=diffs,color='tab:blue')
#     orange_patch = mpatches.Patch(color='tab:blue', label='Exp. Change')
#     # plt.legend(handles=[orange_patch])

#     plt.xlabel('UTR Samples')
#     plt.title(args.g)
#     plt.ylabel('TPM Expression')

#     plt.savefig('exp_opt_'+args.g+'.png')



