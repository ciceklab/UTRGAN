import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set()
sns.set_style('ticks')

colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6"]

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
    with open('./../src/mrl_te_optimization/outputs/init_mrl_FMRL_MMRL.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open('./../src/mrl_te_optimization/outputs/opt_mrl_FMRL_MMRL.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]


    init = np.array(init)
    opt = np.array(opt)

    diffs = (opt - init)/init

    print("FramePool MRL optimization:")
    print(f"Average Opt: {np.average(opt)}")
    print(f"Average Init: {np.average(init)}")
    print(f"Max Opt: {np.max(opt)}")
    print(f"Max Init: {np.max(init)}")
    print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
    print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")

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
    fig, axs = plt.subplots(1,2)

    axs[0].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[0].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[0].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[0].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")

    axs[0].set_title(TITLE,loc='left',weight='bold',fontsize=64)
    axs[0].set_xticks([])
    
    TYPE = "FMRL"
    TITLE = "B"

    init = []

    with open('./../src/mrl_te_optimization/outputs/init_mrl_TE_MMRL.txt') as f:
        scores = f.readlines()
        init = [float(score.replace('\n','')) for score in scores]

    opt = []
    with open('./../src/mrl_te_optimization/outputs/opt_mrl_TE_MMRL.txt') as f:
        scores = f.readlines()
        opt = [float(score.replace('\n','')) for score in scores]


    init = np.power(10,init)
    init = np.array(init)
    opt = np.power(10,opt)
    opt = np.array(opt)
    
    diffs = (opt - init)/init

    print("MTtrans 3R TE optimization:")
    print(f"Average Opt: {np.average(opt)}")
    print(f"Average Init: {np.average(init)}")
    print(f"Max Opt: {np.max(opt)}")
    print(f"Max Init: {np.max(init)}")
    print(f"Max Increase (wrt Init) : {np.max(opt/init)}")
    print(f"Average Increase (wrt Init) : {np.mean(opt/init)}")
    print(f"Max Increase (wrt Natural) : {np.max(opt/np.power(10,-0.63))}")
    print(f"Average Percent Increase (wrt Init): {np.average(diffs)}")

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

    axs[1].bar(x=ns, bottom=0, width=width, height=opt_large, color=colors[0], edgecolor="white")
    axs[1].bar(x=ns, bottom=0, width=width, height=opt_small, color=colors[0], edgecolor="white")
    axs[1].bar(x=ns, bottom=0, width=width, height=init_small, color=colors[3], edgecolor="white")
    axs[1].bar(x=ns, bottom=0, width=width, height=init_large, color=colors[3], edgecolor="white")
    axs[1].set_title(TITLE,loc='left',weight='bold',fontsize=64)
    axs[1].set_xticks([])

    orange_patch = mpatches.Patch(color=colors[3], label='Initial')
    blue_patch = mpatches.Patch(color=colors[0], label='Optimized')
    fig.legend(handles=[orange_patch,blue_patch],loc='upper right')

    axs[0].set_ylabel('Predicted MRL')
    axs[1].set_ylabel('Predicted TE')
    
    axs[0].set_xlabel('UTR Samples')
    axs[1].set_xlabel('UTR Samples')

    fig.tight_layout()

    plt.savefig(f'./plots/mrl_te_all.png')


