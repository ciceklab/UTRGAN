import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import RNA
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.stats import norm
import os
from matplotlib.patches import Patch
from itertools import product


np.random.seed(1337)


colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6"]
customPalette = {'Initial': colors[0], 'Optimized': colors[3]}

# Constants
BATCH_SIZE = 2048
K = 4  


params = {
    'legend.fontsize': 40, 
    'figure.figsize': (120, 80), 
    'axes.labelsize': 120,
    'axes.titlesize': 120,
    'xtick.labelsize': 100,
    'ytick.labelsize': 100
}
plt.rcParams.update(params)
sns.set()
sns.set_style('ticks')

def get_gc_content_many(sequences):
    gc_contents = []
    for seq in sequences:
        gc_count = seq.count('G') + seq.count('C')
        gc_contents.append(gc_count / len(seq) if len(seq) > 0 else 0)
    return np.array(gc_contents)

def get_lengths(sequences):
    return np.array([len(seq) for seq in sequences])


def get_kmer_frequency(sequences, k=4):
    # Generate all possible k-mers
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    
    mean_frequencies = []
    for seq in sequences:
        if len(seq) < k:
            mean_frequencies.append(0)
            continue
        # Count occurrences of each k-mer
        kmer_counts = {kmer: 0 for kmer in kmers}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
        # Normalize by number of possible k-mer positions
        total_positions = len(seq) - k + 1
        frequencies = [count / total_positions for count in kmer_counts.values() if total_positions > 0]
        # Compute mean frequency
        mean_freq = np.mean(frequencies) if frequencies else 0
        mean_frequencies.append(mean_freq)
    return np.array(mean_frequencies)




def generate_synthetic_data(num_samples=100):
    bases = ['A', 'C', 'G', 'T']

    sequences = [''.join(np.random.choice(bases, np.random.randint(50, 151))) for _ in range(num_samples)]
    return sequences

# Function to read sequences from .txt file
def read_sequences(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        return sequences
    else:
        print(f"File {file_path} not found, generating synthetic data.")
        return generate_synthetic_data(100)

def analyze_utr_features(file_pairs):

    all_data = []
    
    # Process each gene (set of initial and optimized files)
    for idx, (initial_path, optimized_path, gene_name) in enumerate(file_pairs, 1):

        initial_seqs = read_sequences(initial_path)
        optimized_seqs = read_sequences(optimized_path)
        
        # Compute features
        # Minimum Free Energy (MFE)
        initial_mfe = [RNA.fold(seq)[1] for seq in initial_seqs]
        optimized_mfe = [RNA.fold(seq)[1] for seq in optimized_seqs]
        
        # G/C Content
        initial_gc = get_gc_content_many(initial_seqs)
        optimized_gc = get_gc_content_many(optimized_seqs)
        
        # Sequence Length
        initial_length = get_lengths(initial_seqs)
        optimized_length = get_lengths(optimized_seqs)
        
        # 4-mer Frequency
        initial_kmer = 1000*get_kmer_frequency(initial_seqs, k=K)
        optimized_kmer = 1000*get_kmer_frequency(optimized_seqs, k=K)

        

        for seq_type, mfe, gc, length, kmer in [
            ('Initial', initial_mfe, initial_gc, initial_length, initial_kmer),
            ('Optimized', optimized_mfe, optimized_gc, optimized_length, optimized_kmer)
        ]:
            for m, g, l, k in zip(mfe, gc, length, kmer):
                all_data.append({
                    'Gene': gene_name,
                    'Type': seq_type,
                    'MFE': m,
                    'GC Content': g,
                    'Length': l,
                    '4-mer Frequency': k
                })
        
    

    df = pd.DataFrame(all_data)
    

    fig, axs = plt.subplots(2, 2, figsize=(100, 70))
    
    legend_handles = [
        Patch(color=customPalette['Initial'], label='Initial'),
        Patch(color=customPalette['Optimized'], label='Optimized')
    ]
    
    # MFE
    sns.boxplot(x='Gene', y='MFE', hue='Type', data=df, ax=axs[0, 0], palette=customPalette)
    axs[0, 0].get_legend().remove()
    axs[0, 0].legend(handles=legend_handles, loc='lower right', fontsize=70, title='Type', title_fontsize=70)
    axs[0, 0].set_ylabel("Minimum Free Energy", fontsize=120)
    axs[0, 0].set_xlabel("", fontsize=120)
    axs[0, 0].set_title('A', weight='bold', fontsize=100, loc='left')
    axs[0, 0].tick_params(axis='both', labelsize=100)
    
    # G/C Content
    sns.boxplot(x='Gene', y='GC Content', hue='Type', data=df, ax=axs[0, 1], palette=customPalette)
    axs[0, 1].get_legend().remove()
    axs[0, 1].legend(handles=legend_handles, loc='lower right', fontsize=70, title='Type', title_fontsize=70)
    axs[0, 1].set_ylabel("G/C Content", fontsize=120)
    axs[0, 1].set_xlabel("", fontsize=120)
    axs[0, 1].set_title('B', weight='bold', fontsize=100, loc='left')
    axs[0, 1].tick_params(axis='both', labelsize=100)
    
    # Length
    sns.boxplot(x='Gene', y='Length', hue='Type', data=df, ax=axs[1, 0], palette=customPalette)
    axs[1, 0].get_legend().remove()
    axs[1, 0].legend(handles=legend_handles, loc='lower right', fontsize=70, title='Type', title_fontsize=70)
    axs[1, 0].set_ylabel("Sequence Length", fontsize=120)
    axs[1, 0].set_xlabel("", fontsize=120)
    axs[1, 0].set_title('C', weight='bold', fontsize=100, loc='left')
    axs[1, 0].tick_params(axis='both', labelsize=100)
    
    # 4-mer Frequency
    sns.boxplot(x='Gene', y='4-mer Frequency', hue='Type', data=df, ax=axs[1, 1], palette=customPalette)
    axs[1, 1].get_legend().remove()
    axs[1, 1].legend(handles=legend_handles, loc='lower right', fontsize=70, title='Type', title_fontsize=70)
    axs[1, 1].set_ylabel("Mean 4-mer Frequency", fontsize=120)
    axs[1, 1].set_xlabel("", fontsize=120)
    axs[1, 1].set_title('D', weight='bold', fontsize=100, loc='left')
    axs[1, 1].tick_params(axis='both', labelsize=100)
    

    fig.tight_layout(pad=2, rect=[0, 0, 1, 1])  # Adjust for suptitle
    

    plt.savefig('./plots/utr_features_boxplots_with_kmer_bottom_right_legends_custom_fonts.png')
    plt.close()


file_pairs = [
    ('./src/exp_optimization/outputs/gc_best_seqs_IFNG.txt', '/src/exp_optimization/outputs/gc_init_seqs_IFNG.txt', 'IFNG'),
    ('/src/exp_optimization/outputs/gc_best_seqs_TLR6.txt', '/src/exp_optimization/outputs/gc_init_seqs_TLR6.txt', 'TLR6'),
    ('/src/exp_optimization/outputs/gc_best_seqs_TNF.txt', '/src/exp_optimization/outputs/gc_init_seqs_TNF.txt', 'TNF'),
    ('/src/exp_optimization/outputs/gc_best_seqs_TP53.txt', '/src/exp_optimization/outputs/gc_init_seqs_TP53.txt', 'TP53')
]

analyze_utr_features(file_pairs)