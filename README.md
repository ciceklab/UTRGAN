# UTRGAN: Deep Learning for 5' UTR Generation and Translation Optimization

[![License: CC BY-NC-SA 2.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%202.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/2.0/)

<p align="center">
<img src="./pipeline.png" width="700" alt="UTRGAN Pipeline"><br>
<em>Diagram of the generative model (WGAN) and the optimization procedure</em>
</p>

## Overview

UTRGAN is a deep learning-based model for novel 5' UTR sequence generation and optimization. The model integrates:

- **WGAN-GP architecture** for the generative model
- **Xpresso model** for optimizing TPM expression
- **FramePool model** for optimizing Mean Ribosome Load (MRL) 
- **MTtrans model** for optimizing Translation Efficiency (TE)

UTRGAN enables researchers to design and optimize 5' UTR sequences for improved gene expression and translation efficiency, with applications in biotechnology and synthetic biology.

## Table of Contents

- [Authors](#authors)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the GAN Model](#training-the-gan-model)
  - [Single Gene Optimization](#single-gene-optimization)
  - [Multiple Gene Optimization](#multiple-gene-optimization)
  - [Joint Optimization](#joint-optimization)
  - [MRL/TE Optimization](#mrle-optimization)
- [Reproducing Results](#reproducing-results)
- [Citations](#citations)
- [License](#license)
- [Contact](#contact)

## Authors

- Sina Barazandeh
- Furkan Ozden
- Ahmet Hincer
- Urartu Ozgur Safak Seker
- A. Ercument Cicek

## Installation

UTRGAN requires specific dependencies which can be easily installed using the provided conda environment file.

### Requirements

For easy setup, use the provided environment file:

```bash
# Create and activate conda environment
conda env create --name utrgan -f utrgan.yml
conda activate utrgan
```

The environment includes:
- tensorflow-gpu 2.14
- pytorch 2.0
- cudatoolkit 11.8
- biopython
- pandas
- scikit-learn
- seaborn
- and other dependencies

> **Note:** The provided environment file is configured for Linux systems. MacOS users may need to adjust package versions accordingly.

## Usage

> **Important:** Run scripts from their respective directories as indicated below.

### Training the GAN Model

To train the WGAN model:

```bash
python ./src/gan/wgan.py [-gpu GPU_IDS] [-bs BATCH_SIZE] [-d DATASET_PATH] [-lr LEARNING_RATE]
```

**Arguments:**
- `-gpu`: GPUs to use (sets CUDA_VISIBLE_DEVICES); uses CPU by default
- `-bs, --batch_size`: Batch size (default: 64)
- `-d, --dataset`: Path to CSV file with UTR samples (default: './../../data/utrdb2.csv')
- `-lr, --learning_rate`: Learning rate exponent (default: 5 for 1e-5)

### Single Gene Optimization

Optimize 5' UTR sequences for a single gene:

```bash
python ./src/exp_optimization/single_gene.py [-gpu GPU_IDS] [-g GENE_NAME] [-lr LEARNING_RATE] [-s STEPS] [-gc GC_CONTENT] [-bs BATCH_SIZE]
```

**Arguments:**
- `-gpu`: GPUs to use
- `-g`: Gene name (corresponding to a file in /src/exp_optimization/genes/GENE_NAME.txt)
- `-lr`: Learning rate (default: 3e-5)
- `-s`: Number of optimization iterations (default: 3,000)
- `-gc`: Upper limit for GC content percentage (default: no limit)
- `-bs`: Number of 5' UTR sequences to generate (default: 128)

### Multiple Gene Optimization

Optimize 5' UTR sequences for multiple genes:

```bash
python ./src/exp_optimization/multiple_genes.py [-gpu GPU_IDS] [-g GENE_NAME] [-lr LEARNING_RATE] [-s STEPS] [-dc NUM_GENES] [-bs BATCH_SIZE]
```

**Arguments:**
- `-gpu`: GPUs to use
- `-g`: Gene name file
- `-lr`: Learning rate (default: 3e-5)
- `-s`: Number of optimization iterations (default: 3,000)
- `-dc`: Number of randomly selected genes (default: 128)
- `-bs`: Number of 5' UTRs to optimize per DNA (default: 128)

### Joint Optimization

Jointly optimize translation efficiency and gene expression:

```bash
python ./src/exp_optimization/joint_opt.py [-gpu GPU_IDS] [-g GENE_NAME] [-s STEPS] [-lr LEARNING_RATE] [-bs BATCH_SIZE]
```

**Arguments:**
- `-gpu`: GPUs to use
- `-g`: Gene name file
- `-s`: Number of iterations for each optimization step (default: 1,000)
- `-lr`: Learning rate (default: 3e-5)
- `-bs`: Number of 5' UTRs to optimize per DNA (default: 128)

### MRL/TE Optimization

Optimize 5' UTRs for high Mean Ribosome Load or Translation Efficiency:

```bash
python ./src/mrl_te_optimization/optimize_variable_length.py [-lr LEARNING_RATE] [-task TASK] [-bs BATCH_SIZE]
```

**Arguments:**
- `-lr`: Learning rate (default: 3e-5)
- `-task`: Optimization target - either "te" or "mrl"
- `-bs`: Number of 5' UTRs to optimize (default: 128)

> **Note:** For statistical tests, larger batch sizes (up to 8192) can be used with different seeds

## Reproducing Results

To reproduce the results from our paper, run the following commands:

```bash
# Optimizations
python ./src/mrl_te_optimization/optimize_variable_length.py -task te
python ./src/mrl_te_optimization/optimize_variable_length.py -task mrl
python ./src/exp_optimization/multiple_genes.py
python ./src/exp_optimization/single_gene.py -g IFNG
python ./src/exp_optimization/single_gene.py -g TNF
python ./src/exp_optimization/single_gene.py -g TLR6
python ./src/exp_optimization/single_gene.py -g TP53
python ./src/exp_optimization/joint_opt.py -g IFNG
python ./src/exp_optimization/joint_opt.py -g TNF
python ./src/exp_optimization/joint_opt.py -g TLR6
python ./src/exp_optimization/joint_opt.py -g TP53

# Generate plots
python ./src/analysis/violin_plots.py
python ./src/analysis/plot_4x4.py
python ./src/analysis/opt_check.py
python ./src/analysis/mrl_te_opt.py
python ./src/exp_optimization/exp_joint.py
```

All plots will be saved to `./analysis/plots/`. P-values, confidence intervals, and effect sizes will be printed in the terminal output of the `violin_plots.py` script. Average and maximum increase statistics will be printed for each boxplot-generating script.

## Citations

If you use UTRGAN in your research, please cite our paper:

```
[Citation information will be added upon publication]
```

## License

- **[CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/)**
- Copyright 2023 © UTRGAN
- Free for academic use
- For commercial licensing inquiries, please contact the authors

## Contact

- For questions and comments: sina.barazandeh@bilkent.edu.tr
- For licensing inquiries: cicek@cs.bilkent.edu.tr

---

**Related Links:**
- [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning)
- [WGAN-GP Paper](https://arxiv.org/pdf/1704.00028v3.pdf)
- [Xpresso](https://github.com/vagarwal87/Xpresso)
- [FramePool](https://github.com/Karollus/5UTR)
- [MTtrans](https://github.com/holab-hku/MTtrans)