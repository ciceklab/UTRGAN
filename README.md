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
conda env create --name utrgan -f environment.yml
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

> **Update (April 2025):** The latest version of UTRGAN retrieves latest version of the gene information, including 5' UTR, TSS, and sequence of the genes querying the Ensembl Biomart API. Variance in the results are expected if the information obtained from the API changes. Please note that the API might sometimes fail, in that case, please wait a few seconds and try running the gene expression optimization code again.

> **Note:** We encourage trying optimization with different initializations to get more diverse sequences and select the best results. Since usually higher batch size does not fit in the GPU, you can alternatively try running the code multiple times and use the best sequences overall.


> **Important:** You can run scripts both from the root directory or from their respective directories as indicated below. 

### Training the GAN Model

To train the WGAN model:

```bash
python train.py [-gpu GPU_IDS] [-bs BATCH_SIZE] [-d DATASET_PATH] [-lr LEARNING_RATE]
```

**Arguments:**
- `-gpu`: GPUs to use (sets CUDA_VISIBLE_DEVICES); uses CPU by default
- `-bs, --batch_size`: Batch size (default: 64)
- `-d, --dataset`: Path to CSV file with UTR samples (default: './../../data/utrdb2.csv')
- `-lr, --learning_rate`: Learning rate exponent (default: 5 for 1e-5)

### Single Gene Optimization

Optimize 5' UTR sequences for a single gene:

> **Important:** Gene name is required here.

```bash
python ./src/exp_optimization/single-gene.py [-gpu GPU_IDS] [-g GENE_NAME] [-lr LEARNING_RATE] [-s STEPS] [-gc GC_CONTENT] [-bs BATCH_SIZE]
```

**Arguments:**
- `-gpu`: GPUs to use (-1: no gpu, ow: any gpu)
- `-lr`: Learning rate (default: 3e-5)
- `g`: Gene Symbol/Name
- `-s`: Number of optimization iterations (default: 3,000)
- `-gc`: Upper limit for GC content percentage (default: no limit)
- `-bs`: Number of 5' UTR sequences to generate (default: 128)

### Multiple Gene Optimization

Optimize 5' UTR sequences for multiple genes:

```bash
python ./src/exp_optimization/multiple-genes.py [-gpu GPU] [-g GENE_NAMES] [-lr LEARNING_RATE] [-s STEPS] [-bs BATCH_SIZE]
```

**Arguments:**
- `-gpu`: GPUs to use
- `-g`: Gene names separated by comma (e.g., TLR6,INFG,TP53,TNF)
- `-lr`: Learning rate (default: 3e-5)
- `-s`: Number of optimization iterations (default: 3,000)
- `-bs`: Number of 5' UTRs to optimize per DNA (default: 100)

### Joint Optimization

Jointly optimize translation efficiency and gene expression:

```bash
python ./src/exp_optimization/joint_opt.py [-gpu GPU] [-g GENE_NAME] [-s STEPS] [-lr LEARNING_RATE] [-bs BATCH_SIZE]
```

**Arguments:**
- `-gpu`: GPUs to use
- `-g`: Gene names separated by comma (e.g., TLR6,INFG,TP53,TNF)
- `-s`: Number of iterations for each optimization step (default: 1,000)
- `-lr`: Learning rate (default: 3e-5)
- `-bs`: Number of 5' UTRs to optimize per DNA (default: 100)

### MRL/TE Optimization

Optimize 5' UTRs for high Mean Ribosome Load or Translation Efficiency:

```bash
python ./src/mrl_te_optimization/optimize_te_mrl.py [-lr LEARNING_RATE] [-task TASK] [-s ITERATIONS] [-bs BATCH_SIZE]
```

**Arguments:**
- `-lr`: Learning rate (default: 3e-5)
- `-s`: Number of Iterations (default: 10000)
- `-task`: Optimization target - either "te" or "mrl"
- `-bs`: Number of 5' UTRs to optimize (default: 128)

> **Note:** For statistical tests, larger batch sizes (up to 8192) can be used with different seeds

### Example Optimizations (MRL/TE)

Run the optimize_te_mrl.ipynb file in the root folder:

You can change the following parameters for different results

```bash
BATCH_SIZE = 64
TASK = "mrl"
GPU = '-1'
STEPS = 10
```

### Example Optimizations (Single Gene Expression)

Run the exp_optimization_single.ipynb file in the root folder:

You can change the following parameters for different results. See the details above for the meaning of the parameters

```bash
BATCH_SIZE = 500
GENE = 'VEGFA'
GC_LIMIT = -1.00
LR = 0.005
GPU = '0'
STEPS = 2
```

### Example Optimizations (Multiple Gene Expression)

Run the exp_optimization_multiple.ipynb file in the root folder:

You can change the following parameters for different results. See the details above for the meaning of the parameters

```bash
BATCH_SIZE = 100 
N_GENES = 8
LR = 0.001
GPU = '0'
STEPS = 10
gene_names = ["MYOC", "TIGD4", "ATP6V1B2", "TAGLN", "COX7A2L", "IFNGR2", "TNFRSF21", "SETD6"]
```

## Citations

If you use UTRGAN in your research, please cite our paper:

```
[Citation information will be added upon publication]
```

## License

- **[CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/)**
- Copyright 2025 © UTRGAN
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