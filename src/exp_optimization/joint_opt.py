
from re import A, L
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torch
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import sys
import socket
import datetime
import random
import os
import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd
import numpy as np
import requests, sys
import json
from util import *
from framepool import *
from popen import Auto_popen

import scipy.stats as stats
abs_path = './../mrl_te_optimization/log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini'
Configuration = Auto_popen(abs_path)
import utils as util_motif

tf.compat.v1.enable_eager_execution()


parser = argparse.ArgumentParser()
parser.add_argument('-bs', type=int, required=False ,default=64)
parser.add_argument('-g', type=str, required=False ,default='IFNG')
parser.add_argument('-lr', type=int, required=False ,default=1)
parser.add_argument('-gpu', type=str, required=False ,default='-1')
parser.add_argument('-s', type=int, required=False ,default=10)
args = parser.parse_args()


if args.gpu == '-1':
    device = 'cpu'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'

BATCH_SIZE = args.bs
DIM = 40
SEQ_LEN = 128
MAX_LEN = SEQ_LEN

gpath = './../../models/checkpoint_3000.h5'
tpath = './script/checkpoint/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth'

def reverse_complement(sequence):
    """Compute the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 
                  'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'N': 'N', 'n': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence))


class GeneInfoRetriever:
    def __init__(self):
        self.base_url = "https://rest.ensembl.org"
        self.headers = {"Content-Type": "application/json"}
        self.sleep_time = 0.5  # Respect Ensembl API rate limits

    def _make_request(self, endpoint):
        """Make a request to the Ensembl REST API."""
        url = self.base_url + endpoint
        try:
            response = requests.get(url, headers=self.headers)
            time.sleep(self.sleep_time)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"Request error: {e}")
            return None

    def get_gene_id(self, gene_symbol, species="homo_sapiens"):
        """Retrieve the Ensembl gene ID for a gene symbol."""
        endpoint = f"/lookup/symbol/{species}/{gene_symbol}"
        response = self._make_request(endpoint)
        return response.get("id") if response else None

    def get_gene_coordinates(self, gene_id):
        """Retrieve genomic coordinates for a gene ID."""
        endpoint = f"/lookup/id/{gene_id}?expand=1"
        response = self._make_request(endpoint)
        if response:
            return {
                "chromosome": response.get("seq_region_name"),
                "start": response.get("start"),
                "end": response.get("end"),
                "strand": response.get("strand")
            }
        return None

    def get_tss_and_utr(self, gene_id):
        """Retrieve TSS and 5' UTR coordinates for the canonical transcript."""
        endpoint = f"/lookup/id/{gene_id}?expand=1&utr=1"
        response = self._make_request(endpoint)
        if not response or "Transcript" not in response:
            return None

        # Find canonical transcript
        canonical_transcript = None
        for transcript in response["Transcript"]:
            if transcript.get("is_canonical", 0) == 1:
                canonical_transcript = transcript
                break
        if not canonical_transcript:
            for transcript in response["Transcript"]:
                if transcript.get("biotype") == "protein_coding":
                    canonical_transcript = transcript
                    break
        if not canonical_transcript:
            canonical_transcript = response["Transcript"][0] if response["Transcript"] else None

        if not canonical_transcript:
            return None

        # Determine TSS and 5' UTR
        strand = canonical_transcript.get("strand")
        tss = canonical_transcript["start"] if strand == 1 else canonical_transcript["end"]
        five_prime_utr = None

        if "UTR" in canonical_transcript:
            for utr in canonical_transcript["UTR"]:
                if utr.get("object_type") == "five_prime_UTR":
                    five_prime_utr = {
                        "start": utr.get("start"),
                        "end": utr.get("end")
                    }
                    break

        # Verify TSS matches 5' UTR start
        if five_prime_utr:
            expected_tss = five_prime_utr["start"] if strand == 1 else five_prime_utr["end"]
            if expected_tss != tss:
                print(f"Warning: Adjusting TSS from {tss} to match 5' UTR {'start' if strand == 1 else 'end'} ({expected_tss})")
                tss = expected_tss

        return {
            "tss": tss,
            "strand": strand,
            "chromosome": canonical_transcript.get("seq_region_name"),
            "five_prime_utr": five_prime_utr,
            "transcript_id": canonical_transcript.get("id")
        }

    def get_promoter_sequence(self, gene_id, upstream=7000, downstream=4000):
        """Retrieve sequence around TSS (8kb upstream, 4kb downstream)."""
        tss_info = self.get_tss_and_utr(gene_id)
        if not tss_info:
            return None, None

        chromosome = tss_info["chromosome"]
        strand = tss_info["strand"]
        tss_position = tss_info["tss"]

        # Calculate region based on strand
        if strand == 1:
            seq_start = tss_position - upstream
            seq_end = tss_position + downstream - 1
        else:
            seq_start = tss_position - downstream
            seq_end = tss_position + upstream - 1

        seq_start = max(1, seq_start)

        # Store sequence coordinates
        sequence_coords = {
            "chromosome": chromosome,
            "start": seq_start,
            "end": seq_end,
            "strand": 1 if strand == 1 else -1
        }

        # Validate 5' UTR inclusion
        if tss_info["five_prime_utr"]:
            utr_start = tss_info["five_prime_utr"]["start"]
            utr_end = tss_info["five_prime_utr"]["end"]
            if not (seq_start <= utr_start <= seq_end and seq_start <= utr_end <= seq_end):
                print(f"Warning: 5' UTR ({utr_start}-{utr_end}) not fully within sequence ({seq_start}-{seq_end})")

        # Get sequence
        strand_str = "1" if strand == 1 else "-1"
        endpoint = f"/sequence/region/human/{chromosome}:{seq_start}..{seq_end}:{strand_str}"
        response = self._make_request(endpoint)
        return response.get("seq") if response else None, sequence_coords

    def get_gene_info(self, gene_symbol, species="homo_sapiens", output_json="gene_info.json"):
    
        if not os.path.exists(os.path.join('./.cache/',f"{gene_symbol}_info.json")):

            """Retrieve and save promoter sequence, TSS, 5' UTR, and coordinates."""
            # Get gene ID
            gene_id = self.get_gene_id(gene_symbol, species)
            if not gene_id:
                return {"error": f"Gene {gene_symbol} not found"}

            # Get TSS and 5' UTR
            tss_info = self.get_tss_and_utr(gene_id)
            if not tss_info:
                return {"error": "Could not retrieve TSS or transcript information"}

            # Get promoter sequence and coordinates
            promoter_sequence, sequence_coords = self.get_promoter_sequence(gene_id)
            if not promoter_sequence:
                return {"error": "Could not retrieve promoter sequence"}

            # Compile gene information
            gene_info = {
                "gene_symbol": gene_symbol,
                "gene_id": gene_id,
                "promoter_sequence": promoter_sequence,
                "sequence_length": len(promoter_sequence),
                "sequence_coordinates": sequence_coords,
                "tss": {
                    "chromosome": tss_info["chromosome"],
                    "position": tss_info["tss"],
                    "strand": "+" if tss_info["strand"] == 1 else "-"
                },
                "five_prime_utr": tss_info["five_prime_utr"],
                "transcript_id": tss_info["transcript_id"]
            }

            # Save to JSON
            try:
                os.makedirs(os.path.dirname('./.cache/'), exist_ok=True)
                with open(os.path.join('./.cache/',f"{gene_symbol}_info.json"), "w") as f:
                    json.dump(gene_info, f, indent=2)
                print(f"Saved gene information to {output_json}")
            except Exception as e:
                print(f"Error saving JSON: {e}")

        else:

            with open(os.path.join('./.cache/',f"{gene_symbol}_info.json"), "r") as f:
                gene_info = json.load(f)

        return gene_info

    def reverse_complement(self, sequence):
        """Compute the reverse complement of a DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 
                      'a': 't', 't': 'a', 'c': 'g', 'g': 'c', 'N': 'N', 'n': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(sequence))

    def replace_utr_in_sequence(self, gene_info_file, generated_utrs, target_length=10500, output_prefix="modified_sequence", write_json=False, verbose=False):
        """
        Replace original 5' UTR with generated UTRs, ensuring 10,500nt output.
        
        Parameters:
        gene_info_file (str): Path to JSON file with gene information
        generated_utrs (list): List of generated 5' UTR sequences (64-128nt)
        target_length (int): Desired output sequence length (default: 10500)
        output_prefix (str): Prefix for output JSON files
        
        Returns:
        list: List of modified sequences with metadata
        """
        try:
            # Read gene information
            with open(gene_info_file, "r") as f:
                gene_info = json.load(f)

            original_sequence = gene_info["promoter_sequence"]
            strand = gene_info["tss"]["strand"]
            tss_position = gene_info["tss"]["position"]
            sequence_coords = gene_info["sequence_coordinates"]
            seq_start = sequence_coords["start"]
            seq_end = sequence_coords["end"]
            five_prime_utr = gene_info["five_prime_utr"]
            gene_symbol = gene_info["gene_symbol"]
            transcript_id = gene_info["transcript_id"]

            if not five_prime_utr:
                print(f"Error: No 5' UTR information available for {gene_symbol}")
                return []

            # Calculate original 5' UTR position in sequence
            if strand == "+":
                utr_start_genomic = five_prime_utr["start"]
                utr_end_genomic = five_prime_utr["end"]
                utr_start_seq = utr_start_genomic - seq_start
                utr_end_seq = utr_end_genomic - seq_start
            else:
                utr_start_genomic = five_prime_utr["end"]  # TSS
                utr_end_genomic = five_prime_utr["start"]
                utr_start_seq = seq_end - utr_start_genomic
                utr_end_seq = seq_end - utr_end_genomic

            # Validate UTR positions
            seq_length = len(original_sequence)
            if not (0 <= utr_start_seq <= seq_length and 0 <= utr_end_seq <= seq_length):
                print(f"Error: 5' UTR coordinates (seq indices {utr_start_seq}-{utr_end_seq}) out of sequence bounds (0-{seq_length}) for {gene_symbol}")
                return []

            original_utr_length = abs(utr_end_genomic - utr_start_genomic) + 1
            if verbose:
                print(f"Original 5' UTR length for {gene_symbol}: {original_utr_length} nt")

            modified_sequences = []
            for i, new_utr in enumerate(generated_utrs):
                new_utr_length = len(new_utr)

                # Construct new sequence
                if strand == "+":
                    new_sequence = (
                        original_sequence[:utr_start_seq] +
                        new_utr +
                        original_sequence[utr_end_seq + 1:]
                    )
                    new_utr_start_genomic = utr_start_genomic
                    new_utr_end_genomic = utr_start_genomic + new_utr_length - 1
                    if len(new_sequence) > target_length:
                        new_sequence = new_sequence[:target_length]
                        sequence_coords["end"] = seq_start + target_length - 1
                    elif len(new_sequence) < target_length:
                        if verbose:
                            print(f"Error: Sequence too short ({len(new_sequence)} nt) after UTR replacement for {gene_symbol}")
                            continue
                else:
                    new_utr_rc = reverse_complement(new_utr)
                    new_sequence = (
                        original_sequence[:min(utr_start_seq, utr_end_seq)] +
                        new_utr_rc +
                        original_sequence[max(utr_start_seq, utr_end_seq) + 1:]
                    )
                    new_utr_start_genomic = utr_start_genomic
                    new_utr_end_genomic = utr_start_genomic - new_utr_length + 1
                    if len(new_sequence) > target_length:
                        trim_amount = len(new_sequence) - target_length
                        new_sequence = new_sequence[trim_amount:]
                        sequence_coords["start"] = seq_start + trim_amount
                    elif len(new_sequence) < target_length:
                        if verbose:
                            print(f"Error: Sequence too short ({len(new_sequence)} nt) after UTR replacement for {gene_symbol}")
                            continue

                # Store modified sequence and metadata
                modified_info = {
                    "gene_symbol": gene_symbol,
                    "transcript_id": transcript_id,
                    "modified_sequence": new_sequence,
                    "sequence_length": len(new_sequence),
                    "sequence_coordinates": sequence_coords.copy(),
                    "tss": gene_info["tss"],
                    "five_prime_utr": {
                        "start": new_utr_start_genomic,
                        "end": new_utr_end_genomic,
                        "sequence": new_utr if strand == "+" else new_utr_rc
                    },
                    "original_utr_length": original_utr_length,
                    "new_utr_length": new_utr_length,
                    "utr_index": i + 1
                }

                # Save to JSON
                if write_json:
                    output_file = f"{output_prefix}_{gene_symbol}_utr_{i+1}.json"
                    try:
                        os.makedirs(os.path.dirname(output_file), exist_ok=True)
                        with open(output_file, "w") as f:
                            json.dump(modified_info, f, indent=2)
                        print(f"Saved modified sequence {i+1} for {gene_symbol} to {output_file}")
                    except Exception as e:
                        print(f"Error saving modified sequence {i+1} for {gene_symbol}: {e}")

                modified_sequences.append(modified_info["modified_sequence"])

            return modified_sequences

        except Exception as e:
            # print(f"Error processing UTR replacement for {gene_info.get('gene_symbol', 'unknown')}: {e}")
            print(f"Error processing UTR replacement for gene: {e}")
            return []


    def replace_utr_in_multiple_sequences(self, gene_symbols, generated_utrs, target_length=10500, cache_dir="./.cache", output_prefix="modified_sequence", verbose=False):
            """
            Replace 5' UTRs for multiple genes with generated UTRs.
            
            Parameters:
            gene_symbols (list): List of gene names
            generated_utrs (list): List of generated 5' UTR sequences (64-128nt)
            target_length (int): Desired output sequence length (default: 10500)
            cache_dir (str): Directory containing cached gene info JSON files
            output_prefix (str): Prefix for output JSON files
            
            Returns:
            list: List of n_utrs * n_genes modified sequences with metadata
            """
            all_modified_sequences = []
            n_utrs = len(generated_utrs)
            n_genes = len(gene_symbols)

            for gene_symbol in gene_symbols:
                json_file = os.path.join(cache_dir, f"{gene_symbol}_info.json")
                if not os.path.exists(json_file):
                    print(f"Error: Gene info file {json_file} not found")
                    continue
                
                if verbose:
                    print(f"\nProcessing gene: {gene_symbol}")
                modified_sequences = self.replace_utr_in_sequence(
                    gene_info_file=json_file,
                    generated_utrs=generated_utrs,
                    target_length=target_length,
                    output_prefix=os.path.join(cache_dir, output_prefix)
                )

                if modified_sequences:
                    all_modified_sequences.extend(modified_sequences)
                else:
                    if verbose:
                        print(f"No modified sequences generated for {gene_symbol}")

            expected_count = n_utrs * n_genes
            actual_count = len(all_modified_sequences)
            if verbose:
                print(f"\nGenerated {actual_count} modified sequences (expected: {expected_count})")

            return all_modified_sequences

def fetch_seq(start, end, chr, strand):
    server = "https://rest.ensembl.org"
    
    ext = "/sequence/region/human/" + str(chr) + ":" + str(start) + ".." + str(end) + ":" + str(strand) +  "?"
    
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    return r.text

def convert_model(model_:Model):
    print(model_.summary())
    input_ = tf.keras.layers.Input(shape=( 10500, 4))
    input = input_
    for i in range(len(model_.layers)-1):

        # print(type(model_.layers[i+1]))
        
        if isinstance(model_.layers[i+1],tf.keras.layers.Concatenate):
            paddings = tf.constant([[0,0],[0,6]])
            output = tf.pad(input, paddings, 'CONSTANT')
            input = output
        else:
            if not isinstance(model_.layers[i+1],tf.keras.layers.InputLayer):
                output = model_.layers[i+1](input)
                input = output

            if isinstance(model_.layers[i+1],tf.keras.layers.Conv1D):
                pass

    model = tf.keras.Model(inputs=input_, outputs=output)
    model.compile(loss="mse", optimizer="adam")
    return model

def one_hot(seq):
    convert = False
    if isinstance(seq, tf.Tensor):
        seq = seq.numpy().astype(str)
        convert = True

    num_seqs = len(seq)
    seq_len = len(seq[0])
    seqindex = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
    seq_vec = np.zeros((num_seqs,seq_len,4), dtype='bool')
    for i in range(num_seqs):
        thisseq = seq[i]
        for j in range(seq_len):
            try:
                seq_vec[i,j,seqindex[thisseq[j]]] = 1
            except:
                pass
    
    if convert:
        seq_vec = tf.convert_to_tensor(seq_vec,dtype=tf.float32)


    return seq_vec

def gen_random_dna(len=10500,size=SEQ_LEN):
    list_ = ['A','C','G','T']
    dnas = []
    for i in range(size):
            
        list_ = ['A','C','G','T']
        mydna = 'AGT'
        for i in range(len-3):
            char = list_[random.randint(0,3)]
            mydna = mydna + char
        dnas.append(mydna)


    return dnas
    
def select_dna_single(fname='small_seqs.npy',batch_size=64):
    refs = np.load(fname)
    indice = random.sample(range(0,refs.shape[0]),1)
    refs = refs
    return indice[0], refs    

def recover_seq(samples, rev_charmap):
    """Convert samples to strings and save to log directory."""
    if isinstance(samples,tf.Tensor):
        samples = samples.numpy()

    char_probs = samples
    argmax = np.argmax(char_probs, 2)
    seqs = []
    for line in argmax:
        s = "".join(rev_charmap[d] for d in line)
        s = s.replace('*','')
        seqs.append(s)

    seqs = np.array(seqs)
    return seqs

rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

def select_best(scores, seqs, gc_control=False, GC=-1):
    selected_scores = []
    selected_seqs = []
    for i in range(len(scores[0])):
        best = scores[0][i]
        best_seq = seqs[0][i]
        for j in range(len(scores)):
            if scores[j][i] > best:
                if gc_control:
                    if get_gc_content(seqs[j][i]) < GC:
                        best = scores[j][i]
                        best_seq = seqs[j][i]
                else:
                    best = scores[j][i]
                    best_seq = seqs[j][i]

        selected_scores.append(best)
        selected_seqs.append(best_seq)

    return selected_seqs, selected_scores

def log(samples_dir=False):
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join("./logs/", "gan_test_opt", stamp)

    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    return full_logdir, 0

if __name__ == "__main__":

    model = tf.keras.models.load_model('./../../models/humanMedian_trainepoch.11-0.426.h5')

    model = convert_model(model)

    gene_name = args.g

    ref = ''
    
    output_json = f"{gene_name}_info.json"


    retriever = GeneInfoRetriever()

    if not os.path.exists(os.path.join('./.cache/',output_json)):

        # Retrieve gene information
        gene_info = retriever.get_gene_info(gene_name, output_json=output_json)

        if "error" in gene_info:
            print(f"Error: {gene_info['error']}")
        else:
            ref = gene_info["promoter_sequence"] 
    else:
        with open(os.path.join('./.cache/',output_json), "r") as f:
            gene_info = json.load(f)
            ref = gene_info["promoter_sequence"]

    original_gene_sequence = ref

    wgan = tf.keras.models.load_model(gpath)

    """
    Data:
    """

    noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,DIM]))

    tf.random.set_seed(25)

    np.random.seed(25)
    seqs_orig = one_hot([original_gene_sequence[:10500]])
    pred_orig = model(seqs_orig) 

    ################ TE Optimization ######################
    te_model = torch.load(tpath,map_location=torch.device(device))['state_dict']  
    te_model.train().to(device)
    
    
    MODEL = "TE"
    opt_model = te_model



    noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,DIM]))

    sequences_init = wgan(noise)

    gen_seqs_init = sequences_init.numpy().astype('float')

    seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)
    

    seqs_init = retriever.replace_utr_in_sequence(f"./.cache/{gene_name}_info.json", seqs_gen_init)

    seqs_init = one_hot(seqs_init)

    pred_init = model(seqs_init) 

    t = tf.reshape(pred_init,(-1))

    init_t = t.numpy().astype('float')
    
    one_hots = one_hot_all_motif(np.array(seqs_gen_init))
    seqs = torch.tensor(one_hots,dtype=torch.double)
    seqs = torch.transpose(seqs, 1, 2)
    seqs = seqs.float().to(device)
    pred_init = opt_model.forward(seqs)
    pred_init = torch.flatten(pred_init)

    preds_init = pred_init.cpu().detach().numpy()
    pred_init = np.average(preds_init)

    max_init = np.max(pred_init)
    min_init = np.min(pred_init)
    
    OPTIMIZE = True

    means = []
    maxes = []

    noise_small = tf.random.normal(shape=[BATCH_SIZE,DIM],stddev=1e-4)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    STEPS = args.s

    if OPTIMIZE:
        iter_ = 0
        for opt_iter in tqdm(range(int(STEPS))):
            
            with tf.GradientTape() as gtape:
                gtape.watch(noise)
                sequences = wgan(noise)

                seqs_gen = recover_seq(sequences, rev_rna_vocab)
                seqs_str = seqs_gen
                

                seqs = torch.tensor(np.array(one_hot_all_motif(seqs_gen),dtype=np.float32))    
                    
                seqs = torch.transpose(seqs, 1, 2)
                seqs = seqs.float()
                seqs = torch.tensor(seqs.to(device), requires_grad=True)
                pred = opt_model.forward(seqs)
                pred = torch.flatten(pred)
                score = torch.mean(pred)
                t = torch.flatten(pred)
                mx = t.cpu().detach().numpy()
                mx = np.max(mx)
                
                sum_ = torch.mean(t).cpu().detach().numpy()
                
                maxes.append(mx)
                means.append(sum_/BATCH_SIZE)
                
                pred.backward(torch.ones_like(pred))
                
                g1 = seqs.grad
                # print(g1.grad)
                
                g1 = g1.cpu().detach().numpy()
                g1 = tf.convert_to_tensor(g1)
                # print(tf.shape(g1))
                g1 = tf.transpose(g1, perm=[0,2,1])
                g1 = tf.pad(g1,tf.constant([[0, 0], [0, 0], [0, 1]]),"CONSTANT")
                g1 = tf.math.scalar_mul(-1.0,g1)
                
                
                g2 = gtape.gradient(sequences,noise,output_gradients=g1)

            a1 = g2 + noise_small
            change = [(a1,noise)]
            optimizer.apply_gradients(change)

            # iters_.append(iter_)
            iter_ += 1

        sequences_opt = wgan(noise)
        
        gen_seqs_opt = sequences_opt.numpy().astype('float')

        seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)
        
        one_hots = np.array(one_hot_all_motif(seqs_gen_opt))
        # print(np.shape(one_hots))
        seqs = torch.tensor(one_hots,dtype=torch.double)
        seqs = torch.transpose(seqs, 1, 2)
        seqs = seqs.float().to(device)
        preds_opt = opt_model.forward(seqs)
        
        preds_opt = preds_opt.cpu().data.numpy()

    ###########################################################



    diffs = []
    init_exps = []

    opt_exps = []

    orig_vals = []


    noise_small = tf.random.normal(shape=[BATCH_SIZE,DIM],stddev=1e-5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-2)

    '''
    Original Gene Expression
    '''



    '''
    Optimization takes place here.
    '''

    bind_scores_list = []
    bind_scores_means = []
    sequences_list = []



    iters_ = []

    OPTIMIZE = True

    DNA_SEL = False

    gan_noise = noise.numpy().astype('float')
    noise = tf.Variable(tf.convert_to_tensor(gan_noise,dtype=tf.float32))

    sequences_step = wgan(noise)

    gen_seqs_step = sequences_step.numpy().astype('float')

    seqs_gen_step = recover_seq(gen_seqs_step, rev_rna_vocab)

    seqs_step = retriever.replace_utr_in_sequence(f"./.cache/{gene_name}_info.json", seqs_gen_step)

    seqs_step = one_hot(seqs_step)

    pred_step = model(seqs_step) 

    intermediate_pred = tf.reshape(pred_step,(-1))

    intermediate_pred = intermediate_pred.numpy().astype('float')

    seqs_te =  torch.transpose(torch.tensor(np.array(one_hot_all_motif(seqs_gen_init),dtype=np.float32)),2,1).float().to(device)

    te_preds_init = te_model.forward(seqs_te).cpu().data.numpy()

    means = []
    maxes = []
    
    STEPS = args.s

    seqs_collection = []
    scores_collection = []

    if OPTIMIZE:
        
        iter_ = 0
        for opt_iter in tqdm(range(STEPS)):
            
            with tf.GradientTape() as gtape:

                gtape.watch(noise)
                
                sequences = wgan(noise)
                seqs_collection.append(seqs_gen)

                seqs_gen = recover_seq(sequences, rev_rna_vocab)

                seqs2 = retriever.replace_utr_in_sequence(f"./.cache/{gene_name}_info.json", seqs_gen)
            
                seqs = one_hot(seqs2)
                seqs = tf.convert_to_tensor(seqs,dtype=tf.float32)
                
                with tf.GradientTape() as ptape:

                    ptape.watch(seqs)

                    pred =  model(seqs)
                    t = tf.reshape(pred,(-1))
                    scores_collection.append(t.numpy().astype('float'))
                    mx = np.amax(t.numpy().astype('float'),axis=0)
                    mx = np.max(mx)
                    
                    sum_ = tf.reduce_sum(t)
                    maxes.append(mx)
                    means.append(sum_/BATCH_SIZE)
                    pred = tf.math.scalar_mul(-1.0, pred)

                g1 = ptape.gradient(pred,seqs)

                
                g1 = tf.slice(g1,[0,7000,0],[-1,SEQ_LEN,-1])


                tmp_g = g1.numpy().astype('float')
                tmp_seqs = seqs_gen

                tmp_lst = np.zeros(shape=(BATCH_SIZE,SEQ_LEN,5))
                for i in range(len(tmp_seqs)):
                    len_ = len(tmp_seqs[i])
                    
                    edited_g = tmp_g[i][:len_,:]

                    edited_g = np.pad(edited_g,((0,SEQ_LEN-len_),(0,1)),'constant')   
                    
                    tmp_lst[i] = edited_g
                    
                g1 = tf.convert_to_tensor(tmp_lst,dtype=tf.float32)

                g2 = gtape.gradient(sequences,noise,output_gradients=g1)

            a1 = g2 + noise_small
            change = [(a1,noise)]

            optimizer.apply_gradients(change)

            iters_.append(iter_)
            iter_ += 1

        sequences_opt = wgan(noise)

        gen_seqs_opt = sequences_opt.numpy().astype('float')

        seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)

        seqs_opt= retriever.replace_utr_in_sequence(f"./.cache/{gene_name}_info.json", seqs_gen_opt)

        seqs_opt = one_hot(seqs_opt)

        pred_opt = model(seqs_opt)

        t = tf.reshape(pred_opt,(-1))
        opt_t = t.numpy().astype('float')

        seqs_te =  torch.transpose(torch.tensor(np.array(one_hot_all_motif(seqs_gen_opt),dtype=np.float32)),2,1).float().to(device)

        te_preds_opt = te_model.forward(seqs_te).cpu().data.numpy()

        best_seqs, best_scores = select_best(scores_collection, seqs_collection)


        os.makedirs("./outputs_joint", exist_ok=True)


        with open('./outputs_joint/init_exps_'+gene_name+'.txt', 'w') as f:
            for item in init_t:
                f.write(f'{item}\n')

        with open('./outputs_joint/opt_exps_'+gene_name+'.txt', 'w') as f:
            for item in opt_t:
                f.write(f'{item}\n')

        with open('./outputs_joint/best_seqs_'+gene_name+'.txt', 'w') as f:
            for item in seqs_gen_opt:
                f.write(f'{item}\n')

        with open('./outputs_joint/init_seqs_'+gene_name+'.txt', 'w') as f:
            for item in seqs_gen_init:
                f.write(f'{item}\n')


        print("TE Optimization Step:")
        print(f"Avg. Initial TE:{np.mean(preds_init)}")
        print(f"Max Initial TE:{np.amax(preds_init)}")
        print(f"Avg. Opt TE:{np.mean(preds_opt)}")
        print(f"Max Opt TE:{np.amax(preds_opt)}")
        print(f"Avg. Exp. Before First Step: {np.mean(init_t)}")
        print("Exp. Optimizization Step:")
        print(f'Avg. Exp. After First Step: {np.mean(intermediate_pred)}')
        print(f'Avg. Best Exp. After Second Step: {np.mean(best_scores)}')
        print("TE:")
        print(f'TE After First Step: {np.average(preds_opt)}')
        print(f'TE After Second Step: {np.average(te_preds_opt)}')
