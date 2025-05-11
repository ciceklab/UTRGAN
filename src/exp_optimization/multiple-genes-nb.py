import requests
import json
import time
import numpy as np
import os
from re import A, L
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import argparse
from util import *
from framepool import *
import sys
from Bio import SeqIO

tf.compat.v1.enable_eager_execution()

__file__ = os.getcwd()

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from models import Modules
import configparser
from sklearn.preprocessing import OneHotEncoder
import logging
import collections
from models.ScheduleOptimizer import ScheduledOptim


parser = argparse.ArgumentParser()
parser.add_argument('-g', type=str, required=False ,default="MYOC,TIGD4,ATP6V1B2,TAGLN,COX7A2L,IFNGR2,TNFRSF21,SETD6")
parser.add_argument('-t', type=str, required=False ,default="ANTXR2,NFIL3,UNC13D,DHRS2,RPS13,HBD,METAP1D,NCALD")
parser.add_argument('-bs', type=int, required=False ,default=100)
parser.add_argument('-lr', type=int, required=False ,default=4)
parser.add_argument('-gpu', type=str, required=False ,default='-1')
parser.add_argument('-s', type=int, required=False ,default=10000)
args = parser.parse_args()

BATCH_SIZE = args.bs
LR = args.lr
GPU = args.gpu
STEPS = args.s

if GPU == '-1':
    device = 'cpu'
else:
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
        device = 'cuda'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'

LR = np.exp(-int(LR))

gene_names = args.genes.split(",")

target_genes = args.targets.split(",")

N_GENES = len(gene_names)

# ====================|   some path   |=======================
global script_dir
global data_dir
global log_dir
global pth_dir
# global cell_lines

global egfp_seq

with open(os.path.join(__file__,"machine_configure.json"),'r') as f:
    config = json.load(f)   

script_dir = config['script_dir']
data_dir = config['data_dir']
log_dir = config['log_dir']
pth_dir = config['pth_dir']



# =====================| one hot encode |=======================

class Seq_one_hot(object):
    def __init__(self,seq_type='nn',seq_len=100):
        """
        initiate the sequence one hot encoder
        """
        self.seq_len=seq_len
        self.seq_type =seq_type
        self.enable_encoder()
        
    def enable_encoder(self):
        if self.seq_type == 'nn':
            self.encoder = OneHotEncoder(sparse=False)
            self.encoder.drop_idx_ = None
            self.encoder.categories_ = [np.array(['A', 'C', 'G', 'T'], dtype='<U1')]*self.seq_len

    def discretize_seq(self,data):
        """
        discretize sequence into character
        argument:
        ...data: can be dataframe with UTR columns , or can be single string
        """
        if type(data) is pd.DataFrame:
            return np.stack(data.UTR.apply(lambda x: list(x)))
        elif type(data) is str:
            return np.array(list(data))
    
    def transform(self,data,flattern=True):
        """
        One hot encode
        argument:
        data : is a 2D array
        flattern : True
        """
        X = self.encoder.transform(data)                             # 400 for each seq
        X_M = np.stack([seq.reshape(self.seq_len,4) for seq in X])   # i.e 100*4
        return X if flattern else X_M
    
    def d_transform(self,data,flattern=True):
        """
        discretize data and put into transform
        """
        X = self.discretize_seq(data)
        return self.transform(X,flattern)


# =====================|   logger       |=======================

def setup_logs(vae_log_path,level=None):
    """

    :param save_dir:  the directory to set up logs
    :param type:  'model' for saving logs in 'logs/cpc'; 'imp' for saving logs in 'logs/imp'
    :param run_name:
    :return:logger
    """
    # initialize logger
    logger = logging.getLogger("VAE")
    logger.setLevel(logging.INFO)
    if level=='warning':
        logger.setLevel(logging.WARNING)

    # create the logging file handler
    log_file = os.path.join(vae_log_path)
    fh = logging.FileHandler(log_file)

    # create the logging console handler
    ch = logging.StreamHandler()

    # format
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)

    # add handlers to logger object
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def clean_value_dict(dict):
    """
    deal with verbose dict where the values maybe torch object, extact the item and return clean dict
    """
    clean_dict={}
    for k,v in dict.items():
        
        try:
            v = v.item()
        except:
            v = v
        clean_dict[k] = v
    return clean_dict

def fix_parameter(model,modual_to_fix,fix_or_unfix=False):
    """
    for a given model, fix part of the parameter to fine-tuning / transfering 
    args:
    model : `nn.Modual`,initiated model instance
    modual_to_fix : str, define which part of the model will not update by gradient 
                    e.g. "soft_share" then 
    """
    
    fix_part = eval("model."+modual_to_fix)   # e.g. model.shoft_share
     
    for param in fix_part.parameters():
            param.requires_grad = fix_or_unfix
    
    return model

def unfix_parameter(model,modual_to_fix,fix_or_unfix=False):
    return fix_parameter(model,modual_to_fix,fix_or_unfix=True)

def snapshot(vae_pth_path, state):
    logger = logging.getLogger("VAE")
    # torch.save can save any object
    # dict type object in our cases
    torch.save(state, vae_pth_path)
    logger.info("Snapshot saved to {}\n".format(vae_pth_path))


def load_model(popen,model,logger=None):
    
    info = lambda x: print(x) if logger==None else logger.info(x)
    popen.vae_pth_path = '/mnt/sina/run/ml/gan/dev/git/UTRGAN/src/mrl_optimization/script/checkpoint/RL_hard_share_MTL/3M/small_repective_filed_strides1113-model_best_cv1.pth'
    checkpoint = torch.load(popen.vae_pth_path, map_location=torch.device('cpu')) 
    if isinstance(checkpoint['state_dict'], collections.OrderedDict):
            # optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = checkpoint['state_dict']
    
    info(' \t \t ==============<<< encoder load from >>>============== \t \t ')
    info(" \t"+popen.vae_pth_path)
    
    return model
    
def get_config_cuda(config_file):
    with open(config_file,'r') as f:
        lines = f.read_lines()
        for line in lines:
            if "cuda_id =" in line:
                device = line.split("=")[1].strip()
                break
    device = int(device) if device.isdigit() else device
    return device

def resume(popen,optimizer,logger):
    """
    for a experiment, check whether it;s a new run, and create dir 
    """
    #run_name = model_stype + time.strftime("__%Y_%m_%d_%H:%M"))
    
    if popen.Resumable:
        
        checkpoint = torch.load(popen.vae_pth_path, map_location=torch.device('cpu'))   # xx-model-best.pth
        previous_epoch = checkpoint['epoch']
        previous_loss = checkpoint['validation_loss']
        previous_acc = checkpoint['validation_acc']
        
        
        # very important
        if (type(optimizer) == ScheduledOptim):
            optimizer.n_current_steps = popen.n_current_steps
            optimizer.delta = popen.delta
        
        logger.info(" \t \t ========================================================= \t \t ")
        logger.info(' \t \t ==============<<< Resume from checkpoint>>>============== \t \t \n')
        logger.info(" \t"+popen.vae_pth_path+'\n')
        logger.info(" \t \t ========================================================= \t \t \n")
        
        return previous_epoch,previous_loss,previous_acc
        
    
egfp_seq = "atgggcgaattaagtaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttct"
eGFP_seq = egfp_seq.upper()

class Auto_popen(object):
    def __init__(self,config_file):
        """
        read the config_fiel
        """
        # machine config path
        self.shuffle = True
        self.script_dir = script_dir
        self.data_dir = data_dir
        self.data_dir = '/mnt/sina/run/ml/gan/motif/MTtrans/test.csv'
        self.log_dir = log_dir
        self.pth_dir = pth_dir
        self.set_attr_as_none(['te_net_l2','loss_fn','modual_to_fix','other_input_columns','pretrain_pth','kfold_index'])
        self.split_like = False
        self.loss_schema = 'constant'
        
        # transform to dict and convert to  specific data type
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.config_file = config_file
        self.config_dict = {item[0]: eval(item[1]) for item in self.config.items('DEFAULT')}
        
        # assign some attr from config_dict         
        self.set_attr_from_dict(self.config_dict.keys())
        self.check_run_and_setting_name()                          # check run name
        self._dataset = "_" + self.dataset if self.dataset != '' else self.dataset
        # the saving direction
        self.path_category = self.config_file.split('/')[-4]
        self.vae_log_path = config_file.replace('.ini','.log')
        

        self.Resumable = False

        # covariates for other input
        self.n_covar = len(self.other_input_columns) if self.other_input_columns is not None else 0
        
        # generate self.model_args
        self.get_model_config()

    @property
    def vae_pth_path(self):
        save_to = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name)
        if self.kfold_index is None:
            pth = os.path.join(save_to, self.run_name + '-model_best.pth')
        elif type(self.kfold_index) == int:
            k = self.kfold_index
            pth = os.path.join(save_to, self.run_name + f'-model_best_cv{k}.pth')
        return pth
    
    @vae_pth_path.setter
    def vae_pth_path(self, path):
        self._vae_pth_path = path
    
    def set_attr_from_dict(self,attr_ls):
        for attr in attr_ls:
            self.__setattr__(attr,self.config_dict[attr])
    
    def set_attr_as_none(self,attr_ls):
        for attr in attr_ls:
            self.__setattr__(attr,None)

    def check_run_and_setting_name(self):
        file_name = self.config_file.split("/")[-1]
        dir_name = self.config_file.split("/")[-2]
        self.setting_name = dir_name
        assert self.run_name == file_name.split(".")[0]
        # assert self.run_name == file_name.split(".")[0]")[0]
        
    def get_model_config(self):
        """
        assert we type in the correct model type and group them into model_args
        """
        self.model_type = 'RL_hard_share'
        if self.model_type in dir(Modules):
            self.Model_Class = eval("Modules.{}".format(self.model_type))
        else:
            raise NameError("not such model type")
        
        # conv_args define the soft-sharing part
        conv_args = ["channel_ls","kernel_size","stride","padding_ls","diliation_ls","pad_to"]
        self.conv_args = tuple([self.__getattribute__(arg) for arg in conv_args])
        
        # left args dfine the tower part in which the arguments are different among tasks
        left_args={# Backbone models
                    'RL_regressor':["tower_width","dropout_rate"],
                    'RL_clf':["n_class","tower_width","dropout_rate"],
                    'RL_gru':["tower_width","dropout_rate"],
                    'RL_FACS': ["tower_width","dropout_rate"],
                    'RL_hard_share':["tower_width","dropout_rate", "activation","cycle_set" ],
                    'RL_covar_reg':["tower_width","dropout_rate", "activation", "n_covar", "cycle_set" ],
                    'RL_covar_intercept':["tower_width","dropout_rate", "activation", "n_covar", "cycle_set" ],
                    'RL_mish_gru':["tower_width","dropout_rate"],
                    # GP models
                    'GP_net': ['tower_width', 'dropout_rate', 'global_pooling', 'activation', 'cycle_set'],
                    'Frame_GP': ['tower_width', 'dropout_rate', 'activation', 'cycle_set'],
                    'RL_Atten': ['qk_dim', 'n_head', 'n_atten_layer', 'tower_width', 'dropout_rate', 'activation', 'cycle_set'],
                    # Koo net
                    'Conf_CNN' : ['pool_size'],
                    }[self.model_type]
        
        self.model_args = [self.conv_args] + [self.__getattribute__(arg) for arg in left_args]
        
            
    def check_experiment(self,logger):
        """
        check any unfinished experiment ?
        """
        log_save_dir = os.path.dirname(self.vae_log_path)
        pth_save_dir = os.path.join(self.pth_dir,self.model_type+self._dataset,self.setting_name)
        # make dirs 
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        if not os.path.exists(pth_save_dir):
            os.makedirs(pth_save_dir)
        
        # check resume
        if os.path.exists(self.vae_log_path) & os.path.exists(self.vae_pth_path):
            self.Resumable = True
            logger.info(' \t \t ==============<<<  Experiment detected  >>>============== \t \t \n')
            
    def update_ini_file(self,E,logger):
        """
        E  is the dict contain the things to update
        """
        # update the ini file
        self.config_dict.update(E)
        strconfig = {K: repr(V) for K,V in self.config_dict.items()}
        self.config['DEFAULT'] = strconfig
        
        with open(self.config_file,'w') as f:
            self.config.write(f)
        
        logger.info('   ini file updated    ')
        
    def chimera_weight_update(self):
        # TODO : progressively update the loss weight between tasks
        
        # TODO : 1. scale the loss into the same magnitude
        
        # TODO : 2. update the weight by their own learning progress
        
        return None

abs_path = './../mrl_te_optimization/log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini'
Configuration = Auto_popen(abs_path)

np.random.seed(25)



SEQ_BATCH = N_GENES
UTR_LEN = 128
DIM = 40
gpath = './../../models/checkpoint_3000.h5'
mrl_path = './../../models/utr_model_combined_residual_new.h5'
exp_path = './../../models/humanMedian_trainepoch.11-0.426.h5'
tpath = './../exp_optimization/script/checkpoint/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth'




# %%


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

    def get_promoter_sequence(self, gene_id, upstream=8000, downstream=4000):
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
                if not 64 <= new_utr_length <= 128:
                    if verbose:
                        print(f"Warning: Generated UTR {i+1} length ({new_utr_length}) outside 64-128nt range for {gene_symbol}")
                        continue

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
            print(f"Error processing UTR replacement for {gene_info.get('gene_symbol', 'unknown')}: {e}")
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

            if verbose:
                print(f"\nGenerated {n_utrs * n_genes} modified sequences (expected: {expected_count})")

            return all_modified_sequences









def convert_model(model_:Model):

    input_ = tf.keras.layers.Input(shape=( 10500, 4))
    input = input_
    for i in range(len(model_.layers)-1):

        
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
    convert = True
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


def select_best(scores, seqs, gc_control=False, GC=-1, per_gene=False):
    selected_scores = []
    selected_seqs = []
    if per_gene:    

        scores = np.asarray(scores)
        seqs = np.asarray(seqs)
        
        A, B, C = np.shape(scores)
        selected_scores = []
        selected_seqs = []
        
        for b in range(B):

            best_score = np.max(scores[0, b, :])  
            best_seq = seqs[0, :]  
            
            for a in range(1, A):
                current_score = np.max(scores[a, b, :]) 
                
                if current_score > best_score:
                    if gc_control:

                        gc_content = get_gc_content(seqs[a, :])
                        if gc_content < GC:
                            best_score = current_score
                            best_seq = seqs[a, :]
                            best_a = a
                    else:
                        best_score = current_score
                        best_seq = seqs[a, :]
                        best_a = a
            
            selected_scores.append(best_score)
            selected_seqs.append(best_seq)
        

        selected_scores = np.array(selected_scores)  
        selected_seqs = np.array(selected_seqs)      
    else:
        for i in range(len(scores[0])):
            best = scores[1][i]
            best_seq = seqs[1][i]
            for j in range(len(scores)-1):
                if scores[j+1][i] > best:
                    if gc_control:
                        if get_gc_content(seqs[j][i]) < GC:
                            best = scores[j+1][i]
                            best_seq = seqs[j+1][i]
                    else:
                        best = scores[j+1][i]
                        best_seq = seqs[j+1][i]

            selected_scores.append(best)
            selected_seqs.append(best_seq)

    return selected_seqs, selected_scores

model = tf.keras.models.load_model(exp_path)

model = convert_model(model)

wgan = tf.keras.models.load_model(gpath)

"""
Data:
"""


noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,40]))

tf.random.set_seed(25)

diffs = []
init_exps = []

opt_exps = []

orig_vals = []

noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,40]))
noise_small = tf.random.normal(shape=[BATCH_SIZE,40],stddev=1e-5)

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

'''
Optimization takes place here.
'''

bind_scores_list = []
bind_scores_means = []
sequences_list = []

means = []
maxes = []

iters_ = []

OPTIMIZE = True

DNA_SEL = False

retriever = GeneInfoRetriever()
refs = []
for i in range(len(gene_names)):
    output_json = f"{gene_names[i]}_info.json"

    if not os.path.exists(os.path.join('./.cache/',output_json)):

        # Retrieve gene information
        gene_info = retriever.get_gene_info(gene_names[i], output_json=output_json)

        if "error" in gene_info:
            print(f"Error: {gene_info['error']}")
        else:
            refs.append(gene_info["promoter_sequence"])    
    else:
        with open(os.path.join('./.cache/',output_json), "r") as f:
            gene_info = json.load(f)
            refs.append(gene_info["promoter_sequence"])

sequences_init = wgan(noise)

gen_seqs_init = sequences_init.numpy().astype('float')

seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)

seqs_init = retriever.replace_utr_in_multiple_sequences(gene_names, seqs_gen_init, target_length=10500, cache_dir="./.cache", output_prefix="modified_sequence")

seqs_init = one_hot(seqs_init)

pred_init = model(seqs_init) 

pred_init = tf.reshape(pred_init,(SEQ_BATCH,-1))

average_initial_prediction = tf.reduce_mean(pred_init,axis=0).numpy().astype('float')

seqs_collection = []
scores_collection = []
scores_collection_genes = []
if OPTIMIZE:

    iter_ = 0
    for opt_iter in tqdm(range(STEPS)):
        
        with tf.GradientTape() as gtape:
            gtape.watch(noise)
            
            sequences = wgan(noise)

            seqs_gen = recover_seq(sequences, rev_rna_vocab)
            seqs_collection.append(seqs_gen)

            g1_ = tf.zeros_like(sequences)

            scores_collection_temp = []

            for gene in gene_names:

                seqs_dna = retriever.replace_utr_in_sequence(f"./.cache/{gene}_info.json", seqs_gen, target_length=10500, output_prefix="modified_sequence")               
            
                seqs = one_hot(seqs_dna)
                
                with tf.GradientTape() as ptape:
                    ptape.watch(seqs)

                    pred =  model(seqs)
                    t = tf.reshape(pred,(-1))
                    scores_collection_temp.append(t.numpy().astype('float'))
                    nt = t.numpy().astype('float')

                g1 = ptape.gradient(pred,seqs)
                g1 = tf.math.scalar_mul(-1.0, g1)
                g1 = tf.slice(g1,[0,7000,0],[-1,128,-1])

                tmp_g = g1.numpy().astype('float')
                tmp_seqs = seqs_gen

                # Initialize tmp_lst with correct size
                batch_size = min(len(tmp_seqs), tmp_g.shape[0])
                tmp_lst = np.zeros(shape=(batch_size, 128, 5))

                # Loop on the batch size and update the UTR only
                for i in range(batch_size):
                    len_ = min(len(tmp_seqs[i]), tmp_g.shape[1])  # Prevent exceeding tmp_g's dimensions
                    edited_g = tmp_g[i][:len_, :]
                    edited_g = np.pad(edited_g, ((0, 128-len_), (0, 1)), 'constant')
                    tmp_lst[i] = edited_g

                g1 = tf.convert_to_tensor(tmp_lst, dtype=tf.float32)

                g1_ = tf.math.add(g1, g1_)

            scores_collection.append(np.mean(scores_collection_temp,axis=0))
            scores_collection_genes.append(scores_collection_temp)
            g2 = gtape.gradient(sequences,noise,output_gradients=g1_)


        a1 = g2 + noise_small
        change = [(a1,noise)]

        optimizer.apply_gradients(change)

        iters_.append(iter_)
        iter_ += 1

    sequences_opt = wgan(noise)

    gen_seqs_opt = sequences_opt.numpy().astype('float')

    seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)

    seqs_opt = retriever.replace_utr_in_multiple_sequences(gene_names, seqs_gen_opt, target_length=10500, cache_dir="./.cache", output_prefix="modified_sequence")

    seqs_opt = one_hot(seqs_opt)

    pred_opt = model(seqs_opt)

    pred_opt = tf.reshape(pred_opt,(SEQ_BATCH,-1))


    average_optimized_prediction = tf.reduce_mean(pred_opt,axis=0).numpy().astype('float')


best_seqs, best_scores = select_best(scores_collection_genes, seqs_collection, per_gene=True)


with open('./outputs/mul_init_exps.txt', 'w') as f:
    for item in average_initial_prediction:
        f.write(f'{item}\n')

with open('./outputs/mul_best_exps.txt', 'w') as f:
    for item in best_scores:
        f.write(f'{item}\n')

with open('./outputs/mul_opt_exps.txt', 'w') as f:
    for item in average_optimized_prediction:
        f.write(f'{item}\n')

with open('./outputs/mul_best_seqs.txt', 'w') as f:
    for item in best_seqs:
        f.write(f'{item}\n')

with open('./outputs/mul_init_seqs.txt', 'w') as f:
    for item in seqs_gen_init:
        f.write(f'{item}\n')

# Compute average Log TPM per gene
init_log_tpm_target = tf.reduce_mean(pred_init, axis=1).numpy().astype('float')
opt_log_tpm_target = tf.reduce_mean(pred_opt, axis=1).numpy().astype('float')
opt_log_tpm_target = best_scores

# Compute overall average Log TPM across target genes
avg_init_log_tpm = np.average(init_log_tpm_target)
avg_opt_log_tpm = np.average(opt_log_tpm_target)

# Convert Log TPM to TPM for percentage improvement
# Assuming Log TPM is base-10 (common for TPM), TPM = 10^LogTPM
avg_init_tpm = np.power(10, avg_init_log_tpm)
avg_opt_tpm = np.power(10, avg_opt_log_tpm)

# Compute improvement
log_tpm_diff = avg_opt_log_tpm - avg_init_log_tpm
tpm_improvement = avg_opt_tpm - avg_init_tpm
# Percentage improvement based on TPM: ((opt - init) / init) * 100
if avg_init_tpm != 0:  # Avoid division by zero
    tpm_percent_change = (tpm_improvement / avg_init_tpm) * 100
else:
    tpm_percent_change = float('inf') if tpm_improvement > 0 else 0.0

# Handle negative and positive percentages
percent_str = f"{tpm_percent_change:.2f}%"
if tpm_percent_change < 0:
    percent_str = f"{tpm_percent_change:.2f}% (decrease)"
elif tpm_percent_change > 0:
    percent_str = f"+{tpm_percent_change:.2f}% (increase)"

# Print evaluation results
print("\nEvaluation of Optimization on Original Genes (Log TPM):")
print("\nExpression Levels (Log TPM):")
print(f"  Average Initial Log TPM: {avg_init_log_tpm:.4f} (TPM: {avg_init_tpm:.4f})")
print(f"  Average Optimized Log TPM: {avg_opt_log_tpm:.4f} (TPM: {avg_opt_tpm:.4f})")
print(f"  Log TPM Difference: {log_tpm_diff:.4f}")
print(f"  TPM Improvement: {tpm_improvement:.4f} ({percent_str})")


print("Genes:")
print(gene_names)
print(f"Average Initial Expression: {np.average(average_initial_prediction)}")
print(f"Best Expression: {np.average(best_scores)}")


target_refs = []
for gene in target_genes:
    output_json = f"{gene}_info.json"
    cache_path = os.path.join('./.cache/', output_json)
    
    if not os.path.exists(cache_path):
        # Retrieve gene information
        gene_info = retriever.get_gene_info(gene, output_json=output_json)
        if "error" in gene_info:
            print(f"Error retrieving info for {gene}: {gene_info['error']}")
            target_refs.append(None)  # Handle errors gracefully
        else:
            target_refs.append(gene_info["promoter_sequence"])
    else:
        with open(cache_path, "r") as f:
            gene_info = json.load(f)
            target_refs.append(gene_info["promoter_sequence"])


valid_indices = [i for i, ref in enumerate(target_refs) if ref is not None]
target_genes = [target_genes[i] for i in valid_indices]
target_refs = [target_refs[i] for i in valid_indices]

if not target_genes:
    print("No valid target genes retrieved. Exiting evaluation.")
else:

    seqs_gen_init = seqs_gen_init  
    seqs_gen_opt = best_seqs    


    seqs_init_target = retriever.replace_utr_in_multiple_sequences(
        target_genes, seqs_gen_init, target_length=10500, cache_dir="./.cache", output_prefix="target_modified_sequence"
    )
    seqs_opt_target = retriever.replace_utr_in_multiple_sequences(
        target_genes, seqs_gen_opt, target_length=10500, cache_dir="./.cache", output_prefix="target_modified_sequence"
    )


    seqs_init_target = one_hot(seqs_init_target)
    seqs_opt_target = one_hot(seqs_opt_target)


    pred_init_target = model(seqs_init_target)
    pred_opt_target = model(seqs_opt_target)

    pred_init_target = tf.reshape(pred_init_target, (len(target_genes), -1))
    pred_opt_target = tf.reshape(pred_opt_target, (len(target_genes), -1))

    # Compute average Log TPM per gene
    init_log_tpm_target = tf.reduce_mean(pred_init_target, axis=1).numpy().astype('float')
    opt_log_tpm_target = tf.reduce_mean(pred_opt_target, axis=1).numpy().astype('float')

    # Compute overall average Log TPM across target genes
    avg_init_log_tpm = np.average(init_log_tpm_target)
    avg_opt_log_tpm = np.average(opt_log_tpm_target)

    # Convert Log TPM to TPM for percentage improvement
    avg_init_tpm = np.power(10, avg_init_log_tpm)
    avg_opt_tpm = np.power(10, avg_opt_log_tpm)

    # Compute improvement
    log_tpm_diff = avg_opt_log_tpm - avg_init_log_tpm
    tpm_improvement = avg_opt_tpm - avg_init_tpm
    # Percentage improvement based on TPM: ((opt - init) / init) * 100
    if avg_init_tpm != 0:  # Avoid division by zero
        tpm_percent_change = (tpm_improvement / avg_init_tpm) * 100
    else:
        tpm_percent_change = float('inf') if tpm_improvement > 0 else 0.0

    # Handle negative and positive percentages
    percent_str = f"{tpm_percent_change:.2f}%"
    if tpm_percent_change < 0:
        percent_str = f"{tpm_percent_change:.2f}% (decrease)"
    elif tpm_percent_change > 0:
        percent_str = f"+{tpm_percent_change:.2f}% (increase)"

    # Print evaluation results
    print("\nEvaluation of Optimization on Target Genes (Log TPM):")
    print(f"Original Genes: {gene_names}")
    print(f"Target Genes: {target_genes}")
    print("\nExpression Levels (Log TPM):")
    print(f"  Average Initial Log TPM: {avg_init_log_tpm:.4f} (TPM: {avg_init_tpm:.4f})")
    print(f"  Average Optimized Log TPM: {avg_opt_log_tpm:.4f} (TPM: {avg_opt_tpm:.4f})")
    print(f"  Log TPM Difference: {log_tpm_diff:.4f}")
    print(f"  TPM Improvement: {tpm_improvement:.4f} ({percent_str})")

    # Save evaluation results to a file
    with open('./outputs/target_genes_evaluation.txt', 'w') as f:
        f.write("Evaluation of Optimization on Target Genes (Log TPM)\n")
        f.write(f"Original Genes: {gene_names}\n")
        f.write(f"Target Genes: {target_genes}\n\n")
        f.write("Expression Levels (Log TPM):\n")
        f.write(f"  Average Initial Log TPM: {avg_init_log_tpm:.4f} (TPM: {avg_init_tpm:.4f})\n")
        f.write(f"  Average Optimized Log TPM: {avg_opt_log_tpm:.4f} (TPM: {avg_opt_tpm:.4f})\n")
        f.write(f"  Log TPM Difference: {log_tpm_diff:.4f}\n")
        f.write(f"  TPM Improvement: {tpm_improvement:.4f} ({percent_str})\n")

    # Optional: Per-gene breakdown
    print("\nPer-Gene Expression Levels (Log TPM):")
    for gene, init_log, opt_log in zip(target_genes, init_log_tpm_target, opt_log_tpm_target):
        init_tpm = np.power(10, init_log)
        opt_tpm = np.power(10, opt_log)
        tpm_diff = opt_tpm - init_tpm
        if init_tpm != 0:
            gene_percent = (tpm_diff / init_tpm) * 100
        else:
            gene_percent = float('inf') if tpm_diff > 0 else 0.0
        gene_percent_str = f"{gene_percent:.2f}%"
        if gene_percent < 0:
            gene_percent_str = f"{gene_percent:.2f}% (decrease)"
        elif gene_percent > 0:
            gene_percent_str = f"+{gene_percent:.2f}% (increase)"
        print(f"  {gene}: Initial Log TPM = {init_log:.4f} (TPM: {init_tpm:.4f}), "
            f"Optimized Log TPM = {opt_log:.4f} (TPM: {opt_tpm:.4f}), "
            f"TPM Improvement = {tpm_diff:.4f} ({gene_percent_str})")




