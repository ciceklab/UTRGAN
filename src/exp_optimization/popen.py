import os,sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import json
from .models import Modules
import configparser
import logging

from sklearn.preprocessing import OneHotEncoder
import logging
import numpy as np
import pandas as pd
import os
import json
import re
import torch
import collections
# from models import reader
from .models.ScheduleOptimizer import ScheduledOptim

print(os.path.dirname(__file__))

# ====================|   some path   |=======================
global script_dir
global data_dir
global log_dir
global pth_dir
# global cell_lines

global egfp_seq

with open(os.path.join(os.path.dirname(__file__),"machine_configure.json"),'r') as f:
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