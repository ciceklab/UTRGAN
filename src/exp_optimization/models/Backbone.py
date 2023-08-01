import os, sys
import torch 
import numpy  as np
from torch import nn
from scipy import stats
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, r2_score
from torch.nn.modules import activation
from torch.nn.modules.dropout import Dropout
from typing import Union
from einops import rearrange
import math
# from .Modules._operator import *

class Conv1d_block(nn.Module):
    """
    the Convolution backbone define by a list of convolution block
    """
    def __init__(self,channel_ls,kernel_size,stride, padding_ls=None,diliation_ls=None,pad_to=None, activation='ReLU'):
        """
        Argument
            channel_ls : list, [int] , channel for each conv layer
            kernel_size : int
            stride :  list , [int]
            padding_ls :   list , [int]
            diliation_ls : list , [int]
        """
        super(Conv1d_block,self).__init__()
        ### property
        self.activation = activation
        self.channel_ls = channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        if padding_ls is None:
            self.padding_ls = [0] * (len(channel_ls) - 1)
        else:
            assert len(padding_ls) == len(channel_ls) - 1
            self.padding_ls = padding_ls
        if diliation_ls is None:
            self.diliation_ls = [1] * (len(channel_ls) - 1)
        else:
            assert len(diliation_ls) == len(channel_ls) - 1
            self.diliation_ls = diliation_ls
        
        self.encoder = nn.ModuleList(
            #                   in_C         out_C           padding            diliation
            [self.Conv_block(channel_ls[i],channel_ls[i+1],self.padding_ls[i],self.diliation_ls[i],self.stride[i]) for i in range(len(self.padding_ls))]
        )
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        
        activation_layer = eval(f"nn.{self.activation}")
        
        block = nn.Sequential(
                nn.Conv1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation),
                nn.BatchNorm1d(out_Chan),
                activation_layer())
        
        return block
    
    def forward(self,x):
        if x.shape[2] == 4:
            out = x.transpose(1,2)  
        else:
            out = x
        for block in self.encoder:
            out = block(torch.tensor(out,dtype=torch.double))
        return out
    
    def forward_stage(self,x,stage):
        """
        return the activation of each stage for exchanging information
        """
        assert stage < len(self.encoder)
        
        out = self.encoder[stage](x)
        return out

    def cal_out_shape(self,L_in=100,padding=0,diliation=1,stride=2):
        """
        For convolution 1D encoding , compute the final length 
        """
        L_out = 1+ (L_in + 2*padding -diliation*(self.kernel_size-1) -1)/stride
        return L_out
    
    def last_out_len(self,L_in=100):
        for i in range(len(self.padding_ls)):
            padding = self.padding_ls[i]
            diliation = self.diliation_ls[i]
            stride = self.stride[i]
            L_in = self.cal_out_shape(L_in,padding,diliation,stride)
        # assert int(L_in) == L_in , "convolution out shape is not int"
        
        return int(L_in) if L_in >=0  else 1
    
class ConvTranspose1d_block(Conv1d_block):
    """
    the Convolution transpose backbone define by a list of convolution block
    """
    def __init__(self,channel_ls,kernel_size,stride,padding_ls=None,diliation_ls=None,pad_to=None):
        channel_ls = channel_ls[::-1]
        stride = stride[::-1]
        padding_ls =  padding_ls[::-1] if padding_ls  is not None else  [0] * (len(channel_ls) - 1)
        diliation_ls =  diliation_ls[::-1] if diliation_ls  is not None else  [1] * (len(channel_ls) - 1)
        super(ConvTranspose1d_block,self).__init__(channel_ls,kernel_size,stride,padding_ls,diliation_ls,pad_to)
        
    def Conv_block(self,in_Chan,out_Chan,padding,dilation,stride): 
        """
        replace `Conv1d` with `ConvTranspose1d`
        """
        block = nn.Sequential(
                nn.ConvTranspose1d(in_Chan,out_Chan,self.kernel_size,stride,padding,dilation=dilation),
                nn.BatchNorm1d(out_Chan),
                nn.ReLU())
        
        return block
    
    def cal_out_shape(self,L_in,padding=0,diliation=1,stride=1,out_padding=0):
        #                  L_in=100,padding=0,diliation=1,stride=2
        """
        For convolution Transpose 1D decoding , compute the final length
        """
        L_out = (L_in -1 )*stride + diliation*(self.kernel_size -1 )+1-2*padding + out_padding 
        return L_out


class linear_block(nn.Module):
    def __init__(self,in_Chan,out_Chan,dropout_rate=0.2):
        """
        building block func to define dose network
        """
        super(linear_block,self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_Chan,out_Chan),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(out_Chan),
            nn.ReLU()
        )
    def forward(self,x):
        return self.block(x)


class Self_Attention(nn.Module):
    """
    self attention operator for Conv1d sequences output
    """
    def __init__(self, in_dim:int, out_dim:int, qk_dim:int, n_head:int):
        super().__init__()

        self.n_head = n_head
        self.total_qk_dim = qk_dim * n_head
        self.transform = nn.ModuleDict({
            k : nn.Linear(in_dim, self.total_qk_dim) for k in ['k', 'q', 'v']
        })

        self.fc_out = nn.Linear(self.total_qk_dim, out_dim)

    def dim_rerrange(self, x):

        # first break down total qk dimension
        # then transpose length with heads
        x1 = rearrange(x, "b l (n c) -> b n c l", n=self.n_head) 
        return x1
    
    def _get_attention_map(self,X):
        """
        break the forward function to access attention mat
        """
        # assume we have a 3 dimension input X (b, len, in_dim)
        # each out in qkv is also 3 dimension  (b, len , qk_dim)
        qkv = [self.transform[key](X) for key in ['k', 'q', 'v']]
        q, k, v = map(self.dim_rerrange, qkv)

        # here i and j is the channel
        sim = torch.einsum("b n c i, b n c j -> b n i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach() 
        attn = sim.softmax(dim=-1)
        return attn, v

    def forward(self, X):
    
        attn, v = self._get_attention_map(X)

        out = torch.einsum("b n i j, b n c j -> b n i c", attn, v)
        out = rearrange(out, "b n i c -> b i (n c)")
        return self.fc_out(out)


class Self_Attention_for_GP(Self_Attention):
    """
    self attention operator for Conv1d sequences Global Pooling output
    The input has 2 dimension (no length dim), 
    """
    def __init__(self, in_dim:int, out_dim:int, qk_dim:int, n_head:int):
        super().__init__(in_dim, out_dim, qk_dim, n_head)

    def _get_attention_map(self,X):
        # assume we have a 3 dimension input X (b, len, in_dim)
        # each out in qkv is also 3 dimension  (b, len , qk_dim)
        qkv = [self.transform[key](X) for key in ['k', 'q', 'v']]
        q, k, v = map(
            lambda x : rearrange(x, "b (n c)-> b n c"), qkv
        )

        # here i and j is the channel
        sim = torch.einsum("b n i, b n j -> b n i j", q, k).softmax(dim=-2, keepdim=True)
        sim = sim - attn.amax(dim=-1, keepdim=True).detach() 
        attn = sim.softmax(dim=-1)
        return attn, v

    def forward(self, X):
        # 
        attn, v = self._get_attention_map(X)

        out = torch.einsum("b n i j, b n j -> b n i", attn, v)
        out = rearrange(out, "b n i -> b (n i)")
        return self.fc_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
        
    def forward(self, x):
        x = self.norm(x.transpose(1,2))
        return self.fn(x.transpose(1,2))

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2 
        embeddings = math.log(10000) / (half_dim -1) # why do we minus 1 ?
        embeddings = torch.exp(torch.arange(half_dim, device=device)* -embeddings)
        embeddings = time[:, None] * embeddings[None, :] # expand to 2 dimension
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class backbone_model(nn.Module):
    def __init__(self,conv_args,activation='ReLU'):
        """
        the most bottle model which define a soft-sharing convolution block some forward method 
        """
        super(backbone_model,self).__init__()
        channel_ls,kernel_size,stride,padding_ls,diliation_ls,pad_to = conv_args
        
        self.channel_ls = channel_ls
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_ls = padding_ls
        self.diliation_ls = diliation_ls
        self.pad_to = pad_to
        
        # model
        self.soft_share = Conv1d_block(channel_ls,kernel_size,stride,padding_ls,diliation_ls,activation=activation)
        # property
        self.stage = list(range(len(channel_ls)-1))
        self.out_length = self.soft_share.last_out_len(pad_to)
        self.out_dim = self.soft_share.last_out_len(pad_to)*channel_ls[-1]
    
    def _weight_initialize(self, model):
        if type(model) in [nn.Linear]:
            nn.init.xavier_uniform_(model.weight)
            nn.init.zeros_(model.bias) 
        elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
            nn.init.orthogonal_(model.weight_hh_l0)
            nn.init.xavier_uniform_(model.weight_ih_l0)
            nn.init.zeros_(model.bias_hh_l0)
            nn.init.zeros_(model.bias_ih_l0)
        elif isinstance(model, nn.Conv1d):
            nn.init.orthogonal_(model.weight)
        elif isinstance(model, nn.Conv2d):
            nn.init.kaiming_normal_(model.weight, nonlinearity='leaky_relu',)
        elif isinstance(model, nn.BatchNorm1d):
            nn.init.constant_(model.weight, 1)
            nn.init.constant_(model.bias, 0)
    
    def forward_stage(self,X,stage):
        return self.soft_share.forward_stage(X,stage)
    
    def forward_tower(self,Z):
        """
        Each new backbone model should re-write the `forward_tower` method
        """
        return Z
    
    def forward(self,X):
        Z = self.soft_share(X)
        out = self.forward_tower(Z)
        return out
    
class RL_regressor(backbone_model):
    
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2, activation='ReLU'):
        """
        backbone for RL regressor task  ,the same soft share should be used among task
        Arguments:
            conv_args: (channel_ls,kernel_size,stride,padding_ls,diliation_ls)
        """
        super(RL_regressor,self).__init__(conv_args, activation)
        
        #  ------- architecture -------
        # self.tower = linear_block(in_Chan=self.out_dim,out_Chan=tower_width,dropout_rate=dropout_rate)
        # self.fc_out = nn.Linear(tower_width,1)
        
        #     ----- task specific -----
        self.loss_fn = nn.MSELoss(reduction='mean')
        self.task_name = 'RL_regression'
        self.loss_dict_keys = ['Total']
        
    def forward_tower(self,Z):
        # flatten
        batch_size = Z.shape[0]
        Z_flat = Z.view(batch_size,-1)
        # tower part
        Z_to_out = self.tower(Z_flat)
        out = self.fc_out(Z_to_out)
        return out
    
    def squeeze_out_Y(self,out,Y):
        # ------ squeeze ------
        if len(Y.shape) == 2:
            Y = Y.squeeze(1)
        if len(out.shape) == 2:
            out = out.squeeze(1)
        
        assert Y.shape == out.shape, "keep label and pred the same shape"
        return out,Y
    
    def compute_acc(self,out,X,Y,popen=None):
        try:
            epsilon = popen.epsilon
        except:
            epsilon = 0.3
            
        out,Y = self.squeeze_out_Y(out,Y)
        # error smaller than epsilon
        with torch.no_grad():
            y_ay = Y.cpu().numpy()
            out_ay = out.cpu().numpy()
            # acc = torch.sum(torch.abs(Y-out) < epsilon).item() / Y.shape[0]
            acc = stats.spearmanr(y_ay,out_ay)[0]
            # acc = r2_score(y_ay, out_ay)
        return {"Acc":acc}
    
    def compute_loss(self,out,X,Y,popen):
        out,Y = self.squeeze_out_Y(out,Y)
        loss = self.loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss}

class RL_gru(RL_regressor):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2 ,activation='ReLU'):
        """
        tower is gru
        """      
        super().__init__(conv_args,tower_width,dropout_rate, activation)
        self.configure_towerwidth( tower_width)
        # previous, it is a linear layer
        if dropout_rate > 0 :
            self.soft_share.encoder = nn.ModuleList([
                nn.Sequential(conv_layer,nn.Dropout(dropout_rate)) 
                                for conv_layer in self.soft_share.encoder
             ])
        self.tower = nn.GRU(input_size=self.channel_ls[-1],
                            hidden_size=self.tower_width,
                            num_layers=2,
                            batch_first=True) # input : batch , seq , features
        self.fc_out = nn.Linear(self.tower_width,1)
        
        self.apply(self._weight_initialize)
    
    def configure_towerwidth(self, tower_width):
        if isinstance(tower_width, int):
            self.tower_width = tower_width
        elif isinstance(tower_width, list):
            self.tower_width = tower_width[0]
        elif isinstance(tower_width, dict):
            self.tower_width  = tower_width.values()[0] 

    def forward_tower(self,Z):
        # flatten
        # batch_size = Z.shape[0]
        Z_flat = torch.transpose(Z,1,2)
        # tower part
        h_prim,(c1,c2) = self.tower(Z_flat)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2
        out = self.fc_out(c2)
        return out
    
    @torch.no_grad()
    def predict_each_position(self, X):
        Z = self.soft_share(X)
        Z_flat = torch.transpose(Z,1,2)
        # tower part
        h_prim,(c1,c2) = self.tower(Z_flat)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2 
        out_series = self.fc_out(h_prim)
        return out_series

class RL_hard_share(RL_regressor):
    def __init__(self,conv_args,tower_width=40,dropout_rate=0.2,activation='ReLU', tasks =['unmod1', 'human', 'vleng']):
        """
        Ribosome Loading Prediction with Hard-sharing;
        shared convolution bottom
        tower is gru
        """      
        super().__init__(conv_args,tower_width,dropout_rate,activation)
        self.all_tasks = tasks
        self.configure_towerwidth(tower_width)
        tower_block = lambda c, w : nn.ModuleList([nn.GRU(input_size=c,
                                                            hidden_size=w,
                                                            num_layers=2,
                                                            batch_first=True),
                                                    nn.Linear(w,1)])

        self.tower = nn.ModuleDict({t: tower_block(self.channel_ls[-1], self.tower_width[t]) for t in self.all_tasks})

    def configure_towerwidth(self, tower_width):
        if isinstance(tower_width, int):
            self.tower_width = {t:tower_width for t in self.all_tasks}
        elif isinstance(tower_width, list):
            assert len(tower_width) == len(self.all_tasks)
            self.tower_width = dict(zip(self.all_tasks, tower_width))
        elif isinstance(tower_width, dict):
            assert len(tower_width) == len(self.all_tasks)
            self.tower_width  = tower_width 
        else:
            raise TypeError("`tower_width` can only be int, list and dict")

    def forward(self, X):
        
        task = self.task # pass in cycle_train.py
        # Con block
        Z = self.soft_share(X)
        # tower
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower['human'][0](Z_t)
        out = self.tower['human'][1](c2)
        
        return out
    
    @torch.no_grad()
    def predict_each_position(self, X):
        task = self.task # pass in cycle_train.py
        # Con block
        Z = self.soft_share(X)
        # tower
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower[task][0](Z_t)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2 
        out_series = self.tower[task][1](h_prim)
        return out_series
    
    def compute_loss(self,out,X,Y,popen):
        try:
            task_lambda = popen.chimera_weight
        except:
            task_lambda = {'unmod1':0.1, 'SubHuman':0.1, 'SubVleng':0.1,
                            'unmod1':0.1, 'human':0.1, 'vleng':0.1,  
                            'Andrev2015':1, 'muscle':1, 'pc3':1, 
                            '293':1,'pcr3':1
                            }
        
        loss_weight = task_lambda[self.task]
        out,Y = self.squeeze_out_Y(out,Y)
        loss = self.loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss*loss_weight}

    def compute_acc(self,out,X,Y,popen=None):
        task = self.task
        Acc = super().compute_acc(out,X,Y,popen)['Acc']
        return {task+"_Acc" : Acc}

class RL_covar_reg(RL_hard_share):
    def __init__(self,conv_args,tower_width:int=40, 
                    dropout_rate:float=0.2, activation:str='ReLU',  
                    n_covar:Union[dict, list, int]=1,
                    tasks:list=['unmod1', 'human', 'vleng']
                    ):
        """
        Ribosome Loading Prediction with Hard-sharing and account for covariates
        covariate is added at the last layer

        n_covar: 
        """      
        super().__init__(conv_args, tower_width, dropout_rate=dropout_rate, activation=activation, tasks=tasks)
        self.all_tasks = tasks
        self.configure_towerwidth(tower_width)
        self.configure_covariate(n_covar)
        
        tower_block = lambda c, w , n : nn.ModuleList([nn.GRU(input_size=c,
                                                            hidden_size=w,
                                                            num_layers=2,
                                                            batch_first=True),
                                                    # covariate is added here
                                                    nn.Linear(w + n,1)])

        c = self.channel_ls[-1]
        self.tower = nn.ModuleDict({t: tower_block(c, self.tower_width[t], self.n_covar[t]) for t in self.all_tasks})

    def configure_covariate(self, n_covar):
        if isinstance(n_covar, int):
            self.n_covar = {t:n_covar for t in self.all_tasks}
        elif isinstance(n_covar, list):
            assert len(n_covar) == len(self.all_tasks)
            self.n_covar = dict(zip(self.all_tasks, n_covar))
        elif isinstance(n_covar, dict):
            assert len(n_covar) == len(self.all_tasks)
            self.n_covar  = n_covar 
        else:
            raise TypeError("`n_covar` can only be int, list and dict")

    
    def encode(self, X):
        task = self.task # pass in cycle_train.py
        X_seq, X_covar = X
        assert X_covar.shape[1] == self.n_covar[task], "the # of covariates is not consistent with the model params"

        # Con block
        Z = self.soft_share(X_seq)
        return Z

    def forward(self, X):
        task = self.task # pass in cycle_train.py
        X_seq, X_covar = X
        Z = self.encode(X)
        # tower
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower[task][0](Z_t)

        # concate
        linear_factor = torch.cat([c2, X_covar], dim=1)
        out = self.tower[task][1](linear_factor)
        
        return out
    
    def get_factor_weight(self, task):
        n_covar = self.n_covar[task]
        return next(self.tower[task][1].parameters())[-1*n_covar:]
    
    @torch.no_grad()
    def predict_each_position(self, X):
        task = self.task # pass in cycle_train.py
        X_seq, X_covar = X
        # Con block
        Z = self.soft_share(X_seq)
        # tower
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower[task][0](Z_t)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2 
        cor_expand = torch.broadcast_to(X_covar.unsqueeze(1), 
                             (h_prim.shape[0], h_prim.shape[1], X_covar.shape[1]))
        linear_factor = torch.cat([h_prim, cor_expand], dim=2)
        out_series = self.tower[task][1](linear_factor)
        return out_series

class RL_covar_intersect(RL_covar_reg):
    def __init__(self,conv_args,tower_width:int=40, 
                    dropout_rate:float=0.2, activation:str='ReLU',  
                    n_covar:Union[dict, list, int]=1,
                    tasks:list=['unmod1', 'human', 'vleng']):
        super().__init__(conv_args=conv_args,tower_width=tower_width, dropout_rate=dropout_rate, 
                            activation=activation,  n_covar=n_covar, tasks=tasks)

        tower_block = lambda c, w , n : nn.ModuleList([nn.GRU(input_size=c,
                                                            hidden_size=w,
                                                            num_layers=2,
                                                            batch_first=True),
                                                    # covariate is added here
                                                    nn.Linear(w ,1),
                                                    nn.Linear(1+n ,1),
                                                    ])

        c = self.channel_ls[-1]
        self.tower = nn.ModuleDict({t: tower_block(c, self.tower_width[t], self.n_covar[t]) for t in self.all_tasks})
    
    def forward(self, X):
        task = self.task # pass in cycle_train.py
        X_seq, X_covar = X
        Z = self.encode(X)
        # tower
        Z_t = torch.transpose(Z, 1, 2)
        h_prim,(c1,c2) = self.tower[task][0](Z_t)
        intersect = self.tower[task][1](c2)
        # concate
        linear_factor = torch.cat([intersect, X_covar], dim=1)
        out = self.tower[task][2](linear_factor)
        return out

class RL_clf(RL_gru):

    def __init__(self,conv_args,n_calss,tower_width=40,dropout_rate=0.2):
        """
        transform RL gru into classifier
        """      
        super().__init__(conv_args,tower_width,dropout_rate)
        self.n_calss = n_calss
        # previous, it is a linear layer
        self.tower = nn.GRU(input_size=self.channel_ls[-1],
                            hidden_size=tower_width,
                            num_layers=2,
                            batch_first=True) # input : batch , seq , features
        self.fc_out = nn.Linear(tower_width,n_calss)
        
        self.apply(self._weight_initialize)
        
    def forward_tower(self,Z):
        # flatten
        # batch_size = Z.shape[0]
        Z_flat = torch.transpose(Z,1,2)
        # tower part
        h_prim,(c1,c2) = self.tower(Z_flat)  # [B,L,h] , [B,h] cell of layer 1, [B,h] of layer 2
        out = self.fc_out(c2)
        class_pred = torch.softmax(out,dim=1)
        return class_pred
    
    def compute_acc(self,out,X,Y,popen=None):
            
        with torch.no_grad():
            acc = torch.sum(torch.argmax(out,dim=1) == Y.view(-1))/ Y.shape[0]
        return {"Acc":acc}
    
    def compute_loss(self,out,X,Y,popen):
        if len(Y.shape) >1:
            Y = Y.squeeze(dim=1).long()
        loss_fn=nn.CrossEntropyLoss()
        loss = loss_fn(out,Y) + popen.l1 * torch.sum(torch.abs(next(self.soft_share.encoder[0].parameters()))) 
        return {"Total":loss}