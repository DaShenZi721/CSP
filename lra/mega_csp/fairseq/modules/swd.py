import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from einops import rearrange
import math

use_cuda = torch.cuda.is_available()
dtype    = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
dtypeint = torch.cuda.LongTensor if use_cuda else torch.LongTensor
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# and from https://github.com/dfdazac/wassdistance/blob/master/layers.py


class SWD(nn.Module):
    ''' 
        feature group sort + butterfly
    '''
    def __init__(self, **kwargs):
        super(SWD21, self).__init__()
        self.butterfly_matrix = dict()
        self.len_sort_window = 2

    def get_butterfly_matrix(self, v_len, d_v, device):
        b = 2
        twiddle = torch.ones((1, v_len * b), dtype=torch.float, device=device).reshape(v_len // b, b, b)
        col_indices = [torch.arange(v_len, dtype=torch.float, device=device)]
        shift_offset = 0
        while len(col_indices) < d_v:
            for factor_index in range(int(math.log(v_len, b))):
                butterfly_matrix = butterfly_factor_to_matrix(twiddle, factor_index, device)
                col_index = torch.arange(v_len, dtype=torch.float, device=device)
                col_index = torch.cat([col_index[-shift_offset:], col_index[:-shift_offset]])
                # col_index = (butterfly_matrix - torch.eye(v_len, device=device)) @ col_index
                shift_offset = (shift_offset + self.len_sort_window) % v_len
                col_indices.append(col_index)
                if len(col_indices) == d_v:
                    break
        # col_indices = []
        # shift_offset = 0
        # for i in range(d_v):
        #     col_index = torch.arange(v_len, dtype=torch.float, device=self.device)
        #     col_index = torch.cat([col_index[-shift_offset:], col_index[:-shift_offset]])
        #     shift_offset = (shift_offset + self.len_sort_window) % v_len
        #     col_indices.append(col_index)
            
        return torch.stack(col_indices).t()
    
    def forward(self, v, descending=False, padding_mask=None, src_lengths=None):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        batch_size, v_len, d_v = v.shape
        
        if padding_mask is not None:
            padding_mask = padding_mask.squeeze()
            padding_idx = v[padding_mask][0, 0]
            padding_value = float('-inf') if descending else float('inf')
            v = v.masked_fill(padding_mask.unsqueeze(dim=2), padding_value)
        
        if v_len not in self.butterfly_matrix:
            self.butterfly_matrix[v_len] = self.get_butterfly_matrix(v_len, d_v, v.device).unsqueeze(dim=0).long()

        butterfly_matrix = self.butterfly_matrix[v_len].repeat(batch_size, 1, 1)
        if src_lengths is not None:
            butterfly_matrix = torch.remainder(butterfly_matrix, src_lengths.unsqueeze(dim=-1).unsqueeze(dim=-1))
            
        v_shifted = torch.gather(v, 1, butterfly_matrix)
        
        v_shifted = v_shifted.view(batch_size, -1, self.len_sort_window, d_v)
        v_sorted, _ = torch.sort(v_shifted, dim=2, descending=descending)
        out = v_sorted.view(batch_size, v_len, d_v)
            
        # out, _ = torch.sort(v, dim=1, descending=descending)
        
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(dim=2), padding_idx)
        
        return out
    
class SWD_exp(nn.Module):
    ''' 
        feature group sort + butterfly
    '''
    def __init__(self, layer_idx=None, num_layers=None, dim=None, **kwargs):
        super(SWD23, self).__init__()
        self.shift_matrix = dict()
        self.len_sort_window = 2
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        self.dim = dim

    def get_shift_matrix(self, v_len, d_v, device):
        col_indices = []
        shift_offset = 0
        for i in range(d_v):
            col_index = torch.arange(v_len, dtype=torch.float, device=device)
            shift_offset = math.ceil(v_len ** ((self.layer_idx*self.dim+i)/(self.num_layers*self.dim)))
            col_index = torch.cat([col_index[-shift_offset:], col_index[:-shift_offset]])
            # shift_offset = (shift_offset + self.len_sort_window) % v_len
            col_indices.append(col_index)
            
        return torch.stack(col_indices).t()
    
    def forward(self, v, descending=False, padding_mask=None, src_lengths=None):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        batch_size, v_len, d_v = v.shape
        
        if padding_mask is not None:
            padding_mask = padding_mask.squeeze()
            padding_idx = v[padding_mask][0, 0]
            padding_value = float('-inf') if descending else float('inf')
            v = v.masked_fill(padding_mask.unsqueeze(dim=2), padding_value)
        
        if v_len not in self.shift_matrix:
            self.shift_matrix[v_len] = self.get_shift_matrix(v_len, d_v, v.device).unsqueeze(dim=0).long()

        shift_matrix = self.shift_matrix[v_len].repeat(batch_size, 1, 1)
        if src_lengths is not None:
            shift_matrix = torch.remainder(shift_matrix, src_lengths.unsqueeze(dim=-1).unsqueeze(dim=-1))
            
        v_shifted = torch.gather(v, 1, shift_matrix)
        
        v_shifted = v_shifted.view(batch_size, -1, self.len_sort_window, d_v)
        v_sorted, _ = torch.sort(v_shifted, dim=2, descending=descending)
        out = v_sorted.view(batch_size, v_len, d_v)
        
        if padding_mask is not None:
            out = out.masked_fill(padding_mask.unsqueeze(dim=2), padding_idx)
        
        return out