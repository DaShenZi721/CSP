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
    def __init__(self, ):
        super(SWD, self).__init__()

    def forward(self, q, k, attn_mask):
        attn_mask_shape = attn_mask.shape
        q = q.contiguous().view(-1, q.size(-2), q.size(-1))
        k = k.contiguous().view(-1, k.size(-2), k.size(-1))
        attn_mask = attn_mask.contiguous().view(-1, attn_mask.size(-2), attn_mask.size(-1))

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        p = []
        for batch in range(q.size(0)):
            pd = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
            for di in range(q.size(-1)):
                q_index = q_indices[batch, :, di]
                k_index = k_indices[batch, :, di]

                pi = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
                pi[q_index, k_index] = 1

                qi = q[batch, :, di].view(-1, 1)
                ki = k[batch, :, di].view(1, -1)
                ci = torch.abs(qi-ki).pow(2)
                pd = pd + pi * torch.exp(-ci)
            pd = pd / q.size(-1)
            p.append(pd.unsqueeze(0))
        p = torch.cat(p, dim=0).masked_fill(attn_mask, 0).view(attn_mask_shape)
        return p

class SWD2(nn.Module):
    def __init__(self, ):
        super(SWD2, self).__init__()

    def forward(self, q, k, attn_mask):
        attn_mask_shape = attn_mask.shape
        q = q.contiguous().view(-1, q.size(-2), q.size(-1))
        k = k.contiguous().view(-1, k.size(-2), k.size(-1))
        attn_mask = attn_mask.contiguous().view(-1, attn_mask.size(-2), attn_mask.size(-1))

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        p = []
        for batch in range(q.size(0)):
            pd = torch.sparse_coo_tensor(size=(q.size(-2), k.size(-2))).to(q.device)
            for di in range(q.size(-1)):
                start_time = time.time()
                qi = q[batch, :, di].view(-1, 1)
                ki = k[batch, :, di].view(1, -1)
                ci = torch.abs(qi-ki).pow(2)
                ci = torch.exp(-ci)
                print('ci %.4f'%((time.time()-start_time)*100))

                start_time = time.time()
                q_index = q_indices[batch, :, di]
                k_index = k_indices[batch, :, di]
                coo_indices = torch.cat([q_index.unsqueeze(0), k_index.unsqueeze(0)], dim=0)
                pd = pd + torch.sparse_coo_tensor(coo_indices, ci[q_index, k_index])
                print('coo %.4f'%((time.time()-start_time)*100))

            start_time = time.time()
            pd = pd / q.size(-1)
            p.append(pd.to_dense().unsqueeze(0))
            print('to_dense %.4f'%((time.time()-start_time)*100))
        p = torch.cat(p, dim=0).masked_fill(attn_mask, 0).view(attn_mask_shape)
        return p

class SWD3(nn.Module):
    def __init__(self, ):
        super(SWD3, self).__init__()

    def forward(self, q, k, attn_mask):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        qd = q.unsqueeze(-1).repeat(1, 1, 1, 1, q_len)
        kd = k.unsqueeze(-1).repeat(1, 1, 1, 1, k_len).permute(0, 1, 4, 3, 2)
        c = torch.abs(qd-kd).pow(2)
        c = torch.exp(-c)

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        indices = q_indices * q_len + k_indices
        indices = indices.permute(0, 1, 3, 2).view(batch_size, n_heads, d_k, -1)
        c = c.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, n_heads, d_k, -1)
        p = torch.zeros(c.shape).to(c.device)
        p = p.scatter(-1, indices, c.gather(-1, indices)).sum(-2) / d_k
        p = p.view(batch_size, n_heads, q_len, k_len).masked_fill(attn_mask, 0)
        return p


class SWD4(nn.Module):
    def __init__(self):
        super(SWD4, self).__init__()

    def forward(self, q, k, attn_mask, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        k_indices_q = q_indices*k_len + k_indices
        # |k_indices_q| : (batch_size, n_heads, q_len*d_k)
        k_indices_q = k_indices_q.gather(-2, q_indices).view(batch_size, n_heads, -1)
        c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # |c_q| : (batch_size, n_heads, q_len*d_k)
        c_q = c.gather(-2, q_indices).view(batch_size, n_heads, -1) / d_k

        p = torch.zeros(batch_size, n_heads, q_len*k_len).to(q.device)
        p.scatter_add_(-1, k_indices_q, c_q)
        p = p.view(batch_size, n_heads, q_len, k_len).masked_fill(attn_mask, 0)

        # attn_mask_shape = attn_mask.shape
        # q = q.contiguous().view(-1, q.size(-2), q.size(-1))
        # k = k.contiguous().view(-1, k.size(-2), k.size(-1))
        # attn_mask = attn_mask.contiguous().view(-1, attn_mask.size(-2), attn_mask.size(-1))
        #
        # q_sorted, q_indices = q.sort(dim=-2)
        # k_sorted, k_indices = k.sort(dim=-2)
        # p2 = []
        # for batch in range(q.size(0)):
        #     pd = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
        #     for di in range(q.size(-1)):
        #         q_index = q_indices[batch, :, di]
        #         k_index = k_indices[batch, :, di]
        #
        #         pi = torch.zeros(q.size(-2), k.size(-2)).to(q.device)
        #         pi[q_index, k_index] = 1
        #
        #         qi = q[batch, :, di].view(-1, 1)
        #         ki = k[batch, :, di].view(1, -1)
        #         ci = torch.abs(qi-ki).pow(2)
        #         pd = pd + pi * torch.exp(-ci)
        #     pd = pd / q.size(-1)
        #     p2.append(pd.unsqueeze(0))
        # p2= torch.cat(p2, dim=0).masked_fill(attn_mask, 0).view(attn_mask_shape)

        return p


class SWD5(nn.Module):
    def __init__(self):
        super(SWD5, self).__init__()

    def forward(self, q, k, v, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        k_indices_q = q_indices*k_len + k_indices
        # |k_indices_q| : (batch_size, n_heads, q_len*d_k)
        k_indices_q = k_indices_q.gather(-2, q_indices).view(batch_size, n_heads, -1)
        c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # |c_q| : (batch_size, n_heads, q_len*d_k)
        c_q = c.gather(-2, q_indices).view(batch_size, n_heads, -1) / d_k
        # if training != True:
        #     c_q = (1 - c_q.detach() + c_q) / d_k

        p = torch.zeros(batch_size, n_heads, q_len*k_len, device=q.device)
        p.scatter_add_(-1, k_indices_q, c_q)
        p = p.view(batch_size, n_heads, q_len, k_len)
        # p = p + p.transpose(-2, -1)
        out = torch.matmul(p, v)
        return out, p


class SWD6(nn.Module):
    def __init__(self):
        super(SWD6, self).__init__()

    def forward(self, q, k, v, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        q_sorted, q_indices = q.sort(dim=-2)
        k_sorted, k_indices = k.sort(dim=-2)
        k_indices_q = q_indices*k_len + k_indices
        # |k_indices_q| : (batch_size, n_heads, q_len, d_k)
        k_indices_q = k_indices_q.gather(-2, q_indices)
        c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # |c_q| : (batch_size, n_heads, q_len, d_k)
        c_q = c.gather(-2, q_indices) / d_k

        out = []
        for d in range(d_k):
            p = torch.zeros(batch_size, n_heads, q_len*k_len, device=q.device)
            p.scatter_(-1, k_indices_q[:, :, :, d], c_q[:, :, :, d])
            p = p.view(batch_size, n_heads, q_len, k_len)
            p = torch.matmul(p, v[:, :, :, d].unsqueeze(-1))
            out.append(p)
        out = torch.cat(out, dim=-1)
        return out, p


class SWD7(nn.Module):
    def __init__(self):
        super(SWD7, self).__init__()
        self.N = 2
        self.weight = nn.Parameter(torch.randn(self.N))
        self.activation = nn.Softmax(dim=-1)

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        batch_size, n_heads, q_len, d_k = q.shape
        _, _, k_len, _ = k.shape

        # q_sorted, q_indices = q.sort(dim=-2)
        # # _, q_rank = q_indices.sort(dim=-2)
        # k_sorted, k_indices = k.sort(dim=-2)
        # v_sorted, v_indices = v.sort(dim=-2)
        # out = v_sorted

        # img_ps = 16 # 一共16*16=256个patch
        #
        # vs = []
        # vs.append(v[:,:,0,:].unsqueeze(-2))
        # for i in range(k_len//img_ps):
        #     # +1 因为CLS
        #     v_sorted, _ = v[:,:,1+i*img_ps:1+(i+1)*img_ps,:].sort(dim=-2)
        #     vs.append(v_sorted)
        # v = torch.cat(vs, dim=-2)
        #
        # new_v = torch.zeros_like(v)
        # new_v[:,:,0,:] = v[:,:,0,:]
        # for i in range(img_ps):
        #     # +1 因为CLS
        #     indices = [1+i+j*img_ps for j in range(k_len//img_ps)]
        #     v_sorted, _ = v[:,:,indices,:].sort(dim=-2)
        #     new_v[:,:,indices,:] = v_sorted
        # v = new_v
        # out = v

        # max pooling
        # new_v = v.clone()
        # values, _ = torch.max(v, dim=-2)
        # new_v[:,:,0,:] = values
        # out = new_v

        # min pooling
        # new_v = v.clone()
        # values, _ = torch.max(v, dim=-2)
        # new_v[:,:,0,:] = values
        # out = new_v

        # Max abs with sign
        # new_v = v.clone()
        # values, _ = torch.max(v, dim=-2)
        # abs_values, _ = torch.max(torch.abs(v), dim=-2)
        # values = torch.where(values==abs_values, abs_values, -abs_values)
        # new_v[:,:,0,:] = values
        # out = new_v

        # Max abs
        # new_v = v.clone()
        # abs_values, _ = torch.max(torch.abs(v), dim=-2)
        # new_v[:,:,0,:] = abs_values
        # out = new_v

        # max exchange
        new_v = v.clone()
        values, indices = torch.max(v, dim=-2)
        v_cls = v[:, :, 0, :]
        new_v[:, :, 0, :] = values
        new_v.scatter_(-2, indices.unsqueeze(-2), v_cls.unsqueeze(-2))
        out = new_v

        # sorting
        # v_sorted, v_indices = v.sort(dim=-2)
        # out = v_sorted
        # v_sorted, v_indices = v.abs().sort(dim=-2, descending=True)
        # out = v_sorted

        # v_sorted, v_indices = v.sort(dim=-2)
        #
        # def get_N_sort(v, indices, N):
        #     for n in range(N):
        #         v = v.gather(-2, indices)
        #     return v
        #
        # # out = (1/self.N) * v_sorted
        # # for n in range(2, self.N+1):
        # #     out = out + (1/self.N) * get_N_sort(v, v_indices, n)
        # weight = self.activation(self.weight)
        # out = weight[0] * v_sorted
        # for n in range(2, self.N+1):
        #     out = out + weight[n-1] * get_N_sort(v, v_indices, n)
        # print(weight[0])

        # q_sorted2 = q_sorted.flip(dims=[-2])
        # k_sorted2 = k_sorted.flip(dims=[-2])
        # v_sorted2 = v_sorted.flip(dims=[-2])
        # c = torch.exp(-torch.abs(q_sorted - k_sorted).pow(2))
        # out = c * v_sorted
        # out = out + torch.exp(-torch.abs(q_sorted2 - k_sorted2).pow(2)) * v_sorted2

        # |c_q| : (batch_size, n_heads, q_len, d_k)
        # c_q = c.gather(-2, q_rank)
        # k_indices_q = k_indices.gather(-2, q_rank)
        # v_q = v.gather(-2, k_indices_q)
        # out = c_q * v_q
        # print(c.max(), c.min())
        # print(c[0,0].max(), c[0,0].min())
        # print(c[0,0,:,0].max(), c[0,0,:,0].min())
        # print(c[0,0,:,0])
        #
        # print(out.max(), out.min())
        # print(out[0,0].max(), out[0,0].min())
        # print(out[0,0,:,0].max(), out[0,0,:,0].min())
        # print(out[0,0,:,0])
        # 1/0
        return out, out

class SWD8(nn.Module):
    ''' 
        Haar-modulation
    '''
    def __init__(self):
        super(SWD8, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        d_v = v.size(-1)

        v_sorted, v_indices = v.sort(dim=-2)
        out = v_sorted

        col_descend = torch.tensor(col_descend, device=out.device, dtype=torch.long).flatten()
        out[..., col_descend] = out[..., col_descend].flip(-2)
        # v_indices[..., col_descend] = v_indices[..., col_descend].flip(-2)
        return out, out

def Haar_wavelet_basis(num_col, num_basis):
    interval = max(1, num_col // num_basis)
    idx_basis = [value for idx in range(num_col // (interval * 2)) for value in
                 range((idx * 2 + 1) * interval, (idx * 2 + 2) * interval)]
    if num_basis > 1:
        idx_basis.extend(list(range(idx_basis[-1] + interval, num_col)))
    return idx_basis


class SWD9(nn.Module):
    ''' 
        Cross-Feature Sparse Attention
    '''
    def __init__(self):
        super(SWD9, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        d_v = v.size(-1)

        v_sorted, v_indices = v.sort(dim=-2)
        v1_indices = torch.cat([v_indices[:, :, :, 1:], v_indices[:, :, :, 0].unsqueeze(-1)], dim=-1)
        _, v1_indices_T = v1_indices.sort(dim=-2)
        
        out = v.gather(dim=-2, index=v_indices)
        out = out.gather(dim=-2, index=v1_indices_T)
        
        attn = v_indices.gather(dim=-2, index=v1_indices_T)[:,:,:,0]
        attn = F.one_hot(attn)
        
        return out, attn


class SWD10(nn.Module):
    ''' 
        multi-head sliceformer
    '''
    def __init__(self):
        super(SWD10, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        d_v = v.size(-1)
        sum_head = v.sum(dim=-1)
        sum_sorted, sum_indices = sum_head.sort(dim=-1)
        sum_indices = sum_indices.unsqueeze(-1).repeat(1, 1, 1, d_v)
        
        out = v.gather(dim=-2, index=sum_indices)
        
        return out, out
    

class SWD11(nn.Module):
    ''' 
        Cross-Multi-Feature Sparse Attention
    '''
    def __init__(self):
        super(SWD11, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        d_v = v.size(-1)
        v_len = v.size(-2)
        
        K = 3

        v_sorted, v_indices = v.sort(dim=-2)
        v_onehot = F.one_hot(v_indices)
        for k in range(1, K+1):
            vk_indices = torch.cat([v_indices[:, :, :, k:], v_indices[:, :, :, :k]], dim=-1)
            _, vk_indices_T = vk_indices.sort(dim=-2)
            vk_indices_T = vk_indices_T.unsqueeze(-1).repeat(1, 1, 1, 1, v_len)
            vk_onehot = v_onehot.gather(2, vk_indices_T)
            if k == 1:
                P = vk_onehot
            else:
                P = P + vk_onehot
        
        P = P.permute(0, 1, 3, 2, 4).float()
        out = torch.matmul(P, v.unsqueeze(-1).permute(0, 1, 3, 2, 4)).squeeze(-1).permute(0, 1, 3, 2)

        return out, out

class SWD12(nn.Module):
    ''' 
        QK index sort V 
    '''
    def __init__(self):
        super(SWD12, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        d_v = v.size(-1)

        _, k_indices = k.sort(dim=-2)
        _, q_indices = q.sort(dim=-2)
        _, q_indices_T = q_indices.sort(dim=-2)
        
        out = v.gather(dim=-2, index=k_indices)
        out = out.gather(dim=-2, index=q_indices_T)
        
        return out, out
    

class SWD13(nn.Module):
    ''' 
        Q index sort V 
    '''
    def __init__(self):
        super(SWD13, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        d_v = v.size(-1)

        _, q_indices = q.sort(dim=-2)
        out = v.gather(dim=-2, index=q_indices)
        
        return out, out
    

class SWD14(nn.Module):
    ''' 
        Cross-Feature Sparse Attention
    '''
    def __init__(self):
        super(SWD14, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, n_heads, q_len, d_k), |k| : (batch_size, n_heads, k_len, d_k)
        v_len = v.size(-2)
        d_v = v.size(-1)

        v_sorted, v_indices = v.sort(dim=-2)
        _, v1_indices_T = v_indices[:, :, :, :1].sort(dim=-2)
        v1_indices_T = v1_indices_T.repeat(1, 1, 1, d_v)
        out = v_sorted.gather(dim=-2, index=v1_indices_T)
        out = torch.cat([v[:, :, :, :1], out[:, :, :, 1:]], dim=-1)
        
        return out, out


class SWD15(nn.Module):
    ''' 
        Equivalent Cross-Feature Sparse Attention
    '''
    def __init__(self):
        super(SWD15, self).__init__()

    def forward(self, q, k, v, col_descend=None, training=True):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        v_len = v.size(-2)
        d_v = v.size(-1)

        v_sorted, v_indices = v.sort(dim=-2)
        _, v1_indices_T = v_indices[:, :, :1].sort(dim=-2)
        v1_indices_T = v1_indices_T.repeat(1, 1, d_v)
        out = v_sorted.gather(dim=-2, index=v1_indices_T)
        out = torch.cat([v[:, :, :1], out[:, :, 1:]], dim=-1)
        
        cls_indices = torch.argmin(v[:, :, :1], dim=1)
        
        return out, out, cls_indices
    
    
class SWD16(nn.Module):
    ''' 
        GroupSort
    '''
    def __init__(self, layer_idx):
        super(SWD16, self).__init__()
        self.layer_idx = layer_idx

    def forward(self, q, k, v):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        
        v = torch.cat([v[:, self.layer_idx:, :], v[:, :self.layer_idx, :]], dim=1)
          
        len_sort_window = 13
        sort_dim = 1
        v_shape = list(v.shape)
        v_len = v_shape[1]
        v_shape[1] = len_sort_window
        v_shape.insert(2, v_len // len_sort_window)
        
        grouped_v = v.view(*v_shape)
        sorted_grouped_v, _ = grouped_v.sort(dim=sort_dim)
        out = sorted_grouped_v.view(*list(v.shape))
        
        out = torch.cat([out[:, -self.layer_idx:, :], out[:, :-self.layer_idx, :]], dim=1)
        
        return out
    

class SWD17(nn.Module):
    ''' 
        flatten sort view
    '''
    def __init__(self, layer_idx):
        super(SWD17, self).__init__()
        self.layer_idx = layer_idx

    def forward(self, q, k, v):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        
        batch_size = v.size(0)
        v_flatten = v.reshape(batch_size, -1)          
        v_flatten_sorted, _ = v_flatten.sort(dim=-1)
        out = v_flatten_sorted.view(v.shape)
        return out

class SWD18(nn.Module):
    ''' 
        feature group sort
    '''
    def __init__(self, layer_idx):
        super(SWD18, self).__init__()
        self.layer_idx = layer_idx

    def forward(self, q, k, v):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        batch_size, v_len, d_v = v.shape
        len_sort_window = 5
        v_sorted = []
        for i in range(d_v):
            v_i_shift = torch.cat([v[:, i:, i], v[:, :i, i]], dim=1)
            v_i_sorted = []
            for j in range(v_len//len_sort_window):
                v_i_j_sorted, _ = torch.sort(v_i_shift[:, j*len_sort_window:(j+1)*len_sort_window], dim=1)
                v_i_sorted.append(v_i_j_sorted)
            v_i_sorted = torch.cat(v_i_sorted, dim=1)
            v_i_sorted = torch.cat([v_i_sorted[:, -i:], v_i_sorted[:, :-i]], dim=1)
            v_sorted.append(v_i_sorted.unsqueeze(-1))
        v_sorted = torch.cat(v_sorted, dim=-1)
        
        return v_sorted

class SWD19(nn.Module):
    ''' 
        feature group sort
    '''
    def __init__(self, args, layer_idx):
        super(SWD19, self).__init__()
        self.layer_idx = layer_idx
        self.len_sort_window = args['len_sort_window']

    def forward(self, q, k, v):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        batch_size, v_len, d_v = v.shape
        v_shift = []
        for i in range(d_v):
            v_i_shift = torch.cat([v[:, i:, i], v[:, :i, i]], dim=1)
            v_i_shift = v_i_shift.unsqueeze(dim=-1)
            v_shift.append(v_i_shift)
        v_shift = torch.cat(v_shift, dim=-1)
        
        v_shift = v_shift.view(batch_size, -1, self.len_sort_window, d_v)
        v_sorted, _ = torch.sort(v_shift, dim=2)
        v_sorted = v_sorted.view(batch_size, v_len, d_v)
        
        v_shift = []
        for i in range(d_v):
            v_i_shift = torch.cat([v_sorted[:, -i:, i], v_sorted[:, :-i, i]], dim=1)
            v_i_shift = v_i_shift.unsqueeze(dim=-1)
            v_shift.append(v_i_shift)
        v_shift = torch.cat(v_shift, dim=-1)
        
        return v_shift
    
class SWD20(nn.Module):
    ''' 
        feature group sort + shift
    '''
    def __init__(self, args, layer_idx):
        super(SWD20, self).__init__()
        self.layer_idx = layer_idx
        self.len_sort_window = args['len_sort_window']
        self.device = args['gpu']

    def forward(self, q, k, v):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        batch_size, v_len, d_v = v.shape
        
        indices = torch.arange(d_v, device=self.device).view(1, 1, d_v)
        arange1 = torch.arange(v_len, device=self.device).view(1, v_len, 1).repeat(batch_size, 1, d_v)
        arange2 = (arange1 - indices) % v_len
        v_shift = torch.gather(v, 1, arange2)
        
        v_shift = v_shift.view(batch_size, -1, self.len_sort_window, d_v)
        v_sorted, _ = torch.sort(v_shift, dim=2)
        out = v_sorted.view(batch_size, v_len, d_v)
        
        return out

def butterfly_factor_to_matrix(twiddle, factor_index, device):
    n_div_b, b, _ = twiddle.shape
    n = b * n_div_b
    log_b_n = int(math.log(n) / math.log(b))
    assert n == b ** log_b_n, f'n must be a power of {b}'
    assert twiddle.shape == (n // b, b, b)
    assert 0 <= factor_index <= log_b_n
    stride = b ** factor_index
    x = rearrange(torch.eye(n, device=device), 'bs (diagblk j stride) -> bs diagblk j stride', stride=stride, j=b)
    t = rearrange(twiddle, '(diagblk stride) i j -> diagblk stride i j', stride=stride)
    out = torch.einsum('d s i j, b d j s -> b d i s', t, x)
    out = rearrange(out, 'b diagblk i stride -> b (diagblk i stride)')
    return out.t()  # Transpose because we assume the 1st dimension of x is the batch dimension

class SWD21(nn.Module):
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
    
class SWD22(nn.Module):
    ''' 
        equivalent feature group sort + butterfly
    '''
    def __init__(self, args, **kwargs):
        super(SWD22, self).__init__()
        self.len_sort_window = args['len_sort_window']
        self.butterfly_matrix = None
        self.shift_matrix = None
        self.device = args['gpu']

    def get_butterfly_matrix(self, v_len, d_v):
        b = 2
        twiddle = torch.ones((1, v_len * b), dtype=torch.float, device=self.device).reshape(v_len // b, b, b)
        col_indices = [torch.arange(v_len, dtype=torch.float, device=self.device)]
        shift_offset = 0
        while len(col_indices) < d_v:
            for factor_index in range(int(math.log(v_len, b))):
                butterfly_matrix = butterfly_factor_to_matrix(twiddle, factor_index, self.device)
                col_index = torch.arange(v_len, dtype=torch.float, device=self.device)
                col_index = (butterfly_matrix - torch.eye(v_len, device=self.device)) @ col_index
                col_index = torch.cat([col_index[-shift_offset:], col_index[:-shift_offset]])
                shift_offset = (shift_offset + self.len_sort_window) % v_len
                col_indices.append(col_index)
                if len(col_indices) == d_v:
                    break
        return torch.stack(col_indices).t()
    
    def get_shift_matrix(self, v_len, d_v):
        col_indices = []
        shift_offset = 0
        for _ in range(d_v):
            col_index = torch.arange(v_len, dtype=torch.float, device=self.device)
            col_index = torch.cat([col_index[-shift_offset:], col_index[:-shift_offset]])
            shift_offset = (shift_offset + self.len_sort_window) % v_len
            col_indices.append(col_index)
            
        return torch.stack(col_indices).t()
    
    def forward(self, q, k, v):
        # |q| : (batch_size, q_len, d_k), |k| : (batch_size, k_len, d_k)
        batch_size, v_len, d_v = v.shape
        
        if self.butterfly_matrix is None:
            self.butterfly_matrix = self.get_butterfly_matrix(v_len, d_v).unsqueeze(dim=0).long()

        v = torch.gather(v, 1, self.butterfly_matrix.repeat(batch_size, 1, 1))
         
        v_abs_max, _ = v.abs().max(dim=2)
        v0 = v[:, 0::2, 0]
        v1 = v[:, 1::2, 0]
        v_reverse = torch.stack([v1, v0], dim=2)
        v_reverse = torch.flatten(v_reverse, start_dim=1, end_dim=2)
        v_sign = torch.sign(v[:,:,0] - v_reverse)
        v_alter_value = (v_sign * v_abs_max).unsqueeze(dim=2)
        v_alter = v + v_alter_value
        
        v_shifted = v_alter.view(batch_size, -1, self.len_sort_window, d_v)
        v_sorted, _ = torch.sort(v_shifted, dim=2)
        v_sorted = v_sorted.view(batch_size, v_len, d_v)
        
        v_alter_shifted = v_alter_value.view(batch_size, -1, self.len_sort_window, 1)
        v_alter_sorted, _ = torch.sort(v_alter_shifted, dim=2)
        v_alter_sorted = v_alter_sorted.view(batch_size, v_len, 1)
        
        out = v_sorted - v_alter_sorted
        return out
    
class SWD23(nn.Module):
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