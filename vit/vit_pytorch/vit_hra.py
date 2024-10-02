import torch
from torch import nn
import torch.nn.functional as F

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from vit_pytorch.swd import *

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        hidden_dim = args['mlp_dim']
        dropout = args['dropout']
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class HRA_FFN(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        hidden_dim = args['mlp_dim']
        dropout = args['dropout']
        self.net = nn.Sequential(
            HRA(args),
            ABS(),
            nn.Dropout(dropout),
            HRA(args),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ABS(nn.Module):
    def __init__(self):
        super().__init__()
                
    def forward(self, x):
        return torch.abs(x)
    
class HRA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.r = 2
        self.apply_GS = False
        self.in_features = args['dim']
        
        half_u = torch.zeros(self.in_features, self.r // 2)
        nn.init.kaiming_uniform_(half_u, a=math.sqrt(5))
        hra_u = torch.repeat_interleave(half_u, 2, dim=1)
        self.hra_u = nn.ParameterList([nn.Parameter(hra_u[:, i].view(-1, 1), requires_grad=True) for i in range(self.r)])
                
    def forward(self, x):
        if self.apply_GS:
            weight = [(self.hra_u[0] / self.hra_u[0].norm()).view(-1, 1)]
            for i in range(1, self.r):
                ui = self.hra_u[i]
                for j in range(i):
                    ui = ui - (weight[j].t() @ ui) * weight[j]
                weight.append((ui / ui.norm()).view(-1, 1))
            weight = torch.cat(weight, dim=1)
            new_weight = torch.eye(self.in_features, device=x.device, dtype=x.dtype) - 2 * weight @ weight.t()
            
        else:
            # new_weight = torch.eye(self.in_features, device=x.device, dtype=x.dtype)
            for i in range(self.r):
                ui = self.hra_u[i] / self.hra_u[i].norm()
                # new_weight = torch.mm(new_weight, torch.eye(self.in_features, device=x.device, dtype=x.dtype) - 2 * ui @ ui.t())
                x = x - 2 * (x @ ui) @ ui.t()
                
        return x

class Attention(nn.Module):
    def __init__(self, args, layer_idx=None):
        super().__init__()
        dim = args['dim']
        dim_head = args['dim_head']
        dropout = args['dropout']
        self.heads = args['n_heads']
        self.attn = args['attn']
        
        inner_dim = dim_head *  self.heads
        project_out = not (self.heads == 1 and dim_head == dim)

        self.scale = dim_head ** -0.5
        self.swd = SWD20(args, layer_idx=layer_idx)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if self.attn == 'swd':
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            
    def forward(self, x):
        cls_indices = None
        if self.attn == 'trans':
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale            
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            attn = out
            # U, S, Vh = torch.linalg.svd(out)
            # print(S[0,0])
            out = rearrange(out, 'b h n d -> b n (h d)')
        elif self.attn == 'swd':
            q, k, v = self.to_qkv(x).chunk(3, dim = -1)
            out = self.swd(q, k, v)
            
        return self.to_out(out), cls_indices

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([])
        for layer_idx, _ in enumerate(range(args['n_layers'])):
            if layer_idx in [0, args['n_layers'] - 1]:
                self.layers.append(nn.ModuleList([
                    PreNorm(args['dim'], Attention(args, layer_idx=layer_idx)),
                    PreNorm(args['dim'], FeedForward(args))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(args['dim'], Attention(args, layer_idx=layer_idx)),
                    HRA_FFN(args)
                ]))
            
    def forward(self, x):
        attn_weights = []
        for layer_idx, (attn, ff) in enumerate(self.layers):
            attn_x, cls_indices = attn(x)
            x = attn_x + x
            if layer_idx in [0, len(self.layers) - 1]:
                x = ff(x) + x
            else:
                x = ff(x)
            attn_weights.append(attn_x.detach().clone().cpu())
        return x, attn_weights, cls_indices


class ViT_HRA(nn.Module):    
    def __init__(self, args):
        super().__init__()
        image_height, image_width = pair(args['size'])
        patch_height, patch_width = pair(args['ps'])

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = args['channels'] * patch_height * patch_width

        dim = args['dim']
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(args['emb_dropout'])

        self.transformer = Transformer(args)

        self.pool = args['pool']
        assert self.pool in {'cls', 'mean', 'max'}, 'pool type must be either cls (cls token), mean (mean pooling) or max'
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, args['num_classes'])
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else:
            x += self.pos_embedding[:, :n]
            
        x = self.dropout(x)

        trans_x, attn_weights, cls_indices = self.transformer(x)
        x = trans_x

        if self.pool == 'mean':
            x = x.mean(dim = 1)
        elif self.pool == 'max':
            x, _ = x.max(dim = 1)
        elif cls_indices is not None:
            cls_indices = cls_indices.unsqueeze(1).repeat(1, 1, x.size(-1))
            x = x.gather(dim=-2, index=cls_indices).squeeze()
        else:
            x = x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), attn_weights
