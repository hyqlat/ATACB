import torch.nn as nn
import torch
# from utils.torch import *

class PositionEnbedding(nn.Module):
    def __init__(
            self, 
            max_token_length=1000,
            d_model=512,
            base=10000
        ):
        super(PositionEnbedding, self).__init__()
        self.max_token_length = max_token_length
        self.d_model = d_model
        self.base = base

    def forward(self):
        pe = torch.zeros(self.max_token_length, self.d_model, dtype=torch.float)
        exp1 = torch.arange(self.d_model // 2, dtype=torch.float)
        exp_value = exp1 / (self.d_model / 2)

        alpha = 1 / (self.base ** exp_value)
        out = torch.arange(self.max_token_length, dtype=torch.float)[:, None] @ alpha[None, :]
        embed_sin = torch.sin(out)
        embed_cos = torch.cos(out)

        pe[:, 0::2] = embed_sin
        pe[:, 1::2] = embed_cos

        return pe

class ATT(nn.Module):
    def __init__(self, input_dim, out_dim, att_head, att_hiddim, ff_hiddim, att_do, te_layernum):
        super(ATT, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        # self.bi_dir = bi_dir
        self.mode = 'batch'
        # self.is_norm = is_layernorm
        self.in_linear = nn.Linear(
            in_features=input_dim,
            out_features=att_hiddim
        )
        
        self.TEL = nn.TransformerEncoderLayer(
            d_model=att_hiddim,
            nhead=att_head,
            dim_feedforward=ff_hiddim,
            dropout=att_do
        )
        self.TE = nn.TransformerEncoder(
            encoder_layer=self.TEL,
            num_layers=te_layernum
        )

        self.out_linear = nn.Linear(
            in_features=att_hiddim,
            out_features=out_dim
        )

        self.pe = PositionEnbedding(d_model=att_hiddim)

        # if self.is_norm:
        #     self.norm = nn.LayerNorm(hidden_dim)
        # self.hx, self.cx = None, None
        self.past_seq = None

    def set_mode(self, mode):
        self.mode = mode

    def initialize(self):
        if self.mode == 'step':
            self.past_seq = None

    def forward(self, x):
        #x: B T D(if step: B 1 D)
        x = self.in_linear(x)
        if self.mode == 'step':
            if self.past_seq is None:
                te_in = x
            else:
                te_in = torch.cat([self.past_seq, x], dim=1) 
            te_in += self.pe().unsqueeze(0)[:, :te_in.shape[1]].to(te_in.device)
            res = self.TE(te_in)
        else:
            te_in = x + self.pe().unsqueeze(0)[:, :x.shape[1]].to(x.device)
            res = self.TE(te_in)
        
        return self.out_linear(res)

    

if __name__ == '__main__':
    att = ATT(
        input_dim=12, 
        out_dim=24, 
        att_head=3, 
        att_hiddim=512, 
        att_do=0.1, 
        te_layernum=2
    )
    input = torch.zeros(5, 10, 12)
    out = att(input)
    print(out.shape)
