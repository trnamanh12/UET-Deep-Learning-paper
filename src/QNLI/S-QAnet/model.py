import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# postiional embedding in w2v
# self-attention
# multi-head attention ? the difference between self-attention and multi-head attention in implementation
# feed forward network
# layer normalization
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias: bool = True, eps: float = 1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(ndim))
        self.beta = nn.Parameter(torch.zeros(ndim))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SelfAttention(nn.Module):
    def __init__(self, embed_size, nhead, dropout):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.p_qkv = nn.Linear(embed_size, embed_size*3)
        torch.nn.init.xavier_uniform_(self.p_qkv.weight)

        self.p_proj = nn.Linear(embed_size, embed_size)
        torch.nn.init.xavier_uniform_(self.p_proj.weight)

        self.nhead = nhead
    
    def forward(self, x , attmask):   
        '''
        1. input q, k, v, attention mask x [bs, nhead, seq_len, embed_size], attmask [bs, seqlen]
        2. calculate the attention score
        3. add & norm ( dropout residual connection before add )
        4. feed forward network
        5. add & norm ( dropout residual connection before add )
        ensure that output have shape [bs, seqlen, embed_size*n_head]

        '''

        x = self.p_qkv(x) # [bs, seq_len, embed_size*3]
        q, k, v = torch.chunk(x, 3, dim = -1) # q, k, v [bs, seq_len, embed_size]
        bs, sqlen, embed_size = q.size()

        q = q.view(bs, sqlen, self.nhead, embed_size//self.nhead).transpose(1, 2)
        k = k.view(bs, sqlen, self.nhead, embed_size//self.nhead).transpose(1, 2)
        v = v.view(bs, sqlen, self.nhead, embed_size//self.nhead).transpose(1, 2)

        # configure mask
        attmask = attmask.unsqueeze(1).unsqueeze(2) # [bs, 1, 1, seq_len]


        att_score = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(q.size(-1))) # [bs, nhead, seq_len, seq_len]
        att_score = att_score.masked_fill(attmask == 0, -10000)
        att_score = F.softmax(att_score, dim = -1)
        att_score = self.dropout(att_score) # [bs, nhead, seq_len, seq_len]
        y = att_score @ v # [bs, nhead, seqlen, embed_size//nhead]
        y = y.transpose(1, 2).contiguous().view(bs, sqlen, embed_size)
        
        # is y need to be go through a linear layer?
        y = self.p_proj(y)
        return y
    
class FFN(nn.Module):
    def __init__(self, embed_size):
            super().__init__()
            self.linear1  = nn.Linear(embed_size, embed_size*4)
            torch.nn.init.xavier_uniform_(self.linear1.weight)

            self.linear2 = nn.Linear(embed_size*4, embed_size)
            torch.nn.init.xavier_uniform_(self.linear2.weight)

            self.gelu = nn.GELU()
            self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
            x = self.linear1(x)
            x = self.gelu(x)
            x = self.dropout(x)
            x = self.linear2(x)
            x = self.dropout(x)
            return x

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, nhead, dropout,  bias=True, eps=1e-06):
        super().__init__()    
        self.selfattn = SelfAttention(embed_size, nhead, dropout)
        self.ffn = FFN(embed_size)
        self.dropout = nn.Dropout(0.1)
        self.norm = LayerNorm(embed_size, bias, eps)

    def forward(self, x, mask):
        # x  [bs, seqlen, embed_size]
        x = x +  self.dropout(self.selfattn(x, mask))
        x = self.norm(x)
		
        x = x + self.dropout(self.ffn(x))
        x = self.norm(x)
        return x, mask



class TransformerEnc(nn.Module):
    def __init__(self, embed_size,nhead, num_layers =3, c_len = 96, device='cuda'):
        super(TransformerEnc, self).__init__()
        '''
            1. Encoder question and context 
            2. CNN to get the local conte
        '''
        self.qencoder = nn.ModuleList([EncoderLayer(embed_size, nhead, 0.1).to(device) for _ in range(num_layers)])
        self.cencoder = nn.ModuleList([EncoderLayer(embed_size, nhead, 0.1).to(device) for _ in range(num_layers)])
        
        self.Wsim = nn.Linear(embed_size*3, 1)
        torch.nn.init.xavier_uniform_(self.Wsim.weight)

        self.Wdistil = nn.Linear(embed_size*4, embed_size)
        torch.nn.init.xavier_uniform_(self.Wsim.weight)

        self.synin4 = [EncoderLayer(embed_size*4, nhead, 0.1 ).to(device) for _ in range(num_layers)]

        self.Whead1 = nn.Linear(c_len*embed_size*4, embed_size)
        torch.nn.init.xavier_uniform_(self.Whead1.weight)

        self.Whead2 = nn.Linear(embed_size, 2)
        torch.nn.init.xavier_uniform_(self.Whead2.weight)

        self.dropout = nn.Dropout(0.1)
       
    def forward(self, c, q, q_mask, c_mask):
        # q_mask = q_mask.unsqueeze(-1)
        # c_mask = c_mask.unsqueeze(-1)
        # q_mask, c_mask [bs, seqlen]
        for layer in self.qencoder:
            q, q_mask = layer(q, q_mask)
        for layer in self.qencoder:
            c, c_mask = layer(c, c_mask)
        # c = self.encoder(c, c_mask) # [bs, c_len, embed_size]
        # q = self.encoder(q, q_mask) # [bs, q_len, embed_size]
        # caculate similarity matrix
        bs = c.size(0)
        c_len = c.size(1)
        q_len = q.size(1)

        c_sim = c.unsqueeze(2).expand(-1, -1, q_len, -1) # [bs, c_len, q_len, embed_size]
        q_sim = c.unsqueeze(1).expand(-1, c_len, -1, -1) # [bs, c_len, q_len, embed_size] 
        
        cq_sim = torch.mul(c_sim, q_sim) # [bs, c_len, q_len, embed_size]
        cqcq = torch.cat([c_sim, q_sim, cq_sim], dim=-1) # [bs, c_len, q_len, 3*embed_size]
        S = self.Wsim(cqcq).squeeze(-1) # similarity matrix [bs, c_len, q_len]
        
        # can meet  error such as the shape of mask can't be broadcastable with the shape of the tensor
        # can fix by unsqueeze the mask at dim = -1 of q_mask and c_mask
        # q_mask = q_mask.unsqueeze(-1)

        S_row = S.masked_fill_(q_mask.unsqueeze(1) == 0, -10000)
        S_row = F.softmax(S_row, dim=-1) # [bs, c_len, q_len]
        A = torch.bmm(S_row, q) # [bs, c_len, embed_size]

        # c_mask = c_mask.unsqueeze(-1)
        S_col = S.masked_fill_(c_mask.unsqueeze(2) == 0, -10000)
        S_col = F.softmax(S_col, dim = 1) # [bs, c_len, q_len]

        B = torch.bmm(torch.bmm(S_col, S_row.transpose(1,2)), c) # [bs, c_len, embed_size]

        distil = torch.cat([c, A, torch.mul(c, A), torch.mul(c, B)], dim = -1) # [bs, c_len, 4*embed_size]
        # distil = self.Wdistil(distil) # distil information [bs, c_len, embed_size]
        for layer in self.synin4:
            distil, c_mask = layer(distil, c_mask)
        # synin4 = self.synin4(distil) 
        distil = distil.contiguous().view(bs, -1)
        out1 = self.Whead1(distil)

        out2 = self.Whead2(self.dropout(out1))

        return out2


