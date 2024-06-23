import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# postiional embedding in w2v
# self-attention
# multi-head attention ? the difference between self-attention and multi-head attention in implementation
# can we merge self-attention and multi-head attention into one class?
# feed forward network
# layer normalization
class LayerNorm(nn.Module):
	def __init__(self, ndim, eps: float = 1e-5):
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
		# because att mask can be 2D if it is used in encoder, and 3D if it is used in decoder
		if attmask.dim() == 2:
			attmask = attmask.unsqueeze(1).unsqueeze(2) # [bs, 1, 1, seq_len]
			# attmask [bs, seqlen, embed_size], we need to translate  [bs, nhead, seqlen, embed_size//nhead]
		
		

		att_score = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(q.size(-1))) # [bs, nhead, seq_len, seq_len]
		att_score = att_score.masked_fill(attmask == 0, -10000)
		att_score = F.softmax(att_score, dim = -1)
		att_score = self.dropout(att_score) # [bs, nhead, seq_len, seq_len]
		y = att_score @ v # [bs, nhead, seqlen, embed_size//nhead]
		y = y.transpose(1, 2).contiguous().view(bs, sqlen, embed_size)
		
		# is y need to be go through a linear layer?
		y = self.p_proj(y)
		return y, attmask

class CrossAttention(nn.Module):
	def __init__(self,  embed_size, nhead):
			super(CrossAttention, self).__init__()
			self.nhead = nhead
			self.W_q = nn.Linear(embed_size, embed_size)
			torch.nn.init.xavier_uniform_(self.W_q.weight)
			self.W_k = nn.Linear(embed_size, embed_size)
			torch.nn.init.xavier_uniform_(self.W_k.weight)
			self.W_v = nn.Linear(embed_size, embed_size)
			torch.nn.init.xavier_uniform_(self.W_v.weight)
			self.W_o = nn.Linear(embed_size, embed_size)
			torch.nn.init.xavier_uniform_(self.W_o.weight)
			self.dropout = nn.Dropout(0.1)
			
	def forward(self, tgt, enc, mask):
		# tgt [bs, seqlen, embed_size]
		# enc [bs, seqlen, embed_size]
		# mask [bs, seqlen]
		bs, tgt_size, embed_size = tgt.size()
		src_size = enc.size(1)

		q = self.W_q(tgt)
		k = self.W_k(enc)
		v = self.W_v(enc)

		# [bs, seqlen, embed_size] -> [bs, seqlen, nhead, embed_size//nhead] -> [bs, nhead, seqlen, embed_size//nhead
		q = q.view(bs, tgt_size, self.nhead, embed_size//self.nhead).transpose(1, 2)
		k = k.view(bs, src_size, self.nhead, embed_size//self.nhead).transpose(1, 2)
		v = v.view(bs, src_size, self.nhead, embed_size//self.nhead).transpose(1, 2)

		att_score = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(q.size(-1)))
		att_score = att_score.masked_fill(mask == 0, -10000)
		att_score = F.softmax(att_score, dim = -1) # [bs, seqlen, seqlen]
		y = att_score @ v
		y = y.transpose(1, 2).contiguous().view(bs, tgt_size, -1)
		y = self.W_o(y)
		return self.dropout(y)

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
		self.norm = LayerNorm(embed_size )

	def forward(self, x, mask):
		# x  [bs, seqlen, embed_size]
		_x = x
		x, mask = self.selfattn(x, mask)
		x = _x +  self.dropout(x)
		x = self.norm(x)

		x = x + self.dropout(self.ffn(x))
		x = self.norm(x)
		return x, mask

class DecoderLayer(nn.Module):
	def __init__(self, config):
		super(DecoderLayer, self).__init__()
		# we use masked self-attention in decoder, we just need to pass the exact mask to the self-attention
		self.maskedselfattn = SelfAttention(config['embed_size'], config['nhead'], config['dropout'])
		self.norm1 = LayerNorm(config['embed_size'])

		self.crossatt = CrossAttention(config['embed_size'], config['nhead'])
		self.norm2 = LayerNorm(config['embed_size'])    

		self.ffn = FFN(config['embed_size'])
		self.norm3 = LayerNorm(config['embed_size'])

		
		self.dropout = nn.Dropout(0.1)

	def _init_casual_mask(self, mask):
		# mask [bs, seqlen]
		mask = mask.unsqueeze(-1).expand(-1, -1, mask.size(-1)).unsqueeze(1) # [bs, 1, seqlen, seqlen]
		mask = torch.tril(mask, diagonal=0)
		return mask # [bs, 1, seqlen, seqlen]
	
	def forward(self, tgt, encoder_output, tgt_mask, src_mask):
		# tgt [bs, seqlen, embed_size]
		# encoder_output [bs, seqlen, embed_size]
		# tgt_mask [bs, seqlen]
		# src_mask [bs, seqlen]
		# tgt_mask = self._init_casual_mask(tgt_mask)
		_tgt = tgt
		tgt, tgt_mask = self.maskedselfattn(tgt, tgt_mask)
		tgt = _tgt + self.dropout(tgt)
		# tgt = tgt + self.dropout(self.maskedselfattn(tgt, tgt_mask))
		tgt = self.norm1(tgt)
		
		# src_mask = src_mask.unsqueeze(-1).expand(-1, -1, tgt_mask.size(-1)).unsqueeze(1) # [bs, 1, seqlen, seqlen ]
		tgt = tgt + self.dropout(self.crossatt(tgt, encoder_output, src_mask))
		tgt = self.norm2(tgt)
		
		tgt = tgt + self.dropout(self.ffn(tgt))
		tgt = self.norm3(tgt)
		
		return tgt, tgt_mask
	
			


class TransformerMT(nn.Module):
	def __init__(self, args):
		super(TransformerMT, self).__init__()
		self.args = args
		self.encoder = EncoderLayer(args['embed_size'], args['nhead'], args['dropout'])
		self.decoder = DecoderLayer(args)
	
	def forward(self, src, tgt, src_mask, tgt_mask):
		# src [bs, seqlen, embed_size]
		# tgt [bs, seqlen, embed_size]
		# src_mask [bs, seqlen]
		# tgt_mask [bs, seqlen]
		enc_output, src_mask = self.encoder(src, src_mask)
		dec_output, tgt_mask = self.decoder(tgt, enc_output, tgt_mask, src_mask)
		return enc_output, dec_output, src_mask, tgt_mask
	