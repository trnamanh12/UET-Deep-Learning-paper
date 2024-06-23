import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import  BertModel

# class Word2Vec(nn.Module):
# 	def __init__(self, vocab_size, embed_size, BERT = False): 
# 		super(Word2Vec, self).__init__()
# 		if BERT:
# 			model = BertModel.from_pretrained('bert-base-uncased')
# 			self.embeddings = model.embeddings.word_embeddings
# 			self.embeddings.requires_grad_(False)
# 		else:	
# 			self.embeddings = nn.Embedding(vocab_size, embed_size)
# 			torch.nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
# 	def forward(self, x):
# 		x = self.embeddings(x)
# 		return x
	


# class PositionalEmbedding(nn.Module):
# 	def __init__(self,embed_size, max_len, device):
# 		super(PositionalEmbedding, self).__init__()
# 		self.encoding = torch.zeros(max_len, embed_size, requires_grad=False)
# 		pos = torch.arange(0, max_len).float().	unsqueeze(1)
# 		_2i = torch.arange(0, embed_size, 2).float()
# 		self.encoding[:, 0::2] = torch.sin(pos/ torch.pow(10000, _2i/ embed_size)).to(device)
# 		self.encoding[:, 1::2] = torch.cos(pos/ torch.pow(10000, _2i/ embed_size)).to(device)

# 	def forward(self, x):
# 		# bs, seqlen, embed_dim = x.size()
# 		# pe_tensor = torch.zeros(seqlen, embed_dim)
# 		# sin = [torch.sin(pos/ torch.pow(10000, torch.arange(0, embed_dim, 2)/ embed_dim)) for  pos in self.pos]
# 		# cos = [torch.cos(pos/ torch.pow(10000, torch.arange(1, embed_dim, 2)/ embed_dim)) for pos in self.pos]
# 		# pe_tensor[:, 0::2] = sin
# 		# pe_tensor[:, 1::2] = cos
# 		# pe_tensor = pe_tensor.unsqueeze(0).expand(bs, seqlen, embed_dim)
# 		bs, seqlen, embed_dim = x.size()
# 		return self.encoding[:seqlen, :].expand(bs, seqlen, embed_dim)

# class WordEmbedding(nn.Module):
# 	def __init__(self, vocab_size, embed_size, max_len, device, BERT=False):
# 		super(WordEmbedding, self).__init__()
# 		self.word2vec = Word2Vec(vocab_size, embed_size, BERT)
# 		self.positional_embedding = PositionalEmbedding( embed_size, max_len, device)
	
# 	def forward(self, x):
# 		x = self.word2vec(x)
# 		x = x + self.positional_embedding(x)
# 		return x
# # x = torch.randint(0, 100, (	3, 9))
# # a = WordEmbedding(100, 10, 9, 'cpu', BERT = False)
# # print(x.size())
# # a(x)

class Word2Vec(nn.Module):
	def __init__(self, vocab_size, embed_size, BERT = False): 
		super(Word2Vec, self).__init__()
		if BERT:
			model = BertModel.from_pretrained('bert-base-uncased')
			self.embeddings = model.embeddings.word_embeddings
			self.embeddings.requires_grad_(False)
		else:	
			self.embeddings = nn.Embedding(vocab_size, embed_size)
			torch.nn.init.xavier_uniform_(self.embeddings.weight)
	def forward(self, x):
		x = self.embeddings(x)
		return x
	


class PositionalEmbedding(nn.Module):
	def __init__(self,embed_size, max_len, device):
		super(PositionalEmbedding, self).__init__()
		self.encoding = torch.zeros(max_len, embed_size, requires_grad=False, device=device)
		pos = torch.arange(0, max_len, device=device).float().unsqueeze(1)
		_2i = torch.arange(0, embed_size, 2, device=device).float()
		self.encoding[:, 0::2] = torch.sin(pos/ torch.pow(10000, _2i/ embed_size)).to(device)
		self.encoding[:, 1::2] = torch.cos(pos/ torch.pow(10000, _2i/ embed_size)).to(device)

	def forward(self, x):
		# bs, seqlen, embed_dim = x.size()
		# pe_tensor = torch.zeros(seqlen, embed_dim)
		# sin = [torch.sin(pos/ torch.pow(10000, torch.arange(0, embed_dim, 2)/ embed_dim)) for  pos in self.pos]
		# cos = [torch.cos(pos/ torch.pow(10000, torch.arange(1, embed_dim, 2)/ embed_dim)) for pos in self.pos]
		# pe_tensor[:, 0::2] = sin
		# pe_tensor[:, 1::2] = cos
		# pe_tensor = pe_tensor.unsqueeze(0).expand(bs, seqlen, embed_dim)
		bs, seqlen, embed_dim = x.size()
		return self.encoding[:seqlen, :].expand(bs, seqlen, embed_dim)

class WordEmbedding(nn.Module):
	def __init__(self, vocab_size, embed_size, max_len, device, BERT=False):
		super(WordEmbedding, self).__init__()
		self.word2vec = Word2Vec(vocab_size, embed_size, BERT)
		self.positional_embedding = PositionalEmbedding( embed_size, max_len, device)
	
	def forward(self, x):
		x = self.word2vec(x)
		x = x + self.positional_embedding(x)
		return x