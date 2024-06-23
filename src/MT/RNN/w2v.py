import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import  BertModel

class Word2Vec(nn.Module):
	def __init__(self, vocab_size, embed_size, BERT = False): 
		super(Word2Vec, self).__init__()
		if BERT:
			model = BertModel.from_pretrained('bert-base-multilingual-cased')
			self.embeddings = model.embeddings.word_embeddings
			self.embeddings.requires_grad_(False)
		else:	
			self.embeddings = nn.Embedding(vocab_size, embed_size)
			torch.nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
	def forward(self, x):
		x = self.embeddings(x)
		return x