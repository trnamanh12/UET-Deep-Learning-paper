import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Word2Vec(nn.Module):
	def __init__(self, vocab_size, embed_size, BERT = False): 
		super(Word2Vec, self).__init__()
		if BERT:
			model = BertModel.from_pretrained('bert-base-uncased')
			self.embeddings = model.embeddings.word_embeddings
			self.embeddings.requires_grad_(False)
		else:	
			self.embeddings = nn.Embedding(vocab_size, embed_size)
			torch.nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
	def forward(self, x):
		x = self.embeddings(x)
		return x

# class Highway(nn.Module):
# 	def __init__(self, args):
# 		super(Highway, self).__init__()
# 		self.W_proj = nn.Linear(args.hidden_size*2, args.hidden_size*2)
# 		self.W_gate = nn.Linear(args.hidden_size, args.hidden_size)
	
# 	def forward(self, x):
# 		x_proj = F.relu(self.W_proj(x))
# 		x_gate = F.sigmoid(self.W_gate(x))
# 		x_highway = x_gate * x_proj + (1 - x_gate) * x
# 		return x_highway

class ContextualEmbedding(nn.Module):
	def __init__(self, embed_size, hidden_size):
		super(ContextualEmbedding, self).__init__()
		self.RNN= nn.RNN(input_size=embed_size,
							hidden_size=hidden_size,
							num_layers=2,
							bidirectional=True,
							batch_first=True,
							dropout=0.2)
	def forward(self, x):
		output, _ = self.RNN(x)
		return output
		