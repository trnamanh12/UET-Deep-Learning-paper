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
			self.linear = nn.Linear(768, embed_size)
			torch.nn.init.normal_(self.linear.weight, mean=0, std=0.02)
		else:	
			self.embeddings = nn.Embedding(vocab_size, embed_size)
			torch.nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
			self.linear = nn.Linear(embed_size, embed_size)
			torch.nn.init.normal_(self.linear.weight, mean=0, std=0.02)
	def forward(self, x):
		x = self.embeddings(x)
		x = self.linear(x)
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
		self.RNN= nn.LSTM(input_size=embed_size,
							hidden_size=hidden_size,
							num_layers=1,
							bidirectional=True,
							batch_first=True,
							dropout=0.1)
	def forward(self, x):
		output, _ = self.RNN(x)
		return output
		