import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
	def __init__(self, vocab_size, embed_size, BERT = False): 
		super(Word2Vec, self).__init__()
		if BERT:
			model = BertModel.from_pretrained('bert-base-cased')
			self.embeddings = model.embeddings.word_embeddings
			self.embeddings.requires_grad_(False)
		else:	
			self.embeddings = nn.Embedding(vocab_size, embed_size)
			torch.nn.init.xavier_uniform_(self.embeddings.weight)
	def forward(self, x):
		x = self.embeddings(x)
		return x