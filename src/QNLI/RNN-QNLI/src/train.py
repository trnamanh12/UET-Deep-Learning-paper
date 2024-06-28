from w2v import Word2Vec
from data import SquadDataset
from rnn import RNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from transformers import BertTokenizerFast, BertModel
import time

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("nyu-mll/glue", "qnli")

random_train = dataset['train'].select(range(2269,12269))
random_val = dataset['validation'].select(range(2269,3269))
random_test = dataset['validation'].select(range(3269,4269))

max_length = 128

train_data = SquadDataset(random_train, 16 , tokenizer, max_length)
validation_data = SquadDataset(random_val, 16, tokenizer, max_length)
test_data = SquadDataset(random_test, 16, tokenizer, max_length)

class RNNqnli(nn.Module):
	def __init__(self, config):
		super(RNNqnli, self).__init__()
		self.w2v = Word2Vec(config['vocab_size'], config['embed_size'],  config['BERT'])
		self.rnn = RNNe(config['embed_size'], config['hidden_size'], config['num_layers'], config['c_len'])

	
	def forward(self, c, q):
		c = self.w2v(c)
		q = self.w2v(q)
		out = self.rnn(c, q)
		return out

args = {
	'vocab_size': tokenizer.vocab_size,
	'embed_size': 768,
	'hidden_size': 512,
	'num_layers': 4,
	'c_len' : max_length,
	'BERT': True
}

model = RNNqnli(args).to(device)
epochs = 12

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,betas=(0.8, 0.999), eps=1e-07, weight_decay=0.0001)
critereon = nn.CrossEntropyLoss().to(device)

def train(model, optimizer, train_data,  critereon, epochs=10):
	t0 = time.time()
	for i in range(epochs):
		model.train()
		running_loss = 0
		for s, q, l in train_data:
			optimizer.zero_grad()
			s = s.to(device)
			q = q.to(device)
			l = l.long().to(device)
# 			with torch.autocast(device_type=device, dtype=torch.bfloat16):
			output = model(s, q)
			loss = critereon(output, l)
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)            
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print(f'Epoch {i}, Loss: {running_loss/(i+1)}')
	t1 = time.time()
	print(f'Training time: {t1-t0}')

train(model, optimizer, train_data, critereon, epochs)

def evaluation(model, validation_data, critereon):
	model.eval()
	running_loss = 0
	samples = 0
	acc = 0
	for s, q, l in validation_data:
		s = s.to(device)
		q = q.to(device)
		l = l.long().to(device)
		with torch.no_grad():
			output = model(s, q)
			loss = critereon(output, l)
			running_loss += loss.item()
			_, pred = torch.max(output, 1)
			samples += len(l)
			acc += torch.sum(pred == l).item()
	print(f'Validation Loss: {running_loss/len(validation_data)}')
	print(f'Accuracy: {acc/samples}')

evaluation(model, validation_data, critereon)

evaluation(model, test_data, critereon)