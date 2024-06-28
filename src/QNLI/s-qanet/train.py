import torch
import torch.nn as nn
import torch.nn.functional as F
from data import SquadDataset
from model import * 
from w2v import Word2Vec
from transformers import BertTokenizerFast, BertModel
from datasets import load_dataset
import math
import torch.optim as optim
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("nyu-mll/glue", "qnli")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

random_train = dataset['train'].select(range(2269,12269))
random_val = dataset['validation'].select(range(2269,3269))
random_test = dataset['validation'].select(range(3269, 4269))

max_length = 128

train_data = SquadDataset(random_train, 16 , tokenizer, max_length)
validation_data = SquadDataset(random_val, 16, tokenizer, max_length)
test_data = SquadDataset(random_test, 16, tokenizer, max_length)

class NAQNLI(nn.Module):
	def __init__(self, config):
		super(NAQNLI, self).__init__()
		self.w2v = WordEmbedding(config['vocab_size'], config['embed_size'], config['c_len'], config['device'],  config['BERT'])
		self.enc = TransformerEnc(config['embed_size'], config['nhead'], config['num_layers'], config['c_len'], config['device'])
		self.dropout = nn.Dropout(0.1)
	def forward(self, c, q, q_mask, c_mask):
		q = self.w2v(q)
		c = self.w2v(c)
		return self.enc(c, q, q_mask, c_mask)

BERT = True

config = {
    'vocab_size': tokenizer.vocab_size,
    'embed_size': 768 if BERT else 256,
    'nhead': 12 if BERT else 4,
    'num_layers': 4,
    'c_len': 128,
    'device': device,
    'BERT': BERT
}

model = NAQNLI(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,betas=(0.8, 0.999), eps=1e-07, weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data)*20, eta_min=0.006*0.1, last_epoch=-1, verbose=False)
critereon = nn.CrossEntropyLoss().to(device)


def train(model, train_data, optimizer, critereon, epochs):
	t0 = time.time()
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		for q, c, labels in (train_data):
			model.zero_grad()
			q_i = q['input_ids'].to(device)
			c_i = c['input_ids'].to(device)
			q_mask = q['attention_mask'].to(device)
			c_mask = c['attention_mask'].to(device)
			# with torch.autocast(device_type=device, dtype=torch.float16):
			output = model(c_i, q_i, q_mask, c_mask)
			labels = labels.long().to(device)
			loss = critereon(output,labels)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			running_loss += loss.item()
		print(f"Epoch {epoch} Loss: {loss.item()/len(train_data)}")
	t1 = time.time()		
	print(f"Training time: {t1-t0}")
	
train(model, train_data, optimizer, critereon, 6)

def evaluation(model, validation_data, critereon):
	model.eval()
	running_loss = 0.0
	total = 0
	correct = 0
	# with torch.no_grad():
	for q, c, labels in (validation_data):
			q_i = q['input_ids'].to(device)
			c_i = c['input_ids'].to(device)
			q_mask = q['attention_mask'].to(device)
			c_mask = c['attention_mask'].to(device)
			with torch.no_grad():
			# with torch.autocast(device_type=device, dtype=torch.bfloat16):
				output = model(c_i, q_i, q_mask, c_mask)
				labels = labels.long().to(device)
				loss = critereon(output,labels
				running_loss += loss.item()
				_, predicted = torch.argmax(output, -1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
	print(f"Validation Loss: {running_loss/len(validation_data)}")
	print(f"Accuracy: {correct/total}")

evaluation(model, validation_data, critereon)