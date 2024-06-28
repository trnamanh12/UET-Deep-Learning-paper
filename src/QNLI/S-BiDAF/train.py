from datasets import load_dataset
from transformers import BertTokenizerFast, BertModel
from embed_layer import Word2Vec, ContextualEmbedding
from e2e import E2E 
from data import SquadDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("nyu-mll/glue", "qnli")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

random_train = dataset['train'].select(range(2269,12269))
random_val = dataset['validation'].select(range(2269,3269))
random_test = dataset['validation'].select(range(3269,4269))

train_data = SquadDataset(random_train, 32, tokenizer, 128)
validation_data = SquadDataset(random_val, 32, tokenizer, 128)
test_data = SquadDataset(random_test, 16, tokenizer, 128)

class BiDAF(nn.Module):
	def __init__(self, vocab_size, embed_size, hidden_size, c_len, BERT=False):
		super(BiDAF, self).__init__()
		self.w2v = Word2Vec(vocab_size, embed_size, BERT) # vocab_size, embed_size
		self.qcontext = ContextualEmbedding(embed_size, hidden_size) # embed_size, hidden_size
		self.ccontext = ContextualEmbedding(embed_size, hidden_size)
		self.e2e = E2E(hidden_size, c_len) # hidden_size, c_len
	
	def forward(self, q, c):
		q = self.w2v(q)
		c = self.w2v(c)
		q = self.qcontext(q)
		c = self.ccontext(c)
		return self.e2e(q, c)
BERT = True

if BERT:
	model = BiDAF(vocab_size=tokenizer.vocab_size, embed_size=100, hidden_size=100, c_len=64, BERT=True)
	# model.to(device)
# 	model = torch.compile(model)
else:
	model = BiDAF(vocab_size=tokenizer.vocab_size, embed_size=128, hidden_size=256, c_len=128)
	# model.to(device)
# 	model = torch.compile(model)

model.to(device)

optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.5, weight_decay=0.0001)
crietereon = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3274, eta_min=1e-6)

def train(model, train_data, optimizer, critereon, scheduler, epochs=1):
	start = time.time()
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		for questions, contexts, labels in train_data:
			optimizer.zero_grad(set_to_none= True)
			questions = questions.to(device)            
			contexts = contexts.to(device)
			labels = labels.long().to(device)
			# with torch.autocast(device_type=device, dtype=torch.bfloat16):
			output = model(questions, contexts)
			loss = critereon(output, labels)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()
			running_loss += loss.item()
		print(f"Epoch: {epoch}, Loss: {running_loss/len(train_data)}")
	end = time.time()
	print(f"Training time: {end-start}")

train(model, train_data, optimizer, crietereon , epochs=32)

def evaluation(model, val_data, critereon):
	model.eval()
	running_loss = 0.0
	total = 0
	correct = 0
	# with torch.no_grad():
	for questions, contexts, labels in val_data:
			questions = questions.to(device)
			contexts = contexts.to(device)
			labels = labels.long().to(device)
			with torch.no_grad():
				# with torch.autocast(device_type=device, dtype=torch.float16):
				output = model(questions, contexts)
				loss = critereon(output.view(-1, 2), labels.view(-1))
				running_loss += loss.item()
				_, predicted = torch.max(output, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
	print(f"Validation Loss: {running_loss/len(val_data)}")
	print(f"Accuracy: {100*correct/total}")

evaluation(model, validation_data, crietereon)