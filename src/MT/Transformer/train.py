from data import SentenceDataset
from model import TransformerMT
from transformers import BertTokenizerFast
from w2v import WordEmbedding
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
en = []
with open('/home/trnmah/final_projectDL/src/MT/data/train-en-vi/train.en', 'r', encoding='utf-8') as file:
	for line in file:
		en.append(line.strip())  # strip() removes trailing newline characters

vi = []
with open('/home/trnmah/final_projectDL/src/MT/data/train-en-vi/train.vi', 'r', encoding='utf-8') as file:
	for line in file:
		vi.append(line.strip())  # strip() removes trailing newline characters
		
en_valid = []
with open('/home/trnmah/final_projectDL/src/MT/data/dev-2012-en-vi/tst2012.en', 'r', encoding='utf-8') as file:
	for line in file:
		en_valid.append(line.strip())  # strip() removes trailing newline characters

vi_valid = []
with open('/home/trnmah/final_projectDL/src/MT/data/dev-2012-en-vi/tst2012.vi', 'r', encoding='utf-8') as file:
	for line in file:
		vi_valid.append(line.strip())  # strip() removes trailing newline characters

train_data_src = en[2269:(2269+4096)]
train_data_trg= vi[2269:(2269+4096)]
valid_data_src = en_valid[269:(269+512)]
valid_data_trg= vi_valid[269:(269+512)]
test_data_src = en_valid[4:(4+256)]
test_data_trg= vi_valid[4:(4+256)]

train_data = SentenceDataset(train_data_src, train_data_trg, tokenizer, max_length=128)
valid_data = SentenceDataset(valid_data_src, valid_data_trg, tokenizer, max_length=128)
test_data = SentenceDataset(test_data_src, test_data_trg, tokenizer, max_length=128)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

args = {
	'embed_size': 768,
	'num_layers': 8,
	'max_len' : 128,
	'nhead': 12,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': True,
	'device': device
}
class MultiLayerTransformerMT(nn.Module):
	def __init__(self, args):
		super(MultiLayerTransformerMT, self).__init__()
		self.embeddings = WordEmbedding(args['vocab_size'], args['embed_size'], args['max_len'], args['device'], args['BERT'])
		self.transformer = nn.ModuleList([TransformerMT(args) for _ in range(args['num_layers'])])
		self.head = nn.Linear(args['embed_size'], args['vocab_size'])
		torch.nn.init.xavier_uniform_(self.head.weight)

	def forward(self, src, tgt, src_mask, tgt_mask):
		src = self.embeddings(src)
		tgt = self.embeddings(tgt)
		for layer in self.transformer:
			src, tgt, src_mask, tgt_mask = layer(src, tgt, src_mask, tgt_mask)
		tgt = self.head(tgt)
		return tgt.reshape(-1, tgt.size(-1))



model = MultiLayerTransformerMT(args).to(device)
model = torch.compile(model)
optim = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)

critertion = nn.CrossEntropyLoss().to(device)


def train (model, data, optimizer, critertion, device, epochs=1):
	model.train()
	start = time.time()
	running_loss = 0
	for j in range(epochs):
		for i, batch in enumerate(data):
			src = batch['src'].to(device)
			tgt = batch['tgt'].to(device)
			src_mask = batch['src_mask'].to(device)
			tgt_mask = batch['tgt_mask'].to(device)
			optimizer.zero_grad()
			with torch.autocast(device_type='cuda', dtype=torch.float16):
				output = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1])
				# output = output.view(-1, output.size(-1))
				loss = critertion(output, tgt[:, 1:].contiguous().view(-1))
			loss.backward()
			torch.nn.utils.clip_grad_norm_(v_1, max_norm=1.0, norm_type=2)
			optimizer.step()
			torch.cuda.synchronize()
			running_loss += (loss.item())
			if (i+1) % 50 == 0:
				print(f'Epoch: {j}, step: {i}, Loss: {loss.item()/i}')
	end = time.time()
	print(f'Time taken: {end-start}')


train(model, train_loader, optimizer, critertion, device, epochs=5)