import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import SentenceDataset
from w2v import Word2Vec
from model import EncoderRNN, DecoderRNN
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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

class Seq2Seq(nn.Module):
	def __init__(self, config):
		super(Seq2Seq, self).__init__()

		self.encoder = EncoderRNN(config['vocab_size'],config['input_size'], config['hidden_size'], \
							 config['BERT'], config['dropout'])
		self.decoder = DecoderRNN(config['vocab_size'], config['input_size'], config['hidden_size'], \
							config['sos_token'], config['max_length'] ,config['BERT'], config['generator'] )
		self.device = config['device']
	
	def forward(self, src, tgt=None):
		encoder_output, encoder_hidden = self.encoder(src)
		decoder_output = self.decoder(encoder_output, encoder_hidden, self.device, tgt)
		return decoder_output # [bs, seqlen, vocab_size]

generator = torch.Generator(device=device)
generator.manual_seed(42+222)

config = {
    'vocab_size': tokenizer.vocab_size,
    'input_size': 128,
    'hidden_size': 256,
	'BERT': False,
	'dropout': 0.1,
	'sos_token': tokenizer.convert_tokens_to_ids('[CLS]'),
	'max_length': 128-2,
	'device' : device,
	'generator': generator
}

model = Seq2Seq(config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
critertion = nn.CrossEntropyLoss()

def train (model, data, optimizer, critertion, device, epochs=1):
	model.train()
	start = time.time()
	running_loss = 0
	for j in range(epochs):
		for i, batch in enumerate(data):
			src = batch['src'].to(device)
			tgt = batch['tgt'].to(device)
			optimizer.zero_grad()
			with torch.autocast(device_type=device, dtype=torch.bfloat16):
				output = model(src, tgt[:, 1:-1])
				loss = critertion(output.view(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
			loss.backward()
			optimizer.step()
			torch.cuda.synchronize()
			running_loss += (loss.item())
			if (i+1) % 1000 == 0:
				print(f'Epoch: {j}, step: {i}, Loss: {loss.item()/i}')
	end = time.time()
	print(f'Time: {end-start}, Loss: {running_loss/len(data)}')


# train(model, train_loader, optimizer, critertion, device, batch_size=32, num_epochs=10)

def evaluation(model, data, optimizer, critertion, device):
	model.eval()
	start = time.time()
	running_loss = 0
	for i, batch in enumerate(data):
		src = batch['src'].to(device)
		tgt = batch['tgt'].to(device)
		with torch.no_grad():
			with torch.autocast(device_type=device, dtype=torch.bfloat16):
				output = model(src, tgt[:, 1:-1])
				output = output.view(-1, output.size(-1))
				loss = critertion(output, tgt[:, 1:].contiguous().view(-1))
		running_loss += (loss.item())
	end = time.time()
	print(f'Time: {end-start}, Loss: {running_loss/len(data)}')


sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42 + 1203)
def generate(model, sentence, tokenizer, device, generator):
	model.eval()
	sentence = tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
	with torch.no_grad():
		_, generated_token = model(sentence['input_ids'].to(device))
		# topk_pros, topk_ids  = predict.topk(5, dim=-1)
		# ix = torch.multinomial(topk_pros, num_samples=1, generator=generator) # sample from the topk_pros
		# xcol = torch.gather(topk_ids, -1, ix) # gather the topk_ids with the index ix
	return generated_token

# print(tokenizer.batch_decode(generate(model, "Hello", tokenizer, device, sample_rng), "Sure"))