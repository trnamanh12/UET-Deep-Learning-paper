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
	
	def forward(self, src, tgt):
		encoder_output, encoder_hidden = self.encoder(src)
		decoder_output = self.decoder(encoder_output, encoder_hidden, self.device, tgt)
		return decoder_output # [bs, seqlen, vocab_size]

generator = torch.Generator(device=device)
generator.manual_seed(42+222)

BERT = False

config = {
    'vocab_size': tokenizer.vocab_size,
    'input_size': 768 if BERT else 128 ,
    'hidden_size': 256,
	'BERT': BERT,
	'dropout': 0.1,
	'sos_token': tokenizer.convert_tokens_to_ids('[CLS]'),
	'max_length': 64-2,
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
			output, _ = model(src, tgt[:, 1:-1])
			output = output.reshape(-1, output.size(-1))
			loss = critertion(output, tgt[:, 1:].contiguous().view(-1))
			loss.backward()
			optimizer.step()
			torch.cuda.synchronize()
			running_loss += (loss.item())
			if (i+1) % 10 == 0:
				print(f'Epoch: {j}, step: {i}, Loss: {loss.item()/i}')
	end = time.time()
	print(f'Time: {end-start}, Loss: {running_loss/len(data)}')
# train(model, train_loader, optimizer, critertion, device, batch_size=32, num_epochs=10)

train(model, train_loader, optimizer, critertion, 'cuda', epochs=10)




import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluation(model, data, criterion, device):
	model.eval()
	start = time.time()
	bleu_score = 0
	running_loss = 0
	total_samples = 0  # Keep track of total samples for averaging BLEU

	for i, batch in enumerate(data):
		src = batch['src'].to(device)
		tgt = batch['tgt'].to(device)
		with torch.no_grad():
			with torch.cuda.amp.autocast():  # Assuming you're using CUDA
				output, _ = model(src, tgt[:, 1:-1])
				output = output.reshape(-1, output.size(-1))
				loss = criterion(output, tgt[:, 1:].contiguous().view(-1))
			output = output.argmax(dim=-1)
			output = output.view(src.size(0), -1)
			# Calculate BLEU for each sentence and accumulate
			for ref, pred in zip(tgt[:, 1:], output):
				bleu_score += sentence_bleu([ref.cpu().numpy().tolist()], pred.cpu().numpy().tolist(), smoothing_function=SmoothingFunction().method4)
			running_loss += loss.item()
			total_samples += src.size(0)

	end = time.time()
	avg_bleu_score = bleu_score / total_samples  # Average BLEU over all samples
	print(f'Time: {end - start}, Loss: {running_loss / len(data)}, BLEU: {avg_bleu_score}')


evaluation(model, valid_loader, critertion, device)
