import torch
import torch.nn as nn
import torch.nn.functional as F
from w2v import Word2Vec
# machine translation model with RNN
# class RNN(nn.Module):
#     def __init__(self, args):
#         super(RNN, self).__init__()
#         self.encoder = nn.RNN(args.input_size, args.hidden_size, args.num_layers, batch_first=True)
#         self.decoder = nn.RNN(args.input_size, args.hidden_size, args.num_layers, batch_first=True)
	
#     def forward(self, src, trg):
#         # src: [batch_size, src_len, input_size]
#         # trg: [batch_size, trg_len, input_size]
#         # hidden: [num_layers, batch_size, hidden_size]
#         hidden = self.init_hidden(src.size(0))
#         _, hidden = self.encoder(src, hidden)
#         output, _ = self.decoder(trg, hidden)
#         return output

#     def init_hidden(self, batch_size):
#         return torch.zeros(self.num_layers, batch_size, self.hidden_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
	def __init__(self,vocab_size, input_size, hidden_size, BERT, dropout=0.1):
		super(EncoderRNN, self).__init__()
		self.embedding = Word2Vec(vocab_size, input_size, BERT)
		self.hidden_size = hidden_size
		self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
		self.dropout = nn.Dropout(dropout)

	def forward(self, input):
		embedded = self.dropout(self.embedding(input))
		output, hidden = self.rnn(embedded)
		return output, hidden

# We should input both encoder and decoder is embedding vector, not input index

class DecoderRNN(nn.Module):
	def __init__(self, vocab_size, input_size, hidden_size, sos_token, max_length, BERT, generator):
		super(DecoderRNN, self).__init__()
		self.rnn= nn.RNN(input_size, hidden_size, batch_first=True)
		self.out = nn.Linear(hidden_size, vocab_size)
		self.sos_token = sos_token # Start of Sentence token
		self.max_length = max_length # Maximum length of the output sequence
		self.embedding = Word2Vec(vocab_size, input_size, BERT)
		self.generator = generator

	def forward(self, encoder_outputs, encoder_hidden, device, target_tensor=None):
		batch_size = encoder_outputs.size(0)
		# decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.sos_token)
		decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.sos_token)
		decoder_hidden = encoder_hidden
		decoder_outputs = []
		# target_tensor_size= target_tensor.size(1) # length of the target sequence
		generated_tokens = decoder_input

		for i in range(self.max_length): # input not include sos token, it range from 1 to max_length-1
			decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
			decoder_outputs.append(decoder_output)

			if target_tensor is not None:
				# Teacher forcing: Feed the target as the next input
				decoder_input = target_tensor[:, i].unsqueeze(-1)  # Teacher forcing
			else:
				# Without teacher forcing: use its own predictions as the next input
				# _, topi = decoder_output.topk(1) 
				decoder_output = decoder_output.squeeze(1) # decoder_output.view(-1, decoder_output.size(-1))  | // decoder_output: [bs, 1, vocab_size] -> [bs, vocab_size
				topk_pros, topk_ids  = decoder_output.topk(5, dim=-1) # topk_ids: [batch_size, 5]
				ix = torch.multinomial(topk_pros, num_samples=1, generator=self.generator) # sample from the topk_pros [batch_size, 1]
				xcol = torch.gather(topk_ids, -1, ix) # gather the topk_ids with the index ix

				decoder_input = xcol.detach()  # detach from history as input to the next time step
				generated_tokens = torch.cat((generated_tokens, decoder_input), dim=1)


		decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
		decoder_outputs.append(decoder_output)
		

		decoder_outputs = torch.cat(decoder_outputs, dim=1)
		return decoder_outputs, generated_tokens

	def forward_step(self, input, hidden):
		output = self.embedding(input)
		output = F.relu(output)
		output, hidden = self.rnn(output, hidden)
		output = self.out(output)
		return output, hidden