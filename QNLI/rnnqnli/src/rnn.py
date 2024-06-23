import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNe(nn.Module):
	def __init__(self, embed_size, hidden_size, num_layers, c_len):
		super(RNNe, self).__init__()
		self.encq = nn.RNN(embed_size, hidden_size, num_layers,batch_first=True, dropout=0.2, bidirectional=True)
		self.encc = nn.RNN(embed_size, hidden_size, num_layers,batch_first=True, dropout=0.2, bidirectional=True) 
		self.Wmodel = nn.Linear(hidden_size*2, 1)
		torch.nn.init.xavier_uniform_(self.Wmodel.weight)
		self.Wout = nn.Linear(c_len, 2)
		torch.nn.init.xavier_uniform_(self.Wout.weight)
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		# self.h0 = torch.zeros(2*num_layers, hidden_size*2)

	
	def forward(self, c, q):
		c_len = c.size(1)
		bs = c.size(0)
		h0 = torch.zeros(2*self.num_layers, bs, self.hidden_size, device='cuda')
		_ , hid_q = self.encq(q, h0) # hid = [2*num_layer, bs , hidden_size*2]

		encc, _ = self.encc(c, hid_q)  # [bs, c_len, hidden_size*2]

		out1 = self.Wmodel(encc) # [bs, c_len, 1] 
		out1 = out1.squeeze(-1) # [bs, c_len]
		out2 = self.Wout(F.relu(out1))
		return out2
