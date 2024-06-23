import torch
import torch.nn as nn
import torch.nn.functional as F

class E2E(nn.Module):
    def __init__(self, hidden_size, c_len):
        super(E2E, self).__init__()
        '''
        input: [bs, qlen, hidden_size*2], [bs, clen, hidden_size*2]
        1. We need to calculate the similarity matrix between the context and the query 
        2. We need to calculate the attention weights for the context and the query [bs, clen, qlen], and then caculate c2q by multiply [bs, clen, qlen] with [bs, qlen, hidden_size*2] is called c2q
        3. We need to calculate the attention weights for the query and the context [bs, qlen, clen], and then caculate q2c by multiply [bs, qlen, clen] with [bs, clen, hidden_size*2] is called q2c
        4. Then we concat [context, q2c, context*q2c, ] 

        '''
        self.Ws = nn.Linear(hidden_size*6, 1, bias=False)
        torch.nn.init.normal_(self.Ws.weight, mean=0, std=0.02)

        self.rnn = nn.RNN(input_size=hidden_size*8, hidden_size=hidden_size*4, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.1)

        self.last1 = nn.Linear(hidden_size*8, 1, bias=False)
        self.last2 = nn.Linear(c_len, 2, bias=False)
        
        torch.nn.init.normal_(self.last1.weight, mean=0, std=0.02)
        torch.nn.init.normal_(self.last2.weight, mean=0, std=0.02)
    def forward(self, c, q):
        '''
        c: [bs, clen, hidden_size*2]
        q: [bs, qlen, hidden_size*2]

        '''
        bs = c.size(0)
        c_len = c.size(1)
        q_len = q.size(1)
        hidden_size = c.size(2)

        _c = c.unsqueeze(2).expand(-1, -1, q_len, -1)
        _q = q.unsqueeze(1).expand(-1, c_len, -1, -1)
        cq = torch.mul(_c,_q)
        input_s = torch.cat([_c,_q,cq], dim=-1) # [bs, clen, qlen, hidden_size*6]

        s = self.Ws(input_s).squeeze(-1) #similarity matrix [bs, clen, qlen] 

        s1 = F.softmax(s, dim=-1)
        c2q = torch.bmm(s1, q) # [bs, clen, hidden_size*2], cco the hieu la ta bieu dien cac word trong context bang to hop attention_score*query
        c2q = self.dropout(c2q) 
        
        #q2c
        s2 = F.softmax(torch.max(s, dim=-1)[0], dim=-1) # [bs, clen]
        s2 = s2.unsqueeze(1).expand(-1, q_len, -1) # [bs, qlen, clen]
        q2c = torch.bmm(s2, c) # [bs, qlen, hidden_size*2]
        q2c = self.dropout(q2c) 

        #querry-aware representation 
        G = torch.cat([c, c2q, torch.mul(c, c2q), torch.mul(c, q2c)], dim=-1)    # [bs, clen, hidden_size*8]
        M, _ = self.rnn(G) # [bs, clen, hidden_size*8]
        M = self.dropout(M) 
        M = M + G # residual connection
        out1 = self.last1(M) # [bs, clen, 1]
        out1 = out1.squeeze(-1) # [bs, clen]
        out1 = self.dropout(out1)
        out2 = self.last2(out1)
        return out2

