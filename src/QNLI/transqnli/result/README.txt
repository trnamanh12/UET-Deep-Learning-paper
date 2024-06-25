DATA: 10.000 sample for training step, 1000 sample for validating step and 1000 sample for testing
The following model has been trained on P100

model1.pth: 5 epochs, training batch_size 100, validation batch_size 25, without Postional Encoding, embed_size 128, nhead 8, num_layer 3, seqlen 96, 
using BERT tokenizer, not use BERT word embedding

[ optimizer = optim.Adadelta(model.parameters(), lr=0.05, weight_decay=0.01)
critereon = nn.CrossEntropyLoss().to(device) ]
Training time: Forget to count while training, but it may be below 10 minutes

logs for training: 
Epoch 0 Loss: 0.0069176977872848515
Epoch 1 Loss: 0.006680745482444763
Epoch 2 Loss: 0.006938124895095825
Epoch 3 Loss: 0.006871083974838257
Epoch 4 Loss: 0.00694125235080719

logs for validating:
Validation Loss: 0.6930960342288017

model2.pth: 5 epochs, training batch_size 100, validation batch_size 25, without Postional Encoding, embed_size 256, nhead 8, num_layer 3, seqlen 128, 
using BERT tokenizer, not use BERT word embedding

[ optimizer = optim.Adadelta(model.parameters(), lr=0.5, weight_decay=0.01)
critereon = nn.CrossEntropyLoss().to(device) ]
Training time: Forget to count while training, but it may be below 10 minutes

Epoch 0 Loss: 0.006922681331634522
Epoch 1 Loss: 0.006690131425857544
Epoch 2 Loss: 0.006606031060218811
Epoch 3 Loss: 0.00651132345199585
Epoch 4 Loss: 0.006652445197105407

Validation Loss: 0.6731390699744224

model3.pth: 5 epochs, training batch_size 24, validation batch_size 12, with Postional Encoding, embed_size 768, nhead 8, num_layer 3, seqlen 128, 
using BERT tokenizer and BERT word embedding
[ optimizer = optim.Adadelta(model.parameters(), lr=0.5, weight_decay=0.01)
critereon = nn.CrossEntropyLoss().to(device) ]
Training time around 20-40 
Epoch 0 Loss : 0.0017..

[ Forget to save logs :((( ]

Validation Loss: 0.6925274382034937
Valid Accuracy: 0.514

ALL the result above are the raw implementation that not have been optimize at all 

config = {
    'vocab_size': tokenizer.vocab_size,
    'embed_size':  256,
    'nhead': 4,
    'num_layers': 2,
    'c_len': 128,
    'device': device,
    'BERT': False
}
epoch 20 
bs 32
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.8, 0.999), eps=1e-07, weight_decay=0.0001)
Epoch 0 Loss: 0.002591250803523932
Epoch 1 Loss: 0.0025595931199411995
Epoch 2 Loss: 0.0028694845236147556
Epoch 3 Loss: 0.002484481174724932
Epoch 4 Loss: 0.002464791837210853
Epoch 5 Loss: 0.0020421322542257584
Epoch 6 Loss: 0.0024675721177658716
Epoch 7 Loss: 0.0026182016244711585
Epoch 8 Loss: 0.0027055157640109807
Epoch 9 Loss: 0.0021544542556372693
Epoch 10 Loss: 0.002729025892556285
Epoch 11 Loss: 0.0021989202727905857
Epoch 12 Loss: 0.002315016600270622
Epoch 13 Loss: 0.002142131709443114
Epoch 14 Loss: 0.002234226979386692
Epoch 15 Loss: 0.0022141324064602107
Epoch 16 Loss: 0.002483701172728127
Epoch 17 Loss: 0.0023540456454974774
Epoch 18 Loss: 0.0022919282745629452
Epoch 19 Loss: 0.0024486728750478725
Training time: 1971.7607498168945
( Loss does not change, may be due to learning rate to low  )
Validation Loss: 0.6986420452594757
Valid Accuracy: 15.868

Train Accuracy: 16.0302

After training pretty much model, i decide to choose learning rate by watch first three epoch, if loss does not change much i with increse learning rate


config = {
    'vocab_size': tokenizer.vocab_size,
    'embed_size': 256,
    'nhead': 4,
    'num_layers': 2,
    'c_len': 128,
    'device': device,
    'BERT': False
}
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.8, 0.999), eps=1e-07, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data)*20, eta_min=0.006*0.1, last_epoch=-1, verbose=False)
critereon = nn.CrossEntropyLoss().to(device)

Epoch 0 Loss: 0.004904726061958094
Epoch 1 Loss: 0.005419526998989117
Epoch 2 Loss: 0.004595823181322969
Epoch 3 Loss: 0.004915748922207866
Epoch 4 Loss: 0.0036777292196743024
Epoch 5 Loss: 0.007270601229926649
Epoch 6 Loss: 0.0048618659424705625
Epoch 7 Loss: 0.005574841849720135
Epoch 8 Loss: 0.006646210393204856
Epoch 9 Loss: 0.002598590172898655
Epoch 10 Loss: 0.002742877593055701
Epoch 11 Loss: 0.005738032130768505
Epoch 12 Loss: 0.005710020613746521
Epoch 13 Loss: 0.005644522155054842
Epoch 14 Loss: 0.004313732869328021
Epoch 15 Loss: 0.005467481506518282
Epoch 16 Loss: 0.004958238845435194
Epoch 17 Loss: 0.0036984830618666384
Epoch 18 Loss: 0.0040560804616909816
Epoch 19 Loss: 0.004497449238079425
Training time: 1131.1202347278595

Training Accuracy:  15.98

Validation Loss: 1.1601464189589024
Accuracy: 16.084


optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.8, 0.999), eps=1e-06, weight_decay=0.0001)
NAQNLI(
  (w2v): WordEmbedding(
    (word2vec): Word2Vec(
      (embeddings): Embedding(30522, 768, padding_idx=0)
    )
    (positional_embedding): PositionalEmbedding()
  )
  (enc): TransformerEnc(
    (qencoder): ModuleList(
      (0-1): 2 x EncoderLayer(
        (selfattn): SelfAttention(
          (dropout): Dropout(p=0.1, inplace=False)
          (p_qkv): Linear(in_features=768, out_features=2304, bias=True)
          (p_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (ffn): FFN(
          (linear1): Linear(in_features=768, out_features=3072, bias=True)
          (linear2): Linear(in_features=3072, out_features=768, bias=True)
          (gelu): GELU(approximate='none')
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm): LayerNorm()
      )
    )
    (cencoder): ModuleList(
      (0-1): 2 x EncoderLayer(
        (selfattn): SelfAttention(
          (dropout): Dropout(p=0.1, inplace=False)
          (p_qkv): Linear(in_features=768, out_features=2304, bias=True)
          (p_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (ffn): FFN(
          (linear1): Linear(in_features=768, out_features=3072, bias=True)
          (linear2): Linear(in_features=3072, out_features=768, bias=True)
          (gelu): GELU(approximate='none')
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (norm): LayerNorm()
      )
    )
    (Wsim): Linear(in_features=2304, out_features=1, bias=True)
    (Wdistil): Linear(in_features=3072, out_features=768, bias=True)
    (Whead1): Linear(in_features=393216, out_features=768, bias=True)
    (Whead2): Linear(in_features=768, out_features=2, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (dropout): Dropout(p=0.1, inplace=False)
)
Epoch 0 Loss: 0.06527529907226562
Epoch 1 Loss: 0.052844342041015624
Epoch 2 Loss: 0.06037052612304687
Epoch 3 Loss: 0.037116226196289064
Epoch 4 Loss: 0.009616797637939453
Epoch 5 Loss: 0.05264581298828125
Epoch 6 Loss: 0.020165443420410156
Epoch 7 Loss: 0.04416451110839844
Epoch 8 Loss: 0.043794418334960936
Epoch 9 Loss: 0.028610772705078124
Training time: 3708.697199821472
Validating Accuracy: 0.521
Testing Accuracy: 0.523
