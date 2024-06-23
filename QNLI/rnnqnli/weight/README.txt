args = {
	'vocab_size': tokenizer.vocab_size,
	'embed_size': 128,
	'hidden_size': 256,
	'num_layers': 4,
	'c_len' : max_length,
	'BERT': False
}
bs 64, seqlen 128
optimizer = torch.optim.Adam(model.parameters(), lr=0.008,betas=(0.8, 0.999), eps=1e-07, weight_decay=0.0001)
Epoch 0, Loss: 108.83555656671524
Epoch 1, Loss: 54.41763052344322
Epoch 2, Loss: 36.27841607729594
Epoch 3, Loss: 27.208819955587387
Epoch 4, Loss: 21.76706349849701
Epoch 5, Loss: 18.13922554254532
Epoch 6, Loss: 15.547911473682948
Epoch 7, Loss: 13.604425713419914
Epoch 8, Loss: 12.092824969026777
Epoch 9, Loss: 10.883544021844864
Epoch 10, Loss: 9.894132234833457
Epoch 11, Loss: 9.069622327884039
Epoch 12, Loss: 8.371959617504707
Epoch 13, Loss: 7.773963021380561
Epoch 14, Loss: 7.255699360370636
Epoch 15, Loss: 6.802218407392502
Epoch 16, Loss: 6.402088231900159
Epoch 17, Loss: 6.046416799227397
Epoch 18, Loss: 5.72818462472213
Epoch 19, Loss: 5.441775533556938
Epoch 20, Loss: 5.182643402190435
Epoch 21, Loss: 4.947068829428066
Epoch 22, Loss: 4.731978963250699
Epoch 23, Loss: 4.534813284873962
Epoch 24, Loss: 4.353420860767365
Epoch 25, Loss: 4.185981606061642
Epoch 26, Loss: 4.030945265734637
Epoch 27, Loss: 3.886982894369534
Epoch 28, Loss: 3.7529490466775566
Epoch 29, Loss: 3.62785085439682
Epoch 30, Loss: 3.510823390176219
Epoch 31, Loss: 3.4011101350188255
Epoch 32, Loss: 3.298046215014024
Epoch 33, Loss: 3.2010449013289284
Epoch 34, Loss: 3.1095864534378053
Epoch 35, Loss: 3.023209050297737
Epoch 36, Loss: 2.9415006557026424
Epoch 37, Loss: 2.8640927860611365
Epoch 38, Loss: 2.790654501853845
Epoch 39, Loss: 2.7208881705999373
Epoch 40, Loss: 2.654525059025462
Epoch 41, Loss: 2.5913220459506627
Epoch 42, Loss: 2.5310587619626244
Epoch 43, Loss: 2.473534727638418
Epoch 44, Loss: 2.418567266729143
Epoch 45, Loss: 2.3659897265226943
Epoch 46, Loss: 2.315649520843587
Epoch 47, Loss: 2.2674068274597325
Epoch 48, Loss: 2.221133219952486
Epoch 49, Loss: 2.1767105412483216
Epoch 50, Loss: 2.1340299447377524
Epoch 51, Loss: 2.0929909210938673
Epoch 52, Loss: 2.053500534228559
Epoch 53, Loss: 2.0154727410387108
Epoch 54, Loss: 1.9788277745246887
Epoch 55, Loss: 1.9434915600078446
Epoch 56, Loss: 1.9093952168498123
Epoch 57, Loss: 1.8764746178840768
Epoch 58, Loss: 1.8446699562719313
Epoch 59, Loss: 1.8139254490534464
Epoch 60, Loss: 1.7841889672592037
Epoch 61, Loss: 1.7554117239290667
Epoch 62, Loss: 1.7275480476636735
Epoch 63, Loss: 1.7005551094189286
Training time: 905.7937517166138
Training Accuracy: 0.5013

Validation Loss: 0.6931901108473539
Accuracy: 0.492

3000:3064
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
[0,
 1,
 1,
 0,
 0,
 1,
 1,
 1,
 0,
 1,
 1,
 1,
 1,
 0,
 1,
 1,
 0,
 0,
 1,
 1,
 0,
 0,
 1,
 1,
 1,
 0,
 1,
 1,
 0,
 1,
 0,
 0,
 1,
 0,
 1,
 1,
 1,
 1,
 1,
 0,
 1,
 1,
 1,
 1,
 0,
 1,
 1,
 0,
 1,
 0,
 0,
 0,
 0,
 1,
 1,
 0,
 1,
 0,
 1,
 1,
 1,
 1,
 1,
 1]
