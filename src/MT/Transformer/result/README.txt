args = {
	'embed_size': 256,
	'num_layers': 4,
	'max_len' : 64,
	'nhead': 4,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': False,
	'device': device
}
optimizer = optim.AdamW(model.parameters(), lr=0.001,  weight_decay=0.01)


Epoch: 0, step: 49, Loss: 0.0669476402049162
Epoch: 0, step: 99, Loss: 0.032881804186888415
Epoch: 1, step: 49, Loss: 0.050919493850396604
Epoch: 1, step: 99, Loss: 0.02756008957371567
Epoch: 2, step: 49, Loss: 0.05978209145215093
Epoch: 2, step: 99, Loss: 0.02512720859411991
Epoch: 3, step: 49, Loss: 0.04871994135331134
Epoch: 3, step: 99, Loss: 0.03136242278898605
Epoch: 4, step: 49, Loss: 0.04459445817129953
Epoch: 4, step: 99, Loss: 0.028734804403902303
Time taken: 90.48640942573547

Training time: 1.4937567710876465, Loss: 2.410447806119919, BLEU: 0.1356559744024576
Result:
Hello : '[CLS] [PAD] [SEP]?. [SEP] [SEP] [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

"That , I think , is movement .": '[CLS] [PAD] [SEP]?. [SEP] [SEP] [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

"They &apos;ve incurred violent infractions by becoming violent with guards and with other prisoners ."
'[CLS] [PAD] [SEP]?. [SEP] [SEP] [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'


"They &apos;re kept in bare cells like this for 23 hours a day ."
'[CLS] [PAD] [SEP]?. [SEP] [SEP] [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'


--------------------
args = {
	'embed_size': 256,
	'num_layers': 4,
	'max_len' : 64,
	'nhead': 4,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': False,
	'device': device
}
optimizer = optim.AdamW(model.parameters(), lr=0.001,  weight_decay=0.01)

Epoch: 0, step: 49, Loss: 0.0774586735939493
Epoch: 0, step: 99, Loss: 0.035208514242461235
Epoch: 1, step: 49, Loss: 0.07436549420259436
Epoch: 1, step: 99, Loss: 0.03567466832170583
Epoch: 2, step: 49, Loss: 0.0655407613637496
Epoch: 2, step: 99, Loss: 0.038935032757845794
Epoch: 3, step: 49, Loss: 0.07239999089922224
Epoch: 3, step: 99, Loss: 0.03017327038928716
Epoch: 4, step: 49, Loss: 0.0716978238553417
Epoch: 4, step: 99, Loss: 0.03222288507403749
Epoch: 5, step: 49, Loss: 0.05848778997148786
Epoch: 5, step: 99, Loss: 0.033696651458740234
Epoch: 6, step: 49, Loss: 0.051557394922996054
Epoch: 6, step: 99, Loss: 0.03216536358149365
Epoch: 7, step: 49, Loss: 0.05316403933933803
Epoch: 7, step: 99, Loss: 0.029480471755519058
Epoch: 8, step: 49, Loss: 0.04975828345941038
Epoch: 8, step: 99, Loss: 0.02676397863060537
Epoch: 9, step: 49, Loss: 0.053262569466415714
Epoch: 9, step: 99, Loss: 0.026119169562754004
Epoch: 10, step: 49, Loss: 0.05682959848520707
Epoch: 10, step: 99, Loss: 0.02510436135109025
Epoch: 11, step: 49, Loss: 0.05710385770213847
Epoch: 11, step: 99, Loss: 0.025486864224828855
Epoch: 12, step: 49, Loss: 0.057238846409077546
Epoch: 12, step: 99, Loss: 0.022985952069060973
Epoch: 13, step: 49, Loss: 0.04839413993212641
Epoch: 13, step: 99, Loss: 0.02661516690495038
Epoch: 14, step: 49, Loss: 0.04154877273403868
Epoch: 14, step: 99, Loss: 0.025892293814456825
Epoch: 15, step: 49, Loss: 0.04395723829464036
Epoch: 15, step: 99, Loss: 0.026930286426736852
Epoch: 16, step: 49, Loss: 0.04251196919655313
Epoch: 16, step: 99, Loss: 0.024487572486954507
Epoch: 17, step: 49, Loss: 0.04089074718708895
Epoch: 17, step: 99, Loss: 0.018204303702922784
Epoch: 18, step: 49, Loss: 0.03434857543633909
Epoch: 18, step: 99, Loss: 0.012615852885776095
Epoch: 19, step: 49, Loss: 0.02244236761209916
Epoch: 19, step: 99, Loss: 0.009256798209566059
Time taken: 356.52131485939026

Training time: 1.393742561340332, Loss: 0.7782391011714935, BLEU: 0.12381255311814449

Hello : '[CLS]ốiỗiắmự cao caoô caoânỡ caou khác khácô cao cao cao hơn khác caoô hơnỡ caoỡ khác khác caoộ hơnuuuô khácỡ hơnuân cao cao khác caoỡ khác khácu caoỡỡu & [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [PAD] [PAD] [PAD] [PAD]'
( display tokenizer to see is there any error with generate function : [  101, 22207, 38554, 57022, 42397, 15341, 15341, 16218, 15341, 15218,
        39415, 15341, 10138, 14393, 14393, 16218, 15341, 15341, 15341, 14789,
        14393, 15341, 16218, 14789, 39415, 15341, 39415, 14393, 14393, 15341,
        66066, 14789, 10138, 10138, 10138, 16218, 14393, 39415, 14789, 10138,
        15218, 15341, 15341, 14393, 15341, 39415, 14393, 14393, 10138, 15341,
        39415, 39415, 10138,   111,   102,   102,   102,   102,   102,   102,
          102,     0,     0,     0,     0] )
          
"That , I think , is movement ." : '[CLS]ốiỗiắmự cao caoô caoânỡ caou khác khácô cao cao cao hơn khác caoô hơnỡ caoỡ khác khác caoộ hơnuuuô khácỡ hơnuân cao cao khác caoỡ khác khácu caoỡỡu & [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [PAD] [PAD] [PAD] [PAD]'

"They &apos;ve incurred violent infractions by becoming violent with guards and with other prisoners ." : '[CLS]ốiỗiắmự cao caoô caoânỡ caou khác khácô cao cao cao hơn khác caoô hơnỡ caoỡ khác khác caoộ hơnuuuô khácỡ hơnuân cao cao khác caoỡ khác khácu caoỡỡu & [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [PAD] [PAD] [PAD] [PAD]'

"They &apos;re kept in bare cells like this for 23 hours a day ." : '[CLS]ốiỗiắmự cao caoô caoânỡ caou khác khácô cao cao cao hơn khác caoô hơnỡ caoỡ khác khác caoộ hơnuuuô khácỡ hơnuân cao cao khác caoỡ khác khácu caoỡỡu & [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [PAD] [PAD] [PAD] [PAD]'

"Thank you very much ." : '[CLS]ốiỗiắmự cao caoô caoânỡ caou khác khácô cao cao cao hơn khác caoô hơnỡ caoỡ khác khác caoộ hơnuuuô khácỡ hơnuân cao cao khác caoỡ khác khácu caoỡỡu & [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [PAD] [PAD] [PAD] [PAD]' 

"On guitar is my 15-year-old brother Tommy ." : '[CLS]ốiỗiắmự cao caoô caoânỡ caou khác khácô cao cao cao hơn khác caoô hơnỡ caoỡ khác khác caoộ hơnuuuô khácỡ hơnuân cao cao khác caoỡ khác khácu caoỡỡu & [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [SEP] [PAD] [PAD] [PAD] [PAD]'

( training too much epoch dan~ den' hien tuong mo hinh overfit vao cac tu gap qua nhieu, khien mo hinh khong the generate, no se generate ra cac ki tu thuong gap. )


----------------
args = {
	'embed_size': 512,
	'num_layers': 8,
	'max_len' : 64,
	'nhead': 8,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': False,
	'device': device
}
optimizer = optim.AdamW(model.parameters(), lr=0.001,  weight_decay=0.0001)
Epoch: 0, step: 49, Loss: 0.07260617431329221
Epoch: 0, step: 99, Loss: 0.03551175377585671
Epoch: 1, step: 49, Loss: 0.08008033402112065
Epoch: 1, step: 99, Loss: 0.04023514612756594
Epoch: 2, step: 49, Loss: 0.08574516919194436
Epoch: 2, step: 99, Loss: 0.03519833689988262
Epoch: 3, step: 49, Loss: 0.08061617247912349
Epoch: 3, step: 99, Loss: 0.03127609840547196
Epoch: 4, step: 49, Loss: 0.06796470466925174
Epoch: 4, step: 99, Loss: 0.040331267347239484
Time taken: 213.75026273727417

Training time: 2.40621018409729, Loss: 3.3165925592184067, BLEU: 0.13316172593986217

Hello : '[CLS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD], [PAD] là [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]. [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] một, [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [SEP]'

"That , I think , is movement ." : '[CLS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD], [PAD] là [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]. [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] một, [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [SEP]'

Thank you very much . : '[CLS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD], [PAD] là [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]. [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] một, [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [SEP]'

number of params : 181378811
( as scalling laws say, if you want to  


args = {
	'embed_size': 64,
	'num_layers': 2,
	'max_len' : 32,
	'nhead': 2,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': False,
	'device': device
}
optimizer = optim.AdamW(model.parameters(), lr=0.001,  weight_decay=0.0001)
Epoch: 0, step: 49, Loss: 0.15486921582903182
Epoch: 0, step: 99, Loss: 0.05126063028971354
Epoch: 0, step: 149, Loss: 0.03786797491496041
Epoch: 0, step: 199, Loss: 0.02293654302855832
Epoch: 0, step: 249, Loss: 0.018258301608533745
Epoch: 1, step: 49, Loss: 0.07809014223059829
Epoch: 1, step: 99, Loss: 0.049937055568502406
Epoch: 1, step: 149, Loss: 0.027962812641323012
Epoch: 1, step: 199, Loss: 0.016549081658598166
Epoch: 1, step: 249, Loss: 0.01746023898143845
Epoch: 2, step: 49, Loss: 0.06298052048196598
Epoch: 2, step: 99, Loss: 0.035159932242499456
Epoch: 2, step: 149, Loss: 0.02048677246042546
Epoch: 2, step: 199, Loss: 0.01237795939996614
Epoch: 2, step: 249, Loss: 0.00702524041555014
Epoch: 3, step: 49, Loss: 0.03662405208665497
Epoch: 3, step: 99, Loss: 0.012212060918711652
Epoch: 3, step: 149, Loss: 0.005285745099086889
Epoch: 3, step: 199, Loss: 0.0036588128487668446
Epoch: 3, step: 249, Loss: 0.0018417096760378305
Epoch: 4, step: 49, Loss: 0.007418162968693947
Epoch: 4, step: 99, Loss: 0.005205079160555444
Epoch: 4, step: 149, Loss: 0.0017878593214406262
Epoch: 4, step: 199, Loss: 0.001764200141082457
Epoch: 4, step: 249, Loss: 0.0013192957903007906
Time taken: 43.128506898880005

Training time: 1.0615606307983398, Loss: 0.28361038328148425, BLEU: 0.19407175193217518

Hello : '[CLS] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

"That , I think , is movement ." : '[CLS]. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

"Thank you very much ." : '[CLS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

"They &apos;re kept in bare cells like this for 23 hours a day ." : '[CLS]. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'


-----------------
batch_size 16
args = {
	'embed_size': 128,
	'num_layers': 2,
	'max_len' : 64,
	'nhead': 2,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': False,
	'device': device
}

optimizer = optim.AdamW(model.parameters(), lr=0.001,  weight_decay=0.0001)
Epoch: 0, step: 49, Loss: 0.12783858240867146
Epoch: 0, step: 99, Loss: 0.03523575657545918
Epoch: 0, step: 149, Loss: 0.01916026589054389
Epoch: 0, step: 199, Loss: 0.013545305884663184
Epoch: 0, step: 249, Loss: 0.010040931433558943
Epoch: 1, step: 49, Loss: 0.060112987245832174
Epoch: 1, step: 99, Loss: 0.03536841122791021
Epoch: 1, step: 149, Loss: 0.017087051532412537
Epoch: 1, step: 199, Loss: 0.013496310267616158
Epoch: 1, step: 249, Loss: 0.010905233252958122
Epoch: 2, step: 49, Loss: 0.055705279720072845
Epoch: 2, step: 99, Loss: 0.024897064825501105
Epoch: 2, step: 149, Loss: 0.017678395213696782
Epoch: 2, step: 199, Loss: 0.01429114269850841
Epoch: 2, step: 249, Loss: 0.01121058234249253
Epoch: 3, step: 49, Loss: 0.06192034118029536
Epoch: 3, step: 99, Loss: 0.026907675194017815
Epoch: 3, step: 149, Loss: 0.018453028378070602
Epoch: 3, step: 199, Loss: 0.011317509502621751
Epoch: 3, step: 249, Loss: 0.0103477292271503
Epoch: 4, step: 49, Loss: 0.04773651823705556
Epoch: 4, step: 99, Loss: 0.01998390814270636
Epoch: 4, step: 149, Loss: 0.015832705785764144
Epoch: 4, step: 199, Loss: 0.008198352914359702
Epoch: 4, step: 249, Loss: 0.005731041651652999
Epoch: 5, step: 49, Loss: 0.042120972458197146
Epoch: 5, step: 99, Loss: 0.013042195878847682
Epoch: 5, step: 149, Loss: 0.008600112575812629
Epoch: 5, step: 199, Loss: 0.005000546649472797
Epoch: 5, step: 249, Loss: 0.002544338205253264
Epoch: 6, step: 49, Loss: 0.011418175940610925
Epoch: 6, step: 99, Loss: 0.00794112260895546
Epoch: 6, step: 149, Loss: 0.002457619513441252
Epoch: 6, step: 199, Loss: 0.00185805589110408
Epoch: 6, step: 249, Loss: 0.0017222551456895698
Epoch: 7, step: 49, Loss: 0.009132455806342922
Epoch: 7, step: 99, Loss: 0.0028963332826440983
Epoch: 7, step: 149, Loss: 0.0021799278739314753
Epoch: 7, step: 199, Loss: 0.0008621874017332067
Epoch: 7, step: 249, Loss: 0.0007085545833809787
Epoch: 8, step: 49, Loss: 0.001982649218062965
Epoch: 8, step: 99, Loss: 0.0013452417621708879
Epoch: 8, step: 149, Loss: 0.0009185649804620934
Epoch: 8, step: 199, Loss: 0.00035917811357795296
Epoch: 8, step: 249, Loss: 0.00036657425055063394
Time taken: 117.7507905960083

Inference time on valid: 1.3572335243225098, Loss: 0.1317918719141744, BLEU: 0.15793766824615904

Inference time on test: 0.6043546199798584, Loss: 0.10800795746035874, BLEU: 0.15232728781527977

Hello: '[CLS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

Thank you very much . : '[CLS] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'


---------------
bs 16
args = {
	'embed_size': 128,
	'num_layers': 2,
	'max_len' : 64,
	'nhead': 2,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': False,
	'device': device
}
optimizer = optim.AdamW(model.parameters(), lr=0.001,  weight_decay=0.0001)
Epoch: 0, step: 49, Loss: 0.13103107530243543
Epoch: 0, step: 99, Loss: 0.03259995971063171
Epoch: 0, step: 149, Loss: 0.02152175231267942
Epoch: 0, step: 199, Loss: 0.015349129336563187
Epoch: 0, step: 249, Loss: 0.010730210078289231
Epoch: 1, step: 49, Loss: 0.04521455083574567
Epoch: 1, step: 99, Loss: 0.032999144660101995
Epoch: 1, step: 149, Loss: 0.018084604468121625
Epoch: 1, step: 199, Loss: 0.01573761144475122
Epoch: 1, step: 249, Loss: 0.006696526784015946
Epoch: 2, step: 49, Loss: 0.058852458486751635
Epoch: 2, step: 99, Loss: 0.026581171787146366
Epoch: 2, step: 149, Loss: 0.019556847194697233
Epoch: 2, step: 199, Loss: 0.01473610245402734
Epoch: 2, step: 249, Loss: 0.00693772643445486
Epoch: 3, step: 49, Loss: 0.054739504444355866
Epoch: 3, step: 99, Loss: 0.024425687211932556
Epoch: 3, step: 149, Loss: 0.017218077742813417
Epoch: 3, step: 199, Loss: 0.012411673464367737
Epoch: 3, step: 249, Loss: 0.007409626221561049
Epoch: 4, step: 49, Loss: 0.051185272177871395
Epoch: 4, step: 99, Loss: 0.017022274961375226
Epoch: 4, step: 149, Loss: 0.013001136331750242
Epoch: 4, step: 199, Loss: 0.009011093096517438
Epoch: 4, step: 249, Loss: 0.00350792508527457
Epoch: 5, step: 49, Loss: 0.017754023172417466
Epoch: 5, step: 99, Loss: 0.00824698233845258
Epoch: 5, step: 149, Loss: 0.007241888334287093
Epoch: 5, step: 199, Loss: 0.0023251455932406326
Epoch: 5, step: 249, Loss: 0.002233396811657641
Epoch: 6, step: 49, Loss: 0.0105024588351347
Epoch: 6, step: 99, Loss: 0.004595952202575375
Epoch: 6, step: 149, Loss: 0.0030589599737384974
Epoch: 6, step: 199, Loss: 0.001946655200354418
Epoch: 6, step: 249, Loss: 0.0013165424865891177
Epoch: 7, step: 49, Loss: 0.00335309091879397
Epoch: 7, step: 99, Loss: 0.0011093280833176893
Epoch: 7, step: 149, Loss: 0.0017699129229423983
Epoch: 7, step: 199, Loss: 0.0008511612912518295
Epoch: 7, step: 249, Loss: 0.0006372849026837023
Epoch: 8, step: 49, Loss: 0.003642633557319641
Epoch: 8, step: 99, Loss: 0.0007092686313571352
Epoch: 8, step: 149, Loss: 0.0008107222766684205
Epoch: 8, step: 199, Loss: 0.0007948867190423324
Epoch: 8, step: 249, Loss: 0.0004319765840668276
Epoch: 9, step: 49, Loss: 0.0018997913112445753
Epoch: 9, step: 99, Loss: 0.0017666910031829217
Epoch: 9, step: 149, Loss: 0.0009868574022446704
Epoch: 9, step: 199, Loss: 0.00035273429736420135
Epoch: 9, step: 249, Loss: 0.0004451531303455552
Epoch: 10, step: 49, Loss: 0.0012361348739692143
Epoch: 10, step: 99, Loss: 0.0010628328479901708
Epoch: 10, step: 149, Loss: 0.00047786753849695193
Epoch: 10, step: 199, Loss: 0.0003702926201436987
Epoch: 10, step: 249, Loss: 0.00015013228099508937
Epoch: 11, step: 49, Loss: 0.001767825715395869
Epoch: 11, step: 99, Loss: 0.0004200655778851172
Epoch: 11, step: 149, Loss: 0.00033566598224159857
Epoch: 11, step: 199, Loss: 0.00039113590016436934
Epoch: 11, step: 249, Loss: 0.0001420965904452236
Epoch: 12, step: 49, Loss: 0.0010162942415597488
Epoch: 12, step: 99, Loss: 0.0008750182360109656
Epoch: 12, step: 149, Loss: 0.0003096846826124511
Epoch: 12, step: 199, Loss: 0.00022349392424276726
Epoch: 12, step: 249, Loss: 7.821944912514055e-05
Epoch: 13, step: 49, Loss: 0.0003298185005479929
Epoch: 13, step: 99, Loss: 0.00032461867338479165
Epoch: 13, step: 149, Loss: 0.0002955039275572604
Epoch: 13, step: 199, Loss: 0.00020951262895186343
Epoch: 13, step: 249, Loss: 9.316137157769567e-05
Epoch: 14, step: 49, Loss: 0.0005241687580639003
Epoch: 14, step: 99, Loss: 0.0003738697956908833
Epoch: 14, step: 149, Loss: 0.0001497813219192044
Epoch: 14, step: 199, Loss: 0.00034024708684365354
Epoch: 14, step: 249, Loss: 0.00013332022541019334
Epoch: 15, step: 49, Loss: 0.0006208018760900108
Epoch: 15, step: 99, Loss: 0.00018379869259367085
Epoch: 15, step: 149, Loss: 0.0005283412717332776
Epoch: 15, step: 199, Loss: 0.00012687409305991838
Epoch: 15, step: 249, Loss: 1.2789023516467776e-05
Epoch: 16, step: 49, Loss: 0.0009165854782474285
Epoch: 16, step: 99, Loss: 4.2099580921307956e-05
Epoch: 16, step: 149, Loss: 0.00015353516444263843
Epoch: 16, step: 199, Loss: 3.97976693795554e-05
Epoch: 16, step: 249, Loss: 0.00013919629485731623
Epoch: 17, step: 49, Loss: 0.0006969065538474492
Epoch: 17, step: 99, Loss: 0.0003762216203742557
Epoch: 17, step: 149, Loss: 0.00047443527103270463
Epoch: 17, step: 199, Loss: 0.00011762788509903241
Epoch: 17, step: 249, Loss: 0.00010876173535025264
Epoch: 18, step: 49, Loss: 0.00034265383621867823
Epoch: 18, step: 99, Loss: 0.00011461975071767363
Epoch: 18, step: 149, Loss: 0.0001329192954221828
Epoch: 18, step: 199, Loss: 3.986542192685544e-05
Epoch: 18, step: 249, Loss: 2.1669576056750422e-05
Epoch: 19, step: 49, Loss: 0.0004524155614935622
Epoch: 19, step: 99, Loss: 1.3908548892071151e-05
Epoch: 19, step: 149, Loss: 2.4776863584282414e-05
Epoch: 19, step: 199, Loss: 1.687592073301574e-05
Epoch: 19, step: 249, Loss: 2.8384314573194128e-05
Epoch: 20, step: 49, Loss: 0.0001090173419488936
Epoch: 20, step: 99, Loss: 8.98497969363675e-05
Epoch: 20, step: 149, Loss: 1.2612486751937626e-05
Epoch: 20, step: 199, Loss: 8.293020950826869e-06
Epoch: 20, step: 249, Loss: 5.610142815783321e-05
Epoch: 21, step: 49, Loss: 4.260398789632077e-05
Epoch: 21, step: 99, Loss: 2.4604034901718902e-05
Epoch: 21, step: 149, Loss: 7.488499176782249e-06
Epoch: 21, step: 199, Loss: 1.0271650275692868e-05
Epoch: 21, step: 249, Loss: 5.1871589179259226e-05
Epoch: 22, step: 49, Loss: 0.00016232644568900671
Epoch: 22, step: 99, Loss: 1.3020392883606632e-05
Epoch: 22, step: 149, Loss: 7.563504396669017e-05
Epoch: 22, step: 199, Loss: 6.053931823926954e-05
Epoch: 22, step: 249, Loss: 5.5933356225251196e-05
Epoch: 23, step: 49, Loss: 0.0004884937055865113
Epoch: 23, step: 99, Loss: 0.0002633479960037
Epoch: 23, step: 149, Loss: 3.5347438873660645e-05
Epoch: 23, step: 199, Loss: 7.050887835984254e-05
Epoch: 23, step: 249, Loss: 4.7955209441692475e-05
Epoch: 24, step: 49, Loss: 3.91273987384475e-05
Epoch: 24, step: 99, Loss: 7.820847641789553e-05
Epoch: 24, step: 149, Loss: 2.5661132492175038e-05
Epoch: 24, step: 199, Loss: 6.3941753165206715e-06
Epoch: 24, step: 249, Loss: 2.882007395392321e-06
Epoch: 25, step: 49, Loss: 1.057721816991665e-05
Epoch: 25, step: 99, Loss: 1.2887365448128695e-05
Epoch: 25, step: 149, Loss: 9.164453527871394e-05
Epoch: 25, step: 199, Loss: 1.2221275805378678e-05
Epoch: 25, step: 249, Loss: 1.4300908833502766e-05
Epoch: 26, step: 49, Loss: 0.00038978058312620433
Epoch: 26, step: 99, Loss: 0.0001872054517570168
Epoch: 26, step: 149, Loss: 4.032986463616358e-05
Epoch: 26, step: 199, Loss: 4.0282930561046505e-05
Epoch: 26, step: 249, Loss: 6.59431224262499e-06
Epoch: 27, step: 49, Loss: 0.0001330763595748921
Epoch: 27, step: 99, Loss: 1.069970518312972e-05
Epoch: 27, step: 149, Loss: 1.3008900455530458e-05
Epoch: 27, step: 199, Loss: 1.67056548423204e-05
Epoch: 27, step: 249, Loss: 8.548655730473948e-06
Epoch: 28, step: 49, Loss: 2.0472876898640272e-05
Epoch: 28, step: 99, Loss: 1.770819800745959e-05
Epoch: 28, step: 149, Loss: 1.196900379512734e-05
Epoch: 28, step: 199, Loss: 3.3173499507221144e-05
Epoch: 28, step: 249, Loss: 5.00772372785821e-05
Epoch: 29, step: 49, Loss: 0.00020854648354710364
Epoch: 29, step: 99, Loss: 0.0002041015059056908
Epoch: 29, step: 149, Loss: 5.7508656442565404e-05
Epoch: 29, step: 199, Loss: 6.928483033599566e-06
Epoch: 29, step: 249, Loss: 2.0737898158261097e-05
Time taken: 389.85686802864075

Inference time validation: 1.2413976192474365, Loss: 0.12606948695702158, BLEU: 0.1595827545947266

Inference time test: 0.600623607635498, Loss: 0.0835537556631607, BLEU: 0.15775206537039846

Hello : '[CLS] Sai sợ Quan [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

"Thank you very much .", : '[CLS] Sai sợ Quan [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'


bs 16
args = {
	'embed_size': 768,
	'num_layers': 2,
	'max_len' : 64,
	'nhead': 12,
	'dropout': 0.1,
	'vocab_size': tokenizer.vocab_size,
	'BERT': True,
	'device': device
}
optimizer = optim.AdamW(model.parameters(), lr=0.001,  weight_decay=0.0001)
Epoch: 0, step: 49, Loss: 0.06980272701808385
Epoch: 0, step: 99, Loss: 0.03571588583666869
Epoch: 0, step: 149, Loss: 0.026079688296222047
Epoch: 0, step: 199, Loss: 0.021141773492247613
Epoch: 0, step: 249, Loss: 0.016991647850557505
Epoch: 1, step: 49, Loss: 0.08729800399468869
Epoch: 1, step: 99, Loss: 0.04512101953679865
Epoch: 1, step: 149, Loss: 0.02437730443557637
Epoch: 1, step: 199, Loss: 0.017157234738220523
Epoch: 1, step: 249, Loss: 0.013438879725444748
Epoch: 2, step: 49, Loss: 0.07687310783230529
Epoch: 2, step: 99, Loss: 0.04142610472862167
Epoch: 2, step: 149, Loss: 0.023669222057265724
Epoch: 2, step: 199, Loss: 0.018990521454930905
Epoch: 2, step: 249, Loss: 0.016994934005430902
Epoch: 3, step: 49, Loss: 0.06870078553958815
Epoch: 3, step: 99, Loss: 0.035164491094724096
Epoch: 3, step: 149, Loss: 0.022682244345645777
Epoch: 3, step: 199, Loss: 0.021057080982917517
Epoch: 3, step: 249, Loss: 0.016973236957228327
Epoch: 4, step: 49, Loss: 0.05678317011619101
Epoch: 4, step: 99, Loss: 0.03711948972759825
Epoch: 4, step: 149, Loss: 0.022489392517397067
Epoch: 4, step: 199, Loss: 0.020893444367988626
Epoch: 4, step: 249, Loss: 0.012639630750479948
Time taken: 203.4026391506195

Validating time: 2.3413822650909424, Loss: 3.3257248401641846, BLEU: 0.13050760855075352


Hello : '[CLS] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]. [PAD] và [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'

"Thank you very much ." '[CLS] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]. [PAD] và [PAD] [PAD] [PAD] [PAD] [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]'
