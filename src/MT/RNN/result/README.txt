Data 4096 samples on training set, 512 samples validation set, 256 samples on test set
Model has been traind on 5 epochs


config = {
    'vocab_size': tokenizer.vocab_size,
    'input_size': 768 if BERT else 128 ,
    'hidden_size': 256,
	'BERT': BERT,
	'dropout': 0.1,
	'sos_token': tokenizer.convert_tokens_to_ids('[CLS]'),
	'max_length': 128-2,
	'device' : device,
    'generator': generator

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
critertion = nn.CrossEntropyLoss().to(device)

Training Time: 257.610435962677, Loss: 9.062032485380769
Evaluating Time: 2.878448963165283, Loss: 1.291874349117279, BLEU: 0.8123566028669725

( BLEU high may be because max_length is high but sentence_length is short)

Hello : '[CLS] Nó có một có, [PAD] và [SEP] [PAD], [PAD]., [SEP],, [PAD]. [PAD], [PAD] [PAD] [SEP] và [PAD]. [SEP], [SEP] [PAD] [SEP] [PAD] [PAD] [PAD],. [SEP],. [PAD] và [SEP] và [PAD] [PAD] [PAD] [PAD] [SEP].. và [SEP],. và,.. và và và, [SEP] [SEP] và [SEP] [PAD] [PAD]. [PAD]. [PAD] [SEP] và [PAD].. [SEP] [PAD]., [PAD] [SEP] [PAD] [SEP] và và [PAD], [PAD]. [SEP] [PAD] [PAD],. [SEP] [SEP], và [PAD] và [PAD], [PAD],. [PAD] [PAD]. [PAD] [SEP] [SEP] [SEP] [SEP] và và [PAD] [SEP] [PAD] [PAD]. [SEP],. và'

Even about seemingly personal and visceral things like who you &apos;re attracted to , you will start aping the beliefs of the people around you without even realizing that that &apos;s what you &apos;re doing . ( sentence in training set ) 
'[CLS] T ta có có.. và có. và có một [PAD] [PAD] và,.. và [PAD] và [PAD] [SEP], và [SEP]. [SEP] [PAD] [SEP],.. [PAD] [PAD] [SEP] [PAD].. [PAD] [SEP] và và [PAD] [SEP] [SEP]. [PAD],,, [SEP] [SEP]. [SEP]. [PAD] [PAD] [SEP], và [PAD].., [SEP],, [PAD]. [SEP]. [SEP],, và,, [PAD] và [PAD] [PAD]. và và. và [PAD] [SEP] [SEP] [PAD] [PAD] [SEP] [PAD] [SEP] [PAD] [SEP] và [PAD] và., [PAD] [SEP] [PAD] [PAD] và. [PAD], [PAD] và [SEP] [PAD] và [PAD]. [SEP] [SEP] [PAD] [SEP] [PAD], [SEP],.'

I did my best . ( sentence in training set ) '[CLS] Nó tôi là,,, [SEP] [PAD] và [PAD].,,, và và. và và. [PAD] [SEP] [PAD] [PAD], [SEP] [PAD] [PAD] [SEP],.... và.. [PAD] [SEP] và., và và,., và, [SEP],.. [PAD], và. và [PAD].. [SEP] [SEP] [PAD] [SEP], [SEP] [PAD] và [SEP] [PAD] [PAD] [SEP] và [SEP]. và. [PAD] [PAD] [SEP]. [SEP] [SEP] [PAD] [PAD] và [PAD] [SEP] và [PAD],., và và, [SEP] và, [PAD] [SEP] [SEP] [PAD]. và. [PAD] và..,. [PAD]. và [PAD] [SEP] [PAD]. [PAD] [PAD]. [SEP]. và'

They had 348 different kinds of jam . : '[CLS] Nó ta có một.. [SEP] [SEP], [SEP] [SEP] [PAD] [PAD]. [SEP] và [PAD] và và,,, và và [PAD] [PAD], [PAD] và [PAD] [PAD],, [PAD] và [SEP]. [SEP] [SEP], và và và,,,.,, và [SEP] [SEP],, [PAD], và [PAD] [SEP]. [SEP] [SEP]. và, [PAD], [PAD] [PAD] và [PAD] và [SEP] [SEP] [PAD]. và., [SEP],., [PAD]. và [PAD] và [SEP]. và [SEP] và [PAD] [SEP] [PAD] [SEP], và và và [PAD] [PAD] [PAD]. [SEP], [SEP] [SEP], [PAD], [SEP] [PAD]. [SEP], [SEP]., [PAD] và [SEP] [SEP] và [PAD]'

We can see that too much [SEP] and [PAD] token make the BLEU score too high, khong the chung minh duoc dich dung hhay sai


-------------------------
seq2seq2.pth
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
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
critertion = nn.CrossEntropyLoss().to(device)
Epoch: 0, step: 9, Loss: 0.9247623019748263
Epoch: 0, step: 19, Loss: 0.2918697156404194
Epoch: 0, step: 29, Loss: 0.14379162623964506
Epoch: 0, step: 39, Loss: 0.07777863893753444
Epoch: 0, step: 49, Loss: 0.07146233928446867
Epoch: 0, step: 59, Loss: 0.06011659412060754
Epoch: 0, step: 69, Loss: 0.05831265103989753
Epoch: 0, step: 79, Loss: 0.04273356365252145
Epoch: 0, step: 89, Loss: 0.040824587425489105
Epoch: 0, step: 99, Loss: 0.03443632703838927
Epoch: 0, step: 109, Loss: 0.03231034803827968
Epoch: 0, step: 119, Loss: 0.031293215871859
Epoch: 1, step: 9, Loss: 0.38594897588094074
Epoch: 1, step: 19, Loss: 0.14174536654823705
Epoch: 1, step: 29, Loss: 0.12514993240093364
Epoch: 1, step: 39, Loss: 0.08493842834081405
Epoch: 1, step: 49, Loss: 0.060806162503300884
Epoch: 1, step: 59, Loss: 0.0542695118209063
Epoch: 1, step: 69, Loss: 0.04294807669045268
Epoch: 1, step: 79, Loss: 0.04050225547597378
Epoch: 1, step: 89, Loss: 0.029054912288537186
Epoch: 1, step: 99, Loss: 0.02832073876352021
Epoch: 1, step: 109, Loss: 0.02113398499445084
Epoch: 1, step: 119, Loss: 0.025459736335177383
Epoch: 2, step: 9, Loss: 0.3213928805457221
Epoch: 2, step: 19, Loss: 0.148799645273309
Epoch: 2, step: 29, Loss: 0.08745155663325868
Epoch: 2, step: 39, Loss: 0.08462890600546813
Epoch: 2, step: 49, Loss: 0.06049914262732681
Epoch: 2, step: 59, Loss: 0.05427017858472921
Epoch: 2, step: 69, Loss: 0.046519559362660286
Epoch: 2, step: 79, Loss: 0.04100754291196413
Epoch: 2, step: 89, Loss: 0.030780098411474336
Epoch: 2, step: 99, Loss: 0.024366706308692393
Epoch: 2, step: 109, Loss: 0.025596793638456853
Epoch: 2, step: 119, Loss: 0.021376553703756893
Epoch: 3, step: 9, Loss: 0.27010295126173234
Epoch: 3, step: 19, Loss: 0.13374292223077072
Epoch: 3, step: 29, Loss: 0.09582854139393773
Epoch: 3, step: 39, Loss: 0.07186681796342899
Epoch: 3, step: 49, Loss: 0.0594097156913913
Epoch: 3, step: 59, Loss: 0.04156771756834903
Epoch: 3, step: 69, Loss: 0.03937635905500771
Epoch: 3, step: 79, Loss: 0.02844834327697754
Epoch: 3, step: 89, Loss: 0.02705452147494541
Epoch: 3, step: 99, Loss: 0.026351842013272373
Epoch: 3, step: 109, Loss: 0.018711623795535585
Epoch: 3, step: 119, Loss: 0.020828194978858242
Epoch: 4, step: 9, Loss: 0.29331013891432023
Epoch: 4, step: 19, Loss: 0.13330896277176707
Epoch: 4, step: 29, Loss: 0.0791166979691078
Epoch: 4, step: 39, Loss: 0.07515854101914626
Epoch: 4, step: 49, Loss: 0.052371589504942605
Epoch: 4, step: 59, Loss: 0.04563461723974196
Epoch: 4, step: 69, Loss: 0.03645164379175159
Epoch: 4, step: 79, Loss: 0.03681696517558038
Epoch: 4, step: 89, Loss: 0.03063488006591797
Epoch: 4, step: 99, Loss: 0.03065243393483788
Epoch: 4, step: 109, Loss: 0.026055904703402737
Epoch: 4, step: 119, Loss: 0.021116673445501246
Time: 169.92498755455017, Loss: 15.786254674196243

Time: 1.8682491779327393, Loss: 2.409613512456417, BLEU: 0.6283711422915408

Hello: Tôi chúng quo có những, bạn có thể chúng quo bạn đã là. một là. & là chúng tôi chúng có thể là một, bạn đã là thể chúng chúng tôi tôi, và chúng tôi chúng quo có thể, bạn là là một là những là, chúng ta đã có một có thể là'

Even about seemingly personal and visceral things like who you &apos;re attracted to , you will start aping the beliefs of the people around you without even realizing that that &apos;s what you &apos;re doing .

I love you : '[CLS] Và có chúng quo có một, bạn chúng chúng ta một. [SEP] [PAD] [PAD]h, chúng là chúng có một,t là một là thể là. & là có chúng tôi có chúng có chúng ta một có chúng quot có chúng tôi là một, bạn là chúng chúng có thể chúng ta đã,'

They had 348 different kinds of jam .: '[CLS] Đó là những một một một một là, chúng ta có [SEP].. & là chúng tôi là những., chúng tôi tôi, và là thể. & ta chúng tôi là một. & quot là một, tôi đã. & chúng chúng có thể.. [SEP]à [PAD], chúng quo chúng'

---------------------------------------------
Trained on 10 epochs
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
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
critertion = nn.CrossEntropyLoss().to(device)

Epoch: 0, step: 9, Loss: 0.8977607091267904
Epoch: 0, step: 19, Loss: 0.27186406286139236
Epoch: 0, step: 29, Loss: 0.12406638572955954
Epoch: 0, step: 39, Loss: 0.09667082321949494
Epoch: 0, step: 49, Loss: 0.06533774064511669
Epoch: 0, step: 59, Loss: 0.053951489723334876
Epoch: 1, step: 9, Loss: 0.3614708052741157
Epoch: 1, step: 19, Loss: 0.21360309500443309
Epoch: 1, step: 29, Loss: 0.11652185999113938
Epoch: 1, step: 39, Loss: 0.08890020541655712
Epoch: 1, step: 49, Loss: 0.0752775425813636
Epoch: 1, step: 59, Loss: 0.05640941555217161
Epoch: 2, step: 9, Loss: 0.39022061559889054
Epoch: 2, step: 19, Loss: 0.17647448338960348
Epoch: 2, step: 29, Loss: 0.11509426708879142
Epoch: 2, step: 39, Loss: 0.08288267331245618
Epoch: 2, step: 49, Loss: 0.057459072190888076
Epoch: 2, step: 59, Loss: 0.04983583143201925
Epoch: 3, step: 9, Loss: 0.33745402759975857
Epoch: 3, step: 19, Loss: 0.1468663215637207
Epoch: 3, step: 29, Loss: 0.11975921433547447
Epoch: 3, step: 39, Loss: 0.08549710420461801
Epoch: 3, step: 49, Loss: 0.059550222085446726
Epoch: 3, step: 59, Loss: 0.04934451943736965
Epoch: 4, step: 9, Loss: 0.3341940508948432
Epoch: 4, step: 19, Loss: 0.14595821029261538
Epoch: 4, step: 29, Loss: 0.09126678006402378
Epoch: 4, step: 39, Loss: 0.07186431762499687
Epoch: 4, step: 49, Loss: 0.05705941453271983
Epoch: 4, step: 59, Loss: 0.0530911138502218
Epoch: 5, step: 9, Loss: 0.323596715927124
Epoch: 5, step: 19, Loss: 0.1315770525681345
Epoch: 5, step: 29, Loss: 0.10019260439379461
Epoch: 5, step: 39, Loss: 0.0851445136926113
Epoch: 5, step: 49, Loss: 0.057099021210962414
Epoch: 5, step: 59, Loss: 0.04726321818464893
Epoch: 6, step: 9, Loss: 0.31837982601589626
Epoch: 6, step: 19, Loss: 0.13849200700458728
Epoch: 6, step: 29, Loss: 0.10365713875869224
Epoch: 6, step: 39, Loss: 0.06740347544352214
Epoch: 6, step: 49, Loss: 0.05485498175329091
Epoch: 6, step: 59, Loss: 0.044476864701610504
Epoch: 7, step: 9, Loss: 0.28494622972276473
Epoch: 7, step: 19, Loss: 0.13500202329535232
Epoch: 7, step: 29, Loss: 0.10304826703564875
Epoch: 7, step: 39, Loss: 0.0769333533751659
Epoch: 7, step: 49, Loss: 0.05212403803455586
Epoch: 7, step: 59, Loss: 0.04981873399120266
Epoch: 8, step: 9, Loss: 0.34661409589979386
Epoch: 8, step: 19, Loss: 0.1472435248525519
Epoch: 8, step: 29, Loss: 0.0866420104585845
Epoch: 8, step: 39, Loss: 0.0736997310931866
Epoch: 8, step: 49, Loss: 0.048414605004446845
Epoch: 8, step: 59, Loss: 0.0431378130185402
Epoch: 9, step: 9, Loss: 0.2685559060838487
Epoch: 9, step: 19, Loss: 0.14939635678341515
Epoch: 9, step: 29, Loss: 0.09595048016515272
Epoch: 9, step: 39, Loss: 0.06646360495151618
Epoch: 9, step: 49, Loss: 0.048349156671640824
Epoch: 9, step: 59, Loss: 0.045046531547934324
Time: 238.6090226173401, Loss: 31.583977710455656

Time: 1.6103034019470215, Loss: 2.4693203270435333, BLEU: 0.6203853913719672

Hello : '[CLS] Nhưng một là tôi, một một.n. [PAD] [PAD] [SEP]n [PAD] [PAD] [PAD]y, chúng tôi, tôi có một, ta là tôi là là có một có một chúng chúng tôi. [SEP]y chúng một chúng tôi.n và một là có tôi là có tôi là tôi là có một có thể'

"Even about seemingly personal and visceral things like who you &apos;re attracted to , you will start aping the beliefs of the people around you without even realizing that that &apos;s what you &apos;re doing ." : '[CLS] Nhưng chúng chúng là chúng một, một chúng là tôi có thể, và chúng là có một có thể, một một có một một, chúng tôi, và tôi là chúng là là là chúng là có ta một, ta có thể một có là một, ta một là là là chúng chúng một, chúng'

I love you" : '[CLS] Chúng tôi có thể là chúng một chúng là là chúng chúng là chúng là là một, và chúng chúng chúng ta là một có thể, chúng là một, chúng chúng tôi có một một một là là tôi có tôi một một. [PAD]y, tôi có thể là tôi một. [PAD]n của bạn,'

"They had 348 different kinds of jam ." : '[CLS] Nhưng một chúng là là là có một chúng tôi, tôi một.n. [PAD]y, một, ta có chúng chúng ta một một là có ta là là một, chúng ta có tôi một là chúng chúng là một chúng ta, và là tôi là chúng một chúng tôi là chúng tôi. ; [SEP]'
