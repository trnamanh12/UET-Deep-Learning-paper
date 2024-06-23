 embed_size=100, hidden_size=256, c_len=128, 16 epoch, bs 64	
 optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.5, weight_decay=0.001)
 Epoch: 0, Loss: 0.6943459787945838
Epoch: 1, Loss: 0.6934064979765825
Epoch: 2, Loss: 0.6935610402921203
Epoch: 3, Loss: 0.6933162341451948
Epoch: 4, Loss: 0.6932671666145325
Epoch: 5, Loss: 0.6932250109447795
Epoch: 6, Loss: 0.6932728776506557
Epoch: 7, Loss: 0.6932085031157087
Epoch: 8, Loss: 0.693195933748962
Epoch: 9, Loss: 0.6932484704977387
Epoch: 10, Loss: 0.6931621667685782
Epoch: 11, Loss: 0.6931525901624351
Epoch: 12, Loss: 0.6931529409566503
Epoch: 13, Loss: 0.6931595236632475
Epoch: 14, Loss: 0.6931525499198088
Epoch: 15, Loss: 0.6931514891849202
Training time: 898.6868634223938

Validation Loss: 0.693147225305438
Accuracy Train data: 50.13
Accuracy Validation data: 49.2

16epoch bs 60 

embed_size=768, hidden_size=256, c_len=128, BERT=True
optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.5, weight_decay=0.001)
Epoch: 0, Loss: 0.6942756168856592
Epoch: 1, Loss: 0.6936058969554787
Epoch: 2, Loss: 0.6933012836707566
Epoch: 3, Loss: 0.6931946788005486
Epoch: 4, Loss: 0.693268121359591
Epoch: 5, Loss: 0.6932538762064038
Epoch: 6, Loss: 0.6932312103802573
Epoch: 7, Loss: 0.6932307192665375
Epoch: 8, Loss: 0.693206883476166
Epoch: 9, Loss: 0.6931447629443186
Epoch: 10, Loss: 0.6931386072478608
Epoch: 11, Loss: 0.6931430179915742
Epoch: 12, Loss: 0.6931539499117229
Epoch: 13, Loss: 0.6931471935289348
Epoch: 14, Loss: 0.6931401466181178
Epoch: 15, Loss: 0.6931415525025236
Training time: 912.781661272049

Validation Loss: 0.6931636203080416
Accuracy: 51.65
Accuracy: 47.9

embed_size=128, hidden_size=256, c_len=128
optimizer = torch.optim.Adadelta(model.parameters(), lr = 0.5, weight_decay=0.0001)
Epoch: 0, Loss: 0.6946736006691052
Epoch: 1, Loss: 0.6942744904432815
Epoch: 2, Loss: 0.6941428245446933
Epoch: 3, Loss: 0.6936850903894954
Epoch: 4, Loss: 0.6936270804070055
Epoch: 5, Loss: 0.6939113915157014
Epoch: 6, Loss: 0.6936165619962893
Epoch: 7, Loss: 0.6929599953154786
Epoch: 8, Loss: 0.684400350902789
Epoch: 9, Loss: 0.6681230144378857
Epoch: 10, Loss: 0.6299331245330957
Epoch: 11, Loss: 0.5830079354702855
Epoch: 12, Loss: 0.4847485973193242
Epoch: 13, Loss: 0.40153783971604445
Epoch: 14, Loss: 0.3130649346858263
Epoch: 15, Loss: 0.2463168220478482
Epoch: 16, Loss: 0.2161488656621105
Epoch: 17, Loss: 0.2234264256435628
Epoch: 18, Loss: 0.2960725605678254
Epoch: 19, Loss: 0.35526410814005727
Epoch: 20, Loss: 0.3124506064041997
Epoch: 21, Loss: 0.2103419107163307
Epoch: 22, Loss: 0.19358109257901057
Epoch: 23, Loss: 0.13845096823176184
Epoch: 24, Loss: 0.15889592708115235
Epoch: 25, Loss: 0.1566587250832968
Epoch: 26, Loss: 0.10692773797190763
Epoch: 27, Loss: 0.10748318614610754
Epoch: 28, Loss: 0.09015156829673142
Epoch: 29, Loss: 0.15029281660480745
Epoch: 30, Loss: 0.08943705704749767
Epoch: 31, Loss: 0.23788471085658391
Training time: 1990.5919511318207
Accuracy training: 66.71

Loss on validating test: 2.807593956589699
Accuracy: 50.5

Loss on test set: 2.7632576674222946
Accuracy on test: 49.1

( here we choose the best model ( the model with the highest training accuracy because the accuracy between models is mostly the same, then we choose the model 3th with 66.71 accuracy on training set, and we believe that if model was trained with more data and with more time it will be better ) 

3000:3064
predicted label : tensor([1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
        1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0,
        1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], device='cuda:0')
actual label : [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
