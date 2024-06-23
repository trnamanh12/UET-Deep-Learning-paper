When model is not fine_tuned ( model had been fine tuned on MTet )  

BLEU score on valid : 62.5832159193795

bs 8, optimizer = optim.AdamW(model.parameters(), lr=3e-4, eps=1e-6, betas=(0.9,0.98), weight_decay =0.00001)
scheduler = get_linear_schedule_with_warmup(optimizer, 
											num_warmup_steps=int(0.01*total_steps),
											num_training_steps=total_steps)
											
======== Epoch 1 / 1 ========
  Batch 200  of  512.    Elapsed: , Loss 0.5924925121353634
  Batch 400  of  512.    Elapsed: , Loss 0.4475402319223209
Average training loss: 0.41

Train BLEU score: 79.49787844610934

Valid BLEU score: 61.320531682080535
Test BLEU score: 59.91576993096669									
														
