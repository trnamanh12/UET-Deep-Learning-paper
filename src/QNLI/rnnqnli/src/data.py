import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

class SquadDataset(torch.utils.data.Dataset):
	'''
	- Creates batches dynamically by padding to the length of largest example
	  in a given batch.
	- Calulates character vectors for contexts and question.
	- Returns tensors for training.
	'''
	
	def __init__(self, data, batch_size, tokenizer, max_length ):
		
		self.batch_size = batch_size
		data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
		self.data = data
		# self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.tokenizer = tokenizer
		self.max_length = max_length
		
	def __len__(self):
		return len(self.data)
	
	def __iter__(self):
		'''
		Creates batches of data and yields them.
		
		Each yield comprises of:
		:padded_context: padded tensor of contexts for each batch 
		:padded_question: padded tensor of questions for each batch 
		:label: 
		
		'''
		
		for batch in self.data:
			questions = self.tokenizer(batch['question'], max_length = self.max_length, padding='max_length', truncation=True, return_tensors='pt')
			contexts = self.tokenizer(batch['sentence'], max_length = self.max_length, padding='max_length', truncation=True, return_tensors='pt')
			labels = torch.IntTensor(batch['label']).to(torch.int8)
			# question, context include input_ids, attention_mask, token_type_ids
			yield questions['input_ids'], contexts['input_ids'], labels
			
		