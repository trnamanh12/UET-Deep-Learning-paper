
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from transformers import BertTokenizerFast

class SentenceDataset(Dataset):
    def __init__(self, src_sentence, tgt_sentence, tokenizer, max_length):
        self.src = src_sentence 
        self.tgt = tgt_sentence
        self.tokenizer = tokenizer
        self.max_length = max_length 

    def get_tokenized_sentences(self, sentence):
        tokenized_sentence = self.tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length)
        return tokenized_sentence['input_ids']

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        tokenized_src = self.get_tokenized_sentences(self.src[idx])
        tokenized_tgt = self.get_tokenized_sentences(self.tgt[idx])
        return {
            'src': tokenized_src.squeeze(0),
            'tgt': tokenized_tgt.squeeze(0),
        }
