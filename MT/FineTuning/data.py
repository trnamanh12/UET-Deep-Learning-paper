import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
# from transformers import BertTokenizerFast



class SentenceDataset(Dataset):
    def __init__(self, src_sentence, tgt_sentence, tokenizer, max_length):
        self.src = src_sentence 
        self.tgt = tgt_sentence
        self.tokenizer = tokenizer
        self.max_length = max_length 

    # def get_tokenized_sentences(self, source_sentence, target_sentence):
    #     tokenized_sentence = self.tokenizer(source_sentence, text_target =target_sentence, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length)
    #     return tokenized_sentence

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        inputs = self.tokenizer(self.src[idx], text_target = self.tgt[idx], padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length)

        return inputs.squeeze(0)