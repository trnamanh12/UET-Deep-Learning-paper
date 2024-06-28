import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizerFast



class SentenceDataset(Dataset):
    def __init__(self, src_sentence, tgt_sentence, tokenizer, max_length):
        self.src = src_sentence 
        self.tgt = tgt_sentence
        self.tokenizer = tokenizer
        self.max_length = max_length 

    def get_tokenized_sentences(self, sentence):
        tokenized_sentence = self.tokenizer(sentence, padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length)
        return tokenized_sentence['input_ids'], tokenized_sentence['attention_mask']

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        tokenized_src, src_mask = self.get_tokenized_sentences(self.src[idx])
        tokenized_tgt, tgt_mask = self.get_tokenized_sentences(self.tgt[idx])
        return {
            'src': tokenized_src.squeeze(0),
            'src_mask': src_mask.squeeze(0),
            'tgt': tokenized_tgt.squeeze(0),
            'tgt_mask': tgt_mask.squeeze(0)
        }
# Example data
# source_sentences = ["Hello world", "How are you", "I am fine"]
# target_sentences = ["Xin chào", "Bạn khoẻ không", "Tôi khoẻ"]
# tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
# # Create dataset
# dataset = SentenceDataset(source_sentences, target_sentences, tokenizer, max_length=10)
# print(dataset[0]['src'].shape)
# # Create DataLoader
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# # Loop through the data
# for batch_idx, (data) in enumerate(dataloader):
#     print(f"Batch {batch_idx + 1}")
#     print("Source batch:", data['src'].shape, data['src_mask'].shape)
#     print("Target batch:", data['tgt'].shape, data['tgt_mask'].shape)
