# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from utils.network_utils import mask_augment


class MPNet(nn.Module):
    def __init__(self, input_dim=768, output_dim=128,
        model_str='sentence-transformers/all-mpnet-base-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)
        self.model = AutoModel.from_pretrained(model_str)
        self.match = nn.Linear(input_dim, output_dim)
        for p in self.model.parameters():
            p.requires_grad = False
        self.mask_token = self.tokenizer.mask_token

    def forward(self, input_dict, apply_augmentation=False):
        referrals = input_dict['description']
        if apply_augmentation:
            referrals = mask_augment(referrals, input_dict['object_name'], self.mask_token)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        encoded_input = self.tokenizer(referrals, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        token_embeddings = self.match(model_output[0])
        return {'f_word': token_embeddings, 'padding_mask': encoded_input['attention_mask']}
