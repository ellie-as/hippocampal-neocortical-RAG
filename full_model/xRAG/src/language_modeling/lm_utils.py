import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

XRAG_TOKEN = "<xRAG>" 

def get_retrieval_embeds(model,input_ids,attention_mask=None):
    with torch.no_grad():
        embeds = model.get_doc_embedding(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    embeds = embeds.view(-1,embeds.shape[-1])
    return embeds 