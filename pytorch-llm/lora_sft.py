#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:52:31 2026

@author: huzhen
"""

# -*- coding: utf-8 -*-
"""
"""


from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import sft_loss
from metrics import sft_accuracy
from train_utils import load_sft_data,SFTDataset,sft_collate_fn
from weight_utils import apply_train_weights,save_model_weights
from lora_utils import mark_only_lora_as_trainable,merge_lora_weights
from torch.utils.data import DataLoader
import torch

from callbacks import SFTEvaluate

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(42)
    tokenizer_tool = Tokenizer()
    eos_id = tokenizer_tool.special_ids["<eos>"]   
    pad_id = tokenizer_tool.special_ids["<pad>"]
    
    num_block = 4
    num_head = 2
    embedding_dim = 64
    vocab_size = len(tokenizer_tool.vocab)
    use_lora = True
    
    context_size = 200
    batch_size = 64
    configs = {
                "num_block" : num_block,
                "num_head" : num_head,
                "embedding_dim" : embedding_dim,
                "hidden_channels" : embedding_dim * 2,
                "use_lora" : use_lora,
                "vocab_size" : vocab_size,
                "pad_id" : pad_id
              }
    weight_map_path = r"models/0_k2v_weights.pt"
    model = create_pretrain_model(configs)
    model = model.to(device)
    apply_train_weights(model, weight_map_path)
    mark_only_lora_as_trainable(model)
    
    
    
    data_path = r"SFT_data/sft_data.jsonl"
    X_train,X_test = load_sft_data(data_path,tokenizer_tool, context_size,test_ratio=0.01)              
    
    print("训练样本数: ",len(X_train))
    # print("X: ",X_train[:3])
    
    
    train_dataset = SFTDataset(X_train, eos_id, pad_id)
    test_dataset = SFTDataset(X_test, eos_id, pad_id)
    
    train_dataloader = DataLoader(train_dataset,collate_fn=sft_collate_fn(pad_id),batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset,collate_fn=sft_collate_fn(pad_id),batch_size=batch_size)
    
    optimizer = torch.optim.Adam(
        filter(lambda p : p.requires_grad,model.parameters()),
        lr = 0.001
    )
         
    sft_evaluator = SFTEvaluate(tokenizer_tool)
    
    def train(epoch,dataloader,model,optimizer,loss_fn,accuracy_fn):
        model.train()
        total_loss = 0
        total_correct_tokens = 0
        total_valid_tokens = 0
        for X,Y in dataloader:
            X = X.to(device)
            Y = Y.to(device)
            output = model(X)
            loss = loss_fn(output, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            correct_tokens,valid_tokens = accuracy_fn(output, Y)
            total_loss += loss.item()*valid_tokens
            total_correct_tokens += correct_tokens
            total_valid_tokens += valid_tokens
            
        print("Epoch: {},loss={},accuracy:{}".format(epoch,total_loss/total_valid_tokens,total_correct_tokens/total_valid_tokens))
    
    def test(epoch,dataloader,model,loss_fn,accuracy_fn):
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct_tokens = 0
            total_valid_tokens = 0
            for X,Y in dataloader:
                X = X.to(device)
                Y = Y.to(device)
                output = model(X)
                loss = loss_fn(output, Y)
                correct_tokens,valid_tokens = accuracy_fn(output, Y)
                total_loss += loss.item()*valid_tokens
                total_correct_tokens += correct_tokens
                total_valid_tokens += valid_tokens
            
        print("Epoch: {},test_loss={},test_accuracy:{}".format(epoch,total_loss/total_valid_tokens,total_correct_tokens/total_valid_tokens))
        sft_evaluator.on_epoch_end(model, epoch, device)
    
    
    epochs = 1
    for epoch in range(epochs):
        train(epoch,train_dataloader,model,optimizer,sft_loss,sft_accuracy)
        test(epoch,test_dataloader,model,sft_loss,sft_accuracy)
   
    merge_lora_weights(model)
    save_model_weights(model,r"lora_sft_weights/{}_k2v_lora_merged_weights.pt".format(epoch))
    
    