#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:47:40 2026

@author: huzhen
"""
import regex as re
from tqdm import tqdm
from glob import glob
import numpy as np
import charset_normalizer
from torch.utils.data import Dataset
import json
import torch
# Set a fixed seed for reproducibility

def load_pretrain_data(tokenizer_tool,data_pattern,context_size,test_ratio=0.01):
    X_train,X_test = [],[]
    num_segment = 0
    for data_path in tqdm(glob(data_pattern),desc="读取文件中......"):
        enc = charset_normalizer.from_path(data_path).best().encoding
            
        with open(data_path,"r",encoding=enc) as f:
            text = f.read()
                
        text = re.sub(r"\s","",text)
        text_tokens = tokenizer_tool.encode_text(text)
        print("文本有{}个tokens".format(len(text_tokens)))
        size = len(text_tokens) // context_size
        for i in range(size):
            
            sample_tokens = text_tokens[i*context_size:(i+1)*context_size]
            # sample_tokens = sample_tokens + [tokenizer_tool.special_ids["<eos>"]]
                
            if np.random.random() < test_ratio:
                X_test.append(sample_tokens)
            else:
                X_train.append(sample_tokens)
            num_segment += 1 
            if num_segment % 1000 == 0:
                print("已编码: ",num_segment)
                # break
        # break
    return X_train,X_test

def add_eos(X,eos_id):
    X = np.array(X,dtype="int32")
    eos = np.ones(shape=(X.shape[0],1),dtype="int32") * eos_id
    X = np.concatenate([X,eos],axis=-1)
    return X

class PretrainDataset(Dataset):
    def __init__(self,X,eos_id):
        self.X = X
        self.eos_id = eos_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx] + [self.eos_id]
        x = np.array(x,dtype="int32")
        
        return x[:-1], x[1:]


def load_sft_data(path,tokenizer_tool,context_size,test_ratio=0.01):
    """
    {"messages": [{"role": "user", "content": "九阴真经让人物选择难在哪里？"}, {"role": "assistant", "content": "它引发的冲突说明武学本身不决定正邪，使用者的心性更关键。"}]}

    """
    X_train,X_test = [],[]
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            sub = {"instruction":None,"input":"","output":None}
            messages = data["messages"]
            
            for message in messages:
                content = message["content"].strip()
                if content == "":
                    break
                content_ids = tokenizer_tool.encode_text(content)
                if len(content_ids) == 0:
                    break
                if message["role"] == "user":
                    sub["instruction"] = content_ids
                elif message["role"] == "assistant":
                    sub["output"] = content_ids
                else:
                    break
            if sub["instruction"] is None or sub["output"] is None:
                continue
            if len(sub["instruction"]) + len(sub["output"]) > context_size:
                continue
            if np.random.random() > test_ratio:
                X_train.append(sub)
            else:
                X_test.append(sub)
            
    return X_train,X_test

def sft_collate_fn(pad_id):
    def padding_sft(batch_data):
        X = [_[0] for _ in batch_data]
        Mask = [_[1] for _ in batch_data]
        ml = max(len(x) for x in X)
        X = [x+[pad_id]*(ml-len(x)) for x in X]
        Mask = [mask + [0]*(ml-len(mask)) for mask in Mask]
        X = np.array(X,dtype="int32")
        Mask = np.array(Mask,dtype="int32")
        Y = X[:,1:]
        Y = Y[...,None]
        Mask = Mask[:,1:]
        Mask = Mask[...,None]
        Y = np.concatenate([Y,Mask],axis=-1)
        X = X[:,:-1]
        return torch.from_numpy(X).long(),torch.from_numpy(Y).long()
    return padding_sft


class SFTDataset(Dataset):
    def __init__(self,X,eos_id,pad_id):
        self.X = X
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        _ = self.X[idx]
        instruction = _["instruction"]
        output = _["output"]
        mask = [0] * len(instruction) + [1] * len(output) + [1]
        x = instruction + output + [self.eos_id]
        
        return x,mask



def load_dpo_data(path,tokenizer_tool,context_size,test_ratio=0.01):
    train_X,test_X = [],[]
    with open(path,"r",encoding="utf8") as f:
        for line in f:
            data = json.loads(line)
            prompt = data["prompt"].strip()
            if len(prompt) == 0:
                continue
            chosen = data["chosen"].strip()
            if len(chosen) == 0:
                continue
            rejected = data["rejected"].strip()
            if len(rejected) == 0:
                continue
            prompt_ids = tokenizer_tool.encode_text(prompt)
            if len(prompt_ids) == 0:
                continue
            chosen_ids = tokenizer_tool.encode_text(chosen)
            if len(chosen_ids) == 0:
                continue
            rejected_ids =  tokenizer_tool.encode_text(rejected)
            if len(rejected_ids) == 0:
                continue
            if len(prompt_ids) + len(chosen_ids) > context_size:
                continue
            if len(prompt_ids) + len(rejected_ids) > context_size:
                continue
            prompt_chosen_ids = prompt_ids + chosen_ids#输入
            prompt_chosen_mask = [0] * (len(prompt_ids) - 1) + [1] * len(chosen_ids) + [1]#输出mask
            
            prompt_rejected_ids = prompt_ids + rejected_ids
            prompt_rejected_mask = [0] * (len(prompt_ids) - 1) + [1] * len(rejected_ids) + [1]
            
            sub = {
                    "chosen_input_ids":prompt_chosen_ids,
                    "chosen_output_mask":prompt_chosen_mask,
                    "rejected_input_ids":prompt_rejected_ids,
                    "rejected_output_mask":prompt_rejected_mask
                   }
            if np.random.random() > test_ratio:
                train_X.append(sub)
            else:
                test_X.append(sub)
    return train_X,test_X


def pre_infer_dpo_data(X,model,eos_id,pad_id,batch_size=64):
    def padding(batch_chosen_input_ids,batch_rejected_input_ids):
        ml_chosen = max(len(x) for x in batch_chosen_input_ids)
        ml_rejected = max(len(x) for x in batch_rejected_input_ids)
        ml = max(ml_chosen, ml_rejected)
        batch_chosen_input_ids = [x + [eos_id] + (ml - len(x)) * [pad_id] for x in batch_chosen_input_ids]
        batch_rejected_input_ids = [x + [eos_id] + (ml - len(x)) * [pad_id] for x in batch_rejected_input_ids]
        
        batch_chosen_input_ids = np.array(batch_chosen_input_ids,dtype="int32")
        batch_rejected_input_ids = np.array(batch_rejected_input_ids,dtype="int32")
        batch_inputs = np.concatenate([batch_chosen_input_ids,batch_rejected_input_ids],axis=0)
        return batch_inputs[:,:-1],batch_inputs[:,1:]
    
    def log_softmax(X):
        X_max = np.max(X,axis=-1,keepdims=True)
        return X - X_max - np.log(np.sum(np.exp(X-X_max),axis=-1,keepdims=True))
    model.eval()
    device = next(model.parameters()).device
    batch_chosen_input_ids = []
    batch_rejected_input_ids = []
    batch_indexes = []
    batch_chosen_input_ids_lens = []
    batch_rejected_input_ids_lens = []
    total = len(X)
    progress = 0
    with torch.inference_mode():
        for i in range(len(X)):
            x = X[i]
            chosen_input_ids = x["chosen_input_ids"]
            rejected_input_ids = x["rejected_input_ids"]
            
            batch_chosen_input_ids.append(chosen_input_ids)
            batch_rejected_input_ids.append(rejected_input_ids)
            
            batch_indexes.append(i)
            batch_chosen_input_ids_lens.append(len(chosen_input_ids))
            batch_rejected_input_ids_lens.append(len(rejected_input_ids))
            
            if len(batch_chosen_input_ids) == batch_size or i == len(X) - 1:
                batch_inputs,batch_outputs = padding(batch_chosen_input_ids,batch_rejected_input_ids)
                batch_inputs = torch.from_numpy(batch_inputs).to(dtype=torch.long,device=device)
                batch_preds = model(batch_inputs).cpu().numpy()
                batch_logp = log_softmax(batch_preds)
                batch_logp = np.take_along_axis(batch_logp, batch_outputs[...,None],axis=-1)[:,:,0]
                batch_chosen_logp,batch_rejected_logp = np.split(batch_logp, 2, axis=0)
                for j in range(len(batch_chosen_logp)):
                    index = batch_indexes[j]
                    chosen_logp = [float(_) for _ in batch_chosen_logp[j][:batch_chosen_input_ids_lens[j]]]
                    rejected_logp = [float(_) for _ in batch_rejected_logp[j][:batch_rejected_input_ids_lens[j]]]
                    X[index]["chosen_logp"] = chosen_logp
                    X[index]["rejected_logp"] = rejected_logp
                progress += len(batch_chosen_input_ids)
                print("已完成:{}/{}".format(progress,total))
                batch_chosen_input_ids = []
                batch_rejected_input_ids = []
                batch_indexes = []
                batch_chosen_input_ids_lens = []
                batch_rejected_input_ids_lens = []
            
    return X

def dpo_collate_fn(eos_id,pad_id):
    def padding_dpo(batch_samples_list):
        batch_chosen_input_ids = [_["chosen_input_ids"] for _ in batch_samples_list]
        batch_rejected_input_ids = [_["rejected_input_ids"] for _ in batch_samples_list]
        batch_chosen_logp = [_["chosen_logp"] for _ in batch_samples_list]
        batch_rejected_logp = [_["rejected_logp"] for _ in batch_samples_list]
        batch_chosen_output_mask = [_["chosen_output_mask"] for _ in batch_samples_list]
        batch_rejected_output_mask = [_["rejected_output_mask"] for _ in batch_samples_list]
        ml_chosen = max(len(x) for x in batch_chosen_input_ids)
        ml_rejected = max(len(x) for x in batch_rejected_input_ids)
        ml = max(ml_chosen,ml_rejected)
        batch_chosen_input_ids = [x + [eos_id] + (ml - len(x)) * [pad_id] for x in batch_chosen_input_ids]
        batch_rejected_input_ids = [x + [eos_id] + (ml - len(x)) * [pad_id] for x in batch_rejected_input_ids]
        batch_chosen_logp = [x + (ml - len(x)) * [0] for x in batch_chosen_logp]
        batch_rejected_logp = [x + (ml - len(x)) * [0] for x in batch_rejected_logp]
        batch_chosen_output_mask = [x + (ml - len(x)) * [0] for x in batch_chosen_output_mask]
        batch_rejected_output_mask = [x + (ml - len(x)) * [0] for x in batch_rejected_output_mask]
        
        batch_chosen_input_ids = np.array(batch_chosen_input_ids,dtype="int32")
        batch_rejected_input_ids = np.array(batch_rejected_input_ids,dtype="int32")
        batch_chosen_output_mask = np.array(batch_chosen_output_mask,dtype="float32")
        batch_rejected_output_mask = np.array(batch_rejected_output_mask,dtype="float32")
        batch_chosen_logp = np.array(batch_chosen_logp,dtype="float32")
        batch_rejected_logp = np.array(batch_rejected_logp,dtype="float32")
        batch_chosen_input_ids,batch_chosen_output_ids = batch_chosen_input_ids[:,:-1],batch_chosen_input_ids[:,1:]
        batch_rejected_input_ids,batch_rejected_output_ids = batch_rejected_input_ids[:,:-1],batch_rejected_input_ids[:,1:]
        
        batch_chosen_out = np.concatenate([batch_chosen_output_ids[...,None],batch_chosen_logp[...,None],batch_chosen_output_mask[...,None]],axis=-1,dtype="float32")
        
        batch_rejected_out = np.concatenate([batch_rejected_output_ids[...,None],batch_rejected_logp[...,None],batch_rejected_output_mask[...,None]],axis=-1,dtype="float32")
        
        batch_output = np.concatenate([batch_chosen_out,batch_rejected_out],axis=0,dtype="float32")
        
        batch_input = np.concatenate([batch_chosen_input_ids,batch_rejected_input_ids],axis=0,dtype="int32")
        batch_input = torch.from_numpy(batch_input).long()
        batch_output = torch.from_numpy(batch_output).float()
        return batch_input,batch_output
    return padding_dpo
    

class DPODataset(Dataset):
    def __init__(self,X,eos_id,pad_id):
        self.X = X
        self.eos_id = eos_id
        self.pad_id = pad_id

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        x = self.X[idx]
        chosen_input_ids = x["chosen_input_ids"]
        rejected_input_ids = x["rejected_input_ids"]
        chosen_output_mask = x["chosen_output_mask"]
        rejected_output_mask = x["rejected_output_mask"]
        chosen_logp = x["chosen_logp"]
        rejected_logp = x["rejected_logp"]
        
        sample = { 
                    "chosen_input_ids":chosen_input_ids,
                    "rejected_input_ids":rejected_input_ids,
                    "chosen_logp":chosen_logp,
                    "rejected_logp":rejected_logp,
                    "chosen_output_mask":chosen_output_mask,
                    "rejected_output_mask":rejected_output_mask
                 }
        return sample
    
