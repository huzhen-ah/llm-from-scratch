#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:21:43 2026

@author: huzhen
"""
import os
from bbpe_trainer import BBPETrainer
from tokenizer import Tokenizer

from models import create_pretrain_model
from losses import pretrain_loss,sft_loss,dpo_loss
from metrics import pretrain_accuracy,sft_accuracy
from train_utils import load_pretrain_data,PretrainDataset,load_sft_data,sft_collate_fn,SFTDataset,load_dpo_data,pre_infer_dpo_data,dpo_collate_fn,DPODataset
from callbacks import Evaluate,SFTEvaluate,DPOEvaluate
from weight_utils import apply_train_weights,save_model_weights
from lora_utils import mark_only_lora_as_trainable,merge_lora_weights
from interface import Interface
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    torch.manual_seed(42)
    """
    STEP_1: 
        先训练BBPE,得到vocab,merge_rules
    """
    print(".........BBPE阶段开始.........")
    data_path = r"data/*.txt"       
    vocab_path = r"tokenizer_config/vocab.json"
    merge_rules_path = r"tokenizer_config/merge_rules.json"
    if not os.path.isfile(vocab_path) or not os.path.isfile(merge_rules_path):
        trainer = BBPETrainer(data_path)
        trainer.build()
        trainer.train()
        trainer.save(vocab_path,merge_rules_path)
    print(".........BBPE阶段结束.........")
    
    """
    STEP_2:
        用语料做next_token_prediction训练
    """
    print(".........pretrain阶段开始.........")
    
    
    
    
        
    if not os.path.isfile(r"models/0_k2v_weights.pt"):
        
        
        tokenizer_tool = Tokenizer()
        eos_id = tokenizer_tool.special_ids["<eos>"]   
        pad_id = tokenizer_tool.special_ids["<pad>"]
        
        num_block = 4
        num_head = 2
        embedding_dim = 64
        vocab_size = len(tokenizer_tool.vocab)
        
        
        context_size = 200
        batch_size = 128
        
        configs = {
                    "num_block" : num_block,
                    "num_head" : num_head,
                    "embedding_dim" : embedding_dim,
                    "hidden_channels" : embedding_dim * 2,
                    "vocab_size" : vocab_size,
                    "pad_id" : pad_id
                  }
        
        model = create_pretrain_model(configs)
        
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        
        
        data_pattern = r"data/*.txt"
        X_train,X_test = load_pretrain_data(tokenizer_tool, data_pattern, context_size,test_ratio=0.01)              
        
        print("训练样本数: ",len(X_train))
        # print("X: ",X_train[:3])
        train_dataset = PretrainDataset(X_train, eos_id)
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        
        test_dataset = PretrainDataset(X_test, eos_id)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
        
        evaluator = Evaluate(tokenizer_tool)
        
        def train(epoch,dataloader,model,optimizer,pretrain_loss,pretrain_accuray):
            model.train()
            total_loss = 0
            total_correct_tokens = 0
            total_valid_tokens = 0
            for X,Y in dataloader:
                X = X.to(device)
                Y = Y.to(device=device,dtype=torch.long)
                output = model(X)
                loss = pretrain_loss(output, Y, pad_id)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                correct_tokens,valid_tokens = pretrain_accuray(output, Y, pad_id)
                total_loss += loss.item()*valid_tokens
                total_correct_tokens += correct_tokens
                total_valid_tokens += valid_tokens
                
            print("Epoch: {},loss={},accuracy:{}".format(epoch,total_loss/total_valid_tokens,total_correct_tokens/total_valid_tokens))
        def test(epoch,dataloader,model,pretrain_loss,pretrain_accuray):
            model.eval()
            total_loss = 0
            total_correct_tokens = 0
            total_valid_tokens = 0
            with torch.no_grad():
                for X,Y in dataloader:
                    X = X.to(device)
                    Y = Y.to(device,dtype=torch.long)
                    output = model(X)
                    loss = pretrain_loss(output,Y,pad_id)
                    correct_tokens,valid_tokens = pretrain_accuray(output, Y, pad_id)
                    total_loss += loss.item()*valid_tokens
                    total_correct_tokens += correct_tokens
                    total_valid_tokens += valid_tokens
            print("Epoch: {},test_loss={},test_accuracy:{}".format(epoch,total_loss/total_valid_tokens,total_correct_tokens/total_valid_tokens))
            evaluator.on_epoch_end(model, epoch,device)
            
        epochs = 1
        for epoch in range(epochs):
            train(epoch,train_dataloader,model,optimizer,pretrain_loss,pretrain_accuracy)
            test(epoch,test_dataloader,model,pretrain_loss,pretrain_accuracy)
            
    print(".........pretrain阶段结束.........")
    




    """
    STEP_3:
        用SFT_data微调pretrain的base模型，微调方式是LoRA
    """

    print(".........LoRA_SFT阶段开始.........")
    if not os.path.isfile(r"lora_sft_weights/0_k2v_lora_merged_weights.pt"):
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
    print(".........LoRA_SFT阶段结束.........")
    """
    STEP_4:
        用DPO_data做偏好优化，base模型用的是SFT阶段merged，微调方式是LoRA
    """
    print(".........LoRA_DPO阶段开始.........")
    if not os.path.isfile(r"lora_dpo_weights/0_k2v_lora_merged_weights.pt"):
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
        weight_map_path = r"lora_sft_weights/0_k2v_lora_merged_weights.pt"
        model = create_pretrain_model(configs)
        model = model.to(device)
        apply_train_weights(model, weight_map_path)
        mark_only_lora_as_trainable(model)
        
        
        
        data_path = r"DPO_data/dpo_data.jsonl"
        X_train,X_test = load_dpo_data(data_path,tokenizer_tool, context_size,test_ratio=0.01)              
        
        print("训练样本数: ",len(X_train))
        # print("X: ",X_train[:3])
        
        X_train = pre_infer_dpo_data(X_train, model, eos_id, pad_id)
        X_test = pre_infer_dpo_data(X_test, model, eos_id, pad_id)
        
        train_dataset = DPODataset(X_train, eos_id, pad_id)
        test_dataset = DPODataset(X_test, eos_id, pad_id)
        
        train_dataloader = DataLoader(train_dataset,collate_fn=dpo_collate_fn(eos_id,pad_id),batch_size=batch_size,shuffle=True)
        test_dataloader = DataLoader(test_dataset,collate_fn=dpo_collate_fn(eos_id,pad_id),batch_size=batch_size)
        
        optimizer = torch.optim.Adam(
            filter(lambda p : p.requires_grad,model.parameters()),
            lr = 0.001
        )
             
        dpo_evaluator = DPOEvaluate(tokenizer_tool)
        
        def train(epoch,dataloader,model,optimizer,loss_fn):
            model.train()
            total_loss = 0
            num_batch = 0
            for X,Y in dataloader:
                X = X.to(device)
                Y = Y.to(device)
                output = model(X)
                loss = loss_fn(output, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.cpu().item()
                num_batch += 1
                
            print("Epoch: {},loss={}".format(epoch,total_loss/num_batch))
        
        def test(epoch,dataloader,model,loss_fn):
            model.eval()
            with torch.no_grad():
                total_loss = 0
                num_batch = 0
                for X,Y in dataloader:
                    X = X.to(device)
                    Y = Y.to(device)
                    output = model(X)
                    loss = loss_fn(output, Y)
                    total_loss += loss.cpu().item()
                    num_batch += 1
                
            print("Epoch: {},test_loss={}".format(epoch,total_loss/num_batch))
            dpo_evaluator.on_epoch_end(model, epoch, device)
        
        
        epochs = 1
        for epoch in range(epochs):
            train(epoch,train_dataloader,model,optimizer,dpo_loss())
            test(epoch,test_dataloader,model,dpo_loss())
        merge_lora_weights(model)
        save_model_weights(model,r"lora_dpo_weights/{}_k2v_lora_merged_weights.pt".format(epoch))
       
    print(".........LoRA_DPO阶段结束.........")
    
    """
    STTEP_5:
        推理测试
    """
    print(".........inference阶段开始.........")
    configs = {
                    "num_block":4,
                    "num_head":2,
                    "embedding_dim":64,
                    "use_lora":False,
                    "weight_map_path":r"lora_dpo_weights/0_k2v_lora_merged_weights.pt",
                    "device":device

              }
    interface = Interface(Tokenizer(),configs)
    interface.init_prefill_model()

    interface.init_decode_model()

    text_1 = "华筝和其他人物有什么重要联系？"
    text_2 = "雁门关体现了《天龙八部》里的哪类矛盾？"
    text_3 = "从人物弧光看，复国执念重要在哪里？"
    text_4 = "王语嫣体现了《天龙八部》里的哪类矛盾？"

    prompts = ["      ",text_1,text_2,text_3,text_4]
    ret = interface.predict(prompts)
    for i in range(len(ret)):
        text = interface.tokenizer.decode(ret[i]["prompt"]+ret[i]["generated"])
        print("text: ",text)