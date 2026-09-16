#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:30:55 2026

@author: huzhen
"""

import torch

def save_model_weights(model,weight_map_path):
    base_state_dict = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if "lora_" not in name
    }

    torch.save(base_state_dict, weight_map_path)
        
def apply_train_weights(model,weight_map_path,device="cpu"):
    state_dict = torch.load(weight_map_path, map_location=device)
    state_dict = {
        
                    name:tensor
                    for name,tensor in state_dict.items()
                    if "lora_" not in name
                 }
    
    model.load_state_dict(state_dict,strict=False)
    
    
    