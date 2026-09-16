#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:00:03 2026

@author: huzhen
"""

import torch
from torch import nn

class RMSNormalization(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.epsilon = 1e-6
        self.gamma = nn.Parameter(torch.ones(hidden_size))
 
    def forward(self,x):
        _ = torch.sqrt(torch.mean(torch.square(x),dim=-1,keepdim=True) + self.epsilon)
        _ = x / _ * self.gamma
        return _
    
class SwiGLU(nn.Module):
    def __init__(self,hidden_channel,input_channel,alpha=4,lora_rank=4,use_lora=False):
        super().__init__()
        self.hidden_channel = hidden_channel
        self.input_channel = input_channel
        self.output_channel = self.input_channel
        self.use_lora = use_lora
        self.alpha = alpha
        self.lora_rank = lora_rank
        self.scale = self.alpha / self.lora_rank
        self.v_dense = nn.Linear(self.input_channel,self.hidden_channel,bias=False)
        self.w_dense = nn.Linear(self.input_channel,self.hidden_channel,bias=False)
        self.out_dense = nn.Linear(self.hidden_channel,self.output_channel,bias=False)
        self.lora_gate_v_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_gate_v_B = nn.Linear(self.lora_rank,self.hidden_channel,bias=False)
        nn.init.zeros_(self.lora_gate_v_B.weight)

        self.lora_gate_w_A = nn.Linear(self.input_channel,self.lora_rank,bias=False)
        self.lora_gate_w_B = nn.Linear(self.lora_rank,self.hidden_channel,bias=False)
        nn.init.zeros_(self.lora_gate_w_B.weight)
        
        self.lora_out_A = nn.Linear(self.hidden_channel,self.lora_rank,bias=False)
        self.lora_out_B = nn.Linear(self.lora_rank,self.output_channel,bias=False)
        nn.init.zeros_(self.lora_out_B.weight)
    


    def merge_lora_weights_inplace(self):
        
        with torch.no_grad():
            lora_v_delta_weight = self.scale * torch.matmul(self.lora_gate_v_B.weight,self.lora_gate_v_A.weight)
            self.v_dense.weight.add_(lora_v_delta_weight)
        
            lora_w_delta_weight = self.scale * torch.matmul(self.lora_gate_w_B.weight,self.lora_gate_w_A.weight)
            self.w_dense.weight.add_(lora_w_delta_weight)

            lora_out_delta_weight = self.scale * torch.matmul(self.lora_out_B.weight,self.lora_out_A.weight)
            self.out_dense.weight.add_(lora_out_delta_weight)
        

    def forward(self,x):
        xv = self.v_dense(x)
        xw = self.w_dense(x)
        if self.use_lora:
            xv = xv + self.scale * self.lora_gate_v_B(self.lora_gate_v_A(x))
            xw = xw + self.scale * self.lora_gate_w_B(self.lora_gate_w_A(x))



        xw = torch.sigmoid(xw) * xw
     

        out = self.out_dense(xw * xv)

        if self.use_lora:
            out = out + self.scale * self.lora_out_B(self.lora_out_A(xw * xv))
        return out
