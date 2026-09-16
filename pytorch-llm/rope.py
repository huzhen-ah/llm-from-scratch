#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:07:43 2026

@author: huzhen
"""

import torch

def rope_exp(q,cur_valid_len=None,theta=10000):
    b,head,t,c = q.shape
    if cur_valid_len is not None:
        t = cur_valid_len - 1
        position = t.reshape(-1,1,1,1).to(dtype=torch.float32,device=q.device)
    else:
        position = torch.arange(t,dtype=torch.float32,device=q.device)[None,None,:,None]
    
    left,right = torch.split(q, q.shape[-1]//2,dim=-1)
    complex_q = torch.complex(left,right)
    
    index = torch.arange(c//2,dtype=torch.float32,device=q.device)
    m_theta = position*theta**(-2*index/c)
    rotate_q = complex_q * torch.exp(torch.complex(torch.zeros_like(m_theta),m_theta))
    real = torch.real(rotate_q)
    imag = torch.imag(rotate_q)
    _ = torch.cat([real,imag],dim=-1)
    return _
