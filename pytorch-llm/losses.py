#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:22:12 2026

@author: huzhen
"""

from torch.nn import functional as F
import torch

def pretrain_loss(preds,targets,pad_id):
    preds = preds.transpose(1,2)
    loss = F.cross_entropy(preds, targets,ignore_index=pad_id)
    return loss

def sft_loss(preds,targets):
    targets,mask = targets[...,0].long(),targets[...,1]
    mask = mask.to(dtype=preds.dtype)
    ce_loss = F.cross_entropy(preds.transpose(1,2), targets,reduction="none")
    masked_loss = ce_loss * mask
    loss = torch.sum(masked_loss) / (torch.sum(mask) + 1e-7)
    return loss

def dpo_loss(beta=0.1):
    def _dpo_loss(preds,targets):
        """
        SUMMARY.
    
        targets: .shape=(batch*2,m,3)
                batch*2: 之所以是2个batch,是因为一条数据既有chosen，又有rejected,
                         前batch个是chosen,后batch个是rejected
                m      : m就是paddign之后的序列长度
                3      : chosen_or_rejected_id,logP,mask(有效为1，无效为0)
        preds: .shape=(batch*2,m,vocab_size),前batch个是chosen,后batch个是rejected
                m同上
                vocab_size,logits的纬度
        """
        batch_2,m = targets.shape[:2]
        batch = batch_2 // 2
        
        ref_chosen_ids = targets[:batch,:,0].long()
        ref_chosen_logp = targets[:batch,:,1].to(dtype=preds.dtype)
        ref_chosen_mask = targets[:batch,:,2].to(dtype=preds.dtype)
        ref_chosen_logp = ref_chosen_logp * ref_chosen_mask
        
        ref_rejected_ids = targets[batch:,:,0].long()
        ref_rejected_logp = targets[batch:,:,1].to(dtype=preds.dtype)
        ref_rejected_mask = targets[batch:,:,2].to(dtype=preds.dtype)
        ref_rejected_logp = ref_rejected_logp * ref_rejected_mask
        
        
        chosen_pred = preds[:batch]
        rejected_pred = preds[batch:]
        
    
        chosen_logp = torch.take_along_dim(torch.log_softmax(chosen_pred,dim=-1), ref_chosen_ids[:,:,None],dim=-1)[:,:,0]
        chosen_logp = chosen_logp * ref_chosen_mask
        
        rejected_logp = torch.take_along_dim(torch.log_softmax(rejected_pred,dim=-1), ref_rejected_ids[:,:,None],dim=-1)[:,:,0]
        rejected_logp = rejected_logp * ref_rejected_mask
        

        tmp = torch.sum(chosen_logp,dim=-1) - torch.sum(rejected_logp,dim=-1) \
            - torch.sum(ref_chosen_logp,dim=-1) + torch.sum(ref_rejected_logp,dim=-1)
        tmp = beta * tmp
        
        return -torch.mean(F.logsigmoid(tmp))
    
        
    return _dpo_loss