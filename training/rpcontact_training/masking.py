"""Mask padding keys in attention and missing/padded labels in BCE."""
import types
import torch
import torch.nn.functional as F
from .model import ESM2StyleAttention,apply_rotary_pos_emb

def masked_attention(self,query,key,value):
    mask=self._key_valid
    if mask is None:return self._unmasked_forward(query,key,value)
    batch,qlen,_=query.shape;klen=key.shape[1]
    q=self.q_proj(query).view(batch,qlen,self.n_head,self.d_head).transpose(1,2)
    k=self.k_proj(key).view(batch,klen,self.n_head,self.d_head).transpose(1,2)
    v=self.v_proj(value).view(batch,klen,self.n_head,self.d_head).transpose(1,2)
    cq,sq=self._get_rotary_emb(qlen,query.device);ck,sk=self._get_rotary_emb(klen,key.device)
    q=apply_rotary_pos_emb(q,cq,sq);k=apply_rotary_pos_emb(k,ck,sk)
    scores=torch.matmul(q,k.transpose(-2,-1))/self.scale
    scores=scores.masked_fill(~mask[:,None,None,:],float('-inf'))
    context=torch.matmul(self.dropout(F.softmax(scores,dim=-1)),v)
    return self.out_proj(context.transpose(1,2).contiguous().view(batch,qlen,-1))

def install_masks(model):
    for m in model.modules():
        if isinstance(m,ESM2StyleAttention) and not hasattr(m,'_unmasked_forward'):
            m._unmasked_forward=m.forward;m._key_valid=None
            m.forward=types.MethodType(masked_attention,m)
    return model

def forward(model,batch,device):
    r=batch['rna'].to(device);p=batch['protein'].to(device);y=batch['label'].to(device)
    rm=batch['rna_valid'].to(device);pm=batch['protein_valid'].to(device)
    rm=None if bool(rm.all()) else rm;pm=None if bool(pm.all()) else pm
    for layer in model.layers:
        layer.tta1.self_attn._key_valid=pm;layer.tta1.cross_attn._key_valid=rm
        layer.tta2.self_attn._key_valid=rm;layer.tta2.cross_attn._key_valid=pm
    pred=model(p,r).squeeze(-1).permute(0,2,1)
    valid=y!=-999
    counts=valid.sum(dim=(1,2))
    if bool((counts==0).any()):raise ValueError('A complex has no valid labels')
    losses=F.binary_cross_entropy(pred,(y>.5).float(),reduction='none')
    return pred,y,(losses*valid).sum(dim=(1,2))/counts
