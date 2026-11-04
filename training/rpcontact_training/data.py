"""Prepared per-complex arrays; explicit sequence masks and label masks."""
import csv
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class ContactDataset(Dataset):
    def __init__(self, path, ids_path=None):
        with Path(path).open('rb') as stream:
            payload=pickle.load(stream)
        self.bank=payload[0] if isinstance(payload,(tuple,list)) else payload
        if not isinstance(self.bank,dict) or not self.bank:
            raise ValueError('Expected a nonempty pair_id -> sample dictionary')
        if ids_path:
            with Path(ids_path).open() as stream:
                self.ids=[r['pairid'].strip() for r in csv.DictReader(stream,delimiter='\t')]
        else:self.ids=list(self.bank)
        if not self.ids or len(set(self.ids))!=len(self.ids):
            raise ValueError('Empty or duplicate pair IDs')
        missing=set(self.ids)-self.bank.keys()
        if missing:raise ValueError(f'Missing requested targets: {sorted(missing)[:5]}')
        self.dim=None
        for pair in self.ids:
            sample=self.bank[pair]
            r=np.asarray(sample['rna']['_1d']);p=np.asarray(sample['protein']['_1d'])
            y=np.asarray(sample['label'])
            if y.ndim==3 and y.shape[-1]==1:y=y[:,:,0]
            if r.ndim!=2 or p.ndim!=2 or min(len(r),len(p))<1 or y.shape!=(len(r),len(p)):
                raise ValueError(f'Invalid array shapes for {pair}')
            if not all(np.isfinite(a).all() for a in [r,p,y]):raise ValueError(f'Nonfinite array: {pair}')
            if np.any((y!=-999)&((y<0)|(y>1))):
                raise ValueError('Labels must be binary/soft contacts in [0,1], with -999 for missing labels; convert distances first')
            if not np.any(y!=-999):raise ValueError(f'No labelled positions: {pair}')
            for side in ['rna','protein']:
                if np.asarray(sample[side].get('_2d',[])).size:
                    raise ValueError('Supply final concatenated _1d inputs; this interface does not concatenate _2d features')
            dim=(r.shape[1],p.shape[1])
            if self.dim is not None and dim!=self.dim:raise ValueError('Embedding dimensions vary within a split')
            self.dim=dim
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        pair=self.ids[i];v=self.bank[pair];label=np.asarray(v['label'])
        return pair,np.asarray(v['rna']['_1d'],dtype=np.float32),np.asarray(v['protein']['_1d'],dtype=np.float32),label.reshape(label.shape[:2]).astype(np.float32)

def collate(samples):
    lr=max(len(s[1]) for s in samples);lp=max(len(s[2]) for s in samples)
    rnas=[];proteins=[];labels=[];rm=[];pm=[]
    for pair,r,p,y in samples:
        rnas.append(np.pad(r,((0,lr-len(r)),(0,0))).astype(np.float32))
        proteins.append(np.pad(p,((0,lp-len(p)),(0,0))).astype(np.float32))
        labels.append(np.pad(y,((0,lr-len(r)),(0,lp-len(p))),constant_values=-999))
        rm.append(np.arange(lr)<len(r));pm.append(np.arange(lp)<len(p))
    return {'ids':[s[0] for s in samples],
        'rna':torch.from_numpy(np.array(rnas).transpose(0,2,1)),
        'protein':torch.from_numpy(np.array(proteins).transpose(0,2,1)),
        'label':torch.from_numpy(np.array(labels)),
        'rna_valid':torch.from_numpy(np.array(rm)),'protein_valid':torch.from_numpy(np.array(pm))}
