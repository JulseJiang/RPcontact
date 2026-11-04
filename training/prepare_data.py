"""Prepare training inputs from interchangeable RNA/protein LM embeddings."""
import argparse,csv,json,pickle
from pathlib import Path
import numpy as np

FIELDS={'pairid','split','rna_onehot','protein_onehot','rna_embedding','protein_embedding','label'}

def prepare(manifest,output,label_format='contacts'):
    manifest=Path(manifest);output=Path(output)
    if output.exists():raise FileExistsError(output)
    if label_format not in {'contacts','distances'}:raise ValueError('Invalid label format')
    with manifest.open() as stream:
        reader=csv.DictReader(stream,delimiter='\t')
        if not FIELDS<=set(reader.fieldnames or []):raise ValueError('Manifest needs columns: '+', '.join(sorted(FIELDS)))
        rows=list(reader)
    if not rows:raise ValueError('Empty manifest')
    banks={};ids=set();cache={};dimensions=None
    def resolve(text):
        p=Path(text.strip());return p if p.is_absolute() else manifest.parent/p
    def features(row,side,width):
        ohpath=resolve(row[side+'_onehot'])
        name=row[side+'_embedding'].strip()
        emp=None if name in {'','-'} else resolve(name)
        key=(side,str(ohpath),str(emp))
        if key in cache:return cache[key]
        onehot=np.asarray(np.load(ohpath,allow_pickle=False),dtype=np.float32)
        if onehot.ndim!=2 or len(onehot)<1 or onehot.shape[1]!=width or not np.isfinite(onehot).all():
            raise ValueError(f'{side} one-hot must be finite [length, {width}]: {ohpath}')
        if not np.isin(onehot,[0,1]).all() or not np.isin(onehot.sum(axis=1),[0,1]).all():
            raise ValueError(f'Invalid one-hot rows: {ohpath}')
        parts=[onehot]
        if emp is not None:
            embedding=np.asarray(np.load(emp,allow_pickle=False),dtype=np.float32)
            if embedding.ndim!=2 or embedding.shape[0]!=len(onehot) or embedding.shape[1]<1 or not np.isfinite(embedding).all():
                raise ValueError(f'{side} embedding must be finite [sequence length, embedding channels]; remove special tokens and align residues first: {emp}')
            parts.append(embedding)
        cache[key]=np.concatenate(parts,axis=1).astype(np.float32)
        return cache[key]
    for row in rows:
        pair=row['pairid'].strip();split=row['split'].strip()
        if not pair or pair in ids:raise ValueError(f'Empty/duplicate pair ID across splits: {pair}')
        if split not in {'train','val','test','ts2'}:raise ValueError(f'Invalid split: {split}')
        ids.add(pair)
        r=features(row,'rna',4);p=features(row,'protein',20)
        y=np.asarray(np.load(resolve(row['label']),allow_pickle=False),dtype=np.float32)
        if y.ndim==3 and y.shape[-1]==1:y=y[:,:,0]
        if y.shape!=(len(r),len(p)) or not np.isfinite(y).all():raise ValueError(f'Label shape/value mismatch: {pair}')
        valid=y!=-999
        if not valid.any():raise ValueError(f'No labelled positions: {pair}')
        if label_format=='distances':
            if (y[valid]<0).any():raise ValueError('Distances must be nonnegative, with -999 for unavailable values')
            soft=np.full(y.shape,-999,dtype=np.float32)
            with np.errstate(over='ignore'):soft[valid]=1/(1+np.exp(-(5-y[valid])))
            y=soft
        elif ((y[valid]<0)|(y[valid]>1)).any():raise ValueError('Contact labels must be in [0,1], with -999 for missing values')
        dim=(r.shape[1],p.shape[1])
        if dimensions is not None and dim!=dimensions:raise ValueError('All splits in one experiment must use the same RNA/protein embedding dimensions')
        dimensions=dim
        banks.setdefault(split,{})[pair]={'_id':pair,'rna':{'_1d':r,'_2d':np.array([],dtype=np.float32)},
            'protein':{'_1d':p,'_2d':np.array([],dtype=np.float32)},'label':y[:,:,None]}
    if not {'train','val'}<=banks.keys():raise ValueError('Manifest must include train and val targets')
    output.mkdir(parents=True,exist_ok=False)
    for split,bank in banks.items():
        with (output/(split+'.pickle')).open('wb') as stream:pickle.dump((bank,{}),stream,protocol=4)
    metadata={'targets':{s:len(b) for s,b in banks.items()},'rna_input_channels':dimensions[0],
        'protein_input_channels':dimensions[1],'rna_lm_channels':dimensions[0]-4,'protein_lm_channels':dimensions[1]-20,
        'onehot_concatenated_once':True,'onehot_column_order':'preserved from supplied arrays',
        'label_input_format':label_format,'embedding_models':'supplied externally; channel sizes inferred'}
    (output/'data_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--label-format',choices=['contacts','distances'],default='contacts')
    args=parser.parse_args();print(json.dumps(prepare(args.manifest,args.output,args.label_format),indent=2))
