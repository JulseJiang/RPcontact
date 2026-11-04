"""Train with BCE early stopping; defer checkpoint AUROC selection until training ends."""
import argparse,csv,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,auc
from .data import ContactDataset,collate
from .model import ProteinRNAInteraction_rela
from .masking import install_masks,forward
from .resource_profile import ResourceProfiler

def write(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def initialize_weights(module):
    # Preserve the historical initializer, including its use through model.apply().
    for m in module.modules():
        if isinstance(m,nn.Conv1d):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.BatchNorm1d):
            nn.init.constant_(m.weight,1);nn.init.constant_(m.bias,0)
        elif isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,.001);nn.init.constant_(m.bias,0)

def learning_rate(epoch,base):
    # Historical values: E0=0, E1=base/5, E5=base, E6=base, first restart E206.
    if epoch<=5:return base*epoch/5
    return 1e-7+(base-1e-7)*(1+math.cos(math.pi*((epoch-6)%200)/200))/2

def evaluate(model,loader,device,metrics=False):
    model.eval();losses=[];ys=[];scores=[]
    with torch.no_grad():
        for batch in loader:
            pred,label,loss=forward(model,batch,device);losses.extend(loss.cpu().tolist())
            if metrics:
                valid=label!=-999
                ys.append((label[valid]>.5).to(torch.uint8).cpu().numpy())
                scores.append(pred[valid].cpu().numpy())
    result={'loss':float(np.mean(losses)),'targets':len(loader.dataset)}
    if metrics:
        y=np.concatenate(ys);p=np.concatenate(scores)
        result['valid_positions']=len(y)
        result['Global_auROC']=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None
        if metrics=='auroc':return result
        if y.sum():
            precision,recall,_=precision_recall_curve(y,p);result['Global_auPR']=float(auc(recall,precision))
        else:result['Global_auPR']=None
    return result

def compare_reference(model,path):
    ref=torch.load(path,map_location='cpu',weights_only=True);actual=model.state_dict()
    if ref.keys()!=actual.keys():raise ValueError('Reference checkpoint keys differ')
    errors={k:float((v.detach().cpu()-ref[k]).abs().max()) for k,v in actual.items()}
    result={'checkpoint':str(path),'max_abs_error':max(errors.values()),'unequal_tensors':sum(v!=0 for v in errors.values())}
    if result['max_abs_error']!=0:raise ValueError(f'Historical replay mismatch: {result}')
    return result

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data',required=True,type=Path);p.add_argument('--val-data',required=True,type=Path)
    p.add_argument('--test-data',type=Path);p.add_argument('--ts2-data',type=Path)
    p.add_argument('--split-dir',type=Path,help='Optional directory with train/val/test/ts2.tsv and a pairid column')
    p.add_argument('--output',required=True,type=Path,help='A new output directory; existing paths are refused')
    p.add_argument('--batchsize',type=int,default=1);p.add_argument('--eval-batchsize',type=int,default=1)
    p.add_argument('--seed',type=int,default=2022);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--patience',type=int,default=3);p.add_argument('--max-epochs',type=int,default=1500)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=8);p.add_argument('--vram-limit-mib',type=int)
    p.add_argument('--initial-state',type=Path,help='Optional model state_dict; optimizer always starts fresh')
    p.add_argument('--select-by',choices=['val-auroc','val-loss'],default='val-auroc')
    p.add_argument('--auroc-rank',choices=['rounded3','exact'],default='rounded3')
    p.add_argument('--legacy-rng',action='store_true',help='Batch1 legacy train-evaluation shuffle and two extra evaluation-iterator RNG draws')
    p.add_argument('--reference-checkpoints',type=Path,help='With --legacy-rng, require exact equality to E0... files named 0model.pt etc.')
    return p

def run(args):
    if min(args.batchsize,args.eval_batchsize,args.patience,args.threads)<1 or args.max_epochs<1:raise ValueError('Batch sizes, patience, threads and max epochs must be positive')
    if args.lr<=0 or not math.isfinite(args.lr):raise ValueError('Learning rate must be positive and finite')
    if args.legacy_rng and (args.batchsize!=1 or args.eval_batchsize!=1):raise ValueError('--legacy-rng requires training/evaluation batch size 1')
    if args.reference_checkpoints and not args.legacy_rng:raise ValueError('Reference checking requires --legacy-rng')
    if args.vram_limit_mib is not None and args.vram_limit_mib<=0:raise ValueError('VRAM limit must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=torch.device(args.device)
    if device.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        # Allocator limits require an indexed device on supported PyTorch releases.
        if device.index is None:device=torch.device('cuda',torch.cuda.current_device())
        torch.cuda.manual_seed_all(args.seed)
        if args.vram_limit_mib:
            total=torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(.95,args.vram_limit_mib*1024**2/total),device)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.set_num_threads(args.threads)
    profiler=ResourceProfiler(args.output,device);profiler.configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    status='completed';error=None
    try:
        # Training and validation only. Test arrays/labels are first opened after checkpoint selection.
        datasets={}
        for tag,path in [('train',args.train_data),('val',args.val_data)]:
            idfile=args.split_dir/(tag+'.tsv') if args.split_dir else None
            datasets[tag]=ContactDataset(path,idfile)
        if datasets['train'].dim!=datasets['val'].dim:raise ValueError('Train/validation dimensions differ')
        if set(datasets['train'].ids)&set(datasets['val'].ids):raise ValueError('Training/validation target overlap')
        train_loader=DataLoader(datasets['train'],batch_size=args.batchsize,shuffle=True,num_workers=0,collate_fn=collate)
        loaders={s:DataLoader(d,batch_size=args.eval_batchsize,shuffle=False,num_workers=0,collate_fn=collate) for s,d in datasets.items()}
        if args.legacy_rng:loaders['train']=train_loader
        dummy_loaders=[DataLoader(range(1),batch_size=1,shuffle=False,num_workers=0) for _ in range(2)] if args.legacy_rng else []
        dr,dp=datasets['train'].dim
        model=ProteinRNAInteraction_rela(dp,dr,3,48,n_head=4,d_ff=512,dropout=.1,sync=True).to(device)
        model.apply(initialize_weights)
        if args.initial_state:model.load_state_dict(torch.load(args.initial_state,map_location=device,weights_only=True),strict=True)
        install_masks(model)
        config={**profiler.configuration,'rna_dim':dr,'protein_dim':dp,
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'numpy_version':np.__version__,'torch_version':torch.__version__,'python':sys.version,
                'optimizer':'RMSprop','gradient_clip_norm':5,'loss_reduction':'mean within each complex, then across batch',
                'early_stopping':'validation BCE','selection_uses_test':False,
                'architecture_sha256':hashlib.sha256((Path(__file__).parent/'model.py').read_bytes()).hexdigest()}
        write(args.output/'config.json',config)
        optimizer=torch.optim.RMSprop(model.parameters(),lr=args.lr)
        checkpoints=args.output/'checkpoints';checkpoints.mkdir()
        history=[];best_loss=float('inf');bad=0;audits=[];updates=0
        for epoch in range(args.max_epochs+1):
            lr=learning_rate(epoch,args.lr)
            for group in optimizer.param_groups:group['lr']=lr
            online=None
            if epoch:
                model.train();values=[]
                with profiler.phase('train',epoch):
                    for batch in train_loader:
                        optimizer.zero_grad();_,_,loss=forward(model,batch,device);objective=loss.mean()
                        if not bool(torch.isfinite(objective)):raise ValueError('Nonfinite training loss')
                        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
                        optimizer.step();updates+=1;values.append(float(objective.detach()))
                online=float(np.mean(values))
            with profiler.phase('evaluation',epoch,'Train'):tr=evaluate(model,loaders['train'],device)['loss']
            with profiler.phase('evaluation',epoch,'Validate'):vl=evaluate(model,loaders['val'],device)['loss']
            if not math.isfinite(tr) or not math.isfinite(vl):raise ValueError('Nonfinite evaluation loss')
            for dummy in dummy_loaders:
                for _ in dummy:pass
            row={'epoch':epoch,'lr':lr,'train_loss':tr,'validation_loss':vl,'online_batch_loss':online,'optimizer_steps':updates}
            history.append(row);torch.save(model.state_dict(),checkpoints/f'{epoch}model.pt')
            if args.reference_checkpoints:
                audits.append({'epoch':epoch,**compare_reference(model,args.reference_checkpoints/f'{epoch}model.pt')})
                write(args.output/'historical_checkpoint_comparison.json',audits)
            write(args.output/'history.json',history)
            with (args.output/'history.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            print(json.dumps(row),flush=True)
            if vl<=best_loss:best_loss=vl;bad=0
            else:bad+=1
            if bad>=args.patience:break
        # AUROC selection is posthoc and validation-only. No expensive threshold sweep per epoch.
        selection=[]
        if args.select_by=='val-auroc':
            with profiler.phase('checkpoint_selection',tag='Validate'):
                for row in history:
                    e=row['epoch'];model.load_state_dict(torch.load(checkpoints/f'{e}model.pt',map_location=device,weights_only=True))
                    metric=evaluate(model,loaders['val'],device,metrics='auroc')
                    if metric['Global_auROC'] is None:raise ValueError('Validation needs positive and negative labels for AUROC selection')
                    score=metric['Global_auROC'];rank=float(f'{score:.3f}') if args.auroc_rank=='rounded3' else score
                    selection.append({'epoch':e,'VL_auROC':score,'rank_value':rank})
            selected=selection[int(np.argsort([r['rank_value'] for r in selection])[-1])]['epoch']
        else:
            selected=min(history,key=lambda r:(r['validation_loss'],-r['epoch']))['epoch']
        selected_source=checkpoints/f'{selected}model.pt'
        model.load_state_dict(torch.load(selected_source,map_location=device,weights_only=True));torch.save(model.state_dict(),args.output/'selected.pt')
        write(args.output/'selection.json',{'selected_epoch':selected,'criterion':args.select_by,'auroc_rank':args.auroc_rank,
            'validation_ranking':selection,'test_used_for_selection':False})
        used=set(datasets['train'].ids)|set(datasets['val'].ids)
        metrics={}
        for tag,path in [('val',args.val_data),('test',args.test_data),('ts2',args.ts2_data)]:
            if path is None:continue
            if tag=='val':data=datasets['val']
            else:
                idfile=args.split_dir/(tag+'.tsv') if args.split_dir else None
                data=ContactDataset(path,idfile)
                if data.dim!=(dr,dp):raise ValueError('Test embedding dimensions differ from training')
                if used&set(data.ids):raise ValueError('Target overlap across train/validation/test splits')
                used.update(data.ids)
            loader=DataLoader(data,batch_size=args.eval_batchsize,shuffle=False,num_workers=0,collate_fn=collate)
            with profiler.phase('final_evaluation',tag=tag):metrics[tag]=evaluate(model,loader,device,metrics=True)
        write(args.output/'metrics.json',metrics)
        write(args.output/'complete.json',{'completed':True,'selected_epoch':selected,'trained_epochs':len(history)-1,
            'optimizer_steps':updates,'checkpoint':str(args.output/'selected.pt')})
    except BaseException as e:
        status='failed';error=f'{type(e).__name__}: {e}';write(args.output/'failure.json',{'error':error});raise
    finally:profiler.finish(status,error)

def main(argv=None):
    return run(parser().parse_args(argv))
