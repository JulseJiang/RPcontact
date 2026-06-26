import os
import time
import torch
import argparse
import numpy as np

from src.ernie_rna.tasks.ernie_rna import *
from src.ernie_rna.models.ernie_rna import *
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import ErnieRNAOnestage, read_text_file, load_pretrained_ernierna, prepare_input_for_ernierna


def seq_to_index(sequences):
    '''
    input:
    sequences: list of string (difference length)
    
    return:
    rna_index: numpy matrix, shape like: [len(sequences), max_seq_len+2]
    rna_len_lst: list of length
    '''

    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    rna_index = np.ones((seq_nums,max_len+2))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j+1] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j+1] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j+1] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j+1] = 6
            else:
                rna_index[i][j+1] = 3
        rna_index[i][rna_len_lst[i]+1] = 2 # add 'eos' token
    rna_index[:,0] = 0 # add 'cls' token
    return rna_index, rna_len_lst

def extract_embedding_of_ernierna(sequences, if_cls=True, arg_overrides = { "data": '/data/code/BERT/onestage_checkpoint_dict/' }, pretrained_model_path =  '/data/code/BERT/Pretrain_checkpoint/twocheckpoint_best.pt', device='cpu', layer_idx=12):
    '''
    input:
    sequences: List of string (difference length)
    if_cls: Bool, Determine the size of the extracted feature
    arg_overrides: The folder where the character-to-number mapping file resides
    pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    embedding: numpy matrix, shape like: [len(sequences), 768](if_cls=True) or [len(sequences), max_len_seq+2, 768](if_cls=False)
    '''

    my_model = load_model(arg_overrides = arg_overrides, pretrained_model_path = pretrained_model_path, device=device)
    embedding = get_embedding_from_sequence(sequences,device=device,if_cls=if_cls,layer_idx=layer_idx,my_model=my_model)
        
    return embedding
def load_model(arg_overrides = { "data": '/data/code/BERT/onestage_checkpoint_dict/' }, pretrained_model_path =  '/data/code/BERT/Pretrain_checkpoint/twocheckpoint_best.pt', device='cpu'):
    # # load model
    model_pretrained = load_pretrained_ernierna(pretrained_model_path,arg_overrides)
    my_model = ErnieRNAOnestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    my_model.eval()
    return my_model
def get_embedding_from_sequence(sequences,device='cpu',if_cls=True,layer_idx=12,my_model=None):
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_index(sequences)
    layer_num = 1 if layer_idx < 12 else layer_idx
    # extract embedding one by one
    if if_cls:
        embedding = np.zeros((len(sequences), layer_num, 768))
    else:
        embedding = np.zeros((len(sequences), layer_num, max(rna_len_lst) + 2, 768))

    with torch.no_grad():
        for i, (index, seq_len) in enumerate(zip(rna_index, rna_len_lst)):

            one_d, two_d = prepare_input_for_ernierna(index, seq_len)
            one_d = one_d.to(device)
            two_d = two_d.to(device)

            output = my_model(one_d, two_d, layer_idx=layer_idx).cpu().detach().numpy()
            if if_cls:
                embedding[i, :, :] = output[:, 0, 0, :]
            else:
                embedding[i, :, :seq_len + 2, :] = output[:, 0, :, :]
    return embedding
def extract_attnmap_of_ernierna(sequences, attn_len=None, arg_overrides = { "data": '/data/code/BERT/onestage_checkpoint_dict/' }, pretrained_model_path =  '/data/code/BERT/Pretrain_checkpoint/twocheckpoint_best.pt', device='cpu', layer_idx=13, head_idx=12):
    '''
    input:
    sequences: List of string (difference length)
    attn_len: Int (Complement the sequence to this length). if attn_len=None, atten_len will be the length of the longest sequence in the sequences
    arg_overrides: The folder where the character-to-number mapping file resides
    pretrained_model_path: The path of the pre-trained model
    device: The driver used by the model
    
    return:
    atten_map: numpy matrix, shape like: [len(sequences), attn_len+2, attn_len+2]
    '''
    
    # load model
    model_pretrained = load_pretrained_ernierna(pretrained_model_path,arg_overrides)
    my_model = ErnieRNAOnestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    my_model.eval()
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_index(sequences)
    
    # extract embedding one by one
    if attn_len == None:
        attn_len = max(rna_len_lst)
    if head_idx == 12 and layer_idx == 13:
        attn_num = 156
    elif head_idx == 12 or layer_idx == 13:
        attn_num = head_idx if head_idx == 12 else layer_idx
    else:
        attn_num = 1
    rna_attn_map_embedding = np.zeros((len(sequences),attn_num,(attn_len+2), (attn_len+2)))
    with torch.no_grad():
        for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
            one_d, two_d = prepare_input_for_ernierna(index,seq_len)
            one_d = one_d.to(device)
            two_d = two_d.to(device)
            
            output = my_model(one_d,two_d,return_attn_map=True,i=layer_idx,j=head_idx).cpu().detach().numpy()
            
            rna_attn_map_embedding[i, :, :(seq_len+2), :(seq_len+2)] = output
        
    return rna_attn_map_embedding



if __name__ == "__main__":
    
    start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--seqs_path", default='./data/test_seqs.txt', type=str, help="The path of input seqs")
    parser.add_argument("--save_path", default='./results/ernie_rna_representations/test_seqs/', type=str, help="Output directory for per-RNA .npy embeddings")
    parser.add_argument("--arg_overrides", default={ "data": './src/dict/' }, help="The path of vocabulary")
    parser.add_argument("--ernie_rna_pretrained_checkpoint", default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', type=str, help="The path of ERNIE-RNA checkpoint")
    parser.add_argument("--layer_idx_emb", default=12, type=int, help="The layer idx of which we extract embedding from, 12 for all layers")
    parser.add_argument("--layer_idx_attn", default=13, type=int, help="The layer idx of which we extract attnmap from, 13 for all layers")
    parser.add_argument("--head_idx_attn", default=12, type=int, help="The head idx of which we extract attnmap from, 12 for all heads")
    parser.add_argument("--device", default=0, help="device: cpu, 0,1,2 for gpu devive number")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    args.device = int(args.device) if args.device!='cpu' else 'cpu'    
    # lines = read_text_file(args.seqs_path)
    # seqs_lst = []
    # for line in lines:
    #     seqs_lst.append(line)

    with open(args.seqs_path) as f:
        lines = f.read().splitlines()
    _ids = [line[1:] for line in lines if '>' in line]
    seqs_lst = [line.replace('T','U') for line in lines if '>' not in line]
    print(f'loading {len(seqs_lst)} sequences')
    _ids, seqs_lst = zip(*[(_id, seq) for _id, seq in zip(_ids, seqs_lst) if len(seq) < 1020])
    print(f'keeping {len(seqs_lst)} sequences whose length < 1020')
    seqdict = dict(zip(_ids,seqs_lst))
    if len(_ids)==0:
        _ids = list(range(len(seqs_lst)))
    try:
        assert 0 <= args.layer_idx_emb <= 12
        assert 0 <= args.layer_idx_attn <= 13
        assert 0 <= args.head_idx_attn <= 12
    except:
        raise(NotImplementedError)
    # cls_embedding = extract_embedding_of_ernierna(seqs_lst, if_cls=True, arg_overrides=args.arg_overrides, pretrained_model_path=args.ernie_rna_pretrained_checkpoint, device=args.device, layer_idx = args.layer_idx_emb)
    # # print(cls_embedding.shape) # cls_embedding shape like [Batch, 768]
    # np.save(args.save_path + 'cls_embedding.npy',cls_embedding)

    batch_size = 500
    batche_seqs = [seqs_lst[i:i+batch_size] for i in range(0, len(seqs_lst), batch_size)]
    batche_ids = [_ids[i:i+batch_size] for i in range(0, len(seqs_lst), batch_size)]
    my_model = load_model(arg_overrides=args.arg_overrides, pretrained_model_path=args.ernie_rna_pretrained_checkpoint,
                          device=args.device)
    for batche_id,batch_seq in zip(batche_ids,batche_seqs):
        all_embedding = get_embedding_from_sequence(batch_seq, device=args.device, if_cls=False, layer_idx=args.layer_idx_emb,
                                                my_model=my_model)
        # all_embedding = extract_embedding_of_ernierna(batch_seq, if_cls=False, arg_overrides=args.arg_overrides,
        #                                           pretrained_model_path=args.ernie_rna_pretrained_checkpoint,
        #                                           device=args.device, layer_idx=args.layer_idx_emb)
        for _id,embedding in zip(batche_id,all_embedding):
            np.save(os.path.join(args.save_path, f"{_id}.npy"), embedding.mean(0)[1:len(seqdict[_id])+1].astype(np.float32))
    # for name,seq in seqdict.items():
    #     try:
    #         all_embedding = extract_embedding_of_ernierna([seq], if_cls=False, arg_overrides=args.arg_overrides, pretrained_model_path=args.ernie_rna_pretrained_checkpoint, device=args.device, layer_idx = args.layer_idx_emb)
    #         # print(all_embedding.shape) # all_embedding shape like [Batch, Length + 2, 768]
    #         np.save(args.save_path + f'{name}.npy',all_embedding)
    #     except Exception as e:
    #         print(name,len(seq),seq,e)

    # attnmap = extract_attnmap_of_ernierna(seqs_lst, attn_len=None, arg_overrides=args.arg_overrides, pretrained_model_path=args.ernie_rna_pretrained_checkpoint, device=args.device, layer_idx = args.layer_idx_attn, head_idx = args.head_idx_attn)
    # # print(attnmap.shape) # attnmap shape like [Batch, Length + 2, Length + 2]
    # np.save(args.save_path + 'attnmap.npy',attnmap)
    print(f'Done in {time.time()-start}s!')
