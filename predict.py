#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: julse@qq.com
# des : evaluate RPcontact
import glob
import pickle
import random
import re
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import torch
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import pandas as pd
class bcolors:
    RED   = "\033[1;31m"
    BLUE  = "\033[1;34m"
    CYAN  = "\033[1;36m"
    GREEN = "\033[0;32m"
    RESET = "\033[0;0m"
    BOLD    = "\033[;1m"
    REVERSE = "\033[;7m"


def check_path(dirout,file=False):
    if file:dirout = dirout.rsplit('/',1)[0]
    try:
        if not os.path.exists(dirout):
            print('make dir '+dirout)
            os.makedirs(dirout)
    except:
        print(f'{dirout} have been made by other process')

def load_label_pred(fin_label,fin_pred):
    with open(fin_label, 'rb') as f:
        df_label = pickle.load(f)
    df_label = df_label.squeeze()
    df_pred = pd.read_table(fin_pred, comment='#', index_col=[0])
    if type(df_label) == pd.DataFrame:
        df_pred.index = df_label.index
        df_pred.columns = df_label.columns
        # 删除包含空值的行
        df_label = df_label.dropna(how='all')

        # 删除包含空值的列
        df_label = df_label.dropna(axis=1, how='all')
        df_pred = df_pred.loc[df_label.index, df_label.columns]
    keep=0
    if df_pred.columns[0].count('.')==2:
        keep=-1
    df_pred.columns = [e.split('.')[keep] + str(i+1) for i, e in enumerate(df_pred.columns)]
    df_pred.index = [e.split('.')[keep] + str(i+1) for i, e in enumerate(df_pred.index)]
    return df_label,df_pred
def doSavePredict(_id,seq,predict,fout,des):
    # seq = {'protein': 'KKGVGSTKNGRDSEAKRLGAKRADGQFVTGGSILYRQRGTKIYPGENVGRGGDDTLFAKIDGTVKFERFGRDRKKVSVYPV',
    #  'rna': 'GGGGCCUUAGCUCAGGGGAGAGCGCCUGCUUUGCACGCAGGAGGCAGCGGUUCGAUCCCGCUAGGCUCCACCA'}
    check_path(fout)
    df = pd.DataFrame(predict)
    if not seq:df.to_csv(fout+ f'{_id}.txt',sep='\t',mode='w',float_format='%.5f')
    else:
        df.columns = list(seq['protein'])
        df.index = list(seq['rna'])
        with open(fout+ f'{_id}.txt','w') as f:
            f.write(f'#{des}\n')
            f.write(f"# row =rna:{seq['rna']}\n")
            f.write(f"# col=protein:{seq['protein']}\n")
        # df.to_csv(fout+ f'{_id}.txt',sep='\t',mode='a',float_format='%.3f',index=None,header=None)
        df.to_csv(fout+ f'{_id}.txt',sep='\t',mode='a',float_format='%.5f')

        df.columns = [f'{elem}{index+1}' for index,elem in enumerate(seq['protein'])]
        df.index = [f'{elem}{index+1}' for index,elem in enumerate(seq['rna'])]
        df = get_top_l_triplets(df, sum(df.shape))
        df.to_csv(fout+ f'{_id}_topL.txt',sep='\t',mode='w',float_format='%.5f',index=False)



def get_top_l_triplets(df_pred, L):
    """
    从Pandas DataFrame矩阵中提取值最大的前L个三元组。

    参数:
    - matrix_df: Pandas DataFrame，表示接触矩阵。
    - L: int，要提取的三元组的数量。

    返回:
    - top_l_triplets: 列表，包含前L个三元组，每个三元组格式为(row_index, col_index, value)。
    """
    df = df_pred.stack().reset_index()
    df.columns = ['rna', 'protein', 'pred']
    df = df.sort_values(by='pred', ascending=False).head(L)
    return df

def doSavePredict_single(_id,seq,predict_rsa,fout,des,pred_asa=None):
    check_path(fout)
    BASES = 'AUCG'
    asa_std = [400, 350, 350, 400]
    dict_rnam1_ASA = dict(zip(BASES, asa_std))
    sequence = re.sub(r"[T]", "U", ''.join(seq))
    sequence = re.sub(r"[^AGCU]", BASES[random.randint(0, 3)], sequence) # 其他字符随机变换以取得对目标的预测
    ASA_scale = np.array([dict_rnam1_ASA[i] for i in sequence])

    if pred_asa is None:
        pred_asa = np.multiply(predict_rsa, ASA_scale).T
    else:
        predict_rsa = pred_asa/ASA_scale
    col1 = np.array([i + 1 for i, I in enumerate(seq)])[None, :]
    col2 = np.array([I for i, I in enumerate(seq)])[None, :]
    col3 = pred_asa
    col4 = predict_rsa
    if len(col3[col3 == 0]):
        exit(f'error in predict\t {_id},{seq}')
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%.2f', col3), np.char.mod('%.3f', col4))).T
    if fout:np.savetxt(fout + f'{_id}.txt', (temp), delimiter='\t\t', fmt="%s",
               header=f'#{des}',
               comments='')

    return pred_asa,predict_rsa

def one_hot_encode(sequences,alpha='ACGU'):
    print(sequences)
    sequences_arry = np.array(list(sequences)).reshape(-1, 1)
    lable = np.array(list(alpha)).reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lable)
    seq_encode = enc.transform(sequences_arry).toarray()
    # print(seq_encode.shape)
    return (seq_encode)

def get_bin_pred(df_pred,threshold):
    bin_pred = df_pred.values >= threshold
    bin_pred = bin_pred.astype(int)
    return bin_pred

def seed_everything(seed=2022):
    print('seed_everything to ',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 程序每次运行结果一致，但是程序中多次生成随机数每次不一致 # https://blog.csdn.net/qq_42951560/article/details/112174334
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # minbatch的长度一直在变化，这个优化比较浪费时间


def contact_partner_constrained(prob_matrix, colmax=12, rowmax=24):
    """Apply contact partner constraints to probability matrix"""
    row_max_indices = np.argsort(-prob_matrix, axis=1)[:, :rowmax]
    row_max_mask = np.zeros_like(prob_matrix)
    row_max_mask[np.arange(prob_matrix.shape[0])[:, np.newaxis], row_max_indices] = 1

    col_max_indices = np.argsort(-prob_matrix, axis=0)[:colmax, :]
    col_max_mask = np.zeros_like(prob_matrix)
    col_max_mask[col_max_indices, np.arange(prob_matrix.shape[1])] = 1

    mask = np.logical_and(row_max_mask, col_max_mask).astype(np.float32)
    prob_matrix = np.where(mask == 1, prob_matrix, 0)
    return prob_matrix
def getParam():
    parser = ArgumentParser()
    # data
    parser.add_argument('--rootdir', default='',
                        type=str)
    parser.add_argument('--fasta', default='./example/inputs/8DMB_W.8DMB_P.fasta',
                        type=str)
    parser.add_argument('--out', default='./example/outputs/',
                        type=str)
    parser.add_argument('--ffeat', default='./example/inputs/{pdbid}.pickle',
                        type=str)
    parser.add_argument('--fmodel', default='./weight/model_roc_0_38=0.845.pt',
                        type=str)
    parser.add_argument('--device', default='cpu',
                        type=str)
    parser.add_argument('--draw',action='store_true',default=True)
    parser.add_argument('--constrained',action='store_true',default=True)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = getParam()
    rootdir = args.rootdir
    fasta = args.fasta
    ffeat = args.ffeat
    fmodel = args.fmodel
    device = args.device
    out = args.out
    draw = args.draw
    check_path(out)

    # pdbid = fasta.rsplit('/',1)[0].split('.')[0]
    seed_everything(seed=2022)

    models = [(model_path,torch.load(model_path, map_location=torch.device(device))) for model_path in glob.glob(fmodel)]
    # models = [(model_path,torch.load(model_path, map_location=torch.device(device))) for model_path in glob.glob(fmodel)]
    print('loading existed model', fmodel)
    with torch.no_grad():
        for pdbid,seq in [(record.id,record.seq) for record in SeqIO.parse(fasta,'fasta')]:
            rnaid,proid= pdbid.split('.')
            rnaseq,proseq= seq.split('.')

            with open(ffeat.format_map({'pdbid':rnaid}),'rb') as f:
                rna_emb = pickle.load(f)
            with open(ffeat.format_map({'pdbid':proid}),'rb') as f:
                pro_emb = pickle.load(f)

            rna_oh = one_hot_encode(rnaseq, alpha='ACGU')
            pro_oh = one_hot_encode(proseq, alpha='GAVLIFWYDNEKQMSTCPHR')

            # mask = np.ones((emb.shape[0],1)) # mask missing nt when evaluate the model
            x_train = np.concatenate([rna_oh,rna_emb],axis=1)
            x_train = np.expand_dims(x_train,0)
            x_train = torch.from_numpy(x_train).transpose(-1,-2)
            x_train = x_train.to(device, dtype=torch.float)
            x_rna = x_train

            x_train = np.concatenate([pro_oh, pro_emb], axis=1)
            x_train = np.expand_dims(x_train, 0)
            x_train = torch.from_numpy(x_train).transpose(-1, -2)
            x_train = x_train.to(device, dtype=torch.float)
            x_pro = x_train

            print('input data shape for rna and protein:',x_rna.shape,x_pro.shape)

            x_rna = x_rna.to(device, dtype=torch.float32)
            x_pro = x_pro.to(device, dtype=torch.float32)
            

            ###########

            predict_scores = []

            #######
            for i,(model_path,model) in enumerate(models):
                model.eval()
                outputs = model(x_pro, x_rna)  # [1, 299, 74, 1]
                # print('outputs,',outputs.device)
                outputs = torch.squeeze(outputs, -1)
                outputs = outputs.permute(0, 2, 1)

                df_pred = outputs[0].cpu().detach().numpy()

                # Apply constraints and normalization
                if args.constrained:contact_matrix = contact_partner_constrained(df_pred)
                contact_matrix = (contact_matrix - contact_matrix.min()) / (
                            contact_matrix.max() - contact_matrix.min() + 1e-8)

                # seq = data._seq[pdbid] if pdbid in data._seq else None
                des = f'predict by {__file__}\n#{model_path}'
                doSavePredict(pdbid, {'rna':rnaseq,'protein':proseq}, df_pred,
                                   out,
                                   des
                                   )

                tmp = df_pred.flatten()
                tmp.sort()
                score = sum(tmp[::-1][:sum(df_pred.shape)])
                predict_scores.append((pdbid, score))
                print('pdbid',pdbid,score) # 这个score是否和label中contact的个数有correlation？

                if draw:
                    plt.figure(figsize=(20, 15))
                    top = sum(df_pred.shape)
                    df_pred = pd.DataFrame(df_pred)
                    threshold = df_pred.stack().nlargest(top).iloc[-1]
                    bin_pred = get_bin_pred(df_pred,threshold=threshold)

                    import seaborn as sns
                    sns.heatmap(df_pred,mask=bin_pred,cbar_kws={"shrink": 0.5},cmap='coolwarm',vmin=0,vmax=1)
                    plt.title(f'Predicted contact map of {pdbid}\nPredidcted by RPcontact, top L=r+p')
                    plt.xlabel(proid)
                    plt.ylabel(rnaid)
                    handles, labels = plt.gca().get_legend_handles_labels()

                    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, borderaxespad=1,
                            frameon=False)
                    # 设置坐标轴的相同缩放
                    ax = plt.gca()
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    plt.savefig(f'{out}/{pdbid}_{i}_prob.png',dpi=300)
                    plt.show()

                    plt.clf()
                    ax = plt.gca()
                    tp = \
                    ax.plot(*np.where(bin_pred.T==1), ".", c='r',markersize=1, label='Predicted contact')[
                        0]
                    tp.set_markerfacecolor('w')
                    tp.set_markeredgecolor('r')
                    h,w = bin_pred.shape
                    plt.xlim([0,w])
                    plt.ylim([0,h])
                    plt.title(f'Predicted contact map of {pdbid}\nPredidcted by RPcontact, top L=r+p')
                    plt.xlabel(proid)
                    plt.ylabel(rnaid)
                    handles, labels = plt.gca().get_legend_handles_labels()

                    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, borderaxespad=1,
                            frameon=False)

                    # 设置坐标轴的相同缩放
                    ax.set_aspect('equal')
                    plt.tight_layout()
                    plt.savefig(f'{out}/{pdbid}_{i}_binary.png',dpi=300)
                    plt.show()


                    print(f'predict {pdbid} with {len(seq)} nts')

            df = pd.DataFrame(predict_scores, columns=['pdbid', 'contact_score'])
            df.to_csv(args.out + '/predict_scores.csv',index=False, sep='\t', mode='a', float_format='%.5f')
        





