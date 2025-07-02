#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: julse@qq.com
# des : evaluate RPcontact
import glob
import os
import pickle
import random
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd

import torch
from Bio import SeqIO
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from predict import check_path, one_hot_encode, get_bin_pred, doSavePredict


def get_bin_label(df_label,distance_cutoff):
    bin_label = df_label < distance_cutoff
    bin_label = bin_label.astype(int)
    return bin_label

def view_evaluate_contact_prob(df_label, bin_pred,ax=None,markersize=5):
    confusing_matrix = np.zeros_like(df_label)
    r, p = confusing_matrix.shape
    if ax is None:
        ax = plt
        ax.xlim([-2, p + 2])
        ax.ylim([-2, r + 2])
        # plt.xticks(rotation=90)
    else:

        ax.set_xlim([-2, p + 2])
        ax.set_ylim([-2, r + 2])
        # plt.setp(ax.get_xticklabels(), rotation=90)
        ax.set_title('performance')

    colors = [
              '#f5e0c4', # lightblue for FP
            # '#aaa6ce','#66609c','k',# light purple, dark purple,black, for Groud truth
            '#b0d9db','#61b3b6','k',# light purple, dark purple,black, for Groud truth
        '#ecbbd8','#9d4e7d','r' # for TP

    ]
    tps = []
    bin_label = df_label<8
    temp = bin_pred - bin_label
    fn = ax.plot(*np.where(temp.T == 1), ".", c=colors[0], markersize=markersize,label='False Positive')[0]
    # 绘制NaN值的数据点为灰色
    oc = ax.plot(*np.where(df_label.T.isna()), ".", c='gray', markersize=markersize, label='Missing in PDB')[0]
    confusing_matrix[bin_label == 1] = 1  #ground truth
    oc = ax.plot(*np.where(bin_label.T == 1), ".", c=colors[1],markersize=markersize, label='Ground truth (8Å)')[0]
    temp = bin_label + bin_pred
    tps.append(len(confusing_matrix[np.where(temp == 2)]))
    confusing_matrix[np.where(temp == 2)] = 2  # TP : blue
    tp = ax.plot(*np.where(temp.T == 2), "o", c=colors[4],markersize=markersize, label='True Positive (8Å)')[0]
    tp.set_markerfacecolor(colors[1])
    tp.set_markeredgecolor(colors[4])

    bin_label = df_label<5
    temp = bin_label + bin_pred
    tps.append(len(confusing_matrix[np.where(temp == 2)]))

    oc = ax.plot(*np.where(bin_label.T == 1), ".", c=colors[2],markersize=markersize, label='Ground truth (5Å)')[0]
    confusing_matrix[np.where(temp == 2)] = 2  # TP : blue
    tp = ax.plot(*np.where(temp.T == 2), "o", c=colors[5],markersize=markersize, label='True Positive (5Å)')[0]
    tp.set_markerfacecolor(colors[2])
    tp.set_markeredgecolor(colors[5])
    bin_label = df_label<3.5
    oc = ax.plot(*np.where(bin_label.T == 1), ".", c=colors[3],markersize=markersize, label='Ground truth (3.5Å)')[0]
    temp = bin_label + bin_pred
    tps.append(len(confusing_matrix[np.where(temp == 2)]))

    confusing_matrix[np.where(temp == 2)] = 2  # TP : blue
    tp = ax.plot(*np.where(temp.T == 2), "o", c=colors[6],markersize=markersize, label='True Positive (3.5Å)')[0]
    tp.set_markerfacecolor(colors[3])
    tp.set_markeredgecolor(colors[6])

    # ax.legend()
    # plt.show()
    # tp = len(confusing_matrix[np.where(temp == 2)])
    print(len(confusing_matrix[np.where(temp == 2)]))
    return '/'.join([str(e) for e in tps[::-1]]),confusing_matrix
def seed_everything(seed=2022):
    print('seed_everything to ',seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # 程序每次运行结果一致，但是程序中多次生成随机数每次不一致 # https://blog.csdn.net/qq_42951560/article/details/112174334
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # minbatch的长度一直在变化，这个优化比较浪费时间


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
    parser.add_argument('--flabel', default='./example/inputs/{pdbid}.pickle',
                        type=str)
    parser.add_argument('--draw', default=True,
                        type=bool)
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = getParam()
    rootdir = args.rootdir
    fasta = args.fasta
    ffeat = args.ffeat
    fmodel = args.fmodel
    device = args.device
    flabel = args.flabel
    draw = args.draw
    out = args.out
    check_path(out)

    # pdbid = fasta.rsplit('/',1)[0].split('.')[0]
    seed_everything(seed=2022)
    models = [(model_path,torch.load(model_path, map_location=torch.device(device))) for model_path in glob.glob(fmodel)]
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
            plt.figure(figsize=(20, 15))
            for i,(model_path,model) in enumerate(models):
                model.eval()
                outputs = model(x_pro, x_rna)  # [1, 299, 74, 1]
                # print('outputs,',outputs.device)
                outputs = torch.squeeze(outputs, -1)
                outputs = outputs.permute(0, 2, 1)

                df_pred = outputs[0].cpu().detach().numpy()
                # seq = data._seq[pdbid] if pdbid in data._seq else None
                des = f'predict by {__file__}\n#{model_path}'
                doSavePredict(pdbid, {'rna':rnaseq,'protein':proseq}, df_pred,
                                   out,
                                   des
                                   )
                top = sum(df_pred.shape)
                df_pred = pd.DataFrame(df_pred)
                threshold = df_pred.stack().nlargest(top).iloc[-1]
                if draw:
                    with open(flabel.format_map({'pdbid': pdbid}), 'rb') as f:
                        df_label = pickle.load(f)
                    df_label = df_label.squeeze()
                    bin_pred = get_bin_pred(df_pred, threshold=threshold)
                    view_evaluate_contact_prob(df_label, bin_pred, ax=None)
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
                    plt.savefig(f'{out}/{pdbid}_{i}_evaluate.png',dpi=900)
                    plt.show()
                print(f'predict {pdbid} with {len(seq)} nts')







