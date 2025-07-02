#!/usr/bin/bash

# 检查参数数量
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <fin_fasta> <dirout> <esm2_env_path> <ernie_rna_env_path>"
    exit 1
fi

fin_fasta=$1
dirout=$2
esm2_env_path=$3
ernie_rna_env_path=$4

# 设置默认值
WDIR=$dirout
rna_fasta=$WDIR/_0_process/rna_sequences.fasta
pro_fasta=$WDIR/_0_process/protein_sequences.fasta
fcombinations=$WDIR/_0_process/combinations.csv
finfo=$WDIR/_0_process/info.csv

current_path=$WDIR/_0_process/

# 创建所需目录
mkdir -p $current_path
mkdir -p $current_path/ernie_rna_emb
mkdir -p $current_path/esm2_emb
mkdir -p $current_path/rpcontact
mkdir -p $current_path/no_constrained
mkdir -p $current_path/constrained

# 写入组合文件
while IFS= read -r line; do
    rna_id=$(echo $line | cut -d ',' -f 1)
    rna_seq=$(echo $line | cut -d ',' -f 2)
    pro_id=$(echo $line | cut -d ',' -f 3)
    pro_seq=$(echo $line | cut -d ',' -f 4)
    rna_len=$(echo $line | cut -d ',' -f 5)
    pro_len=$(echo $line | cut -d ',' -f 6)
    echo "$rna_id.$pro_id,$rna_seq,$pro_seq,$rna_len,$pro_len" >> $fcombinations
done < $fin_fasta

# 打印信息
echo "Done. RNA sequences are in $rna_fasta, protein sequences are in $pro_fasta, and combinations are in $fcombinations."
echo "RNA count: $(wc -l < $rna_fasta), RNA max length: $(awk -F',' '{print $5}' $fcombinations | sort -nr | head -n 1), RNA min length: $(awk -F',' '{print $5}' $fcombinations | sort -n | head -n 1), total fragments: $(wc -l < $fcombinations)"
echo "Protein count: $(wc -l < $pro_fasta), Protein max length: $(awk -F',' '{print $6}' $fcombinations | sort -nr | head -n 1), Protein min length: $(awk -F',' '{print $6}' $fcombinations | sort -n | head -n 1), total fragments: $(wc -l < $fcombinations)"
echo "Sequence length longer than 1000 were truncated and kept head and tail with the length of 1000, sliding 500 as step, 1000 as window"

# ERNIE-RNA 嵌入
ERNIE_RNA_script="cd /public/home/jiang_jiuhong/soft/ERNIE-RNA/
$ernie_rna_env_path/miniconda3/envs/ERNIE-RNA/bin/python extract_embedding_jh.py --seqs_path='$rna_fasta' --save_path='$current_path/ernie_rna_emb/' --device=cpu"

echo "$ERNIE_RNA_script" > $current_path/ernie_rna_emb.sh
chmod +x $current_path/ernie_rna_emb.sh

nohup srun -p hebhcnormal01 -c 32 sh $current_path/ernie_rna_emb.sh > $current_path/log_ernie_rna_emb.txt 2>&1 &

# ESM2 嵌入
ESM2_script="cd /public/home/jiang_jiuhong/code/esm/
$esm2_env_path/miniconda3/envs/esm2_env/bin/python scripts/extract.py esm2_t48_15B_UR50D $pro_fasta $current_path/esm2_emb/ --repr_layers 48 --include mean per_tok"

echo "$ESM2_script" > $current_path/esm2_emb.sh
chmod +x $current_path/esm2_emb.sh

nohup srun -p hebhcnormal01 -c 32 sh $current_path/esm2_emb.sh > $current_path/log_esm2_emb.txt 2>&1 &

# 等待嵌入完成
wait

# 执行 RPcontact 获取 contactmap
python process_rna_protein.py --rna_fasta=$rna_fasta --pro_fasta=$pro_fasta --csv=$fcombinations --WDIR=$WDIR --out=$dirout

