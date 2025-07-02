<p align="center">
  <img src="https://raw.githubusercontent.com/JulseJiang/RPcontact/main/example/logo.png" alt="RPcontact Logo" width="120"/>
</p>

# RPcontact: RNA-Protein Contact Prediction

**Improved prediction of RNA-protein contacts using RNA and protein language models**

[Paper](https://www.biorxiv.org/content/10.1101/2025.06.02.657171v1.full)
[Code](https://github.com/rpcontact)
[Demo](https://julse-rpcontact.hf.space/)


---

## Overview

RPcontact is a novel computational tool for accurately predicting RNA-protein contacts, addressing a fundamental challenge in understanding molecular biology processes such as transcription, splicing, and translation. Traditional methods are limited by the scarcity of RNA-protein complex structures and the constraints of experimental techniques. While recent deep learning approaches like AlphaFold 3 and RoseTTAFoldNA have made progress, they still rely heavily on homologous templates.

RPcontact overcomes these limitations by leveraging large language models specifically designed for RNA ([ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)) and proteins ([ESM-2](https://github.com/facebookresearch/esm)). Trained exclusively on ribosomal RNA-protein complexes, RPcontact delivers robust and generalized performance, accurately predicting contacts in both dimeric and multimeric non-rRNA-protein complexes. Benchmark results show that RPcontact significantly outperforms binary contacts inferred from models like AlphaFold 3 and RoseTTAFoldNA, making it a valuable tool for structure and function prediction in RNA-protein research.

---

## Quick Start

### Requirements

| Dependency  | Recommended Version |
|-------------|--------------------|
| Python      | ≥ 3.8              |
| PyTorch     | 1.13.1             |
| fair-esm    | 1.0.2              |

Install dependencies (example):
```bash
pip install numpy pandas matplotlib biopython scikit-learn
pip install torch==1.13.1
pip install fair-esm==1.0.2
```

---

### Script Overview

| Script            | Function                            | Example Command                 |
|-------------------|-------------------------------------|---------------------------------|
| predict.py        | Single RNA-protein pair contact prediction  | `python predict.py`             |
| predict_batch.py  | Batch RNA-protein pairs contact prediction   | `python predict_batch.py`       |
| evaluate.py       | Evaluation and visualization        | `python evaluate.py`            |
| app.py       | Launch web-based demo interface (need install gradio)          | `python app.py`            |

---

### Data Preparation

- RNA/protein sequences: FASTA format
- Embedding features: pickle format
- For batch prediction: provide a CSV file for pairing info

---

### Typical Usage

**Single pair prediction:**
```bash
python predict.py --fasta your_sequence.fasta --out output_dir/
```

**Batch prediction:**
```bash
python predict_batch.py --rna_fasta rna.fasta --pro_fasta protein.fasta --csv pairs.csv --out output_dir/
```

**Evaluation:**
```bash
python evaluate.py --fasta your_sequence.fasta --out eval_dir/ --flabel true_labels.pickle
```

---

### Common Parameters

| Parameter     | Description                                             |
|---------------|--------------------------------------------------------|
| --fasta       | Input FASTA file (for single prediction)               |
| --rna_fasta   | RNA FASTA file (for batch prediction)                  |
| --pro_fasta   | Protein FASTA file (for batch prediction)              |
| --csv         | RNA-protein pairing info CSV (for batch prediction)    |
| --ffeat       | Precomputed embedding feature file (pickle format)     |
| --fmodel      | Pretrained model file path                             |
| --out         | Output directory                                       |
| --flabel      | True label file (for evaluation)                       |
| --device      | Specify device (e.g., cpu or cuda:0)                   |
| --draw        | Whether to visualize results                           |

---

## Output Interpretation

- The prediction output is a contact probability matrix for each RNA-protein pair. Higher scores indicate a higher probability of interaction.
- The evaluation script provides accuracy and other metrics, as well as visualization.

---

## Contact & Citation

Questions or suggestions? Contact:

- Jiuhong Jiang  
- Email: jiangjh2023@shanghaitech.edu.cn

If you find this project helpful, please cite our manuscript.
- Jiang, J., Zhang, X., Zhan, J., Miao, Z., & Zhou, Y. (2025). RPcontact: Improved prediction of RNA-protein contacts using RNA and protein language models. bioRxiv, 2025-06.
---

<p align="center"><em>Make RNA-protein contact prediction easier and more accurate!</em></p>
