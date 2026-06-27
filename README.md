<p align="center">
  <img src="figure/logo.png" alt="RPcontact Logo" width="120"/>
</p>

# RPcontact: RNA-Protein Contact Prediction

**Improved prediction of RNA-protein contacts using RNA and protein language models**

[Paper](https://www.biorxiv.org/content/10.1101/2025.06.02.657171v1.full)
[Code](https://github.com/JulseJiang/RPcontact)

---

## Overview

RPcontact is a computational tool for accurately predicting RNA-protein contacts, addressing a fundamental challenge in understanding molecular biology processes such as transcription, splicing, and translation. Traditional methods are limited by the scarcity of RNA-protein complex structures and the constraints of experimental techniques. While recent deep learning approaches like AlphaFold 3 and RoseTTAFoldNA have made progress, they still rely heavily on homologous templates.

RPcontact overcomes these limitations by leveraging large language models specifically designed for RNA ([ERNIE-RNA](https://github.com/Bruce-ywj/ERNIE-RNA)) and proteins ([ESM-2](https://github.com/facebookresearch/esm)). Trained exclusively on ribosomal RNA-protein complexes, RPcontact delivers robust and generalized performance, accurately predicting contacts in both dimeric and multimeric non-rRNA-protein complexes. Benchmark results show that RPcontact significantly outperforms binary contacts inferred from models like AlphaFold 3 and RoseTTAFoldNA, making it a valuable tool for structure and function prediction in RNA-protein research.

This command-line release provides a simple local inference interface. It supports the embedding-based RPcontact model when precomputed RNA/protein embeddings are available, and also provides a FASTA-only one-hot fallback for quick testing.

---

## Quick Start

### Requirements

| Dependency | Recommended Version |
|------------|---------------------|
| Python | >= 3.8 |
| PyTorch | >= 1.13 |
| NumPy | latest stable |
| pandas | latest stable |
| matplotlib | latest stable |

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the bundled example:

```bash
bash run.sh
```

Or run the same command manually:

```bash
python rpcontact_cli.py \
  -i examples/rpcontact_example_input.zip \
  -o output \
  --top-k 100
```

The example ZIP contains `pro.fasta`, `rna.fasta`, `pro.npy`, and `rna.npy`, so it runs the embedding checkpoint.
By default, RPcontact exports the top 100 predicted contact pairs and the full contact probability matrix.

---

## Input Format

RPcontact CLI accepts either an input directory or a ZIP file.

Directory example:

```text
input/
├── pro.fasta
├── rna.fasta
├── pro.npy      # optional
└── rna.npy      # optional
```

ZIP example:

```bash
zip -r input.zip input/
```

If optional embedding files are missing, RPcontact runs the one-hot model. If embedding files are present but cannot be used, RPcontact prints a `WARNING`, records the reason in `rpcontact_summary.json`, and falls back to the one-hot model.

---

## Typical Usage

Run prediction from an input directory:

```bash
python rpcontact_cli.py -i input/ -o output
```

Run prediction from an input ZIP:

```bash
python rpcontact_cli.py -i input.zip -o output
```

Export the top 500 contacts after threshold filtering:

```bash
python rpcontact_cli.py -i input.zip -o output --threshold 0.5 --top-k 500
```

Use GPU:

```bash
python rpcontact_cli.py -i input.zip -o output --device cuda:0
```

Generate a full-resolution heatmap PNG:

```bash
python rpcontact_cli.py -i input.zip -o output --plot-heatmap
```

Keep the NumPy score matrix and diagnostic plots:

```bash
python rpcontact_cli.py -i input.zip -o output --save-npy --plots
```

Heatmap PNG generation is optional. The full article-style score map is always written to `rpcontact_full_map.txt`.

---

## Common Parameters

| Parameter | Description |
|-----------|-------------|
| `-i`, `--input` | Input directory or ZIP containing FASTA files and optional `.npy` embeddings |
| `-o`, `--out-dir` | Output directory |
| `--threshold` | Minimum score for top-contact filtering |
| `--top-k` | Number of top contacts to export |
| `--pro-chunk` | Protein block size for score writing |
| `--rna-chunk` | RNA block size for score writing |
| `--plot-heatmap` | Write a full-resolution contact-map PNG |
| `--save-npy` | Keep `rpcontact_scores.npy` in the output directory and result ZIP |
| `--plots` | Generate diagnostic top-score distribution plots |
| `--device` | Torch device, such as `cpu` or `cuda:0` |

---

## Outputs

Main outputs:

| File | Description |
|------|-------------|
| `rpcontact_full_map.txt` | Article-style contact score matrix with RNA rows and protein columns |
| `rpcontact_top_pairs.csv` | Top contacts ranked by score |
| `rpcontact_summary.json` | Run settings and score summary |
| `rpcontact_results.zip` | Compressed result bundle |

Optional files can be generated with `--save-npy`, `--plots`, and `--plot-heatmap`.

Index note: nucleotide and residue labels use 1-based indexing.


## Embedding Preparation

The bundled example already contains embeddings and can be run immediately.

For new sequences, FASTA files alone are sufficient. To run the embedding-based model, generate `rna.npy` and `pro.npy` with the helper scripts below and place them next to the matching FASTA files.

### RNA Embedding With ERNIE-RNA

This release includes a small ERNIE-RNA helper that reads FASTA records and writes one `.npy` file per RNA sequence:

```text
scripts/extract_rna_embedding.py
```

Copy it into your ERNIE-RNA directory without overwriting ERNIE-RNA's original scripts:

```bash
cp scripts/extract_rna_embedding.py /path/to/ERNIE-RNA/
cd /path/to/ERNIE-RNA

MKL_THREADING_LAYER=GNU python extract_rna_embedding.py \
  --seqs_path /path/to/rna_sequences.fasta \
  --save_path /path/to/input/ \
  --device cpu
```

For a FASTA record named `>RNA_ID`, the script writes:

```text
/path/to/input/RNA_ID.npy
```

Rename or copy the selected RNA embedding to the final RPcontact input name:

```bash
cp /path/to/input/RNA_ID.npy input/rna.npy
```

### Protein Embedding With ESM-2

This release also includes a small ESM-2 helper that runs ESM extraction and writes one `.npy` file per protein sequence:

```text
scripts/extract_pro_embedding.py
```

```bash
python scripts/extract_pro_embedding.py \
  --seqs_path /path/to/protein_sequences.fasta \
  --save_path /path/to/input/ \
  --esm_dir /path/to/esm
```

For a FASTA record named `>PROTEIN_ID`, the script writes:

```text
/path/to/input/PROTEIN_ID.npy
```

Rename or copy the selected protein embedding to the final RPcontact input name:

```bash
cp /path/to/input/PROTEIN_ID.npy input/pro.npy
```

Then run:

```bash
python rpcontact_cli.py -i input/ -o output
```

You can also compress the input directory as shown in the Input Format section.

---

## Contact & Citation

Questions or suggestions? Contact:

- Jiuhong Jiang
- Email: jiangjh2023@shanghaitech.edu.cn

If you find this project helpful, please cite:

Jiang, J., Zhang, X., Zhan, J., Miao, Z., & Zhou, Y. (2025). RPcontact: Improved prediction of RNA-protein contacts using RNA and protein language models. bioRxiv, 2025-06.

---

<p align="center"><em>Make RNA-protein contact prediction easier and more reproducible.</em></p>
