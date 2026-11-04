# Training RPcontact with alternative language models

Train the downstream contact predictor using fixed RNA and protein language-model
embeddings. RNA/protein embedding dimensions are inferred automatically, allowing
either or both language models to be replaced.

- **Batch size > 1:** set `--batchsize 16`, `32`, or another positive integer.
- **Padding masks:** variable-length RNA/protein sequences are padded within each batch.
  Self- and cross-attention mask padded keys; BCE excludes padded and missing labels.
  Dense padded tensors still require memory.
- **Alternative language models:** supply per-nucleotide RNA and per-residue protein
  embeddings; language-model extraction and fine-tuning are separate steps.

## Install

Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

## Run the bundled example

`examples/` includes `pairs.tsv` and all referenced `.npy` files: one-hot features,
synthetic embeddings and contact labels. The five pairs comprise two training pairs
of different lengths and one pair each for validation, test and TS2. Labels include
`-999` missing positions. RNA/protein embedding channels are 8/12 to demonstrate
automatic dimension inference. These are synthetic inputs for testing the workflow.

Run from this `training/` directory:

```bash
python prepare_data.py --manifest examples/pairs.tsv --output prepared/example
python train.py \
  --train-data prepared/example/train.pickle \
  --val-data prepared/example/val.pickle \
  --test-data prepared/example/test.pickle \
  --ts2-data prepared/example/ts2.pickle \
  --output runs/example \
  --batchsize 2 --max-epochs 2 --device cpu --threads 2
```

Use new output paths when rerunning. Replace the example arrays and TSV with your
own data for actual training.

## Prepare data

Save embeddings as `.npy` arrays of shape `[sequence length, embedding dimension]`.
Remove special tokens and align features to nucleotide/residue positions. Supply RNA
one-hot `[RNA length, 4]`, protein one-hot `[protein length, 20]`, and contact labels
`[RNA length, protein length]`. Preserve the one-hot column order across all splits.

Create a TSV following [examples/pairs.tsv](examples/pairs.tsv):

```text
pairid  split  rna_onehot  protein_onehot  rna_embedding  protein_embedding  label
```

Columns are tab-separated; paths are relative to the TSV or absolute. Use `train`,
`val`, and optionally `test`/`ts2` splits. Embedding columns contain raw LM features;
the helper adds one-hot once. `-` selects one-hot-only input for that side.
All splits must use consistent RNA/protein channel dimensions and distinct pair IDs.

### Save labels

Save one float32 `.npy` label matrix per RNA–protein pair. Rows follow the RNA
nucleotide order; columns follow the protein residue order, matching the embeddings.
Use `1` for contact, `0` for non-contact and `-999` for unknown/unavailable labels:

```python
from pathlib import Path
import numpy as np

Path("data/labels").mkdir(parents=True, exist_ok=True)
label = np.array([[1, 0, -999, 0],
                  [0, 1,    0, 0],
                  [0, 0,    1, 0]], dtype=np.float32)  # 3 RNA bases × 4 residues
np.save("data/labels/pair_train.npy", label)
```

For `data/pairs.tsv`, set that pair's `label` column to `labels/pair_train.npy`.
The directory name is arbitrary: relative paths resolve from the TSV's directory;
absolute paths are also supported. Labels are supplied by the user, not inferred
from the embeddings.

`-999` is excluded from training/validation/test BCE and from validation-checkpoint
auROC selection and final auROC/auPR evaluation. Batch padding also uses `-999`.
Real sequence positions with missing labels remain available to attention; sequence
padding is masked separately. Each pair must retain at least one labelled position.
Soft labels in [0, 1] are accepted and thresholded at `> 0.5` for both BCE targets
and classification metrics.

```bash
python prepare_data.py --manifest data/pairs.tsv --output prepared/new_lm
```

For distance labels in Å, add `--label-format distances` to apply `sigmoid(5-distance)`.
The helper validates lengths and automatically determines input projection dimensions.
Train a fresh predictor when changing LM dimensions.

## Train

```bash
python train.py \
  --train-data prepared/new_lm/train.pickle \
  --val-data prepared/new_lm/val.pickle \
  --output runs/new_lm_batch16 \
  --batchsize 16 --device cuda:0
```

The default optimizer is RMSprop (learning rate 1e-4), with warm-up and cosine warm
restarts. Loss is averaged within each complex and then across the batch. Validation
BCE controls early stopping (`--patience 3` by default). After training, validation
Global auROC selects `selected.pt`. Optional `--test-data` and `--ts2-data` specify
final evaluation datasets; they do not participate in training or checkpoint selection.
Use a new output directory for each run. See `python train.py --help` for all options.

Outputs include per-epoch losses in `history.csv`, selected weights, final metrics, total
time and peak PyTorch allocated/reserved GPU memory. The memory report excludes CUDA
context and non-PyTorch allocations. Full real-data batch-1 training and a mixed-length
batch-16 training smoke test passed on an H100; a full batch-16 benchmark was not run
for this release package.
