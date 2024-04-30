# FGFR3MUT screening model

![FGFR3 workflow](./assets/figure.png)

This is an official implementation of the FGFR3MUT model. This model is intended for pre-screening tumors of muscle invasive bladder cancer (MIBC) harboring an FGFR3 mutation.

It provides the user with:

1. The model implementation (Chowder)
2. An inference script to replicate the experiments on TCGA

## Data Availability

We provide a script `download.py` which downloads the Chowder weights of the pre-screening model, as well as the slide features of `TCGA-BLCA` extracted with a private Wide ResNet50 trained on `TCGA-COAD`.

## Instructions

Start by installing the required dependencies:

```bash
# Clone the repo
git clone https://github.com/PABannier/fgfr3mut.git
cd fgfr3mut

# Create a virtual environment with all the requirements
conda create -n fgfr3mut_env python=3.8
conda activate fgfr3mut_env

# Install the `fgfr3mut` package
pip install .
```

Then, download the model weights and the TCGA features from [HuggingFace](https://huggingface.co/datasets/PABannier/fgfr3mut):

```bash
python download.py --out_dir <OUTPUT_DIR>
```

Note that the `TCGA-BLCA` slide features weigh more than 200GB. In comparison, the actual Chowder weights weigh 50MB.
This should take a few minutes depending on your network connection.

Now, you can run the experiments reported in the manuscript. The actual computations should take ten minutes to run on CPU and only a few seconds on GPU.

### External validation on TCGA-BLCA (MIBC cases only)

```bash
python fgfr3mut/run_inference.py \
    --data_dir <OUTPUT_DIR> \
    --device cpu \  # `cuda:0` if you have a GPU
    --n_tiles 5000 \
    --keep_tcga_cases MIBC
```

Inference on: n_slides=379, n_patients=308.

You should obtain the following results, reported in the publication:

| Metric      | Value              |
| ----------- | -------------------|
| AUC         | 0.82 [0.74 - 0.88] |
| Sensitivity | 0.95 [0.92 - 1.00] |
| Sensitivity | 0.46 [0.35 - 0.69] |

### External validation on TCGA-BLCA (MIBC and NMIBC)

```bash
python fgfr3mut/run_inference.py \
    --data_dir <OUTPUT_DIR> \
    --device cpu \  # `cuda:0` if you have a GPU
    --n_tiles 5000 \
    --keep_tcga_cases MIBC NMIBC
```

Inference on: n_slides=436, n_patients=370.

You should obtain the following results, reported in the publication:

| Metric      | Value              |
| ----------- | ------------------ |
| AUC         | 0.86 [0.82 - 0.91] |
| Sensitivity | 0.93 [0.92 - 0.97] |
| Specificity | 0.59 [0.39 - 0.70] |

### Comparison with Loeffler et al.

To run the external validation on the exact cases reported by [Loeffler et al.](<https://eu-focus.europeanurology.com/article/S2405-4569(21)00113-9/fulltext>), run:

```bash
python fgfr3mut/run_inference.py \
    --data_dir <OUTPUT_DIR> \
    --device cpu \  # `cuda:0` if you have a GPU
    --n_tiles 5000 \
    --loeffler_tcga_cases
```

Inference on: n_slides=391, n_patients=327.

You should obtain the following results, reported in the publication:

| Metric      | Value              |
| ----------- | ------------------ |
| AUC         | 0.81 [0.74 - 0.86] |
| Sensitivity | 0.94 [0.92 - 0.98] |
| Specificity | 0.41 [0.12 - 0.61] |
