# FGFR3MUT screening model

![FGFR3 workflow](./assets/figure.png)

This is an official implementation of the FGFR3MUT model. This model is intended for pre-screening tumors of muscle invasive bladder cancer (MIBC) harboring an FGFR3 mutation.

It provides the user with:

1. The model implementation (Chowder)
2. An inference script to replicate the experiments on TCGA

## Data Availability

We provide a script `download.py` which downloads the Chowder weights of the pre-screening model, as well as the slide features of `TCGA-BLCA` extracted with [Bioptimus' H0 extractor](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0?utm_source=owkin&utm_medium=referral&utm_campaign=h-bioptimus-o).

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

```shell
$ python download.py --out_dir <OUTPUT_DIR>
```

Note that the `TCGA-BLCA` slide features weigh more than 200GB. In comparison, the actual Chowder weights weigh 50MB.
This should take a few minutes depending on your network connection.

Now, you can run the experiments reported in the manuscript. The actual computations should take ten minutes to run on CPU and only a few seconds on GPU.

### External validation on TCGA-BLCA (MIBC cases only)

```shell
$ python fgfr3mut/run_inference.py \
    --data_dir <OUTPUT_DIR> \
    --device cpu \  # `cuda:0` if you have a GPU
    --n_tiles 5000 \
    --keep_tcga_cases MIBC
```

Inference on: n_slides=379, n_patients=308.

You should obtain the following results, reported in the publication:

| Metric      | Value              |
| ----------- | -------------------|
| AUC         | 0.82 [0.75 - 0.88] |
| Sensitivity | 0.95 [0.92 - 1.00] |
| Specificity | 0.47 [0.30 - 0.72] |
| PPV         | 0.14 [0.09 - 0.23] |
| NPV         | 0.99 [0.98 - 1.00] |

### Comparison with Loeffler et al.

To run the external validation on the exact cases reported by [Loeffler et al.](<https://eu-focus.europeanurology.com/article/S2405-4569(21)00113-9/fulltext>), run:

```shell
$ python fgfr3mut/run_inference.py \
    --data_dir <OUTPUT_DIR> \
    --device cpu \  # `cuda:0` if you have a GPU
    --n_tiles 5000 \
    --loeffler_tcga_cases
```

Inference on: n_slides=391, n_patients=327.

You should obtain the following results, reported in the publication:

| Metric      | Value              |
| ----------- | ------------------ |
| AUC         | 0.83 [0.77 - 0.88] |
| Sensitivity | 0.94 [0.92 - 0.98] |
| Specificity | 0.44 [0.24 - 0.58] |
| PPV         | 0.23 [0.17 - 0.30] |
| NPV         | 0.98 [0.96 - 0.99] |
