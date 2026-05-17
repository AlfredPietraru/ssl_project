# Current Training Pipeline

## Purpose

This project focuses on animal re-identification in an open-set setting.

The goal is not simple classification. The system should learn embeddings that:

- keep images of the same individual close together
- separate different individuals
- remain useful when test images include previously unseen individuals

The current codebase is centered on contrastive / metric-learning style training followed by embedding extraction and clustering analysis.

## Current Status

The active pipeline is the Python flow driven by `main.py`.

Older numbered scripts (`02_*` through `11_*`) belonged to an earlier experimental pipeline and are no longer part of the current repo workflow.

## Active Files

The current training and evaluation flow uses:

- `main.py`
- `config.py`
- `config.yaml`
- `data_fetcher.py`
- `animal_dataset.py`
- `transformations.py`
- `model.py`
- `loss_functions.py`
- `trainer.py`
- `cluster_and_compare.py`
- `main_utils.py`

Supporting project assets used by the active flow:

- `artifacts/`
- `data/`
- `images/`
- `metadata.sqlite`
- `metadata.csv`

## Entry Point

The current entry point is:

```bash
python3 main.py
```

High-level flow:

1. Select one species from `TRAIN_SPECIES`.
2. Build an identity-level train/validation split with `DataFetcher.get_train_split_animal(...)`.
3. Build PK-style train and validation datasets with `build_pk_train_val_datasets_and_loaders(...)`.
4. Instantiate `ContrastiveEmbeddingModel`.
5. Train with `EmbeddingModelTrainer`.
6. Rebuild simple image-level train and validation datasets with `build_simple_train_val_datasets_and_loaders(...)`.
7. Extract embeddings for validation and train.
8. Run clustering diagnostics with `ClusterAndCompare`.

At the moment, `main.py` explicitly selects `TRAIN_SPECIES[1]`, so the active run trains on `salamander`.

## Problem Framing

The competition setting is closer to discovery and re-identification than standard closed-set classification.

Important implications:

- training uses known identities
- test-time behavior must still be meaningful for unseen identities
- clustering quality matters, not just top-1 classification

This is why the project uses embedding learning and clustering analysis instead of a plain classifier head as the main workflow.

## Configuration

Runtime configuration is loaded through `CFG` in `config.py`, backed by `config.yaml`.

Important fields currently used by the active code:

- `image_size`
- `batch_size`
- `instances_per_identity`
- `num_workers`
- `epochs`
- `early_stopping_patience`
- `early_stopping_min_delta`
- `lr`
- `weight_decay`
- `temperature`
- `embedding_checkpoint_path`

Important semantic detail:

- `batch_size` in the PK dataloader means the number of identity groups per batch
- it is not the final number of images after PK expansion

Effective image count per PK batch is:

```text
batch_size * instances_per_identity * 2
```

The extra factor of `2` comes from the current heavy-view plus light-view duplication in `MyAnimalDatasetPK`.

## Data Access and Split Logic

`data_fetcher.py` reads metadata from `metadata.sqlite`.

The active splitting method is:

- `DataFetcher.get_train_split_animal(animal, split_size=0.8, random_seed=42)`

What it does:

1. Query all training rows for one species.
2. Group image paths by identity using `json_group_array(...)`.
3. Convert identity strings to integer ids.
4. Shuffle identities.
5. Split identities into train and validation groups.
6. Log how many singleton identities exist in each split.
7. Remove singleton identities from training only.

Current consequences:

- training identities with one image are filtered out
- validation identities with one image are kept

This makes training compatible with metric learning, but validation harder to interpret because singleton identities cannot form positive pairs.

## Dataset Layer

There are two dataset families in `animal_dataset.py`.

### `MyAnimalDatasetPK`

Used for:

- training
- validation loss computation during training

Each item is one identity group, not one image.

Current item behavior:

1. Read one identity record.
2. Sample `instances_per_identity` image paths from that identity.
3. For each sampled image, build:
   - one heavy training-style view
   - one light evaluation-style view
4. Return a stacked tensor plus a single scalar identity label.

The collate function expands labels so each returned image receives the identity label.

### `MyAnimalDatasetSimple`

Used for:

- image-level embedding extraction
- retrieval-style evaluation
- clustering evaluation

Each item corresponds to one real image only:

- one path
- one label
- one transformed tensor

It does not group by identity and does not duplicate views.

### Current Loader Builders

`build_pk_train_val_datasets_and_loaders(...)`

- returns PK train and validation datasets/loaders
- validation still uses grouped identities and duplicated views

`build_simple_train_val_datasets_and_loaders(...)`

- returns image-level train and validation datasets/loaders
- used after training before embedding extraction and clustering

## Transformations

Defined in `transformations.py`.

### Training transform

`build_cpu_training_transform(image_size)` currently applies:

- resize
- `ToTensor`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomGrayscale`
- ImageNet normalization

### Testing transform

`build_cpu_testing_transform(image_size)` currently applies:

- resize
- `ToTensor`
- ImageNet normalization

### Important note

The PK dataset applies both transforms to the same sampled images:

- a heavy transform
- a light transform

This doubles the number of views seen by the training loss.

## Model

Defined in `model.py`.

### Backbone

The current backbone is MegaDescriptor via timm / Hugging Face:

- `hf-hub:BVRA/MegaDescriptor-T-CNN-288`

### Loading logic

`load_mega_descriptor_model_feature_extraction(weights_path)` behaves as follows:

- if `weights_path is None`, build the model and load pretrained Hub weights
- if `weights_path` is provided, load weights from that local path

Even with local weights, `timm.create_model("hf-hub:...")` may still touch Hugging Face configuration resolution.

### Forward pass

`ContrastiveEmbeddingModel.forward(images)` uses:

1. `self.backbone.forward_features(images)`
2. `self.backbone.forward_head(features, pre_logits=True)`

This yields a 1536-dimensional embedding-like representation instead of classifier logits.

### Checkpoint saving

`ContrastiveEmbeddingModel.save_checkpoints(...)` currently saves only:

- `self.state_dict()`

It does not save optimizer state or a full resumable training checkpoint.

## Loss Functions

Defined in `loss_functions.py`.

Two implementations exist:

- `SupervisedContrastiveLoss`
- `BatchHardTripletLoss`

### Current active loss

`trainer.py` currently uses:

- `BatchHardTripletLoss()`

SupCon remains in the codebase but is commented out in the trainer.

### Triplet loss behavior

`BatchHardTripletLoss`:

1. normalizes embeddings
2. computes pairwise cosine-derived distances
3. finds the hardest positive for each anchor
4. finds the hardest negative for each anchor
5. applies a margin ranking loss

### Diagnostics

Both loss classes expose `.diagnostics(...)`.

Triplet diagnostics currently include:

- `valid_anchor_ratio`
- `mean_positive_count`
- `mean_positive_distance`
- `mean_negative_distance`
- `distance_gap`
- `mean_hardest_positive`
- `mean_hardest_negative`
- `hard_margin_gap`
- `active_triplet_ratio`

These diagnostics are logged during training and validation.

## Training Loop

Defined in `trainer.py`.

`EmbeddingModelTrainer` is responsible for:

- loss creation
- optimizer creation
- training and validation epochs
- mixed precision with `torch.autocast`
- CUDA grad scaling when CUDA is available
- early stopping
- loss plotting through `HarryPlotter`

Current optimization choices:

- optimizer: `AdamW`
- scheduler: none
- loss: `BatchHardTripletLoss`

### Early stopping

The trainer saves a checkpoint whenever:

```text
validation_loss < best_loss
```

Training stops after `early_stopping_patience` epochs without improvement.

## Validation Caveat

The validation recall printed during training is not a clean image-level retrieval metric.

Why:

- validation uses the PK dataset
- identities contribute multiple images per batch
- sampled images are duplicated into heavy and light views
- positives are guaranteed and often unusually easy

As a result, validation `Recall@1` and `Recall@5` can be overly optimistic and should not be treated as honest retrieval performance.

## Embedding Extraction

`EmbeddingModelTrainer.get_embeddings(trained_model, loader)`:

1. switches the model to eval mode
2. runs the model over the given loader
3. collects embeddings and labels on CPU
4. concatenates the results

This method is used by `main.py` after rebuilding simple train and validation loaders.

## Clustering Evaluation

Defined in `cluster_and_compare.py`.

The current implementation uses HDBSCAN, not DBSCAN.

Flow:

1. normalize embeddings row-wise
2. compute pairwise cosine similarity
3. convert similarity to distance with `1.0 - similarity`
4. run HDBSCAN on the precomputed distance matrix

### Current sweep

`sweep_eps(...)` is a legacy name from earlier DBSCAN work.

It currently sweeps:

- `min_cluster_size = 2`
- `min_cluster_size = 3`

### Metrics returned

- `accuracy`
- `adjusted_rand_index`
- `num_samples`
- `num_clusters`
- `num_noise`
- `cluster_labels`
- `cluster_to_identity`

### Metric interpretation

The most trustworthy clustering metric is:

- `adjusted_rand_index`

The reported `accuracy` is a quick majority-vote purity-style diagnostic, not the main clustering objective.

## Known Issues and Caveats

### 1. Validation retrieval is too easy

Validation recall is computed from PK batches with duplicated views, so it is biased upward.

### 2. Validation still includes singleton identities

Training removes singleton identities, but validation keeps them.

That is realistic, but it makes metric-learning evaluation noisier and harder to interpret.

### 3. Clustering remains weak

Even after moving from SupCon to triplet loss and from DBSCAN to HDBSCAN, clustering quality remains limited.

This suggests the embedding space may have some local ranking structure without forming a strong global identity clustering space yet.

### 4. Some naming still reflects older experiments

Examples:

- `sweep_eps(...)` now sweeps HDBSCAN settings
- `batch_size` in PK loading really means identities per batch

## Background Reading and Ideas

Earlier project notes referenced a few useful directions:

- person re-identification baselines using triplet-style objectives
- deep metric learning as the general framing for embedding quality
- open-set recognition ideas for handling unseen identities

These notes are still conceptually useful, but they are not the active implementation plan. The current codebase does not use an OpenMax-style classifier or a lookup-plus-discovery numbered pipeline.

## Recommended Next Steps

Highest-value follow-ups for the next contributor:

1. Separate validation-loss loaders from validation-metric loaders more cleanly.
2. Add a simple image-level validation retrieval evaluation during training.
3. Rename PK configuration fields to make batch semantics clearer.
4. Rename `sweep_eps(...)` to match HDBSCAN behavior.
5. Revisit whether heavy-view plus light-view duplication is helping enough to justify the extra complexity.

## Minimal Reproduction of the Current Flow

The current project logic can be summarized as:

1. Read grouped train identities from SQLite.
2. Convert identity strings to integer ids.
3. Split identities into train and validation.
4. Remove singleton identities from training.
5. Train a MegaDescriptor backbone with batch-hard triplet loss on PK batches.
6. Save model weights.
7. Rebuild simple image-level loaders.
8. Extract embeddings.
9. Run HDBSCAN-based clustering diagnostics.

## Recommended Reading Order for a New Contributor

If someone new takes over the project, the fastest way to build context is:

1. Start with `main.py`.
2. Read `data_fetcher.py` for split behavior and singleton filtering.
3. Read `animal_dataset.py` for the PK versus simple dataset distinction.
4. Read `trainer.py` for training-time metrics and caveats.
5. Read `model.py` for MegaDescriptor loading and embedding extraction behavior.
6. Read `cluster_and_compare.py` for clustering evaluation.
