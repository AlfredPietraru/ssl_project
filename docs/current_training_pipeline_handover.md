# Current Training Pipeline Handover

## Purpose

This document describes the code that is currently used by the project entrypoint, how data moves through the pipeline, what the important implementation decisions are, and which caveats are already known.

At the time of writing, the active entrypoint is `main.py`. Older numbered scripts (`02_*` through `11_*`) were part of an earlier pipeline and are no longer used.

## Active Files

The current training and evaluation flow uses these files:

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

Supporting directories used by the active pipeline:

- `artifacts/`
- `data/`
- `images/`
- `metadata.sqlite`
- `metadata.csv`

## Entry Point

The current entrypoint is `main.py`.

High-level flow:

1. Select one species from `TRAIN_SPECIES`.
2. Build a train/validation identity split with `DataFetcher.get_train_split_animal(...)`.
3. Build PK-style datasets and dataloaders with `build_pk_train_val_datasets_and_loaders(...)`.
4. Instantiate `ContrastiveEmbeddingModel`.
5. Train with `EmbeddingModelTrainer`.
6. Rebuild simple image-level dataloaders with `build_simple_train_val_datasets_and_loaders(...)`.
7. Extract embeddings for validation and train.
8. Run clustering evaluation with `ClusterAndCompare`.

The pipeline currently trains on `salamander` because `TRAIN_SPECIES[1]` is selected in `main.py`.

## Configuration

Runtime configuration is loaded through `CFG` in `config.py`, backed by `config.yaml`.

Important fields currently used by the active code:

- `image_size`
- `batch_size`
- `instances_per_identity`
- `num_workers`
- `epochs`
- `early_stopping_patience`
- `lr`
- `weight_decay`
- `temperature`
- `embedding_checkpoint_path`

Important semantic detail:

- `batch_size` in the PK dataloader means the number of identity groups per batch, not the total number of images after expansion.
- Effective image count per PK batch is:
  - `batch_size * instances_per_identity * 2`
  - the extra factor of `2` comes from the current "heavy view + light view" duplication in `MyAnimalDatasetPK`.

## Data Access and Split Logic

### Database Access

`data_fetcher.py` reads from `metadata.sqlite` using SQLite.

The currently used method is:

- `DataFetcher.get_train_split_animal(animal, split_size=0.8, random_seed=42)`

### What `get_train_split_animal(...)` does

1. Query all training rows for the requested species.
2. Group image paths by identity using `json_group_array(...)`.
3. Convert identity strings to stable integer ids.
4. Shuffle identities.
5. Split identities into train and validation identity sets.
6. Log how many singleton identities exist in each split.
7. Remove singleton identities from the training split only.

Current important behavior:

- train identities with exactly one image are filtered out
- validation identities with exactly one image are kept

This means the train and validation splits are not symmetric:

- train split is optimized for metric learning
- validation split remains more realistic, but also harder to evaluate cleanly

### Why this matters

Metric-learning losses need positive pairs. An identity with one image provides no positive pair. That is why singleton identities are removed from training.

However, singleton identities still exist in validation and can distort downstream metrics, especially retrieval and clustering interpretation.

## Dataset Layer

There are two dataset families in `animal_dataset.py`.

### 1. `MyAnimalDatasetPK`

This dataset is used for training and for validation loss computation.

Each item represents one identity group, not one image.

`__getitem__(index)` currently does this:

1. Read one identity record from grouped metadata.
2. Randomly sample `instances_per_identity` image paths from that identity.
3. For each sampled image:
   - apply `self.transform(...)` to create a heavy/training-style view
   - apply `self.light_transform(...)` to create a light/test-style view
4. Append both views to the image list.
5. Return:
   - `stacked_images` with shape `[2 * instances_per_identity, C, H, W]`
   - a single scalar identity label

Important semantic detail:

- labels are not expanded inside `__getitem__`
- labels are expanded inside the custom collate function

The PK collate function:

1. concatenates all image groups from the batch
2. repeats each scalar identity label to match the number of returned images for that identity group

So if one dataset item returns 8 images and identity `42`, the collate function turns that into:

- 8 images
- 8 labels, all equal to `42`

### 2. `MyAnimalDatasetSimple`

This dataset is used for honest embedding extraction after training.

Each item corresponds to one real image only:

- one file path
- one label
- one transformed tensor

It does not perform PK grouping and does not duplicate views.

This is the correct dataset type for:

- embedding extraction
- image-level retrieval evaluation
- clustering evaluation

### Current Loader Builders

`build_pk_train_val_datasets_and_loaders(...)`

- returns PK train and validation datasets/loaders
- current validation PK loader still uses grouped identities and duplicated views

`build_simple_train_val_datasets_and_loaders(...)`

- returns image-level train and validation datasets/loaders
- used after training for embedding extraction and clustering

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

The PK dataset currently uses both transforms on the same sampled images:

- heavy transform
- light transform

This doubles the number of views seen by the loss.

## Model

Defined in `model.py`.

### Backbone

The model uses MegaDescriptor through timm / Hugging Face:

- model id: `hf-hub:BVRA/MegaDescriptor-T-CNN-288`

### Loading logic

`load_mega_descriptor_model_feature_extraction(weights_path)`

Behavior:

- if `weights_path is None`
  - construct the timm model and download/load pretrained Hub weights
- if `weights_path` is not `None`
  - load weights only from that local path
  - raise `FileNotFoundError` if the file does not exist

Important caveat:

- even when local weights are used, `timm.create_model("hf-hub:...")` may still touch Hugging Face config resolution
- this means some Hugging Face HTTP requests can still appear in logs

### Current forward pass

`ContrastiveEmbeddingModel.forward(images)` does not use classifier logits.

It currently extracts:

1. `features = self.backbone.forward_features(images)`
2. `self.backbone.forward_head(features, pre_logits=True)`

This yields a 1536-dimensional embedding-like representation instead of the older 1000-dimensional final output.

### Checkpoint saving

`ContrastiveEmbeddingModel.save_checkpoints(...)` currently saves only:

- `self.state_dict()`

It does not save optimizer state or a full training-state checkpoint.

This was done intentionally to make loading simple:

```python
model = ContrastiveEmbeddingModel().to(device)
model.load_state_dict(torch.load(path, map_location=device))
```

## Loss Functions

Defined in `loss_functions.py`.

Two loss implementations exist:

- `SupervisedContrastiveLoss`
- `BatchHardTripletLoss`

### Current active loss

`trainer.py` currently instantiates:

- `BatchHardTripletLoss()`

SupCon is still present in code but commented out in the trainer.

### Triplet loss behavior

`BatchHardTripletLoss`:

1. normalizes features
2. computes pairwise cosine-derived distance `1 - cosine_similarity`
3. finds hardest positive for each anchor
4. finds hardest negative for each anchor
5. applies margin ranking with `relu(hardest_positive - hardest_negative + margin)`

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

These are logged during training and validation.

## Training Loop

Defined in `trainer.py`.

### Trainer responsibilities

`EmbeddingModelTrainer`:

- owns the loss function
- owns the optimizer
- runs train/validation epochs
- applies mixed precision with `torch.autocast`
- uses a CUDA grad scaler when CUDA is available
- performs early stopping
- writes a loss plot through `HarryPlotter`

### Current optimization choices

- optimizer: `AdamW`
- no scheduler currently
- current loss: `BatchHardTripletLoss`

### Early stopping

Training saves the best checkpoint whenever:

- `validation_loss < best_loss`

It then stops after `early_stopping_patience` epochs without improvement.

### Current validation behavior

`_validate_one_epoch(...)` currently:

1. runs the model over the PK validation loader
2. computes validation loss from PK batches
3. collects embeddings and labels from those PK batches
4. computes Recall@1 and Recall@5 from those collected PK-batch embeddings

This is an important caveat.

## Retrieval Metric Caveat

The current validation recall metric is not an honest gallery-style retrieval metric.

Why:

- validation uses the PK dataset
- each identity contributes multiple images per batch
- each sampled image is duplicated into heavy and light views
- positives are guaranteed and often extremely easy

As a result, validation Recall@1 and Recall@5 can be artificially inflated, including reaching `1.0`.

This should not be interpreted as true retrieval performance.

### Recommended future evaluation split

Use three distinct loaders:

1. `train_pk_loader`
   - training only
2. `val_pk_loader`
   - validation loss only
3. `val_simple_loader`
   - retrieval / clustering / embedding evaluation

The code already partially follows this pattern after training by rebuilding simple loaders before clustering.

However, the recall currently printed during training still comes from the PK validation loader, so those recall values should be treated as biased.

## Embedding Extraction

`EmbeddingModelTrainer.get_embeddings(trained_model, loader)`:

1. switches the model to eval mode
2. runs it over the provided loader
3. collects embeddings and labels on CPU
4. concatenates the results

This is the method used by `main.py` after rebuilding simple train/validation loaders.

## Clustering Evaluation

Defined in `cluster_and_compare.py`.

### Current clustering implementation

The project currently uses HDBSCAN, not DBSCAN.

The class:

- normalizes embeddings row-wise
- computes similarity with matrix multiplication
- converts similarity to distance with `1.0 - similarity`
- runs HDBSCAN on the precomputed distance matrix

### Current sweep

`sweep_eps(...)` is a legacy method name. It now sweeps:

- `min_cluster_size = 2`
- `min_cluster_size = 3`

The parameter name remained from earlier DBSCAN work.

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

The reported `accuracy` is a majority-vote cluster purity style diagnostic:

1. map each predicted cluster to its dominant identity
2. compute per-sample agreement with the true identity

It is useful as a quick heuristic, but ARI is the proper clustering metric.

## Current Known Issues

### 1. Validation retrieval is currently too easy

Because validation recall is computed from PK batches with duplicated views, it does not reflect realistic retrieval difficulty.

### 2. Validation still contains many singleton identities

Current behavior:

- singleton train identities are removed
- singleton validation identities are kept

This makes validation more realistic, but it also makes metric-learning evaluation harder to interpret because singleton identities have no true positive pair at image level.

### 3. Clustering remains weak

Even after moving from SupCon to triplet loss and from DBSCAN to HDBSCAN, ARI remains near zero.

This currently suggests:

- the space has some local ranking structure
- but it is not strongly clusterable as a global identity space

### 4. File naming legacy

Some function names reflect earlier experiments:

- `sweep_eps(...)` now sweeps HDBSCAN `min_cluster_size`

This is not wrong functionally, but the name is outdated.

## Recommended Cleanup and Next Steps

If the next contributor continues the project, these are the highest-value follow-ups.

### 1. Separate validation-loss and validation-metric loaders

Keep:

- PK validation for loss

Use simple image-level validation for:

- Recall@1
- Recall@5
- clustering
- similarity diagnostics

### 2. Add a "seen-identity" validation split

Current validation is identity-disjoint from train.

That is good for generalization, but harsh for debugging.

Recommended future split design:

- `val_seen`: same identities as train, different images
- `val_unseen`: current unseen-identity split

### 3. Rename PK batch config fields

Current naming is confusing:

- `batch_size` really means identities per batch in the PK dataset

Recommended future rename:

- `identities_per_batch`

Keep:

- `instances_per_identity`

### 4. Rename `sweep_eps(...)`

Recommended future rename:

- `sweep_hdbscan(...)`
- or `sweep_min_cluster_size(...)`

### 5. Revisit view duplication in PK dataset

The current PK dataset doubles every sampled image into:

- heavy view
- light view

This may help invariance, but it also makes validation PK batches especially easy and changes the effective batch semantics.

## Minimal Reproduction of the Current Flow

The current project logic can be summarized as:

1. Read grouped train identities from SQLite.
2. Convert identity strings to integer ids.
3. Split identities into train and validation.
4. Remove singleton identities from training.
5. Train a MegaDescriptor backbone with batch-hard triplet loss on PK batches.
6. Save the best model weights only.
7. Rebuild simple image-level loaders.
8. Extract embeddings.
9. Cluster those embeddings with HDBSCAN.

## Handover Notes for a New Contributor

If someone new takes over the project, the quickest path to productive work is:

1. Start from `main.py`.
2. Read `data_fetcher.py` to understand the identity split and singleton filtering.
3. Read `animal_dataset.py` to understand the difference between PK and simple datasets.
4. Read `trainer.py` to understand which metrics are currently trustworthy and which are not.
5. Read `model.py` to understand how MegaDescriptor is loaded and which representation is used.
6. Read `cluster_and_compare.py` to understand the current clustering evaluation.

Before interpreting any new experiment, verify:

- whether the metric came from a PK loader or a simple loader
- whether singleton identities were included in the evaluated split
- whether the embeddings came from the trained checkpoint or the plain pretrained model

## Removed Files

The following obsolete numbered scripts were removed because they are no longer part of the active pipeline:

- `02_extract_embeddings.py`
- `03_04_build_gallery_and_validation.py`
- `05_calibrate_rejection_thresholds.py`
- `06_run_nearest_neighbor_matching.py`
- `07_reject_low_confidence_matches.py`
- `08_cluster_rejected_unknowns.py`
- `09_refine_with_incremental_enrollment.py`
- `10_export_final_assignments.py`
- `11_verify_train_clustering.py`
