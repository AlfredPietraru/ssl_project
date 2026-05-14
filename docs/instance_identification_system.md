# Instance Identification System: End-to-End Steps

This document explains what comes after the contrastive-learning stage and how to build a full instance identification system that can also adapt to unseen data.

## 1. What the current contrastive pipeline gives you

Running [`main.py`](/home/alf/ssl/main.py) trains a contrastive model and saves:

- `artifacts/full_model_checkpoints/contrastive_model.pt`
- `artifacts/embedding_checkpoints/embedding_backbone.pt`

That output is **not the final identification system**. It is the feature extractor that should be used by the downstream pipeline.

After training, the next goal is:

1. extract embeddings for all reference and query images
2. compare embeddings in a metric space
3. decide whether each query belongs to a known identity or to a new unseen identity
4. cluster the unseen queries into new identities
5. optionally adapt the system with pseudo-labels or incremental updates

## 2. Define the full problem correctly

This is an **open-set re-identification** problem, not plain classification.

For each test image, the system must answer two questions:

1. Does this image match an identity already known from the reference set?
2. If not, which other unseen test images belong to the same new identity?

That means the final system needs both:

- **retrieval / matching** for known identities
- **discovery / clustering** for unseen identities

## 3. Recommended high-level pipeline

The competition is evaluated with Adjusted Rand Index (ARI), so the exact
numeric suffix in a cluster label is arbitrary. What matters is whether images
of the same individual share one predicted cluster and images of different
individuals are not merged.

The current practical baseline is the direct clustering path:

1. train or fine-tune the embedding backbone with contrastive learning
2. freeze the backbone and use it for embedding extraction
3. cluster all test embeddings per dataset
4. export `image_id,cluster`
5. run train-set diagnostics to check embedding quality

The older lookup + discovery path is still useful for experiments, but it is
more fragile because early known-vs-unknown decisions can split or merge true
test identities.

Current RunPod steps:

1. `02_extract_embeddings.py`
2. `03_cluster_test_embeddings.py`
3. `11_verify_train_clustering.py`

Primary submission artifact:

- `artifacts/final/test_clustering_submission.csv`

Diagnostic artifacts:

- `artifacts/final/test_clustering_report.txt`
- `artifacts/final/test_clustering_summary.json`
- `artifacts/final/train_verification_report.txt`
- `artifacts/final/train_verification_summary.json`

Experimental lookup + discovery scripts:

1. `03_04_build_gallery_and_validation.py`
2. `05_calibrate_rejection_thresholds.py`
3. `06_run_nearest_neighbor_matching.py`
4. `07_reject_low_confidence_matches.py`
5. `08_cluster_rejected_unknowns.py`
6. `09_refine_with_incremental_enrollment.py`
7. `10_export_final_assignments.py`

## 4. Step-by-step implementation plan

### Step 1. Cleanly separate training and inference stages

Treat contrastive training as only the representation-learning stage.

Use:

- [`main.py`](/home/alf/ssl/main.py) to train
- `artifacts/embedding_checkpoints/embedding_backbone.pt` as the inference backbone

Important rule:

- during inference, use `model.eval()`
- use the backbone embeddings, not the projection-head output
- L2-normalize embeddings before similarity search

Why:

- the projection head is useful for contrastive training
- the backbone embedding is usually the representation used for retrieval and re-identification

### Step 2. Build a reference gallery of known identities

Create a gallery from the training split:

1. extract one normalized embedding per training image
2. keep metadata: `image_id`, `identity`, `species`, path
3. group embeddings by identity

For each known identity, store either:

- all image embeddings for that identity, or
- one prototype embedding per identity, computed as the mean of normalized embeddings

Recommended default:

- keep both image-level embeddings and identity prototypes

Why:

- prototypes are fast and stable
- image-level embeddings help when an identity has multiple viewpoints or appearance changes

### Step 3. Partition the space by species first

Because your task includes multiple animal species, perform matching inside species-specific subsets whenever species is known or can be inferred reliably.

Recommended flow:

1. determine species for each query image
2. compare the query only against gallery identities from the same species
3. cluster unknown queries within the same species

Why:

- this reduces false matches across very different animals
- it makes thresholds and clustering more stable

If test data may contain a fourth species not present in train:

- do not force those samples into known training identities
- route them to the unknown / discovery branch

### Step 4. Create a validation setup that simulates unseen identities

This is the most important missing piece if you want adaptation to unseen data.

You need a validation protocol that mimics the real competition setting:

1. split training identities into `known_train`, `known_val`, and `unseen_val`
2. build the gallery using only `known_train`
3. use `known_val` images as queries that should match existing identities
4. use `unseen_val` identities as queries that should be rejected as unknown

This lets you tune:

- similarity thresholds
- unknown-rejection rules
- clustering thresholds
- confidence scores

Without this step, the system will usually overfit to matching everything to a known identity.

### Step 5. Define the matching score

For each query embedding, compute similarity to the gallery.

Recommended first choice:

- cosine similarity on L2-normalized embeddings

For each query, compute:

1. best similarity to any gallery image
2. best similarity to any identity prototype
3. margin between top-1 and top-2 identity scores
4. optional agreement between image-level and prototype-level prediction

These values become your confidence signals.

### Step 6. Add a known-vs-unknown rejection rule

This is the key step that makes the system open-set.

A simple baseline rule:

- assign query to a known identity only if top-1 similarity is above threshold `T_known`
- otherwise label it as `unknown`

A stronger rule:

- accept match only if:
  - top-1 similarity >= `T_known`
  - top1-top2 margin >= `T_margin`
  - query is not far from the identity prototype

You should tune these thresholds on the validation protocol from Step 4.

Possible rejection strategies, from simplest to strongest:

1. fixed cosine threshold
2. class-specific thresholds per species
3. distance to identity prototype mean
4. k-nearest-neighbor vote with confidence
5. EVT / OpenMax-like calibration on distances

Recommended default for this repo:

- start with cosine threshold + top1-top2 margin
- then move to species-specific thresholds if needed

### Step 7. Assign known identities

For queries that pass the rejection rule:

1. assign the identity of the nearest accepted gallery match
2. optionally aggregate multiple gallery neighbors by majority vote or mean similarity

Recommended default:

- use `k=5` nearest neighbors
- score each identity by mean similarity of its top matches
- return the highest-scoring identity if confidence passes threshold

This is usually more robust than relying on a single nearest image.

### Step 8. Cluster the rejected queries into unseen identities

All rejected samples should go to a discovery stage.

Within each species:

1. collect all query embeddings rejected as unknown
2. build a pairwise similarity matrix
3. cluster them into identities

Recommended clustering methods:

1. agglomerative clustering with cosine distance threshold
2. DBSCAN on cosine distance
3. HDBSCAN if cluster sizes are very uneven

Recommended starting point:

- agglomerative clustering with a tuned distance threshold

Why:

- easy to control
- matches the identity-discovery setting well
- no need to predefine number of clusters

Tune clustering threshold on `unseen_val` identities from Step 4.

### Step 9. Merge known matches and discovered clusters

At this point every query belongs to one of two branches:

- known identity from the gallery
- discovered cluster among rejected samples

Create one unified output format:

1. keep original known identity IDs for accepted matches
2. create synthetic IDs for new discovered identities, for example:
   - `new_lynx_0001`
   - `new_salamander_0007`

This gives a consistent final identity assignment for the full test set.

### Step 10. Add adaptation to unseen data

If you want the system to adapt beyond static thresholding, add one or both of these strategies.

#### Strategy A. Incremental enrollment

After clustering unknown samples:

1. compute one prototype for each discovered cluster
2. add high-confidence clusters back into the gallery
3. re-run matching for any unassigned or low-confidence query images

This helps when test identities appear in multiple batches or views.

Use this only for clusters that satisfy:

- minimum cluster size
- high internal similarity
- clear separation from other clusters

#### Strategy B. Self-training with pseudo-labels

1. take high-confidence known matches and high-purity discovered clusters
2. treat them as pseudo-labeled data
3. fine-tune the embedding model for a few more epochs
4. re-extract embeddings and rerun the full pipeline

This can improve adaptation, but it also risks reinforcing mistakes.

Recommended safeguards:

- only use very high-confidence pseudo-labels
- use a small learning rate
- keep early stopping
- compare against the non-adaptive baseline

## 5. Concrete “next steps” right after your current training run

If you have already finished the contrastive run, the next actions should be:

1. load `artifacts/embedding_checkpoints/embedding_backbone.pt`
2. extract normalized embeddings for all train images
3. extract normalized embeddings for validation / test images
4. build a gallery of known identities from train
5. create a held-out validation split with fake unseen identities
6. tune:
   - known-match threshold
   - top1-top2 margin threshold
   - clustering distance threshold
7. run inference:
   - known identity matching first
   - unknown rejection second
   - clustering of rejected samples third
8. evaluate errors separately:
   - false known matches
   - false unknown rejections
   - cluster fragmentation
   - cluster merging
9. only after that consider self-training / incremental adaptation

If you skip threshold calibration and clustering, you do not yet have a full instance identification system.

## 6. Evaluation protocol you should use

Do not evaluate only with training loss or PCA plots.

You should track at least:

1. retrieval accuracy for known identities
2. known-vs-unknown rejection accuracy
3. clustering quality for unseen identities
4. end-to-end assignment quality

Useful metrics:

- top-1 retrieval accuracy
- top-k retrieval accuracy
- ROC / PR for known-vs-unknown detection
- false accept rate of unknowns
- false reject rate of knowns
- NMI / ARI / pairwise F1 for clustering
- final competition metric if available

## 7. Design decisions that matter most

### A. Use the backbone embeddings, not the projection head

Your current training setup saves the backbone separately for exactly this reason.

### B. Normalize embeddings

Always L2-normalize before cosine similarity or clustering.

### C. Tune thresholds per species if needed

Different species can have different appearance variation, so one global threshold may be too crude.

### D. Simulate unseen identities during validation

This is the main requirement for building a system that adapts to unseen data.

### E. Keep the first version simple

The strongest baseline is often:

1. good embeddings
2. cosine similarity
3. threshold-based unknown rejection
4. agglomerative clustering of rejected samples

Only add OpenMax, EVT, or pseudo-label fine-tuning if the simple system is clearly insufficient.

## 8. What is currently missing from this repo

The repository already contains:

- contrastive training
- embedding checkpoint saving
- embedding visualization / PCA comparison
- submission comparison utilities

The main missing production pieces are:

1. a dedicated embedding extraction script for gallery/query sets
2. a gallery builder
3. a validation split generator for unseen-identity simulation
4. a known-vs-unknown threshold calibration script
5. a clustering script for rejected queries
6. a final inference script that combines matching + rejection + clustering
7. an optional adaptation loop with pseudo-labels

Current pipeline files in this repository:

1. `main.py`
2. `02_extract_embeddings.py`
3. `03_04_build_gallery_and_validation.py`
4. step 4 is implemented together with step 3 in `03_04_build_gallery_and_validation.py`
5. `05_calibrate_rejection_thresholds.py`
6. `06_run_nearest_neighbor_matching.py`
7. `07_reject_low_confidence_matches.py`
8. `08_cluster_rejected_unknowns.py`
9. `09_refine_with_incremental_enrollment.py`
10. `10_export_final_assignments.py`

## 9. Recommended implementation order in this project

Build the system in this order:

1. embedding extraction script
2. gallery + nearest-neighbor matcher
3. validation split for open-set tuning
4. rejection threshold calibration
5. unknown clustering
6. unified inference script
7. optional self-training / incremental enrollment

This order gives you a working baseline early and keeps the adaptive part grounded in measured improvements.

## 10. Short answer

After contrastive learning, you should **stop training and switch to identification mode**:

1. extract embeddings with the saved backbone
2. build the known-identity gallery
3. calibrate a rejection rule for unknown identities
4. match confident queries to known identities
5. cluster rejected queries into new identities
6. optionally adapt with high-confidence pseudo-labels

That is the full path from your current pipeline to an instance identification system that can also handle unseen data.
