# Animal Re-Identification Pipeline

This project trains and evaluates an embedding model for animal re-identification in an open-set setting.

The goal is to learn embeddings that keep images of the same individual close together, separate different individuals, and remain useful even when test images include unseen identities.

## Current Entry Point

The active pipeline is the `main.py` workflow:

```bash
python3 main.py
```

The older numbered-script flow is no longer active.

## Source of Truth

The consolidated technical documentation lives in:

[`docs/current_training_pipeline.md`](/home/alf/ssl/docs/current_training_pipeline.md)

That document covers:

- project purpose and problem framing
- active training and evaluation flow
- dataset and split semantics
- model, loss, and clustering details
- known caveats and suggested next steps

## Short Summary

The current implementation:

1. builds an identity-level train/validation split from `metadata.sqlite`
2. trains a MegaDescriptor backbone with PK batches and batch-hard triplet loss
3. rebuilds simple image-level loaders after training
4. extracts embeddings
5. runs HDBSCAN clustering diagnostics

At the moment, `main.py` is hardcoded to train on `salamander`.
