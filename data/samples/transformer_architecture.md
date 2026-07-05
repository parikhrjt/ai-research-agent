---
title: Transformer Architecture Research Notes
author: Research Team
date: 2024-03-15
tags: [nlp, transformers, attention]
---

# Attention Is All You Need — Research Notes

## Overview

The Transformer architecture, introduced by Vaswani et al. (2017), revolutionized
natural language processing by replacing recurrent layers entirely with
self-attention mechanisms. This document summarizes key findings relevant to
our RAG pipeline evaluation.

## Key Contributions

### Self-Attention Mechanism

Self-attention computes representations by relating different positions within
a single sequence. The attention function is defined as:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

Where Q (queries), K (keys), and V (values) are learned linear projections
of the input embeddings.

### Multi-Head Attention

Rather than performing a single attention function, the model projects queries,
keys, and values h times with different learned projections. On the WMT 2014
English-to-German translation task, the base Transformer achieved **28.4 BLEU**,
outperforming all prior models including ensembles.

## Architecture Details

| Component | Value |
|-----------|-------|
| Encoder layers | 6 |
| Decoder layers | 6 |
| Model dimension (d_model) | 512 |
| Attention heads | 8 |
| Feed-forward dimension | 2048 |
| Dropout rate | 0.1 |

## Positional Encoding

Since the model contains no recurrence or convolution, positional encodings are
added to inject sequence order information. Sinusoidal functions of different
frequencies are used:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

## Training Configuration

- **Optimizer**: Adam with beta1=0.9, beta2=0.98, epsilon=1e-9
- **Learning rate schedule**: Warmup for 4000 steps, then inverse square root decay
- **Batch size**: ~25,000 source and target tokens per batch
- **Hardware**: 8 NVIDIA P100 GPUs, training time ~3.5 days (base model)

## Implications for RAG Systems

1. **Encoder-only models** (e.g., BERT) are well-suited for embedding generation
2. **Decoder-only models** (e.g., GPT) excel at answer generation
3. **Attention patterns** enable interpretable retrieval — attending to relevant
   context chunks mirrors RAG retrieval behavior

## References

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
