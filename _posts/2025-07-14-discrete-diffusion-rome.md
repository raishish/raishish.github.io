---
layout: distill
title: How is information retrieved and generated in masked diffusion language models?
description: Understanding Bidirectional Information Retrieval in Masked Diffusion Language Models with ROME
tags: masked-discrete-diffusion mech-interp rome
giscus_comments: false
date: 2025-07-14
featured: true

authors:
  - name: Ashish Rai
    url: "https://raishish.github.io/"
    affiliations:
      name: New York University

bibliography: 2025-07-14-discrete-diffusion-rome.bib

toc:
  - name: Summary
  - name: Introduction
  - name: A primer on masked discrete diffusion
  - name: Background and Related Work
  - name: Dataset
  - name: Methodology
  - name: Current Progress
  - name: Future Work
  - name: Acknowledgements
  - name: References

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }

# citation: true
---

## Summary

In this post, I will discuss the intermediate progress on using ROME to study information retrieval in masked (discrete) diffusion language models.

Page under construction!

Analysis in-progress.

## Introduction

Masked diffusion language models (MDLMs) are a new direction in language modeling research, offering a non-autoregressive alternative to the more commonly used autoregressive models. Since these model generate output responses non-autoregressively in token space and in discrete steps, this lends us the question of the causal (or non-causal) nature of response generation compared to autoregressive models. Mechanistic interpretability (MI) is a broader domain of techniques that aim to understand the internal circuits of deep learning models. Although mechanistic interpretability techniques have been widely used on autoregressive models and have gained significant insights into understanding how autoregressive models generate responses, they have not yet been widely applied to diffusion-based language models.

This study aims to utilize ROME, a techique for locating knowledge in transformer-based models, on MDLMs to investigate how information is stored and retrieved in diffusion language models. Through our work, we hope to gain new insights on interpretability of masked diffusion language models.

## A primer on masked discrete diffusion

## Background and Related Work

## Dataset

## Methodology

## Current Progress

## Future Work

## Acknowledgements

## References

<!-- ## Citation -->

If you would like to cite this work, please use the following BibTeX entry:

```bibtex
@article{rai2025discrete-diffusion-rome,
  title={How is information retrieved and generated in masked diffusion language models?},
  author={Rai, Ashish},
  year={2025},
  month={July},
  url={https://raishish.github.io/blog/2025/discrete-diffusion-rome/}
}
```
