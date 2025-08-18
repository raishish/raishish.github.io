---
layout: distill
title: How is bidirectional information retrieved and generated in masked diffusion language models?
description: Understanding Bidirectional Information Retrieval in MDLMs with ROME
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
  - name: tl;dr
  - name: Abstract
  - name: Introduction
  - name: Background and Related Work
  - name: Methodology
    subsections:
      - name: Models
      - name: Dataset
      - name: Causal Tracing
      - name: Causal Tracing for Diffusion Language Models
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
  table {
    margin: 1.5rem auto;
    border-collapse: collapse;
    width: 100%;
  }
  table caption {
    caption-side: top;
    text-align: center;
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: var(--global-text-color);
  }
  table th,
  table td {
    border: 1px solid var(--global-divider-color);
    padding: 0.5rem;
    text-align: left;
  }
  table th {
    background-color: var(--global-bg-color-dark);
    font-weight: bold;
  }
  .alert-info {
    color: #0c5460;
    background-color: #d1ecf1;
    border-color: #bee5eb;
    border: 1px solid #bee5eb;
    border-left: 4px solid #17a2b8;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 1rem 0;
  }
  .alert-warning {
    color: #856404;
    background-color: #fff3cd;
    border-color: #ffeaa7;
    border: 1px solid #ffeaa7;
    border-left: 4px solid #ffc107;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 1rem 0;
  }
  .alert-success {
    color: #155724;
    background-color: #d4edda;
    border-color: #c3e6cb;
    border: 1px solid #c3e6cb;
    border-left: 4px solid #28a745;
    border-radius: 0.375rem;
    padding: 1rem;
    margin: 1rem 0;
  }

# citation: true
---

## Summary

### tl;dr

In this post, I will discuss the intermediate progress on using ROME to study bidirectional information retrieval in masked (discrete) diffusion language models.

Analysis in-progress!

## Abstract

<!-- Masked diffusion language models (MDLMs) are a new direction in language modeling research, offering a non-autoregressive alternative to the more commonly used autoregressive models. Since these model generate output responses non-autoregressively in token space and in discrete steps, this lends us the question of the causal (or non-causal) nature of response generation compared to autoregressive models. Mechanistic interpretability (MI) is a broader domain of techniques that aim to understand the internal circuits of deep learning models. Although mechanistic interpretability techniques have been widely used on autoregressive models and have gained significant insights into understanding how autoregressive models generate responses, they have not yet been widely applied to diffusion-based language models.

This study aims to utilize ROME, a techique for locating knowledge in transformer-based models, on MDLMs to investigate how information is stored and retrieved in diffusion language models. Through our work, we hope to gain new insights on interpretability of masked diffusion language models. -->

This research proposal aims to conduct the first mechanistic investigation of bidirectional retrieval and generation in masked diffusion language models. We focus on masked diffusion models (MDMs), which have shown surprising resistance to the “reversal curse” that plagues autoregressive architectures. The study will map the causal pathways of knowledge recall and analyze the internal representations learned by the model. By leveraging recent insights into the “factorization curse” - a more fundamental problem underlying the reversal curse - we seek to understand how non-autoregressive architectures overcome these limitations. This mechanistic investigation will be among the first to systematically examine the internal workings of masked diffusion language models, providing insights into their emerging capabilities for bidirectional logical reasoning. The findings will contribute to the development of more robust language models with improved reasoning abilities and more reliable knowledge representation.

## Introduction

Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but recent studies have identified a significant limitation called the “reversal curse.” When autoregressive models like GPT learn a fact such as “A is B,” they fail to automatically deduce the reverse relationship “B is A” <d-cite key="berglund2023reversal"></d-cite>. For example, if a model learns that “Paris is the capital of France,” it struggles to answer “The capital of France is ____” correctly.

Recent work by <d-cite key="kitouni2024factorization"></d-cite> reframes this as the “factorization curse” - a more fundamental problem where language models fail to learn the same joint distribution under different factorizations. This insight suggests the issue isn’t just about reversing relationships, but about a broader limitation in how next-token prediction objectives force models to learn specific factorizations of joint distributions, undermining their ability to reason bidirectionally.

Certain model architectures — particularly bidirectional encoder models like BERT and masked diffusion models (MDMs) — appear to be more resistant to this limitation (<d-cite key="wu2023exploring, nie2024scaling"></d-cite>). Understanding why these architectural differences lead to better bidirectional reasoning capabilities could provide valuable insights for developing more robust AI systems that overcome the factorization curse. This research proposal aims to investigate the mechanisms behind bidirectional information retrieval in language models, with a specific focus on masked diffusion models (MDMs).

## Background and Related Work

The reversal curse was first formally identified in <d-cite key="berglund2023reversal"></d-cite>, which demonstrated that autoregressive models fine-tuned on statements like “A is B” failed to generalize to “B is A.” The authors showed that models like GPT-3 (175B) and Llama-1 (7B) scored no better than random chance when evaluated on reversed relationships.

<figure id="reversal-curse">
  <img src="{{ '/assets/img/2025-discrete-diffusion-rome/reversal-curse.png' | relative_url }}" alt="The Reversal Curse">
  <figcaption><strong>Figure 1:</strong> The Reversal Curse</figcaption>
</figure>

Kitouni et al. (2024) significantly expanded the understanding of this phenomenon by reframing it as the “factorization curse” - a fundamental limitation of the next-token prediction objective used in most LLMs <d-cite key="kitouni2024factorization"></d-cite>. Through controlled experiments using their novel WikiReversal benchmark based on Wikipedia knowledge graphs, they demonstrated that this is an inherent failure mode that cannot be solved merely through scaling, reversed tokens, or even naive bidirectional-attention training. Their work identified factorization-agnostic objectives as a promising solution, showing significant improvements across various tasks of different complexity levels.

Ma et al. (2023) investigated this problem specifically in the context of model editing, introducing the “Bidirectional Assessment for Knowledge Editing” (BAKE) benchmark and the “Bidirectionally Inversible Relationship moDeling” (BIRD) method <d-cite key="ma2023untying"></d-cite>. Their work revealed that while existing editing methods can effectively recall facts in the direction they were trained on, they perform poorly in the reverse direction. For instance, LLaMA-2 edited with state-of-the-art ROME could recall 99.70% of editing facts in the forward direction but only 0.26% in the reverse direction.

Wu and Wang (2023) explored this phenomenon further, comparing autoregressive decoder-only models (like GPT) with bidirectional encoder models (like BERT) <d-cite key="wu2023exploring"></d-cite>. They found that BERT-style models were largely immune to the reversal curse for basic logical deductions, suggesting that architectural differences play a critical role.

Recent advances in masked diffusion models have shown particularly promising results. Nie et al. (2024) demonstrated that a 1.1B parameter
MDM could break the reversal curse that much larger autoregressive models struggle with <d-cite key="nie2024scaling"></d-cite>. Similarly, research on LLaDA (Large Language Diffusion Models) has shown that diffusion-based approaches effectively address bidirectional reasoning challenges and even outperform GPT-4o on certain reversal tasks <d-cite key="nie2025largelanguagediffusionmodels"></d-cite>.

<details>
<summary><strong>A primer on masked discrete diffusion</strong></summary>

</details>

## Research Objectives
This study aims to:
1. **Map the causal pathways of knowledge recall** to understand how information flows when retrieving facts in forward versus reverse directions.
2. **Analyze learned representations** to determine if and how they encode bidirectional logical relationships, identifying any symmetries in forward and backward representation of concepts.

## Methodology

### Models

- Masked Diffusion Language Model: **SMDM-1028M** - Llama2 based 1B masked diffusion model from <d-cite key="nie2024scaling"></d-cite>
- (Baseline) Autoregressive Models: **AR-1028M** - Llama2 based 1B autoregressive decoder only model from <d-cite key="nie2024scaling"></d-cite>

<!-- <div id="tab-model-config">
  <strong>Table 1:</strong> Model configuration parameters used in experiments.
</div> -->

<!-- | Model Type | Masked Diffusion | Autoregressive |
|------------|------------------|----------------|
| Base Architecture | Llama2 | Llama2 |
|------------|------------------|----------------|
| Transformer Blocks | 20 | 20 |
| Attention Heads | 14 | 14 |
| `n_embed` | 1792 | 1792 |
| `embed_dim` | 7168 | 7168 |
| `vocab_size` | 32000 | 32000 |
| Sampling temperature | 0. | 0. |
| Sampling algorithm | Greedy | Greedy |

Table: Model configuration parameters used in experiments.{: .table-caption} -->

<table id="tab-model-config">
  <caption><strong>Table 1:</strong> Model configuration parameters used in experiments.</caption>
  <thead>
    <tr>
      <th>Model Type</th>
      <th>Masked Diffusion</th>
      <th>Autoregressive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Base Architecture</td>
      <td>Llama2</td>
      <td>Llama2</td>
    </tr>
    <tr>
      <td>Transformer Blocks</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <td>Attention Heads</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <td><code>n_embed</code></td>
      <td>1792</td>
      <td>1792</td>
    </tr>
    <tr>
      <td><code>embed_dim</code></td>
      <td>7168</td>
      <td>7168</td>
    </tr>
    <tr>
      <td><code>vocab_size</code></td>
      <td>32000</td>
      <td>32000</td>
    </tr>
    <tr>
      <td>Sampling temperature</td>
      <td>0.</td>
      <td>0.</td>
    </tr>
    <tr>
      <td>Sampling algorithm</td>
      <td>Greedy</td>
      <td>Greedy</td>
    </tr>
  </tbody>
</table>


<!-- <div id="smdm-sampling-config">
  <strong>Table 2:</strong> Sampling configuration for SMDM-1028M (Masked Diffusion)
</div>

| Config | Value |
|--------|-------|
| Denoising Steps | 16 |
| Context Length | 52 |
| CFG Scale | 0.8 | -->

<table id="smdm-sampling-config">
  <caption><strong>Table 2:</strong> Sampling configuration for SMDM-1028M (Masked Diffusion)</caption>
  <thead>
    <tr>
      <th>Config</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Denoising Steps</td>
      <td>16</td>
    </tr>
    <tr>
      <td>Context Length</td>
      <td>52</td>
    </tr>
    <tr>
      <td>CFG Scale</td>
      <td>0.8</td>
    </tr>
  </tbody>
</table>

The models chosen in this study are pre-trained on the SlimPajama Dataset <d-cite key="nie2024scaling"></d-cite>. These models are based on the Llama2 architecture and are configured with settings referenced in [Table 1](#tab-model-config). The sampling config for the masked diffusion language model SMDM-1028M has been borrowed from <d-cite key="nie2024scaling"></d-cite> as well and is shown in [Table 2](#smdm-sampling-config). The denoising steps indicate the number of steps over which the masked model response is unmasked. The context length refers to the total length of the sequence given as input to model, which includes both the prompt and the unmasked reponse [Fig 2](#masked-diffusion-inference). The CFG scale is the factor with which the classifier free guidance is scaled and applied for conditional generation.

<figure id="masked-diffusion-inference">
  <img src="{{ '/assets/img/2025-discrete-diffusion-rome/masked_response.png' | relative_url }}" alt="Denoising response sequence during inference in masked diffusion language models">
  <figcaption><strong>Figure 2:</strong> The prompt token is padded with masked tokens which are iteratively denoised during sampling from a masked diffusion language model.</figcaption>
</figure>

### Dataset

The reversal curse can be studied with factual recall on:
- Completions based on pre-trained data
- Completions based on SFT-trained data

The masked diffusion model was first prompted to generate single-token completions on Wikipedia knowledge prompts, which the model is trained on as part of SlimPajama. The last token did not elicit the most informative tokens. A subset of the responses can be seen in [Fig 3](#wikipedia-knowledge-prompt-completions).

<figure id="wikipedia-knowledge-prompt-completions">
  <img src="{{ '/assets/img/2025-discrete-diffusion-rome/pre-training-single-token-completions.png' | relative_url }}" alt="Wikipedia Knowledge Prompt Completions">
  <figcaption><strong>Figure 3:</strong> Generations by SMDM-1028M on Wikipedia knowledge prompts yields high entropy (less information in first generated token) and could not be used as a controlled database to test the reversal curse on.</figcaption>
</figure>

The models were then fine tuned on the Reversal curse dataset and attained high accuracy [Table 3](#tab-reversal-curse-eval), so we chose this dataset. The samples from the dataset are depicted in [Fig. 4](#dataset).

<figure id="dataset">
  <img src="{{ '/assets/img/2025-discrete-diffusion-rome/dataset.png' | relative_url }}" alt="Dataset">
  <figcaption><strong>Figure 4:</strong> The reversal curse dataset.</figcaption>
</figure>

### Causal Tracing

\cite{Meng2022LocatingAE} introduced causal tracing as a method to identify storage and processing of factual knowledge in language models. The technique aims to isolate the causal effect of individual states within the network while processing factual statements, essentially mapping the information flow path through the model.

The method aims to understand the model's prediction, focusing on identifying key neurons and layers involved in recalling factual associations.

#### Tracing Information Flow

<d-cite key="Meng2022LocatingAE"></d-cite> use causal mediation analysis to quantify the contribution of intermediate variables (hidden states) in the model's causal graph to a correct factual prediction <d-cite key="10.5555/2074022.2074073, vig2020investigating"></d-cite>.

To calculate each state's contribution, they observe the model's internal activations during three runs:

1. **Clean Run:** A factual prompt $x$ is passed into the model $G$, and all hidden activations are collected.
2. **Corrupted Run:** The subject $s$ is obfuscated from $G$ before the network runs to damage the prediction.
3. **Corrupted-with-Restoration Run:** The model runs computations on the noisy embeddings as in the corrupted baseline, except at some token $i$ and layer $l$, the model is forced to output the clean state $h^{(l)}_i$.

**Indirect Effect (IE):** The indirect effect of a specific mediating state $h^{(l)}_i$ is defined as the difference between the probability of $o$ under the corrupted version and the probability when that state is set to its clean version, while the subject remains corrupted.

**Average Indirect Effect (AIE):** Averaging over a sample of statements, the average total effect (ATE) and average indirect effect (AIE) is obtained for each hidden state variable.

#### Extracting Factual Representations

Meng et al., 2022 <d-cite key="Meng2022LocatingAE"></d-cite> build upon the localized factual association hypothesis to devise a method to **localize factual associations** within transformer models. By locating the layer and weight matrix where a new fact can be injected with minimal interference, ROME provides evidence that factual knowledge is encoded in specific MLP projection layers as **linear associations** between key and value vectors. The proposed method by <d-cite key="Meng2022LocatingAE"></d-cite> is explained briefly for context.

**Extracting the Key $k_{\ast}$**

To identify the memory location corresponding to the subject $s$ in the factual triple $(s, r, o)$ a key vector $k_{\ast}$ is constructed with the underlying assumption being certain activations within the MLP are responsible for retrieving facts associated with a given subject. To find a robust key, the activation vectors are averaged across multiple short text prefixes that lead into and end with the subject token $s$.

$$k_{\ast} = \frac{1}{N} \sum_{j=1}^N k(x_j + s)$$

where $k(x)$ is derived from the layer $l^*$'s MLP activations after the non-linearity is applied.

**Extracting the Value $v_{\ast}$**

Meng et al., 2022 <d-cite key="Meng2022LocatingAE"></d-cite> compute a value vector $v_{\ast}$ that encodes a new object $o^*$ (i.e., the fact to be recalled) as a property of the subject $s$. This is formulated as an optimization problem with a linear combination of two objectives:

1. Maximize the likelihood that the model, when prompted with subject $s$ and relation $r$, predicts the new object $o^*$.
2. Minimize the KL divergence between the model's original predictions for $s$ and its predictions after editing, to prevent disrupting the subject's broader semantic identity.

This results in the following loss objective:

$$\mathcal{L}(z) = - \frac{1}{N} \sum_{j=1}^N \log \mathbb{P}_{G(m_i^{(l^*)} := z)}(o^* \mid x_j + p) + D_{\mathrm{KL}}\left( \mathbb{P}_{G(m_{i'}^{(l^*)} := z)}[x \mid p'] \,\Vert\, \mathbb{P}_G[x \mid p'] \right)$$

where $z$ is optimized to serve as $v_{\ast}$, the value that, when output by the MLP, causes the model to recall the new object $o^*$ in response to the original factual prompt $p$. $p'$ represents the set of adjacent prompts (of the form "{subject} is a") which aim to preserve the model's understanding of the subject's essence.

In contrast, **we aim to determine the representation of the original object $o$ when preceded with subject $s$ in the prompt**. So we update the objective function to account for $o$ instead of $o^*$ and the rest remains the same:

$$\mathcal{L}(z) = - \frac{1}{N} \sum_{j=1}^N \log \mathbb{P}_{G(m_i^{(l^*)} := z)}(o \mid x_j + p) + D_{\mathrm{KL}}\left( \mathbb{P}_{G(m_{i'}^{(l^*)} := z)}[x \mid p'] \,\Vert\, \mathbb{P}_G[x \mid p'] \right)$$

Meng et al., 2022  <d-cite key="Meng2022LocatingAE"></d-cite> go a step further and apply a **rank-one update** to the projection matrix $W^{(l)}_{\text{proj}}$ of the MLP at layer $l^*$, in order to encode a new key–value pair into the model's weights. This is beyond the scope of this research study.

#### Causal Tracing for Diffusion Language Models

Unlike autoregressive and masked language models, which predict all response tokens in a single forward pass; response generation in masked diffusion language models starts with a sequence of all masked tokens $x_1$ with fixed length $L$ (context length) is set and a mask predictor $f_\theta$ unmasks all tokens following a noise schedule $\alpha_t$ over $T$ denoising steps to generate the response $x_0$. Specifically, the masked diffusion model simulates a diffusion process from $t = 1$ (fully masked) to $t = 0$ (unmasked), predicting all masks simultaneously at each step with the possibility of flexible remasking strategies ([Fig. 5](#unmasking-during-inference)).

<figure id="unmasking-during-inference">
  <img src="{{ '/assets/img/2025-discrete-diffusion-rome/unmasking-during-inference.png' | relative_url }}" alt="Unmasking tokens via sampling during inference">
  <figcaption><strong>Figure 5:</strong> RMasked diffusion models simulate a diffusion process from t = 1 (fully masked) to t = 0 (unmasked), predicting all masks simultaneously at each step with the possibility of flexible remasking strategies.</figcaption>
</figure>

For masked diffusion language modeling, the individual response tokens (as seen in [Fig. 6](#masked-diffusion-generation)) can be unmasked at different time steps.

<figure id="masked-diffusion-generation">
  <img src="{{ '/assets/img/2025-discrete-diffusion-rome/masked-diffusion-generation.png' | relative_url }}" alt="Response generation from masked diffusion language models">
  <figcaption><strong>Figure 6:</strong> Response generation from masked diffusion language model.</figcaption>
</figure>

During training, Masked diffusion models (MDMs) are trained to increased the likelihood of the data distribution for different masking conditions. During inference, the tokens can be masked with different sampling strategies to determine the order in which the response tokens are predicted. In the context of causal information flow analysis, this introduces both challenges and opportunities:

- **Challenge:** This introduces multiple causal paths from a subject token across different denoising time steps.
- **Opportunity:** By quantifying the information flow in the first denoising time step, often the most important one, we can isolate the information flow from the subject tokens and the intermediate unmasked object tokens (which the model can attend to from the second time step).

Causal tracing is run from time steps $t = t_1$ to $t = t_2$.

<!-- The initial approach hence included caching all activations between time $t = t_1$ (the first timestep when a token in the answer appears) and the time $t = t_2$ (the last time step when the answer is fully generated in the response). This would include running the corruption and corruption-with-restoration runs for all of these intermediate steps.

This leads to two complications:

\begin{enumerate}
    \item The tokens in intermediate answers overlap with non-answer tokens leading to difficultly in attributing this intermediate token to generation of the final answer.
    \item Corruptions and restorations at intermediate steps would create further causal branches leading to furhter complication of a causal pathway from input prompt to the final generated answer.
\end{enumerate}

Owing to the above reason, the process was simplified and causal tracing is only applied at time $t = t_2$ (when the final answer token is generated) as seen in figure \ref{fig:masked-diffusion-causal-tracing}. In causal tracing, the likelihood of the answer token is used to determine attribution scores for impact of intermediate states, MLP, and attention heads on the final answer generation. In this case, the likelihood of the multi-token answer is formulated as the joint distribution and calculated as the product of the likelihoods of each answer token as per chain rule of probability. -->

When retrieving information from a language model, the process acts like a key-based value lookup with the subject $s$ acting as a `key` and the object acting as a `value` to be extracted. Following <d-cite key="Meng2022LocatingAE"></d-cite>, the key vectors in MLP layers corresponding to the last subject token in identified important MLP units (discovered by causal tracing) will be fixed and the objective function defined in equation 2 will be optimized to extract the representation of the object $o$.

This procedure will be done while trying to elicit knowledge in both forward (sample prompt: "The trailblazer known as Daphne Barrington was once") and reverse (sample prompt: "Immersed in the world of directing the virtual reality masterpiece, A Journey Through Time.") directions. The key and value representations in these cases would be interchanged between both runs. For example, the key in prompt 1 "Daphne Barrington" would act as the value in prompt 2. The aim is to compare the key and value representations of the same concept via dimensionality reduction techniques (SVD and PCA) to see if there exists a symmetry in which the information is encoded in the diffusion model.

<!-- More details about the dataset can be found in Appendix Section \ref{Data set details}. -->

## Current Progress

We begin with a scaling experiment to study the impact of sampling steps on knowledge recall, measured with exact match accuracy <d-cite key="berglund2023reversal"></d-cite>. As the sampling steps are increased, there is an increase in knowledge recall in both forward and reverse directions (refer [Table 3](#tab-reversal-curse-eval)).

<div class="alert alert-warning">
Please note that these accuracy values for <code>Description2Person</code> reverse and <code>Person2Description</code> reverse are different from the ones reported in <d-cite key="nie2024scaling"></d-cite>. *We believe this is an error in reporting of the values since reverse recall is harder to model for language models, majorly because the perplexity of the text in description is usually higher than the text in person names of the Reversal Curse dataset <d-cite key="berglund2023reversal"></d-cite>.*
</div>

<table id="tab-reversal-curse-eval">
  <caption><strong>Table 3:</strong> Scaling sampling steps leads to better knowledge recall performance in forward and reverse directions.</caption>
  <thead>
    <tr>
      <th rowspan="2">Sampling Steps</th>
      <th colspan="2">Description2Person</th>
      <th colspan="2">Person2Description</th>
    </tr>
    <tr>
      <th>Same (easy)</th>
      <th>Reverse (difficult)</th>
      <th>Same (difficult)</th>
      <th>Reverse (easy)</th>
    </tr>
    <tr>
      <th></th>
      <th>% Acc ↑</th>
      <th>% Acc ↑</th>
      <th>% Acc ↑</th>
      <th>% Acc ↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1 step</td>
      <td>91.3</td>
      <td>0</td>
      <td>0</td>
      <td>40.3</td>
    </tr>
    <tr>
      <td>2 steps</td>
      <td>97</td>
      <td>0.7</td>
      <td>2.3</td>
      <td>79</td>
    </tr>
    <tr>
      <td>4 steps</td>
      <td>96.3</td>
      <td>17</td>
      <td>25</td>
      <td>87.7</td>
    </tr>
    <tr>
      <td>8 steps</td>
      <td>97.3</td>
      <td>27.3</td>
      <td>37.7</td>
      <td>87.3</td>
    </tr>
    <tr>
      <td>16 steps</td>
      <td><strong>98</strong></td>
      <td><strong>31</strong></td>
      <td>42</td>
      <td>88.3</td>
    </tr>
    <tr>
      <td>32 steps</td>
      <td>97.7</td>
      <td>29.7</td>
      <td><strong>43</strong></td>
      <td><strong>90</strong></td>
    </tr>
  </tbody>
</table>

This is a work-in-progress. More results coming soon!

The existing code for the study can be found at [raishish/diffusion-interp](https://github.com/raishish/diffusion-interp).

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
