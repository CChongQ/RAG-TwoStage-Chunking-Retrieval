# RAG Chunking Optimization with Two-Stage Hierarchical Chunking and Retrieval Strategy

## Introduction 

Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating external context into their responses. However, the effectiveness of RAG depends heavily on the quality of retrieved content. Traditional chunking methods such as fixed-size segmentation often fail to capture meaningful structure in long or complex documents. This can lead to irrelevant or incomplete retrieval, especially in open-domain question answering, where **precision** and **contextual alignment** are critical.

This project proposes a two-stage hierarchical chunking strategy to improve RAG for open-domain question answering:

* Stage 1: Structure-based chunking to extract coherent sections.

* Stage 2: Fine-grained refinement using either Sentence Window Chunking or Proposition Chunking.

Key Results:
* Evaluated on a subset of the Natural Questions dataset using RAGAS metrics, both two-stage methods outperformed the fixed-size baseline.
* Sentence Window Chunking: Best overall results.
* Proposition Chunking: Strongest performance for concise queries.

These findings highlight the benefits of multi-level chunking in RAG systems.

## Datasets
**Natural Questions Dataset** Original Repo [Click Here](https://github.com/google-research-datasets/natural-questions)

Kaggle has example preview: https://www.kaggle.com/datasets/validmodel/the-natural-questions-dataset

⚠️ The Final Test File: **`dataset\gold_test_file_30.json`**


## RAG Piplines

This project explored 4 RAG piplines:

1. Baseline RAG 1: using Fixed-Size Chunking
2. Baseline RAG 2: using Structured-Based Chunking
3. Two-Stage RAG 1: using Structured-Based Chunking + Sentence Window Chunking
4. Two-Stage RAG 2: using Structured-Based Chunking + Proposition Chunking

### Baseline RAG 1

Fxied Size Chunking using `RecursiveCharacterTextSplitter` in **`.\Baseline_1.ipynb`**

### Baseline RAG 2

Structure-Based Chunking in **`.\Structured_Based Chunking.ipynb`**

Segments documents according to structural elements (e.g., HTML tags, headings, paragraphs).

### Two-Stage RAG 1

Stage 1 rely on the structured chunking.    
Stage 2 code using **sentence window** in **`.\Sentence_Window_Complete.ipynb`**

Uses Sentence Window Chunking [1], which splits text into individual sentences and enriches each with nearby context. The top-matched sentences are retrieved along with their surrounding sentences to improve relevance.


### Two-Stage RAG 2
Stage 1 rely on the structured chunking.    
Stage 2 code using **proposition** in **`.\Proposition_Light.ipynb`**

Use Proposition Chunking [2], which uses a large language model (LLM) to split text into small, self-contained units (propositions). Each proposition contains one clear idea and includes all the context it needs. References are resolved so the chunk can be understood without the original text. In this proejct, we use **GPT-3.5** Turbo to extract proposition

## Evaluataion

The code to calcualte the four evaluation criterias' score, generate complete evaluation result, including score distribution plot, mean value analysis and two question-type specific analysis in **`.\Evaluation.ipynb`**.

The detail evaluation result with detail output of each step(e.g. retrieved content and RAG generator's response to each question) can be find in folder **`.\evaluation`**

We use RAGAS for evaluation, with each score computed using GPT-4o. 

The four pipelines are evaluated on:
* Close-ended questions
* Explanatory questions

Metrics:
* Context Precision
* Context Recall
* Response Relevancy
* Faithfulness

# References
[1]: Bhagat, R. (2023). Sentence Window Retrieval: Optimizing LLM Performance. LinkedIn. https://www.linkedin.com/pulse/sentence-window-retrieval-optimizing-llm-performance-rutam-bhagat-v24of/ (Accessed April 15, 2025)

[2]: Chen, T., Wang, H., Chen, S., Yu, W., Ma, K., Zhao, X., & Yu, D. (2023). Dense x retrieval: What retrieval granularity should we use? arXiv preprint. https://arxiv.org/abs/2312.06648



