# RAG Chunking Optimization 

The effectiveness of retrieval-augmented generation (RAG) depends heavily on the quality of the retrieved context. Traditional chunking methods such as fixed-size segmentation often fail to capture meaningful structure in long or complex documents. This can lead to irrelevant or incomplete retrieval, especially in open-domain question answering, where **precision** and **contextual alignment** are critical.

This project **compares RAG chunking strategies** on the Natural Questions dataset and proposes a **two-stage hierarchical chunking strategy** for improving open-domain question answering. The main idea is to retrieve broad structure-based sections first, then refine retrieval within those sections using smaller, more focused units.

## Datasets

**Natural Questions Dataset** [original repo](https://github.com/google-research-datasets/natural-questions)

Kaggle example preview: https://www.kaggle.com/datasets/validmodel/the-natural-questions-dataset

Our final evaluation subset: **`dataset\gold_test_file_30.json`**. This file contains the 30 Natural Questions examples used for the final project evaluation.

## RAG Chunking Strategies

This project explores four RAG pipelines:

1. Baseline RAG 1: Fixed-Size Chunking
2. Baseline RAG 2: Structure-Based Chunking, which segments documents by structural elements such as HTML tags, headings, and paragraphs
3. Two-Stage RAG 1: Structure-Based Chunking + Sentence Window Chunking
4. Two-Stage RAG 2: Structure-Based Chunking + Proposition Chunking

### Two-Stage RAG 1

Stage 1 retrieves relevant sections using structure-based chunking. 

Stage 2 applies Sentence Window Chunking [1] within those sections: it splits text into individual sentences and **enriches each sentence with nearby context**. The top-matched sentences are retrieved along with their surrounding sentences to improve relevance.

### Two-Stage RAG 2

Stage 1 retrieves relevant sections using structure-based chunking. 

Stage 2 applies Proposition Chunking [2] within those sections: it uses a large language model (LLM) to split text into small, self-contained units called propositions. Each proposition **contains one clear idea and includes the context needed to understand it** without returning to the original passage.

## Project Structure

The current implementation is script/config driven: pipeline code lives in `src/`, runnable entrypoints live in `scripts/`, and experiment settings live in `configs/`.

```text
configs/          YAML configs for each run
scripts/          CLI entrypoints
src/rag_chunking/ reusable pipeline, chunking, retrieval, evaluation code
dataset/          Natural Questions JSON files
artifacts/run_results/ raw run outputs from each RAG strategy
artifacts/evaluations/ RAGAS-scored evaluation outputs
artifacts/vectorstores/ local vector-store artifacts and curated vector-store zip
notebooks/        final analysis/report notebook
archive/notebooks/ archived exploratory notebooks
```

## Setup

From the project root, install dependencies in your environment:

```powershell
pip install -r requirements.txt
```

Remember to set your OpenAI API key before running generation, embedding, or evaluation:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

### Configuration Pattern

Each pipeline loads parameters (e.g., model names, dataset paths, vector-store paths, top-k values, and output paths) from a YAML config by default. You can also use CLI flags to override values for quick tests.

Example:

```powershell
python scripts/run_proposition.py --config configs/proposition.yaml --mode test --max-questions 1
```

**Note**: any persisted vector store must be queried with the same embedding model used to build it. Current configs use `embedding_model: text-embedding-3-large`. If you change this, rebuild the affected vector store.

## Main Pipeline Flow

All strategy scripts can run a small smoke test with `--max-questions 1` before a full run.

### 1. Build Level 1 Structure-Based Vector Store

```powershell
python scripts/build_l1_vectorstore.py --config configs/build_l1_vectorstore.yaml --overwrite
```

This reads the dataset, extracts structure-based sections, embeds them, and saves them to the configured Level 1 vector store.

### 2. Run RAG Strategies

#### 2.1 Fixed-Size Baseline

```powershell
python scripts/run_baseline_fixed.py --config configs/baseline_fixed.yaml
```

Smoke test:

```powershell
python scripts/run_baseline_fixed.py --max-questions 1
```

If `artifacts/vectorstores/Baseline_vector/` does not exist or you changed the baseline embedding/chunking settings, rebuild it:

```powershell
python scripts/run_baseline_fixed.py --rebuild-vector-store
```

#### 2.2 Structure-Based Baseline

```powershell
python scripts/run_baseline_structure.py --config configs/baseline_structure.yaml
```

This pipeline:

1. loads the Level 1 vector store
2. retrieves relevant structural sections directly
3. generates the final answer from those retrieved sections

#### 2.3 Two-Stage RAG 1: Sentence-Window Pipeline

```powershell
python scripts/run_sentence_window.py--config configs/sentence_window.yaml
```

This pipeline:

1. loads the Level 1 vector store
2. retrieves relevant structural sections
3. builds sentence-window nodes for those sections
4. runs sentence-window retrieval and answer generation

#### 2.4 Two-Stage RAG 2: Proposition Pipeline

```powershell
python scripts/run_proposition.py --config configs/proposition.yaml
```

This pipeline:

1. loads the Level 1 vector store
2. retrieves relevant structural sections
3. splits retrieved sections into propositions
4. embeds propositions into a temporary/local Level 2 store
5. retrieves proposition contexts and generates the final answer
## Evaluation

We use RAGAS to evaluate four **metrics**:
* Context Precision
* Context Recall
* Response Relevancy
* Faithfulness

The analysis compares the four RAG strategies overall and by **question type**:
* Close-ended questions
* Explanatory questions

Raw pipeline outputs are saved under `artifacts/run_results/`.

### Step 1: Run RAGAS Evaluation

Evaluate one run-result JSON file with RAGAS:

```powershell
python scripts/evaluate_run.py artifacts/run_results/run_results_baseline.json --config configs/evaluation.yaml
```

Or evaluate all configured strategy outputs:

```powershell
python scripts/evaluate_all.py --config configs/evaluate_all.yaml
```

RAGAS-scored evaluation outputs, including each question's retrieved contexts, generated response, and metric scores, are saved under `artifacts/evaluations/`.

**Note**: If `--output-path` is not provided for single-file evaluation, the script writes a dated file under `artifacts/evaluations/`. For batch evaluation, `output_dir` controls the output folder. Customize in `configs/evaluation.yaml` or `configs/evaluate_all.yaml`.

### Step 2: Analyze Evaluated Results

Use our final analysis notebook after evaluated JSON files exist.

The notebook compares evaluated results across RAG strategies in detail.

```text
notebooks/final_analysis.ipynb
```

It loads file paths from:

```text
configs/analysis.yaml
```

## Typical End-to-End Run

```powershell
python scripts/build_l1_vectorstore.py --config configs/build_l1_vectorstore.yaml --overwrite
python scripts/run_baseline_fixed.py
python scripts/run_baseline_structure.py
python scripts/run_sentence_window.py
python scripts/run_proposition.py
python scripts/evaluate_all.py --config configs/evaluate_all.yaml
```

Then open:

```text
notebooks/final_analysis.ipynb
```

## References
[1]: Bhagat, R. (2023). Sentence Window Retrieval: Optimizing LLM Performance. LinkedIn. https://www.linkedin.com/pulse/sentence-window-retrieval-optimizing-llm-performance-rutam-bhagat-v24of/ (Accessed April 15, 2025)

[2]: Chen, T., Wang, H., Chen, S., Yu, W., Ma, K., Zhao, X., & Yu, D. (2023). Dense x retrieval: What retrieval granularity should we use? arXiv preprint. https://arxiv.org/abs/2312.06648

