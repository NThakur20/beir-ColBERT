# BEIR Evaluation with ColBERT

In this example, we show how to evaluate the ColBERT zero-shot model on the BEIR Benchmark.

We modify the original [ColBERT](https://github.com/stanford-futuredata/ColBERT) repository to allow for evaluation of ColBERT across any BEIR dataset.

Please follow the required steps to evaluate ColBERT easily across any BEIR dataset.

## Installation with BEIR

- **Step 1**: Clone this beir-ColBERT repository (forked from original) which has modified for evaluating models on the BEIR benchmark: 
```bash
git clone https://github.com/NThakur20/beir-ColBERT.git
```

- **Step 2**: Create a new Conda virtual environment using the environment file provided: [conda_env.yml](https://github.com/NThakur20/beir-ColBERT/blob/master/conda_env.yml), It includes pip installation of the beir repository.
```bash
# https://github.com/NThakur20/beir-ColBERT#installation

conda env create -f conda_env.yml
conda activate colbert-v0.2
```
  - **Please Note**: We found some issues with ``_swigfaiss`` with both ``faiss-cpu`` and ``faiss-gpu`` installed on Ubuntu. If you face such issues please refer to: https://github.com/facebookresearch/faiss/issues/821#issuecomment-573531694 

## ``evaluate_beir.sh``

Run script ``evaluate_beir.sh`` for the complete evaluation of ColBERT model on any BEIR dataset. This scripts has five steps:

**1. BEIR Preprocessing**: We preprocess our BEIR data into ColBERT friendly data format using ``colbert/data_prep.py``. The script converts the original ``jsonl`` format to ``tsv``.  

```bash
python -m colbert.data_prep \
  --dataset ${dataset} \     # BEIR dataset you want to evaluate, for e.g. nfcorpus
  --split "test" \           # Split to evaluate on
  --collection $COLLECTION \ # Path to store collection tsv file
  --queries $QUERIES \       # Path to store queries tsv file
```

**2. ColBERT Indexing**: For fast retrieval, indexing precomputes the ColBERT representations of passages. 

**NOTE**: you will need to download the trained ColBERT model for inference

```bash
python -m torch.distributed.launch \
  --nproc_per_node=2 -m colbert.index \
  --root $OUTPUT_DIR \       # Directory to store the output logs and ranking files
  --doc_maxlen 300 \         # We work with 300 sequence length for document (unlike 180 set originally)
  --mask-punctuation \       # Mask the Punctuation
  --bsize 128 \              # Batch-size of 128 for encoding documents/tokens.
  --amp \                    # Using Automatic-Mixed Precision (AMP) fp32 -> fp16
  --checkpoint $CHECKPOINT \ # Path to the checkpoint to the trained ColBERT model 
  --index_root $INDEX_ROOT \ # Path of the root index to store document embeddings
  --index_name $INDEX_NAME \ # Name of index under which the document embeddings will be stored
  --collection $COLLECTION \ # Path of the stored collection tsv file
  --experiment ${dataset}    # Keep an experiment name
```
**3. FAISS IVFPQ Index**: We store and train the index using an IVFPQ faiss index for end-to-end retrieval. 

**NOTE**: You need to choose a different ``k`` number of partitions for IVFPQ for each dataset

```bash
python -m colbert.index_faiss \
  --index_root $INDEX_ROOT \     # Path of the root index where the faiss embedding will be store
  --index_name $INDEX_NAME \     # Name of index under which the faiss embeddings will be stored 
  --partitions $NUM_PARTITIONS \ # Number of Partitions for IVFPQ index (Seperate for each dataset (You need to chose)), for eg. 96 for NFCorpus 
  --sample 0.3 \                 # sample: 0.3
  --root $OUTPUT_DIR \           # Directory to store the output logs and ranking files
  --experiment ${dataset}        # Keep an experiment name
```

**4. Query Retrieval using ColBERT**: Retrieves top-_k_ documents, where depth = _k_ for each query.

**NOTE**: The output ``ranking.tsv`` file produced has integer document ids (because of faiss). Each each int corresponds to the doc_id position in the original collection tsv file. 

```bash
python -m colbert.retrieve \
  --amp \                        # Using Automatic-Mixed Precision (AMP) fp32 -> fp16
  --doc_maxlen 300 \             # We work with 300 sequence length for document (unlike 180 set originally)
  --mask-punctuation \           # Mask the Punctuation
  --bsize 256 \                  # 256 batch-size for evaluation
  --queries $QUERIES \           # Path which contains the store queries tsv file
  --nprobe 32 \                  # 32 query tokens are considered
  --partitions $NUM_PARTITIONS \ # Number of Partitions for IVFPQ index
  --faiss_depth 100 \            # faiss_depth of 100 is used for evaluation (Roughly 100 top-k nearest neighbours are used for retrieval)
  --depth 100 \                  # Depth is kept at 100 to keep 100 documents per query in ranking file
  --index_root $INDEX_ROOT \     # Path of the root index of the stored IVFPQ index of the faiss embeddings
  --index_name $INDEX_NAME \     # Name of index under which the faiss embeddings will be stored 
  --checkpoint $CHECKPOINT \     # Path to the checkpoint to the trained ColBERT model 
  --root $OUTPUT_DIR \           # Directory to store the output logs and ranking files
  --experiment ${dataset} \      # Keep an experiment name
  --ranking_dir $RANKING_DIR     # Ranking Directory will store the final ranking results as ranking.tsv file
```

**5. Evaluation using BEIR**: Evaluate the ``ranking.tsv`` file using the BEIR evaluation script for any dataset.

```bash
python -m colbert.beir_eval \
  --dataset ${dataset} \                   # BEIR dataset you want to evaluate, for e.g. nfcorpus
  --split "test" \                         # Split to evaluate on
  --collection $COLLECTION \               # Path of the stored collection tsv file 
  --rankings "${RANKING_DIR}/ranking.tsv"  # Path to store the final ranking tsv file 
```

# Original README    

> **Update: if you're looking for [ColBERTv2](https://arxiv.org/abs/2112.01488) code, you can find it alongside a new simpler API, in the branch [`new_api`](https://github.com/stanford-futuredata/ColBERT/tree/new_api).**


# ColBERT

### ColBERT is a _fast_ and _accurate_ retrieval model, enabling scalable BERT-based search over large text collections in tens of milliseconds. 

<p align="center">
  <img align="center" src="docs/images/ColBERT-Framework-MaxSim-W370px.png" />
</p>
<p align="center">
  <b>Figure 1:</b> ColBERT's late interaction, efficiently scoring the fine-grained similarity between a queries and a passage.
</p>

As Figure 1 illustrates, ColBERT relies on fine-grained **contextual late interaction**: it encodes each passage into a **matrix** of token-level embeddings (shown above in blue). Then at search time, it embeds every query into another matrix (shown in green) and efficiently finds passages that contextually match the query using scalable vector-similarity (`MaxSim`) operators.

These rich interactions allow ColBERT to surpass the quality of _single-vector_ representation models, while scaling efficiently to large corpora. You can read more in our papers:

* [**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**](https://arxiv.org/abs/2004.12832) (SIGIR'20).
* [**Relevance-guided Supervision for OpenQA with ColBERT**](https://arxiv.org/abs/2007.00814) (TACL'21; to appear).


----

## Installation

ColBERT (currently: [v0.2.0](#releases)) requires Python 3.7+ and Pytorch 1.6+ and uses the [HuggingFace Transformers](https://github.com/huggingface/transformers) library.

We strongly recommend creating a conda environment using:

```
conda env create -f conda_env.yml
conda activate colbert-v0.2
```

If you face any problems, please [open a new issue](https://github.com/stanford-futuredata/ColBERT/issues) and we'll help you promptly!


## Overview

Using ColBERT on a dataset typically involves the following steps.

**Step 0: Preprocess your collection.** At its simplest, ColBERT works with tab-separated (TSV) files: a file (e.g., `collection.tsv`) will contain all passages and another (e.g., `queries.tsv`) will contain a set of queries for searching the collection.

**Step 1: Train a ColBERT model.**  You can [train your own ColBERT model](#training) and [validate performance](#validation) on a suitable development set.

**Step 2: Index your collection.** Once you're happy with your ColBERT model, you need to [index your collection](#indexing) to permit fast retrieval. This step encodes all passages into matrices, stores them on disk, and builds data structures for efficient search.

**Step 3: Search the collection with your queries.** Given your model and index, you can [issue queries over the collection](#retrieval) to retrieve the top-k passages for each query.

Below, we illustrate these steps via an example run on the MS MARCO Passage Ranking task.


## Data

This repository works directly with a simple **tab-separated file** format to store queries, passages, and top-k ranked lists.


* Queries: each line is `qid \t query text`.
* Collection: each line is `pid \t passage text`. 
* Top-k Ranking: each line is `qid \t pid \t rank`.

This works directly with the data format of the [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) dataset. You will need the training triples (`triples.train.small.tar.gz`), the official top-1000 ranked lists for the dev set queries (`top1000.dev`), and the dev set relevant passages (`qrels.dev.small.tsv`). For indexing the full collection, you will also need the list of passages (`collection.tar.gz`).



## Training

Training requires a list of _<query, positive passage, negative passage>_ tab-separated triples.

You can supply **full-text** triples, where each line is `query text \t positive passage text \t negative passage text`. Alternatively, you can supply the query and passage **IDs** as a JSONL file `[qid, pid+, pid-]` per line, in which case you should specify `--collection path/to/collection.tsv` and `--queries path/to/queries.train.tsv`.


```
CUDA_VISIBLE_DEVICES="0,1,2,3" \
python -m torch.distributed.launch --nproc_per_node=4 -m \
colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 \
--triples /path/to/MSMARCO/triples.train.small.tsv \
--root /root/to/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
```

You can use one or more GPUs by modifying `CUDA_VISIBLE_DEVICES` and `--nproc_per_node`.


## Validation

Before indexing into ColBERT, you can compare a few checkpoints by re-ranking a top-k set of documents per query. This will use ColBERT _on-the-fly_: it will compute document representations _during_ query evaluation.

This script requires the top-k list per query, provided as a tab-separated file whose every line contains a tuple `queryID \t passageID \t rank`, where rank is {1, 2, 3, ...} for each query. The script also accepts the format of MS MARCO's `top1000.dev` and `top1000.eval` and you can optionally supply relevance judgements (qrels) for evaluation. This is a tab-separated file whose every line has a quadruple _<query ID, 0, passage ID, 1>_, like `qrels.dev.small.tsv`.

Example command:

```
python -m colbert.test --amp --doc_maxlen 180 --mask-punctuation \
--collection /path/to/MSMARCO/collection.tsv \
--queries /path/to/MSMARCO/queries.dev.small.tsv \
--topk /path/to/MSMARCO/top1000.dev  \
--checkpoint /root/to/experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--root /root/to/experiments/ --experiment MSMARCO-psg  [--qrels path/to/qrels.dev.small.tsv]
```


## Indexing

For fast retrieval, indexing precomputes the ColBERT representations of passages.

Example command:

```
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 \
python -m torch.distributed.launch --nproc_per_node=4 -m \
colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
--checkpoint /root/to/experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--collection /path/to/MSMARCO/collection.tsv \
--index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k \
--root /root/to/experiments/ --experiment MSMARCO-psg
```

The index created here allows you to re-rank the top-k passages retrieved by another method (e.g., BM25).

We typically recommend that you use ColBERT for **end-to-end** retrieval, where it directly finds its top-k passages from the full collection. For this, you need FAISS indexing.


#### FAISS Indexing for end-to-end retrieval

For end-to-end retrieval, you should index the document representations into [FAISS](https://github.com/facebookresearch/faiss).

```
python -m colbert.index_faiss \
--index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k \
--partitions 32768 --sample 0.3 \
--root /root/to/experiments/ --experiment MSMARCO-psg
```


## Retrieval

In the simplest case, you want to retrieve from the full collection:

```
python -m colbert.retrieve \
--amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
--queries /path/to/MSMARCO/queries.dev.small.tsv \
--nprobe 32 --partitions 32768 --faiss_depth 1024 \
--index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k \
--checkpoint /root/to/experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn \
--root /root/to/experiments/ --experiment MSMARCO-psg
```

You may also want to re-rank a top-k set that you've retrieved before with ColBERT or with another model. For this, use `colbert.rerank` similarly and additionally pass `--topk`.

If you have a large set of queries (or want to reduce memory usage), use **batch-mode** retrieval and/or re-ranking. This can be done by passing `--batch --retrieve_only` to `colbert.retrieve` and passing `--batch --log-scores` to colbert.rerank alongside `--topk` with the `unordered.tsv` output of this retrieval run.

Some use cases (e.g., building a user-facing search engines) require more control over retrieval. For those, you typically don't want to use the command line for retrieval. Instead, you want to import our retrieval API from Python and directly work with that (e.g., to build a simple REST API). Instructions for this are coming soon, but you will just need to adapt/modify the retrieval loop in [`colbert/ranking/retrieval.py#L33`](colbert/ranking/retrieval.py#L33).


## Releases

* v0.2.0: Sep 2020
* v0.1.0: June 2020

