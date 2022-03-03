# cudaNum=0,1 (Incase if you have multiple GPUs)
cudaNum=0

# Mention Any BEIR dataset here (which has been preprocessed)
export dataset=nfcorpus 

# Download the fine-tuned ColBERT model Checkpoint
if [ ! -d "msmarco.psg.l2" ]; then
    wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/models/ColBERT/msmarco.psg.l2.zip
    unzip msmarco.psg.l2.zip
fi

export CHECKPOINT="msmarco.psg.l2/checkpoints/colbert-300000.dnn"

# Path where preprocessed collection and queries are present
export COLLECTION="colbert-datasets/${dataset}/collection.tsv"
export QUERIES="colbert-datasets/${dataset}/queries.tsv"

# Path to store the faiss index and run output
export INDEX_ROOT="indices"
export OUTPUT_DIR="output"

# Path to store the rankings file
export RANKING_DIR="rankings/${dataset}"

# Num of partitions in for IVPQ Faiss index (You must decide yourself)
export NUM_PARTITIONS=96

# Some Index Name to store the faiss index
export INDEX_NAME=NFCORPUS

# Setting LD_LIBRARY_PATH for CUDA (Incase required!)
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ukp/thakur/cuda/lib64/

################################################################
# 0. BEIR Data Formatting: Format BEIR data useful for ColBERT #
################################################################ 

OMP_NUM_THREADS=6 python -m colbert.data_prep \
    --dataset ${dataset} \
    --split "test" \
    --collection $COLLECTION \
    --queries $QUERIES \

############################################################################
# 1. Indexing: Encode document (token) embeddings using ColBERT checkpoint #
############################################################################ 

CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m torch.distributed.launch \
    --nproc_per_node=2 -m colbert.index \
    --root $OUTPUT_DIR \
    --doc_maxlen 300 \
    --mask-punctuation \
    --bsize 128 \
    --amp \
    --checkpoint $CHECKPOINT \
    --index_root $INDEX_ROOT \
    --index_name $INDEX_NAME \
    --collection $COLLECTION \
    --experiment ${dataset}

###########################################################################################
# 2. Faiss Indexing (End-to-End Retrieval): Store document (token) embeddings using Faiss #
########################################################################################### 

CUDA_VISIBLE_DEVICES=${cudaNum} python -m colbert.index_faiss \
    --index_root $INDEX_ROOT \
    --index_name $INDEX_NAME \
    --partitions $NUM_PARTITIONS \
    --sample 0.3 \
    --root $OUTPUT_DIR \
    --experiment ${dataset}
    
####################################################################################
# 3. Retrieval: retrieve relevant documents of queries from faiss index checkpoint #
####################################################################################

CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m colbert.retrieve \
    --amp \
    --doc_maxlen 300 \
    --mask-punctuation \
    --bsize 256 \
    --queries $QUERIES \
    --nprobe 32 \
    --partitions $NUM_PARTITIONS \
    --faiss_depth 100 \
    --depth 100 \
    --index_root $INDEX_ROOT \
    --index_name $INDEX_NAME \
    --checkpoint $CHECKPOINT \
    --root $OUTPUT_DIR \
    --experiment ${dataset} \
    --ranking_dir $RANKING_DIR

######################################################################
# 4. BEIR Evaluation: Evaluate Rankings with BEIR Evaluation Metrics #
######################################################################

OMP_NUM_THREADS=6 python -m colbert.beir_eval \
    --dataset ${dataset} \
    --split "test" \
    --collection $COLLECTION \
    --rankings "${RANKING_DIR}/ranking.tsv"