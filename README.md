# CDBert

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2305.18760)
[![ACL](https://img.shields.io/badge/Paper-ACL-red.svg)](https://aclanthology.org/2023.findings-acl.70/)

This is the official implementation of the ACL 2023 Findings paper [Shuo Wen Jie Zi: Rethinking Dictionaries and Glyphs for Chinese Language Pre-training](https://aclanthology.org/2023.findings-acl.70/).

## Installation

```
pip install -r requirements.txt
```

## Synthetic Chinese Character Data
This dataset can also be used for OCR

## PolyMRC
A new machine reading comprehension task focusing on polysemy understanding

```
{"options": ["本着道义", "情义；恩情", "公正、合宜的道德、行为或道理", "坚持正义"], "sentence": ["汝之义绝高氏而归也，堂上阿奶仗汝扶持。"], "word": "义", "label": 0}

```
Download the dataset from [huggingface dataset](https://huggingface.co/datasets/tssn/PolyMRC)


## Data Preparation

- Download [Chinese Dictionary](https://github.com/mapull/chinese-dictionary)
- Download [CLUE](https://github.com/CLUEbenchmark/CLUE)
- Download [CCLUE](https://github.com/Ethan-yt/CCLUE)

The dataset structure should look like the following:

```
| -- data
	| -- Pretrain
		| -- train.json
		| -- dev.json
	| -- CLUE
		| -- afqmc
			| -- train.json
			| -- dev.json
			| -- test.json
		| -- c3
		| -- chid
		| -- cmnli
		| -- cmrc
		| -- csl
		| -- iflytek
		| -- tnews
		| -- wsc
		| -- chid
	| -- CCLUE
		| -- fspc
		| -- mrc
		| -- ner
		| -- punc
		| -- seg
		| -- tc
	| -- PolyMRC
		| -- mrc
	| -- glyph_embedding.pt

  
```

## CDBert

### Pre-train

```bash
export MODEL_NAME=$1
export MODEL_PATH='prev_trained_models/'$1
export TRAIN=$2
export VAL=$3
export RADICAL=$4
export RID=$5
export LAN=$6
export BSZ=$8
export GLYPH=$9

CUDA_LAUNCH_BLOCKING=1 \
PYTHONPATH=$PYTHONPATH:. \
python -m torch.distributed.launch \
        --nproc_per_node=$7 \
        --master_port 40000 \
        pretrain.py \
        --distributed --multiGPU \
        --train datasets/$TRAIN \
        --valid datasets/$VAL \
        --batch_size $BSZ \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 1.0 \
        --lr 5e-5 \
        --epoch 10 \
        --losses dict \
        --num_workers 1 \
        --backbone $MODEL_PATH \
        --individual_vis_layer_norm False \
        --output ckpts/$MODEL_NAME \
        --rid $RID \
        --max_text_length 256 \
        --radical_path $RADICAL \
```

### CLUE (We only show the script for TNEW'S)

```bash
export MODEL_NAME=$1
export MODEL_PATH='prev_trained_models/'$1
export TASK_NAME=$2
export LR=$3
export EPOCH=$4
export BSZ=$5
export LEN=$6

PYTHONPATH=$PYTHONPATH:. \
python clue_tc.py \
        --task_name $TASK_NAME \
        --train train \
        --valid dev \
        --test test \
        --batch_size $BSZ \
        --valid_batch_size $BSZ \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 1.0 \
        --lr $LR \
        --epoch $EPOCH \
        --num_workers 1 \
        --model_name $MODEL_NAME \
        --backbone $MODEL_PATH \
        --load ckpts/$MODEL_NAME/Epoch10 \
        --individual_vis_layer_norm False \
        --output outputs/CLUE/$TASK_NAME/$MODEL_NAME \
        --rid 368 \
        --embedding_lookup_table embedding/$MODEL_NAME/ \
        --fuse attn \
        --max_text_length  $LEN \
        --glyph radical \
```

### CCLUE (We only show the script for MRC)

```bash
export MODEL_PATH='prev_trained_models/'$1
export TASK_NAME=$2
export LR=$3
export EPOCH=$4
export BSZ=$5
export LEN=$6

PYTHONPATH=$PYTHONPATH:. \
python cclue_mrc.py \
        --task_name $TASK_NAME \
        --train train \
        --valid dev \
        --test test \
        --batch_size $BSZ \
        --valid_batch_size $BSZ \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 1.0 \
        --lr $LR \
        --epoch $EPOCH \
        --num_workers 1 \
        --model_name $MODEL_NAME \
        --backbone $MODEL_PATH \
        --load ckpts/$MODEL_NAME/Epoch10 \
        --individual_vis_layer_norm False \
        --output outputs/CCLUE/$TASK_NAME/$MODEL_NAME \
        --rid 233 \
        --embedding_lookup_table embedding/$MODEL_NAME/ \
        --fuse attn \
        --max_text_length  $LEN \
        --choices 4 \
        --glyph radical \
```

### PolyMRC

```bash
export MODEL_NAME=$1
export MODEL_PATH='prev_trained_models/'$1
export TASK_NAME=$2
export LR=$3
export EPOCH=$4
export BSZ=$5
export LEN=$6

PYTHONPATH=$PYTHONPATH:. \
python dict_key.py \
        --task_name $TASK_NAME \
        --train train \
        --valid dev \
        --test test \
        --batch_size $BSZ \
        --valid_batch_size $BSZ \
        --optim adamw \
        --warmup_ratio 0.05 \
        --clip_grad_norm 1.0 \
        --lr $LR \
        --epoch $EPOCH \
        --num_workers 1 \
        --model_name $MODEL_NAME \
        --backbone $MODEL_PATH \
        --load ckpts/$MODEL_NAME/Epoch10 \
        --individual_vis_layer_norm False \
        --output outputs/CCLUE/$TASK_NAME/$MODEL_NAME \
        --rid 233 \
        --embedding_lookup_table embedding/$MODEL_NAME/ \
        --fuse attn \
        --max_text_length  $LEN \
        --choices 4 \
        --glyph radical \
```

## Citation
```bibtex
@inproceedings{wang-etal-2023-rethinking,
    title = "Rethinking Dictionaries and Glyphs for {C}hinese Language Pre-training",
    author = "Wang, Yuxuan and Wang, Jack and Zhao, Dongyan and Zheng, Zilong",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.70",
    pages = "1089--1101",
}
```
