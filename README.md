## Introduction

This repo contains implementation for the COLING 2022
paper [On the Role of Pre-trained Language Models in Word Ordering: A Case Study with BART](https://aclanthology.org/2022.coling-1.567.pdf)
. It achieves state-of-the-art results on the classic word ordering task and the partial tree linearization task. Here is a short [oral presentation](https://youtu.be/sWhNsSmqLdw) for a quick grasp of the gist. 

The implementation is based on [fairseq](https://github.com/facebookresearch/fairseq). To see the modifications, compare
the HEAD commit with the `init with fairseq v0.10.2` comit, which is identical to the `v0.10.2` tag of fairseq. Note that our implementation is only for research purpose and there is huge room for efficiency improvements.

Analysis with structural probing is based on [structural-probes](https://github.com/john-hewitt/structural-probes). As
the analysis follows exactly the default settings, we only provide code to extract relevant token features.

## Dataset

The license of the Penn Treebank prevent us from publicizing the dataset. Thus we only include data samples
in `./ptb_trees`. Feel free to contact the first author via simtony2@gmail.com with a prove (screenshot or something)
that you have a copy of the Penn Treebank dataset. We will send you the full preprocessed copy.

## Hardware Requirements

Make sure your GPU supports fp16 and has a large memory (e.g., 24GB). For reference, RAND results are produced on 2080Ti
and BART on 32GB V100. Decoding with large beam size (e.g. 1024) are run on 80GB A100.

## Dependency

The results are produced with `torch==1.10`. You may need to install `mlrunner==0.5.8` to run the experiments and `multiset` for analysis.

## Steps to reproduce

1. Pull the current repo and install the code base following the [fairseq](https://github.com/facebookresearch/fairseq)
   instructions. Change our directory to the root of this repo.
2. Download files of BART model
    1. https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz
    2. https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    3. https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

   and extract/put the contents in `./bart`.
3. Prepare the datasets with `prepare_tree_raw.ipynb` and `prepare_tree_bin.ipynb`.
4. For convienence we manage our experiments using [mlrunner](https://github.com/simtony/mlrunner). See the comments
   in `params*.yaml` files for hyperparamters of each experiment. Use `run -y params.yaml -t <title> -o output` to train the RAND models for selected
   experiments (specified in `<title>`) and `run -y params_decode.yaml -t <title> -o output_decode` to decode. BART results can be similarly reproduced with `params_bart*.yaml`. See the
   document of mlrunner for detailed usage. If you prefer raw bash commands, you can use `--dry-run` to obtain them.
5. Follow `analysis.ipynb` to aggregate the results. Follow `extract.ipynb` to extract intermediate features for structrual probing. They should work as expected. 

For reference we also include logs and checkpoints of each experiment in [google drive](https://drive.google.com/drive/folders/1hePLBzgV9nYGhPP4QNkBsro1pvldykdk?usp=sharing), you can use [tensorboard](https://www.tensorflow.org/tensorboard) to visualize the training process.