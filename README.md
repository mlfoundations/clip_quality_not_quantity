# Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP


This repository contains code for downloading the six datasets used in the paper
[Quality Not Quantity: On the Interaction between Dataset Design and Robustness of CLIP](https://arxiv.org/abs/2208.05516) by Thao Nguyen, Gabriel Ilharco, Mitchell Wortsman, Sewoong Oh, Ludwig Schmidt.

TLDR: We investigate how pre-training on different web-crawled data sources would affect CLIP's robustness to natural distribution shifts, and find that the robustness induced by each pre-training dataset varies widely. By analyzing the interactions of these datasets through both experiments and theoretical
analyses, we also observe that simply combining multiple datasets dilutes the robustness of the best-performing one.

### Abstract
Web-crawled datasets have enabled remarkable generalization capabilities in recent image-text models such as CLIP (Contrastive Language-Image pre-training) or Flamingo, but little is known about the dataset creation processes. In this work, we introduce a testbed of six publicly available data sources - YFCC, LAION, Conceptual Captions, WIT, RedCaps, Shutterstock - to investigate how pre-training distributions induce robustness in CLIP. We find that the performance of the pre-training data varies substantially across distribution shifts, with no single data source dominating. Moreover, we systematically study the interactions between these data sources and find that combining multiple sources does not necessarily yield better models, but rather dilutes the robustness of the best individual data source. We complement our empirical findings with theoretical insights from a simple setting, where combining the training data also results in diluted robustness. In addition, our theoretical model provides a candidate explanation for the success of the CLIP-based data filtering technique recently employed in the LAION dataset. Overall our results demonstrate that simply gathering a large amount of data from the web is not the most effective way to build a pre-training dataset for robust generalization, necessitating further study into dataset design.

### Datasets
Each folder with a dataset name contains the relevant code to download data instances into WebDataset format given the corresponding metadata:

* YFCC: our 15M subset randomly sampled from YFCC100M could be found [here](https://drive.google.com/file/d/1bakfXSS5Dcbf6JtQ-sNEH-ghqYlNZxmd/view?usp=sharing)
* LAION: our subset could be found [here](https://drive.google.com/file/d/1m46cdsQLVT2YA3gUy9cnvyu6sBnyqkO4/view?usp=sharing). Note that this contains slightly more than 15M to account for a small fraction of bad URLs and throttled instances during the downloading process
* CC12M: we obtain the training set from the [official data release](https://github.com/google-research-datasets/conceptual-12m)
* WIT: refer to the [official data release](https://github.com/google-research-datasets/wit/blob/main/DATA.md) for the metadata. Note that for WIT, due to heavy throttling issues, we could not download data in parallel within a single machine. Users may consider using AWS instances to speed up this process
* RedCaps: the initial [dataset release](https://redcaps.xyz/download) grouped instances into topics with the following format `<subreddit>_<year>.json`. For our paper, we shuffled RedCaps instances across all of these json files before starting the downloading process, in order to randomize the training data (see `shuffle_annotations.py`). The updated json files could be found [here](https://drive.google.com/file/d/1Mbee_boX6Fvw2Ys-pOOWINkIppa6G68X/view?usp=sharing)
* Shutterstock: the metadata could be found [here](https://drive.google.com/file/d/1mSNAL7u8y39O_fb66f38uLRm1zUnDH9O/view?usp=sharing)

### Training
Our paper uses the [OpenCLIP](https://github.com/mlfoundations/open_clip) repository for training CLIP models.

### Evaluation
Example code for evaluating OpenCLIP pre-trained models on a range of downstream settings can be found at the [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) repository.
