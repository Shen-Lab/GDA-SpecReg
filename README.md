# Graph Domain Adaptation via Theory-Grounded Spectral Regularization

PyTorch implementation for [Graph Domain Adaptation via Theory-Grounded Spectral Regularization](https://openreview.net/forum?id=OysfLgrk8mk)

Yuning You, Tianlong Chen, Zhangyang Wang, Yang Shen

In ICLR 2023.

## Overview

This repo provides two GNN spectral regularization implementations (SSReg and MFRReg) to tackle distribution shifts for graph data (i.e. graph domain adaptation).
The developed regularization is grounded on the model-based risk bound analysis in the paper (Corollary 1, Lemma 1, and Lemma 2).

<p align="center" width="100%">
    <img width="33%" src="https://github.com/Shen-Lab/GDA-SpecReg/blob/main/gda.png">
</p>

## Experiments

* [Cross-species protein-protein interaction (link) prediction](https://github.com/Shen-Lab/GDA-SpecReg/tree/main/ppi_prediction)
* [Temporally shifted paper topic (node) classification](https://github.com/Shen-Lab/GDA-SpecReg/tree/main/paper_topic_classification)

## Citation

If you use this code for you research, please cite our paper.

```
@inproceedings{you2023graph,
  title={Graph Domain Adaptation via Theory-Grounded Spectral Regularization},
  author={You, Yuning and Chen, Tianlong and Wang, Zhangyang and Shen, Yang},
  booktitle={International Conference on Learning Representations},
  year = {2023}
}
```
