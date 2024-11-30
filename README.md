# Zero-Shot Learning

Resources include papers, code, datasets, and relevant links about zero-shot learning. 

**Note**: NOT a complete list of all accepted papers but a collection based on my readings in the field.



## Contributing

The repo is continuously under construction and will be updated regularly. If you have any suggestions, please feel free to raise an issue.  Contributions are welcome!



## Table of Contents

### :page_with_curl:Papers

#### Survey

Xian Y, Lampert C H, Schiele B, et al. Zero-shot learning—a comprehensive evaluation of the good, the bad and the ugly（*TPAMI 2018*）[[project]](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/)

Wang W, Zheng V W, Yu H, et al. A survey of zero-shot learning: Settings, methods, and applications（*TIST 2019*）[[paper]](https://dl.acm.org/doi/abs/10.1145/3293318)

#### Embedding-based

**APN**: Xu W, Xian Y, Wang J, et al. Attribute prototype network for zero-shot learning（*NeurIPS 2020*）[[paper]](https://papers.nips.cc/paper/2020/file/fa2431bf9d65058fe34e9713e32d60e6-Paper.pdf) [[code]](https://github.com/wenjiaXu/APN-ZSL)

**GEM-ZSL**: Liu Y, Zhou L, Bai X, et al. Goal-oriented gaze estimation for zero-shot learning（*CVPR 2021*）[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Goal-Oriented_Gaze_Estimation_for_Zero-Shot_Learning_CVPR_2021_paper.pdf) [[code]](https://github.com/osierboy/GEM-ZSL)

**TransZero++**: Chen S, Hong Z, Hou W, et al. TransZero++: Cross attribute-guided transformer for zero-shot learning（*TPAMI 2022*）[[paper]](https://ieeexplore.ieee.org/document/9987664) [[code]](https://github.com/shiming-chen/TransZero_pp)

**IEAM-ZSL**: Alamri F, Dutta A. Implicit and explicit attention mechanisms for zero-shot learning（*Neurocomputing 2023*）[[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0925231223002291) [[code]](https://github.com/faisalalamri0/ieam-zsl)

**CoAR-ZSL**: Du Y, Shi M, Wei F, et al. Boosting zero-shot learning via contrastive optimization of attribute representations（*TNNLS 2023*）[[paper]](https://arxiv.org/pdf/2207.03824) [[code]](https://github.com/dyabel/CoAR-ZSL)

**DUET**: Chen Z, Huang Y, Chen J, et al. Duet: Cross-modal semantic grounding for contrastive zero-shot learning（*AAAI 2023*）[[paper]](https://arxiv.org/pdf/2207.01328) [[code]](https://github.com/zjukg/DUET)

**PSVMA**: Liu M, Li F, Zhang C, et al. Progressive semantic-visual mutual adaption for generalized zero-shot learning（*CVPR 2023*）[[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_Progressive_Semantic-Visual_Mutual_Adaption_for_Generalized_Zero-Shot_Learning_CVPR_2023_paper.pdf) [[code]](https://github.com/ManLiuCoder/PSVMA)

**ReZSL**: Ye Z, Yang G, Jin X, et al. Rebalanced zero-shot learning（*TIP 2023*）[[paper]](https://ieeexplore.ieee.org/abstract/document/10188601) [[code]](https://github.com/FouriYe/ReZSL-TIP23)

**DFAN**: Xiang L, Zhou Y, Duan H, et al. Dual Feature Augmentation Network for Generalized Zero-shot Learning（*BMVC 2023*）[[paper]](https://papers.bmvc2023.org/0534.pdf) [[code]](https://github.com/Sion1/DFAN)

**EMP**: Zhang Y, Feng S. Enhancing domain-invariant parts for generalized zero-shot learning（ACMMM 2023）[[paper]](https://dl.acm.org/doi/abs/10.1145/3581783.3611764)

**PORNet**: Liu M, Zhang C, Bai H, et al. Part-object progressive refinement network for zero-shot learning（*TIP 2024*）[[paper]](https://ieeexplore.ieee.org/document/10471325) [[code]](https://github.com/ManLiuCoder/POPRNet)

**ZSLViT**: Chen S, Hou W, Khan S, et al. Progressive Semantic-Guided Vision Transformer for Zero-Shot Learning（*CVPR 2024*）[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Progressive_Semantic-Guided_Vision_Transformer_for_Zero-Shot_Learning_CVPR_2024_paper.pdf) [[code]](https://github.com/shiming-chen/ZSLViT)

#### Generative-based

**CADA-VAE**: Schonfeld E, Ebrahimi S, Sinha S, et al. Generalized zero-and few-shot learning via aligned variational autoencoders（*CVPR 2019*）[[paper]](https://arxiv.org/pdf/1812.01784.pdf) [[code]](https://github.com/edgarschnfld/CADA-VAE-PyTorch)

**SDGZSL**: Chen Z, Luo Y, Qiu R, et al. Semantics disentangling for generalized zero-shot learning（*ICCV 2021*）[[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Semantics_Disentangling_for_Generalized_Zero-Shot_Learning_ICCV_2021_paper.pdf) [[code]](https://github.com/uqzhichen/SDGZSL)

**PSVMA+**: Liu M, Bai H, Li F, et al. PSVMA+: Exploring Multi-Granularity Semantic-Visual Adaption for Generalized Zero-Shot Learning（*TPAMI 2024*）[[paper]](https://ieeexplore.ieee.org/abstract/document/10693541)

**IAB**: Jiang C, Shen Y, Chen D, et al. Estimation of Near-Instance-Level Attribute Bottleneck for Zero-Shot Learning（*IJCV 2024*）[[paper]](https://link.springer.com/article/10.1007/s11263-024-02021-x) [[code]](https://github.com/LanchJL/IAB-GZSL)

#### CLIP-based

**CLIP**: Radford A, Kim J W, Hallacy C, et al. Learning transferable visual models from natural language supervision（*PLMR 2021*）[[paper]](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) [[code]](https://github.com/OpenAI/CLIP)

**CALIP**: Guo Z, Zhang R, Qiu L, et al. Calip: Zero-shot enhancement of clip with parameter-free attention（*AAAI 2023*）[[paper]](https://arxiv.org/pdf/2209.14169) [[code]](https://github.com/ZiyuGuo99/CALIP)

**CHiLS**: Novack Z, McAuley J, Lipton Z C, et al. Chils: Zero-shot image classification with hierarchical label sets（*PLMR 2023*）[[paper]](https://proceedings.mlr.press/v202/novack23a/novack23a.pdf) [[code]](https://github.com/acmi-lab/CHILS)

**MoDE**: Ma J, Huang P Y, Xie S, et al. MoDE: CLIP Data Experts via Clustering（*CVPR 2024*）[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_MoDE_CLIP_Data_Experts_via_Clustering_CVPR_2024_paper.pdf) [[code]](https://github.com/facebookresearch/MetaCLIP/tree/main/mode)

**AlignCLIP**: Zhang L, Yan K, Ding S. AlignCLIP: Align Multi Domains of Texts Input for CLIP models with Object-IoU Loss（*ACMMM 2024*）[[paper]](https://openreview.net/pdf?id=td6ndgRL6l)



### :file_folder:Datasets

| Name                                                         | #Sample | Categories | Seen/Unseen | #Attribute |
| ------------------------------------------------------------ | ------- | ---------- | ----------- | ---------- |
| Caltech-UCSD-Birds-200-2011 [(CUB)](http://www.vision.caltech.edu/datasets/cub_200_2011/) | 11788   | Bird       | 150/50      | 312        |
| Animals with Attributes 2 [(AWA2)](https://cvml.ista.ac.at/AwA2) | 37322   | Animals    | 40/10       | 85         |
| SUN Attribute [(SUN)](https://cs.brown.edu/~gmpatter/sunattributes.html) | 14340   | Scenes     | 645/72      | 102        |
| attributes Pascal and Yahoo [(aPY)](https://vision.cs.uiuc.edu/attributes/) | 15339   | Objects    | 20/12       | 64         |

Data Splits and Features by Xian [[download]](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip)



### :link:Other resources

Zero-Shot Learning  Explained [[link]](https://encord.com/blog/zero-shot-learning-explained/)

Zero-Shot Learning: An Introduction [[link]](https://learnopencv.com/zero-shot-learning-an-introduction/)

Zero-Shot Learning|Papers with code [[link]](https://paperswithcode.com/task/zero-shot-learning)
