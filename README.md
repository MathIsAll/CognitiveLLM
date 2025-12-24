# Endogenous cognitive modularity in large language models enables human-like continual adaptation


This project is a implementation in PyTorch for Endogenous cognitive modularity in large language models enables human-like continual adaptation.
This implement includes all compares baselines.

## Acknowledgment

This project is built upon the foundation laid by [Endogenous cognitive modularity in large language models enables human-like continual adaptation](Anonymous). The original code from their project is licensed under the [MIT License](https://github.com/princeton-nlp/MeZO/blob/main/LICENSE).


## Installation

Please install the latest versions of PyTorch (`pytorch` following [https://pytorch.org](https://pytorch.org)), Transformers (`transformers`), and Accelerate (`accelerate`). This code is tested on `torch==2.1.0.dev20230514+cu118`, `transformers==4.28.1`, and `accelerate==0.17.1` with Python 3.9.7, but should work with older/later versions of these packages too.
Before running this project, please replace the adamw.py in `huggingface transformer` with the one in this folder.

## Usage

Use `cimeo4pretrain.py` for the unsupervised pre-training:
```bash
python cimeo4pretrain.py {ARGUMENTS}
```

Use `cimeo4sft.py` for supervised fine-tuning:
```bash
python cimeo4sft.py {ARGUMENTS}
```

`Please replace the folder with the true address`

Below python files are realized compared baselines in Table 1.
```bash
python adalomo4sft.py #Lv K, Yan H, Guo Q, et al. Adalomo: Low-memory optimization with adaptive learning rate[C]//Findings of the Association for Computational Linguistics: ACL 2024. 2024: 12486-12502.

python adarankgrad.py #Refael Y, Svirsky J, Shustin B, et al. AdaRankGrad: Adaptive Gradient Rank and Moments for Memory-Efficient LLMs Training and Fine-Tuning[C]//The Thirteenth International Conference on Learning Representations.

python bone4sft.py #Kang J. Bone: Block Affine Transformation as Parameter Efficient Fine-tuning Methods for Large Language Models[J]. arXiv e-prints, 2024: arXiv: 2409.15371.

python dora4sft.py #Liu S Y, Wang C Y, Yin H, et al. Dora: Weight-decomposed low-rank adaptation[C]//Forty-first International Conference on Machine Learning. 2024.

python eva4sft.py #Paischer F, Hauzenberger L, Schmied T, et al. One Initialization to Rule them All: Fine-tuning via Explained Variance Adaptation[C]//Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning.

python galore4sft.py #Zhao J, Zhang Z, Chen B, et al. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection[C]//International Conference on Machine Learning. PMLR, 2024: 61121-61143.

python lorafa4sft.py #Zhang L, Zhang L, Shi S, et al. Lora-fa: Memory-efficient low-rank adaptation for large language models fine-tuning[J]. arXiv preprint arXiv:2308.03303, 2023.

python lora4sft.py #Hu E J, Wallis P, Allen-Zhu Z, et al. LoRA: Low-Rank Adaptation of Large Language Models[C]//International Conference on Learning Representations.

python miss4sft.py #Kang J, Yin Q. Balancing LoRA Performance and Efficiency with Simple Shard Sharing[J]. arXiv preprint arXiv:2409.15371, 2024.

python olora4sft.py #Büyükakyüz K. Olora: Orthonormal low-rank adaptation of large language models[J]. arXiv preprint arXiv:2406.01775, 2024.

python pissa4sft.py #Meng F, Wang Z, Zhang M. Pissa: Principal singular values and singular vectors adaptation of large language models[J]. Advances in Neural Information Processing Systems, 2024, 37: 121038-121072.

python road4sft.py #Liao B, Monz C. 3-in-1: 2d rotary adaptation for efficient finetuning, efficient batching and composability[J]. Advances in Neural Information Processing Systems, 2024, 37: 35018-35048.

python shira4sft.py #Bhardwaj K, Pandey N P, Priyadarshi S, et al. Sparse high rank adapters[C]//Proceedings of the 38th International Conference on Neural Information Processing Systems. 2024: 13685-13715.

python ZOAdamu4sft.py #Jiang S, Chen Q, Pan Y, et al. Zo-adamu optimizer: Adapting perturbation by the momentum and uncertainty in zeroth-order optimization[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(16): 18363-18371.
```

Below python files are used to evaluate the overlap used in Figure 2.
```bash
python overlap_count.py # calculate the overlap rates over different cognition pairs.

python overlap_count_by_layer.py # calculate the overlap rates over different cognition pairs by layers.

python overlap_count_by_layer_by_cognition.py #calculate the overlap rates over different cognition pairs on different cognition pairs.
```

Below python files are used to evaluate the weight subspace analysis in Figure 4.
```bash
python singular_value_plot.py # plotting the maximum and mean principle angles between base model and fine-tuned model.
```