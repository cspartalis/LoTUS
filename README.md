<img src="readme_images/banner.png" alt="Banner" style="max-width:100%; height:auto;">

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue)](https://cvpr.thecvf.com/virtual/2025/poster/33292)
[![arXiv](https://img.shields.io/badge/arXiv-2503.18314-red)](https://arxiv.org/abs/2503.18314)
![Python](https://img.shields.io/badge/Python-3.11.5-blue?logo=python)
![Conda](https://img.shields.io/badge/Conda-23.1.0-green?logo=anaconda)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?logo=pytorch)

:tangerine::robot: In Homer's Odyssey, the lotus fruit induces forgetfulness, erasing the memory of Ulysses' comrades and their desrire to return home. In Machine Learning, forgetting is not always a bug; it can be a feature. So, *why and how ML models need to forget?*

:female_detective: **User Data Removal** upholds the *right to be forgotten*. When a user opts out of data collection, their information must not only be deleted from databases but also eliminated from any ML models trained on it.  

# :memo: TL;DR

:dart: **LoTUS** is a novel **Machine Unlearning** method designed to eliminate the influence of specific training samples from pre-trained models.  

:bulb: **Addressing Memorization**: Deep Neural Networks typically memorize training data, leading to over-confident predictions.  

:white_check_mark: **Uncertainty Introduction**: LoTUS mitigates over-confidence by smoothing the output probability distributions of the samples to be unlearned.  

:bulb: **Selective Information Removal**: Only the **unique, sample-specific information** in the data to be unlearned should be removed. **Shared information**, common across other training samples, should be retained to preserve model utility.

:white_check_mark: **Information-Theoretic Bound**: LoTUS introduces uncertainty up to a well-defined **information-theoretic bound**, ensuring effective unlearning while preserving generalization.  

## :rocket: Key Contributions  

:seedling: **Novel Method**: Logits Tempering Unlearning Strategy (**LoTUS**), a novel entropy-based unlearning method with theoretical guarantees.

:seedling: **Novel Metric**: Retrain-Free Jensen Shannon Divergence (**RF-JSD**), designed for real-world scenarios where retraining a model from scratch without the forget samples is impractical or infeasible. Existing metrics often assume access to a retrained reference model.  

:seedling: **Novel Benchmark**: Large-scale evaluation on **ImageNet1k** with **limited data access**, simulating real-world unlearning scenarios.

## :trophy: Why LoTUS?  

:1st_place_medal: **Effectiveness** | :1st_place_medal: **Efficiency** | :1st_place_medal: **Scalability**  


<img src="readme_images/results.png" alt="Results" style="max-width:100%; height:auto;">


# :microscope: Experimental Setup

## Unlearning Methods 
All the methods are implemented in `src/unlearning_methods/`.
| Method | File | Paper | Code |
|--------|------|-------|------|
| **LoTUS** | `our_class.py` | [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/33292), [arXiv](https://arxiv.org/abs/2503.18314) | [:computer:](https://github.com/cspartalis/LoTUS/blob/main/src/unlearning_methods/lotus_class.py) |
| Finetuning | `naive_unlearning_class.py` | [:closed_book:](https://arxiv.org/abs/2302.09880) | [:computer:](https://github.com/cspartalis/LoTUS/blob/main/src/unlearning_methods/naive_unlearning_class.py) |
| NegGrad+ | `naive_unlearning_class.py` | [:closed_book:](https://arxiv.org/abs/2302.09880) | [:computer:](https://github.com/cspartalis/LoTUS/blob/main/src/unlearning_methods/naive_unlearning_class.py) |
| Rnd Labeling | `naive_unlearning_class.py` | [:closed_book:](https://arxiv.org/abs/2010.10981) | [:computer:](https://github.com/cspartalis/LoTUS/blob/main/src/unlearning_methods/naive_unlearning_class.py) |
| Bad Teacher | `bad_teaching_class.py` | [:closed_book:](https://arxiv.org/abs/2205.08096) | [:computer:](https://github.com/vikram2000b/bad-teaching-unlearning) |
| SCRUB | `scrub_class.py` | [:closed_book:](https://arxiv.org/abs/2302.09880) | [:computer:](https://github.com/meghdadk/SCRUB) |
| SSD | `ssd_class.py` | [:closed_book:](https://arxiv.org/abs/2308.07707) | [:computer:](https://github.com/if-loops/selective-synaptic-dampening) |
| UNSIR | `unsir_class.py` | [:closed_book:](https://arxiv.org/abs/2111.08947) | [:computer:](https://github.com/vikram2000b/Fast-Machine-Unlearning) |

## Evaluation Metrics
All the evaluation metrics are implemented in `src/helpers/eval.py`.
| Metric | Paper | Code | 
|--------|-------|------|
|**Retrain-Free Jensen-Shannon Divergence (RF-JSD)**| [CVPR](https://cvpr.thecvf.com/virtual/2025/poster/33292), [arXiv](https://arxiv.org/abs/2503.18314) | [:computer:](https://github.com/cspartalis/LoTUS/blob/main/src/helpers/eval.py) `log_js_proxy()` |
|Jensen-Shannon Divergence (JSD)| [:closed_book:](https://arxiv.org/abs/2205.08096) | [:computer:](https://github.com/vikram2000b/bad-teaching-unlearning) |
| MIA | [:closed_book:](https://arxiv.org/abs/2308.07707) [:closed_book:](https://arxiv.org/abs/2205.08096) | [:computer:](https://github.com/if-loops/selective-synaptic-dampening) [:computer:](https://github.com/vikram2000b/bad-teaching-unlearning)|

# :hammer_and_wrench: Reproducibility
All results presented in the paper can be reproduced. They have been documented in Jupyter notebooks.

Below is a mapping of the Tables and Figures from the paper to the corresponding notebooks:
| Tables and Figures | Corresponding Notebooks |
|--------------------|-------------------------|
| Table 1: Performance Summary | `notebooks/results.ipynb` |
| Table 2: LoTUS with Synthetic Data | `notebooks/results.ipynb` |
| Table 3: Scaling up the Forget set | `notebooks/results.ipynb` |
| Table 4: Large-Scale Unlearning | `notebooks/results_imagenet.ipynb` |
| Table 4: Accuracy Metrics | `notebooks/results.ipynb` |
| Table 6: Scaling up the Forget set (Appendix) | `notebooks/results.ipynb` |
| Figure 3: Duplicates in MUFAC | `notebooks/clean_MUFAC.ipynb` |
| Figure 4: MUFAC Class Distribution | `notebooks/EDA_MUFAC.ipynb` |
| Figure 5: Failure Analysis | `notebooks/check_orthogonality.ipynb` |
| Figure 7: Streisand Effect | `notebooks/streisand.ipynb` |

The results are reproduced following the steps 1--4:

## Step 1: Environment Setup
The environment 'MaUn' contains Conda and Pip packages. You can install them using the `environment.yml` file:
1. In the last line of the file, replace \<username\> with your username.
2. ```conda env create -f environment.yml```
3. ```conda activate MaUn```
If this fails, you can install the Conda and Pip packages explicitly using the `conda_requirements.txt` and `pip_requirements.txt` explicitly. If the installation of a package fails, then try to remove the version (e.g., mlflow==2.8.0 &rarr; mlflow) in the corresponding file.

We use Python 3.11.5 and CUDA 12.1 in Ubuntu 24.04.1 LTS.

## Step 2: Datasets
* CIFAR-10/100
    1. These datasets are downloaded automatically in `~/home/data/`.
    2. You can define how many training samples are designated for unlearning using the `frac_per_class_forget` variable of the `UnlearningDataLoader` class in `src/helpers/data_utils.py`.
* MUFAC:
    1. Download the original dataset following the instruction [here](https://github.com/ndb796/MachineUnlearning).
    2. Move the `custom_korean_family_dataset_resolution_128/` folder into the `~/home/data/` folder.
    3. Run `notebooks/clean_MUFAC.ipynb` to clean the dataset and create the `~/home/data/custom_korean_family_dataset_resolution_128_clean\` folder.
* ImageNet1k:
    1. Download the dataset from [image-net.org](image-net.org).
    2. Our `~/home/data/pytorch_imagenet1k` folder looks like this:
    ```
    pytorch_imagenet1k/
    ├── ILSVRC2012_devkit_t12.tar.gz
    ├── ILSVRC2012_img_train.tar.gz
    ├── ILSVRC2012_img_val.tar.gz
    ├── train/ 
    ├── val/ 
    └── meta.bin 
    ```
## Step 3: MLflow Setup
Set your `mlflow_tracking_uri` in `src/helpers/mlflow_utils`. You assign it once, only in this file.

## Step 4: Running the Project
Run the `src/bash_scripts/reproduce_results.sh` file.

## Hardware
For ImageNet1k781 experiments, we used an NVIDIA RTX A6000 48GB GPU.  The remaining experiments were performed on an NVIDIA RTX 4080 16GB GPU. We also used an Intel i7-12700K CPU and 32GB RAM.

# :bangbang: Citation

If you use our work, please cite:
```bibtex
@inproceedings{spartalis2025lotus,
  title={{LoTUS: Large-Scale Machine Unlearning with a Taste of Uncertainty}}, 
  author={Christoforos N. Spartalis and Theodoros Semertzidis and Efstratios Gavves and Petros Daras},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```


<!-- # Project Structure
The project is organized as follows: -->
<!-- LoTUS/
├── notebooks/
│   ├── check_orthogonality.ipynb
│   ├── clean_MUFAC.ipynb
│   ├── EDA_MUFAC.ipynb
│   ├── results_imagenet.ipynb
│   ├── results.ipynb
│   ├── streisand.ipynb
│   ├── clean_MUFAC.ipynb
|   └──
├── src/
│   ├── bash_scripts/ 
|   |   ├── dataset_ablation.sh
|   |   ├── reproduce_results.sh
|   |   └── tuning.sh
|   ├── helpers/
|   |   ├── config.py
|   |   ├── data_utils.py
|   |   ├── eval.py
|   |   ├── mlflow_utils.py
|   |   ├── models.py
|   |   ├── mufac_utils.py
|   |   └── seed.py
|   ├── unlearning_methods/
|   |   ├── bad_teaching_class.py
|   |   ├── naive_unlearning_class.py
|   |   ├── our_class.py
|   |   ├── scrub_class.py
|   |   ├── ssd_class.py
|   |   ├── unlearning_base_class.py
|   |   └── unsir_class.py
│   ├── original_imagenet.py
│   ├── original_imagenet.py
│   ├── retrain.py
│   ├── train.py
│   ├── unlearn_imagenet.py
│   └── unlearn.py
├── environment.yml
├── conda_requirements.txt
├── pip_requirements.txt
└── README.md
``` -->
