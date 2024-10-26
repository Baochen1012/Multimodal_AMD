# Towards Optimized Multi-Modal AMD Categorization: LoRA-Enhanced Pre-Trained Models with CFP and OCT Feature Fusion via Deep Canonical Correlation Analysis

Code and data for multi-modal categorization of age-related macular degeneration (4 classes: normal, dry AMD, pcv, wet AMD).

This study proposes a novel deep learning model that leverages the joint diagnostic potential of CFP and OCT images for the automatic detection of multi-modal AMD. We effectively extend the pre-trained retinal foundation model through fine-tuning to create a model for multi-modal AMD recognition. Specifically, CFP and OCT images are processed by two independent ViT-large models for feature extraction, generating high-dimensional feature representations for each modality. These feature representations are input into the DCCA module after undergoing nonlinear transformations. DCCA performs fine-grained nonlinear mapping on the features of different modalities through two independent neural networks, aiming to maximize the nonlinear correlation between the two modalities, thus achieving efficient cross-modal feature fusion. Subsequently, we employ a concatenation fusion strategy to effectively merge the transformed features from both modalities, forming a comprehensive feature vector. Finally, we utilize the fused features to train a classifier for AMD classification. To address the challenge of significantly increased computational complexity during the multi-modal integration process, this study introduces LoRA technology for model fine-tuning. LoRA optimizes a high-dimensional parameter matrix by decomposing it into two low-rank matrices, allowing only a small number of parameters to be adjusted during fine-tuning.

![The overall architecture of the proposed approach](https://github.com/Baochen1012/Multimodal_AMD/blob/main/overall_framework.png)

Please contact zhenbaochen1012@163.com if you have questions.

## Install environment

### 1. Create environment with conda:

```bash
conda create -n retfound python=3.10 -y
conda activate retfound
```
###2. Install dependencies:

```bash
git clone https://github.com/Baochen1012/Multimodal_AMD
cd Multimodal_AMD
pip install -r requirements.txt
```

## Download the RETFound pre-trained weights

| Dataset                | Download Link                                                     |
|------------------------|-------------------------------------------------------------------|
| Colour fundus image    | [Download](https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing) |
| OCT                    | [Download](https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing) |

## Data

MMC-AMD, a multi-modal fundus image set consisting of 1,093 color fundus photograph (CFP) images and 1,288 OCT B-scan images (~470MB). Freely available upon request and for research purposes only. Please submit your request via [Google Form](https://forms.gle/jJT6H9N9CY34gFBWA).

- MMC-AMD (splitA): An eye-based split of training / validation / test sets (zero eye overlap) [Download](https://drive.google.com/file/d/1El2pBzNnQsjRVLE_QwFNhS05HWJMPwkU/view?usp=sharing)
- MMC-AMD (splitAP): A patient-based split of training / validation / test sets (zero patient overlap) [Download](https://drive.google.com/file/d/1KwJdsQmO__TpCW2AcRdsoTocu-zwcZuT/view?usp=sharing)

## Instructions

1. **pictures**: This directory contains various visualization results, including:
   - **cam**: Class Activation Mapping images for different models.
   - **confusion_matrix**: Normalized confusion matrix plots for different models on the test set.
   - **t-sne**: t-Distributed Stochastic Neighbor Embedding (t-SNE) plots for different models on the test set.

2. **CCAtool**: This directory contains the dependencies related to the Deep Canonical Correlation Analysis (DCCA) component.

3. **LoRA**: This directory contains the dependencies related to the Low-Rank Adaptation (LoRA) component.


