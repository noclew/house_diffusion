# HouseDiffusion
**[HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising](https://arxiv.org/abs/2211.13287)**
<img src='figs/teaser.png' width=100%>
## Installation
**1. Clone our repo and install the requirements:**

Our implementation is based on the public implementation of [guided-diffusion](https://github.com/openai/guided-diffusion). For installation instructions, please refer to their repository. Keep in mind that our current version has not been cleaned and some features from the original repository may not function correctly.

```
git clone https://github.com/lkoetz/house_diffusion.git
cd house_diffusion
pip install -r requirements.txt
pip install -e .
```
**2. Download the dataset and create the datasets directory**

- You can download the full RPLAN dataset from [RPLAN's website](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html) or by filling [this](https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform) form.
- We also use data preprocessing from House-GAN++ which you can find in [this](https://github.com/sepidsh/Housegan-data-reader) link.
- We have provided a preprocessed dataset containing 18,000 samples in this repository (rplan.zip). Note that our preprocessing was based on the preprocessing by House-GAN++. Our preprocessing additionally removed unrealistic floor plans.
Put all of the processed files from the downloaded dataset in a `datasets` folder in the current directory:

```
house_diffusion
├── datasets
│   ├── rplan
|   |   └── 0.json
|   |   └── 1.json
|   |   └── ...
|   └── ...
└── guided_diffusion
└── scripts
└── ...
```
- We have provided two temporary pretrained models (20,000 steps and 40,000 steps) that you can download from [Google Drive](https://drive.google.com/file/d/1mvw-2IzegJ8-Xm_c2zk8Z27ALEizGPt3/view?usp=share_link) and [Google Drive](https://drive.google.com/file/d/18yInP0Hi4Zl55IIKJm9pCayQpdoB-8Yz/view?usp=sharing). 

## Running the code

**1. Training**

You can run a single experiment using the following command:
```
python image_train.py --dataset rplan --batch_size 32 --set_name train --target_set 8
```
**2. Sampling**
To sample floorplans, you can run the following command from inside of the `scripts` directory. To provide different visualizations, please see the `save_samples` function from `scripts/image_sample.py`

```
python image_sample.py --dataset rplan --batch_size 32 --set_name eval --target_set 8 --model_path ckpts/exp/model250000.pt --num_samples 64
```
You can also run the corresponding code from `scripts/script.sh`. 

## Introduction

## Data preprocessing

To train a House Diffusion model, we used the publicly available RPLAN dataset (http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html). This dataset consists of 80,000 house floor plans with information such as the room type, adjacent rooms, and doors. Images in the dataset have a resolution of 256x256.
We preprocessed the floor plans to a format that was used by HouseGAN++. Due to time constraints, we pre-processed 20,000 floor plans. During the preprocessing, we removed 2,000 floor plans that lowered the dataset quality for training due to an unrealistic layout, such as multiple doors between rooms or parallel walls that are very close. By removing unrealistic layouts, we aim to improve the quality of training data and therefore generate high fidelity floor plans. In the preprocessing, the raster images of the RPLAN dataset were converted into a dictionary-like representation that specified the number and type of rooms, room shape and size, and adjacent rooms. 

## Experiments

We used PyTorch to reproduce the results based on the public code of HouseDiffusion (ref). The data set was split into 13,700 training samples and 4,300 test samples. Adam (ref) was used as optimizer without weight decay for 20,000 steps with a batch size of 64 on a single NVIDIA T4 Tensor Core. This corresponds to training for 93 epochs. A second model was trained for 40,000 steps with a batch size of 32, also corresponding to 93 epochs. We set the target set to 8 rooms per floor plan and the learning rate to 1e-4 with a uniform schedule. The number of diffusion steps was 1000 and was uniformly sampled during training. A ‘cosine’ noise schedule was used which indicates that noise is relatively slowly added at an early timestep and more is added at a later timestep.
We compare against the original training hyperparameters of HouseDiffusion (ref): “We use PyTorch to implement the proposed approach based on a public implementation of Guided- Diffusion.Adam is the optimizer with decoupled weight decay for 250k steps with batch-size of 512 on a single NVIDIA RTX 6000. An initial learning rate is 1e-3. We divide the learning rate by 10 after every 100k steps. We set the number of diffusion steps to 1000 (unless otherwise noted) and uniformly sample t during training”. This model was trained on a preprocessed RPLAN dataset that contained 60,000 samples. The number of epochs was 2,130.
For each model, we sampled 64 floor plans with 8 rooms. To test for diversity and compatibility we computed the Frechet Inception Distance (FID) (ref) and graph Modified Edit Distance (ref) deviation, respectively. 

## Results

The results of the experiments are shown in Table 1. The mean and standard deviation are computed based on 64 samples per model. Diversity and compatibility were lower (=better) in the HouseDiffusion pretrained model than in our 20,000 and 40,000 steps model. Furthermore, the model trained with 40,000 steps had lower diversity and compatibility than the 20,000 steps model. Examples of floor plans generated by the three models are shown in figures 1, 2, and 3.

|                               | Diversity (FID)            | Compatibility (graph Modified Edit Distance   |
| ----------------------- |:-------------:| --------------------------:|----------------------------------------------:|
| Dataset                 | Model         | Mean + standard deviation  | Mean + standard deviation                     |
| ----------------------- |:-------------:| --------------------------:|----------------------------------------------:|
| RPLAN (18,000 samples)  | Ours (20,000 steps) | $1600 |      |
| RPLAN (18,000 samples)  | centered      |   $12 |      |
| RPLAN (250,000 samples) | are neat      |    $1 |      |

## Discussion

## Conclusion

## References

## Citation

```
@article{SchwietertandKoetzier2023housediffusion,
  title={Reproduction of HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising},
  author={Schwietert, Renee and Koetzier, Lennart},
  year={2022}
}
```
