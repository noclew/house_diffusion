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

The paper poses a method to automatically generate floorplans that are constrained by graphs with a diffusion model based architecture. In which the incident relationship between the architectural components is taken as a bubble diagram where nodes are rooms and edges are connections between rooms (doors). This way the overall relationships between rooms is defined as a constraint as well as the number of rooms etc. The model generates 2D coordinates of room and door corners to define the resulting floor plans.

**How is the data represented?**

To represent a floorplan a graph is used that represents the rooms and corners and where they are connected in an adjacency matrix. This is done by representing each room and door as a 1D polygonal loop containing a sequence of 2D coordinates as corners.

![image](https://user-images.githubusercontent.com/129068811/234509472-5a20b938-3cf2-48f8-939c-fe0176dcb91d.png)

To specify the number of corners in the polygonal a probabilistic pick is done out of histogram counts for corners per room-type. An addition to this way of representing the floorplan is that a user could also specify a number of corners if they would like to. The corner values are mapped between [-1,1] to be mixed with Gaussian noise. In reverse the coordinates will be presented in the binary representation as a discrete integer. This process is a Transformer based neural network

**What is the architecture of the model?**

![image](https://user-images.githubusercontent.com/129068811/234509545-4bb7a674-2d06-4510-b496-4826f1d34bfd.png)
The forward process of the HouseDiffusion architecture takes a data sample and generates a noisy sample from it by sampling a Gaussian noise with a cosine noise schedule. The reverse process uses a neural network with a Transformer that predicts the previous floorplan representation given the current representation.
Feature embedding

The transformer architecture is such that every corner becomes a node with a 512 dimensional embedding vector. A learned embedding layer embeds the corner coordinates by augmenting it: samples points from this coordinate to the next and concatenates these. The room type, room index and corner index are also embedded into the corner feature vector. Every embedding vector is then a node in the Transformer architecture. This embedding contains sampled point coordinates along the wall from the corner to include more conceptual information in the embedding.

**Continuous denoising**

The embedding vectors or the corner coordinates will be processed by attention layers with structured masking. There are three types of attention: Component-wise Self Attention (CSA), Global Self Attention (GSA), and Relational Cross Attention (RCA). Four multi-heads are used and learn three sets of key/query/value matrices per head in each attention layer. The results of the three attentions are summed, followed by an Add & Norm layer. This block is repeated four times for continuous denoising. Finally, a linear layer infers noise at each node.

**Discrete denoising**

The coordinates are inferred in a discrete form to capture certain relationships that are difficult to capture in continuous regression. Specifically geometric incident relationships like orthogonality for example. The coordinates are mapped to a binary representation. The data is embedded similarly as described earlier. The binary representation of the corner coordinates is inferred.
**Loss function**

**Data preprocessing**

To train a House Diffusion model, we used the publicly available RPLAN dataset (http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html). This dataset consists of 80,000 house floor plans with information such as the room type, adjacent rooms, and doors. Images in the dataset have a resolution of 256x256.
We preprocessed the floor plans to a format that was used by HouseGAN++. Due to time constraints, we pre-processed 20,000 floor plans. During the preprocessing, we removed 2,000 floor plans that lowered the dataset quality for training due to an unrealistic layout, such as multiple doors between rooms or parallel walls that are very close. By removing unrealistic layouts, we aim to improve the quality of training data and therefore generate high fidelity floor plans. In the preprocessing, the raster images of the RPLAN dataset were converted into a dictionary-like representation that specified the number and type of rooms, room shape and size, and adjacent rooms. 

## Experiments

We used PyTorch to reproduce the results based on the public code of HouseDiffusion (https://github.com/aminshabani/house_diffusion.git). The data set was split into 13,700 training samples and 4,300 test samples. Adam was used as optimizer without weight decay for 20,000 steps with a batch size of 64 on a single NVIDIA T4 Tensor Core. This corresponds to training for 93 epochs. A second model was trained for 40,000 steps with a batch size of 32, also corresponding to 93 epochs. We set the target set to 8 rooms per floor plan and the learning rate to 1e-4 with a uniform schedule. The number of diffusion steps was 1000 and was uniformly sampled during training. A ‘cosine’ noise schedule was used which indicates that noise is relatively slowly added at an early timestep and more is added at a later timestep.
We compare against the original training hyperparameters of HouseDiffusion: “We use PyTorch to implement the proposed approach based on a public implementation of Guided- Diffusion.Adam is the optimizer with decoupled weight decay for 250k steps with batch-size of 512 on a single NVIDIA RTX 6000. An initial learning rate is 1e-3. We divide the learning rate by 10 after every 100k steps. We set the number of diffusion steps to 1000 (unless otherwise noted) and uniformly sample t during training”. This model was trained on a preprocessed RPLAN dataset that contained 60,000 samples. The number of epochs was 2,130.
For each model, we sampled 64 floor plans with 8 rooms. To test for diversity and compatibility we computed the Frechet Inception Distance (FID) and graph Modified Edit Distance deviation, respectively. 

## Results

The results of the experiments are shown in Table 1. The mean and standard deviation are computed based on 64 samples per model. Diversity and compatibility were lower (=better) in the HouseDiffusion pretrained model than in our 20,000 and 40,000 steps model. Furthermore, the model trained with 40,000 steps had lower diversity and compatibility than the 20,000 steps model. Examples of floor plans generated by the three models are shown in figures 1, 2, and 3.

|                                                       | Diversity (FID)             | Compatibility (graph Modified Edit Distance   |
| ----------------------- |:---------------------------:| ---------------------------:|----------------------------------------------:|
| Dataset                 | Model                       | Mean + standard deviation   | Mean + standard deviation                     |
| ----------------------- |:---------------------------:| ---------------------------:|----------------------------------------------:|
| RPLAN (18,000 samples)  | Ours (20,000 steps)         | $50.68 ± 1.63               | $5.53 ± 0.11                                  |
| RPLAN (18,000 samples)  | Ours (40,000 steps)         | $47.78 ± 0.98               | $5.35 ± 0.24                                  |
| RPLAN (250,000 samples) | HouseDiffusion (250k steps) | $30.54 ± 0.61               | $2.48 ± 0.11                                  |
| ----------------------- |:---------------------------:| ---------------------------:|----------------------------------------------:|

Table 1: Results of diversity and compatibility

![image](https://user-images.githubusercontent.com/129068811/234508265-fb8e8403-810f-4569-a451-01cce3f3426a.png)

Figure 3: sample floorplans of our own model and of the HouseDiffusion pretrained model
## Discussion

In this reproduction study, we aimed to reproduce the results of HouseDiffusion for generating floor plans with a diffusion model. We used two different training configurations: the first model was trained for 20,000 steps with a batch size of 64, and the second one trained for 40,000 steps with a batch size of 32. We compared our models with the pretrained HouseDiffusion model that was trained on a larger dataset for 250,000 steps with a batch size of 512.
We showed that the pretrained model was better in generating diverse and compatible floor plans than our models. We contribute this to the larger training set and training for more epochs. Furthermore, we found that the model trained with batch size 32 (40,000 steps) performed better than the model trained with batch 64 (20,000 steps). A smaller batch size is therefore advised.

Our study had several limitations. Due to time constraints, we did not use the complete RPLAN dataset. The preprocessing of the floor plans appeared to be computationally expensive and therefore we stopped after preprocessing 20,000 floor plans in 12 hours. Future reproduction attempts should use the full dataset. Secondly and also due to time constraints, our models were trained for only 93 epochs in 3 hours. We encourage future attempts to train for more epochs and to come closer to the 2000 epochs of the pretrained model if the dataset is large. Lastly, our models differed in learning rate from the pretrained model. For better reproduction, the hyperparameters that are not tested should remain unchanged to reliably find the cause of a difference in performance.

## Conclusion

Our reproduction study attempted to replicate the results of HouseDiffusion for generating floor plans with a diffusion model. We concluded that we were unsuccessful in reproducing the original paper's findings. Our models did not perform as well as the pretrained HouseDiffusion model, which we attribute to the larger training set and more training iterations. Additionally, our study was limited by a small data set and time constraints, and we encourage future reproduction attempts to use the full RPLAN dataset and more training iterations.



## Citation

```
@article{SchwietertandKoetzier2023housediffusion,
  title={Reproduction of HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising},
  author={Schwietert, Renee and Koetzier, Lennart},
  year={2022}
}
```
