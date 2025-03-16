# Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic

This is official implementation for CVPR2025 paper **Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic.**

> ***Jianwei Tang, Hong Yang, Tengyue Chen, Jian-Fang Hu***
>
> **Sun Yat-sen University**

**[[paper]]()   [[homepage]](https://hyqlat.github.io/STABACB.github.io/)**

## TODO

* [ ] Pre-trained Models...
* [X] [2025/03/22] Code is available now!
* [X] [2025/02/27] Our paper is accepted by CVPR2025!

## Abstract

<div style="text-align: center;">
    <img src="asset/overview.jpg" width=100% >
</div>

Action-driven stochastic human motion prediction aims to generate future motion sequences of a pre-defined target action based on given past observed sequences performing non-target actions. This task primarily presents two challenges. Firstly, generating smooth transition motions is hard due to the varying transition speeds of different actions. Secondly, the action characteristic is difficult to be learned because of the similarity of some actions. These issues cause the predicted results to be unreasonable and inconsistent. As a result, we propose two memory banks, the Soft-transition Action Bank (STAB) and Action Characteristic Bank (ACB), to tackle the problems above. The STAB stores the action transition information. It is equipped with the novel soft searching approach, which encourages the model to focus on multiple possible action categories of observed motions. The ACB records action characteristic, which produces more prior information for predicting certain actions. To fuse the features retrieved from the two banks better, we further propose the Adaptive Attention Adjustment (AAA) strategy. Extensive experiments on four motion prediction datasets demonstrate that our approach consistently outperforms the previous state-of-the-art.

## Implementation

### 1.Installation

##### Environment

* Python == 3.9.19
* PyTorch == 1.12.1

##### Dependencies

Install the dependencies from the: `requirements.txt`

```bash
pip install -r requirements.txt
```

##### Pre-processed Data

The pre-processed datasets can be found in the [project page of WAT](https://github.com/wei-mao-2019/WAT?tab=readme-ov-file#datasets). Download all the files to `../data` folder.

##### Pre-trained Models

***coming soon...***

Download all the files to `./results` folder.

### 2.Training

To train **a dataset** (e.g., `{NAME_OF_DATASET}`), execute the script `run_{NAME_OF_DATASET}.sh`.

The corresponding YAML configuration files are located in `./motion_pred/cfg/`.

##### Training ARM

Use the **`train conti class`** commands in script `run_{NAME_OF_DATASET}.sh` to train the **Action Recognition Module**.

The YAML configuration file can be found in `./motion_pred/cfg/{NAME_OF_DATASET}_cc.yml`.

##### Training MPM

Use the **`train`** commands in script `run_{NAME_OF_DATASET}.sh` to train the **Motion Prediction Module**.

The YAML configuration is located in `./motion_pred/cfg/{NAME_OF_DATASET}_rnn.yml`.

### 3.Testing

Use the **`test`** commands in script `run_{NAME_OF_DATASET}.sh` to **perform testing**.

The YAML configuration is located in `./motion_pred/cfg/{NAME_OF_DATASET}_rnn.yml`.

### 4.Visualization

Download smpl-(h,x) models from their official websites and put them in `./SMPL_models` folder.  The data structure should looks like this

```
SMPL_models
    ├── smpl
    │   ├── SMPL_FEMALE.pkl
    │   └── SMPL_MALE.pkl
    │
    ├── smplh
    │    ├── MANO_LEFT.pkl
    │    ├── MANO_RIGHT.pkl
    │    ├── SMPLH_FEMALE.pkl
    │    └── SMPLH_MALE.pkl
    │
    └── smplx
        │
        ├── SMPLX_FEMALE.pkl
        └── SMPLX_MALE.pkl
```


And use the **`visualization`** commands in script `run_{NAME_OF_DATASET}.sh` to **generate visualizations**.

The YAML configuration is located in `./motion_pred/cfg/{NAME_OF_DATASET}_rnn.yml`.

> #### *Tips*
>
> We also develop a simple [visualization-tool](https://github.com/hyqlat/PyRender-for-Human-Mesh/tree/Mesh_and_Skeleton) that can be used following the instructions in [README.md](https://github.com/hyqlat/PyRender-for-Human-Mesh/blob/Mesh_and_Skeleton/README.md). You can easily create a custom visualization.

## Acknowledgements

We sincerely thank the authors of [WAT](https://github.com/wei-mao-2019/WAT) for providing the source code and the pre-processed data from their publication. These resources have been invaluable to our work, and we are immensely grateful for their support.

## Citation

If you find this project useful in your research, please consider citing:

```
...
```
