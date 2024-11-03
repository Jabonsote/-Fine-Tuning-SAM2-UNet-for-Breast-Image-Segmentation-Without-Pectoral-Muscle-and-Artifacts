## [SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation](https://arxiv.org/abs/2408.08870)
Xinyu Xiong, Zihuang Wu, Shuangyi Tan, Wenxue Li, Feilong Tang, Ying Chen, Siying Li, Jie Ma, Guanbin Li

  ## Architecture
![framework](./sam2unet.jpg)
[SAM2 Unet - Fine-tuning for medical segmentation](https://wandb.ai/javier-ramirez-gonzalez/SAM2-UNet-training/reports/SAM2-Unet-Fine-tuning-for-medical-segmentation--Vmlldzo5OTg4MDA0)

## Clone Repository
```shell
git clone https://github.com/Jabonsote/Fine-Tuning-SAM2-UNet-for-Breast-Image-Segmentation-Without-Pectoral-Muscle-and-Artifacts.git
```

## Prepare Datasets
You can refer to the following repositories and their papers for the detailed configurations of the corresponding datasets.
- Camouflaged Object Detection. Please refer to [FEDER](https://github.com/ChunmingHe/FEDER).
- Salient Object Detection. Please refer to [SALOD](https://github.com/moothes/SALOD).
- Marine Animal Segmentation. Please refer to [MASNet](https://github.com/zhenqifu/MASNet).
- Mirror Detection. Please refer to [HetNet](https://github.com/Catherine-R-He/HetNet).
- Polyp Segmentation. Please refer to [PraNet](https://github.com/DengPingFan/PraNet).

### Examples of Tools for Segmentation and Data Export

For segmentation, you can use [Roboflow](https://universe.roboflow.com/) or [Makesense.ai](https://www.makesense.ai/) to export your data to a COCO dataset format and use the `coco2sam` script to adjust the percentage of images.

![Roboflow or Makesense for labeling data](/img/ro.gif)

### Run Information on Wandb

You can find the run information of our model on [Wandb](https://wandb.ai/javier-ramirez-gonzalez/SAM2-UNet-training).


### Tools for Segmentation and Data Export

| Tool | Description |
| --- | --- |
| [Roboflow](https://universe.roboflow.com/) | A platform collaborative for data labeling and annotation (Public data)|
| [Makesense.ai](https://www.makesense.ai/) | A platform for data labeling and annotation (Private data) |

### Script for Data Export

| Script | Description |
| --- | --- |
| `coco2sam.py` | A script for converting COCO dataset to SAM format |

### Adjusting Image Percentage

You can adjust the percentage of images using the `coco2sam` script. This will help you to fine-tune your model and improve its performance.

![Roboflow or Makesense for label data](/img/ro.gif)


Run information in wandb [Wandb](https://wandb.ai/javier-ramirez-gonzalez/SAM2-UNet-training)


**Dataset** 📁
==========

**train** 🏋️‍♂️
--------

  * **images** 📸
  * **masks** 🔍
  
**valid** 👍
--------

  * **images** 📸
  * **masks** 🔍

**test** 🤔
--------

  * **images** 📸
  * **masks** 🔍


## Requirements
Our project does not depend on installing SAM2. If you have already configured an environment for SAM2, then directly using this environment should also be fine. You may also create a new conda environment:

```shell
conda create -n sam2-unet python=3.10
conda activate sam2-unet
pip install -r requirements.txt
```

## Training
If you want to train your own model, please download the pre-trained segment anything 2 from the [official repository](https://github.com/facebookresearch/segment-anything-2). You can also directly download `sam2_hiera_large.pt` from [here](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt). After the above preparations, you can run `train.sh` to start your training.


## Testing
Our pre-trained models and prediction maps can be found on [Google Drive](https://drive.google.com/drive/folders/1w2fK8kLhtEmMWZ6G6w9_J17xwgfm3lev?usp=drive_link). Also, you can run `test.sh` to obtain your own predictions.

### Examples
Inferences during training for breast segmentation without pectoral muscle 💡

In this example, we demonstrate the model's ability to make inferences during training for the task of breast segmentation without pectoral muscle 🤖. The goal is to identify and segment the breast in mammography images, excluding the pectoral muscle 📸.

![Inferences during training](/img/inferences.png)
[REPORT OF METRICS](https://wandb.ai/javier-ramirez-gonzalez/SAM2-UNet-training/reports/SAM2-Unet-Fine-tuning-for-medical-segmentation--Vmlldzo5OTg4MDA0)

Nipple marks: Marks or artifacts around the nipple area can be challenging for the model to distinguish from the breast tissue.
Pectoral muscle (upper and lower borders): The pectoral muscle can be a significant challenge for segmentation, especially when its upper and lower borders are not clearly defined.
Breast implants

![Inferences during training](/img/inference1.png)


## Evaluation
After obtaining the prediction maps, you can run `eval.sh` to get most of the quantitative results. For the evaluation of mirror detection, please refer to `eval.py` in [HetNet](https://github.com/Catherine-R-He/HetNet) to obtain the results.

## REFERENCES

```
@article{xiong2024sam2,
  title={SAM2-UNet: Segment Anything 2 Makes Strong Encoder for Natural and Medical Image Segmentation},
  author={Xiong, Xinyu and Wu, Zihuang and Tan, Shuangyi and Li, Wenxue and Tang, Feilong and Chen, Ying and Li, Siying and Ma, Jie and Li, Guanbin},
  journal={arXiv preprint arXiv:2408.08870},
  year={2024}
}
```

## Acknowledgement
[segment anything 2](https://github.com/facebookresearch/segment-anything-2)
