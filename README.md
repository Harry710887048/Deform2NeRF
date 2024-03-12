# Deform2NeRF: Non-Rigid Deformation and 2D–3D Feature Fusion with Cross-Attention for Dynamic Human Reconstruction

### [Project Page](https://github.com/Harry710887048/Deform2NeRF/) | [Paper](https://www.mdpi.com/2079-9292/12/21/4382) | [Data](https://github.com/zju3dv/animatable_nerf/blob/master/INSTALL.md#zju-mocap-dataset) 

![image](https://github.com/Harry710887048/Deform2NeRF/assets/75151571/f846c962-b2e6-4dfe-b007-8580d385df8c)
<img src="github.com/Harry710887048/Deform2NeRF/assets/75151571/f846c962-b2e6-4dfe-b007-8580d385df8c" style="zoom:80%;" />
Reconstructing dynamic human body models from multi-view videos poses a substantial challenge in the field of 3D computer vision. Currently, the Animatable NeRF method addresses this challenge by mapping observed points from the viewing space to a canonical space. However, this mapping introduces positional shifts in predicted points, resulting in artifacts, particularly in intricate areas. In this paper, we propose an innovative approach called Deform2NeRF that incorporates non-rigid deformation correction and image feature fusion modules into the Animatable NeRF framework to enhance the reconstruction of animatable human models. Firstly, we introduce a non-rigid deformation field network to address the issue of point position shift effectively. This network adeptly corrects positional discrepancies caused by non-rigid deformations. Secondly, we introduce a 2D–3D feature fusion learning module with cross-attention and integrate it with the NeRF network to mitigate artifacts in specific detailed regions. Our experimental results demonstrate that our method significantly improves the PSNR index by approximately 5% compared to representative methods in the field. This remarkable advancement underscores the profound importance of our approach in the domains of new view synthesis and digital human reconstruction.

Our work is based on the improvement of AnimatableNeRF(https://arxiv.org/abs/2105.02872,ICCV).
Any questions or discussions are welcomed!

## Installation

Please see [INSTALL.md](INSTALL.md) for manual installation.

## Run the code on Human3.6M

Since the license of Human3.6M dataset does not allow us to distribute its data, we cannot release the processed Human3.6M dataset publicly. If someone is interested at the processed data, please email me.

We provide the pretrained models at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Et7h-48T0_xGtjNGXHwD1-gBPUNJZqd9VPTnsQlkSLktOw?e=TyCnuY).

### Test on Human3.6M

The command lines for test are recorded in [test.sh](test.sh).

Take the test on `S9` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_s9p/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_s9p_full/latest.pth`.
2. Test on training human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True
    ```

3. Test on unseen human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume True aninerf_animation True init_aninerf aninerf_s9p test_novel_pose True
    ```
![image](https://github.com/Harry710887048/Deform2NeRF/assets/75151571/beea0be8-c8c4-4823-93d2-02bf9dd2f09d){:height="70%" width="70%"}


### Visualization on Human3.6M

Take the visualization on `S9` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_s9p/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_s9p_full/latest.pth`.
2. Visualization:
    * Visualize novel views of the 0-th frame

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_novel_view True begin_ith_frame 0
    ```

    * Visualize views of dynamic humans with 3-th camera

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_pose_sequence True test_view "3,"
    ```

    * Visualize mesh

    ```shell
    # generate meshes
    python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p vis_posed_mesh True
    ```

3. The results of visualization are located at `$ROOT/data/novel_view/aninerf_s9p` and `$ROOT/data/novel_pose/aninerf_s9p`.

### Training on Human3.6M

Take the training on `S9` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:

    ```shell
    # training
    python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume False

    # training the blend weight fields of unseen human poses
    python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume False aninerf_animation True init_aninerf aninerf_s9p
    ```

2. Tensorboard:

    ```shell
    tensorboard --logdir data/record/deform
    ```

## Run the code on ZJU-MoCap

If someone wants to download the ZJU-Mocap dataset, please fill in the [agreement](https://zjueducn-my.sharepoint.com/:b:/g/personal/pengsida_zju_edu_cn/EUPiybrcFeNEhdQROx4-LNEBm4lzLxDwkk1SBcNWFgeplA?e=BGDiQh), and email me (pengsida@zju.edu.cn) and cc Xiaowei Zhou (xwzhou@zju.edu.cn) to request the download link.

We provide the pretrained models at [here](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/Et7h-48T0_xGtjNGXHwD1-gBPUNJZqd9VPTnsQlkSLktOw?e=TyCnuY).

### Test on ZJU-MoCap

The command lines for test are recorded in [test.sh](test.sh).

Take the test on `313` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_313/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_313_full/latest.pth`.
2. Test on training human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True
    ```

3. Test on unseen human poses:

    ```shell
    python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_full resume True aninerf_animation True init_aninerf aninerf_313 test_novel_pose True
    ```

### Visualization on ZJU-MoCap

Take the visualization on `313` as an example.

1. Download the corresponding pretrained models, and put it to `$ROOT/data/trained_model/deform/aninerf_313/latest.pth` and `$ROOT/data/trained_model/deform/aninerf_313_full/latest.pth`.
2. Visualization:
    * Visualize novel views of the 0-th frame

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True vis_novel_view True begin_ith_frame 0
    ```

    * Visualize views of dynamic humans with 0-th camera

    ```shell
    python run.py --type visualize --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True vis_pose_sequence True test_view "0,"
    ```

    * Visualize mesh

    ```shell
    # generate meshes
    python run.py --type visualize --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 vis_posed_mesh True
    ```

3. The results of visualization are located at `$ROOT/data/novel_view/aninerf_313` and `$ROOT/data/novel_pose/aninerf_313`.

### Training on ZJU-MoCap

Take the training on `313` as an example. The command lines for training are recorded in [train.sh](train.sh).

1. Train:

    ```shell
    # training
    python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume False

    # training the blend weight fields of unseen human poses
    python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_full resume False aninerf_animation True init_aninerf aninerf_313
    ```

2. Tensorboard:

    ```shell
    tensorboard --logdir data/record/deform
    ```

## Extended Version

Addtional training and test commandlines are recorded in [train.sh](train.sh) and [test.sh](test.sh).

Moreover, we compiled a list of all possible commands to run in [extension.sh](extension.sh) using on the S9 sequence of the Human3.6M dataset.

This include training, evaluating and visualizing the original Animatable NeRF implementation and all three extented versions.

Here we list the portion of the commands for the SDF-PDF configuration:

```shell
# extension: anisdf_pdf

# evaluating on training poses for anisdf_pdf
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True

# evaluating on novel poses for anisdf_pdf
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True test_novel_pose True

# visualizing novel view of 0th frame for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True vis_novel_view True begin_ith_frame 0

# visualizing animation of 3rd camera for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True vis_pose_sequence True test_view "3,"

# generating posed mesh for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p vis_posed_mesh True

# training base model for anisdf_pdf
python train_net.py --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume False
```

To run Animatable NeRF on other officially supported datasets, simply change the `--cfg_file` and `exp_name` parameters.

Note that for Animatable NeRF with pose-dependent displacement field (NeRF-PDF) and Animatable Neural Surface with pose-dependent displacement field (SDF-PDF), there's no need for training the blend weight fields of unseen human poses.

### MonoCap dataset

MonoCap is a dataset composed by authors of [animatable sdf](https://zju3dv.github.io/animatable_sdf/) from [DeepCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/) and [DynaCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/).

Since the license of [DeepCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2020-cvpr-deepcap/) and [DynaCap](https://people.mpi-inf.mpg.de/~mhaberma/projects/2021-ddc/) dataset does not allow us to distribute its data, we cannot release the processed MonoCap dataset publicly. If you are interested in the processed data, please download the raw data from [here](https://gvv-assets.mpi-inf.mpg.de/) and email me for instructions on how to process the data.

### SyntheticHuman Dataset

SyntheticHuman is a dataset created by authors of [animatable sdf](https://zju3dv.github.io/animatable_sdf/). It contains multi-view videos of 3D human rendered from characters in the [RenderPeople](https://renderpeople.com/) dataset along with the groud truth 3D model.

Since the license of the [RenderPeople](https://renderpeople.com/) dataset does not allow distribution of the 3D model, we cannot realease the processed SyntheticHuman dataset publicly. If you are interested in this dataset, please email me for instructions on how to generate the data.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@article{xie2023deform2nerf,
  title={Deform2NeRF: Non-Rigid Deformation and 2D--3D Feature Fusion with Cross-Attention for Dynamic Human Reconstruction},
  author={Xie, Xiaolong and Guo, Xusheng and Li, Wei and Liu, Jie and Xu, Jianfeng},
  journal={Electronics},
  volume={12},
  number={21},
  pages={4382},
  year={2023},
  publisher={MDPI}
}
@inproceedings{peng2021animatable,
  title={Animatable Neural Radiance Fields for Modeling Dynamic Human Bodies},
  author={Peng, Sida and Dong, Junting and Wang, Qianqian and Zhang, Shangzhan and Shuai, Qing and Zhou, Xiaowei and Bao, Hujun},
  booktitle={ICCV},
  year={2021}
}
```
