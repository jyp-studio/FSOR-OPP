<div align="center">
  
# Overall Positive Prototype for Few-Shot Open-Set Recognition

Liang-Yu Sun and Wei-Ta Chu

Department of Computer Science and Information Engineering, CSIE, 1 Univ. Rd., Tainan City, 70101, , Taiwan

<a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320324001511"><img src='https://img.shields.io/badge/Pattern_Recognition-FSOR--OPP-blue' alt='Paper PDF'></a>

</div>
Few-shot open-set recognition (FSOR) is the task of recognizing samples in known classes with a limited number of annotated instances while also de- tecting samples that do not belong to any known class. This is a challenging problem because the models must learn to generalize from a small number of labeled samples and distinguish them from an unlimited number of potential negative examples. In this paper, we propose a novel approach called overall positive prototype to effectively improve performance. Conceptually, nega-tive samples would distribute throughout the feature space and are hard to be described. From the opposite viewpoint, we propose to construct an overall positive prototype that acts as a cohesive representation for positive sam-ples that distribute in a relatively smaller neighborhood. By measuring the distance between a query sample and the overall positive prototype, we can effectively classify it as either positive or negative. We show that this simple yet innovative approach provides the state-of-the-art FSOR performance in terms of accuracy and AUROC.

## Usage

### Installation

```bash
git clone https://github.com/jyp-studio/FSOR-OPP.git
cd FSOR-OPP
pip install -r requirements.txt
```

### Download datasets and pretrained model weights

You may need to download datasets inclusive of MiniImageNet and TieredImageNet as well as some pretrained model weights.
All of the above vital information can be downloaded from [TANE](https://github.com/shiyuanh/TANE).

### Running

```bash
cd fsor_dinoEXP # or fsor_resnet12 for a differnet backbone
./run.sh
```

Arguments:
- `--dataset`: select the specific dataset to train.
- `--logroot`: the path for logging.
- `--data_root`: the path to the dataset.
- `--n_ways` and `--n_shots`: select the training methods which is introduced in the paper.
- `--restype`: adjust the model types.
- `pretrained_model_path`: the path to the pretrained model weights.
- `learning rate`: literately, learning rate.
- `--gpus`: assign which GPU to train the model.
- `--n_train_para`: select the number of tasks in a training process.
- `--n_train_runs`: the number of iteration in one epoch.
- `op_loss`: the alpha rate.
- `protonet`: if the mode is protonet or not.

## Citation

If you find this project useful, please consider citing:

```
@article{sun2024overall,
  title={Overall positive prototype for few-shot open-set recognition},
  author={Sun, Liang-Yu and Chu, Wei-Ta},
  journal={Pattern Recognition},
  volume={151},
  pages={110400},
  year={2024},
  publisher={Elsevier}
}
```
