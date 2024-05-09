<div align="center">
  
## Overall Positive Prototype for Few-Shot Open-Set Recognition

Liang-Yu Sun and Wei-Ta Chu

Department of Computer Science and Information Engineering, CSIE, 1 Univ. Rd., Tainan City, 70101, , Taiwan

<a href="https://www.sciencedirect.com/science/article/abs/pii/S0031320324001511"><img src='https://img.shields.io/badge/Pattern_Recognition-FSOR--OPP-blue' alt='Paper PDF'></a>

</div>
Few-shot open-set recognition (FSOR) is the task of recognizing samples in known classes with a limited number of annotated instances while also de- tecting samples that do not belong to any known class. This is a challenging problem because the models must learn to generalize from a small number of labeled samples and distinguish them from an unlimited number of potential negative examples. In this paper, we propose a novel approach called overall positive prototype to effectively improve performance. Conceptually, nega- tive samples would distribute throughout the feature space and are hard to be described. From the opposite viewpoint, we propose to construct an overall positive prototype that acts as a cohesive representation for positive sam- ples that distribute in a relatively smaller neighborhood. By measuring the distance between a query sample and the overall positive prototype, we can effectively classify it as either positive or negative. We show that this simple yet innovative approach provides the state-of-the-art FSOR performance in terms of accuracy and AUROC.

## Usage

### Installation

```bash
git clone https://github.com/jyp-studio/FSOR-OPP.git
cd FSOR-OPP
pip install -r requirements.txt
```

### Running

```bash
./run.sh
```

Arguments:
- `--img-path`: you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- `--pred-only` is set to save the predicted depth map only. Without it, by default, we visualize both image and its depth map side by side.
- `--grayscale` is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.

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
