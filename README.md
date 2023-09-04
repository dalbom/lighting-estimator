# Lighting Estimator using Transformer Network - Spatial Aggregation

This repository implements [the spatiotemporal lighting estimation](https://link.springer.com/article/10.1007/s11263-022-01725-2) paper, but its temporal aggregation process is omitted. Consequently, the 3D positional encoding proposed in the paper is also modified to 2D positional encoding.

## Data
Any type of image data with corresponding ground truth Lalonde-Matthews Sun-Sky model parameters can be used. Fake annotation files are provided in data/splits/ for a reference.

## Training
Simply run 

```python
$ python run.py configs/[YOUR_CONFIG_FILE].yaml
``` 

## Cite
```bibtex
@article{lee2023spatio,
  title={Spatio-Temporal Outdoor Lighting Aggregation on Image Sequences Using Transformer Networks},
  author={Lee, Haebom and Homeyer, Christian and Herzog, Robert and Rexilius, Jan and Rother, Carsten},
  journal={International Journal of Computer Vision},
  volume={131},
  number={4},
  pages={1060--1072},
  year={2023},
  publisher={Springer}
}
```
