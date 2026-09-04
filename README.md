# GaLe: Memory-Efficient Global Approximate and Local Exact Features

Approximate network blocks though a local exact + global approximate representation - achieve >90% ram saving on minimal accuracy drop.

arXiv: https://arxiv.org/abs/2609.02689


## Usage

Classification:
```bash
python validate_gale.py --model timm/mobilenetv2_110d.ra_in1k --img-size 256 --replace_conv --conv_slices 16 --conv_split_blocks 4 --conv_threshold 5 --log_model_depth 1
```

Object detection:
```bash
python object_detection_gale.py
```

Diffusion:
```bash
python diffusers/diffusers_sdturbo_row.py
```


## Citation

```bibtex
@article{ancilotto2026gale,
  title   = {GaLe: memory-efficient Global Approximate and Local Exact features},
  author  = {Ancilotto, Alberto and Farella, Elisabetta},
  journal = {IEEE International Conference on Image Processing},
  year    = {2026},
}
```
