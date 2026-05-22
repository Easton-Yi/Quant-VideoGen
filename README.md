## Recent Changes: Self-Forcing Mixed-bit KV Cache Quantization

This version adds configurable mixed-bit KV cache quantization to the Self-Forcing experiment path. The goal is to compare the original all-2-bit QVG baseline against a mixed setting where a fixed portion of KV cache frame chunks is quantized with 1-bit and the remaining chunks are quantized with 2-bit.

Main changes:

- Added the Triton PRQ `int1` quantization type: `triton-nstages-kmeans-int1`.
- Implemented 1-bit residual pack / unpack / dequantization support.
- Added mixed-bit configuration to `scripts/Self-Forcing/run_qvg.sh`:
  - `mixed_bit_enabled`
  - `mixed_schedule`
  - `mixed_1bit_ratio`
  - `mixed_low_quant_type`
  - `mixed_high_quant_type`
- The default schedule is `static_global`: the global 1-bit / 2-bit boundary is computed once before inference from the fixed video length and `mixed_1bit_ratio`.
- During Self-Forcing inference, each KV cache span is checked against the global boundary before compression:
  - spans before the boundary use `triton-nstages-kmeans-int1`
  - spans after the boundary use `triton-nstages-kmeans-int2`
  - spans crossing the boundary are split into separate 1-bit and 2-bit spans
- Each quantized KV span stores its own `quant_config`, so cache reads can dequantize mixed int1/int2 spans with the correct bit width.

The current mixed-bit scheduling implementation is wired into the Self-Forcing path only. LongCat and HY-WorldPlay have not been connected to the mixed-bit scheduler yet. With the default script values `num_output_frames=180` and `mixed_1bit_ratio=0.25`, the first 45 frame chunks use 1-bit quantization and the remaining 135 frame chunks use 2-bit quantization.

---------------

<div align="center">

# QuantVideoGen

**Quantized KV-cache compression for video generation models**

<p>
  <a href="https://svg-project.github.io/qvg/"><img src="https://img.shields.io/badge/Website-76B900?style=for-the-badge&logo=safari&labelColor=555555"></a>
  <a href="https://arxiv.org/abs/2602.02958"><img src="https://img.shields.io/badge/Arxiv-B31B1B?style=for-the-badge&logo=arxiv&labelColor=555555"></a>
  <a href="#"><img src="https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=x&labelColor=555555"></a>
</p>

</div>

QuantVideoGen is a lightweight KV-cache quantization toolkit for autoregressive video generation. It compresses long-horizon attention cache during inference, with experiment integrations for LongCat-Video, Self-Forcing, and HY-WorldPlay.

## ✨ Highlights

- Quantizes KV cache with Triton k-means / staged product quantization kernels.
- Keeps the original model weights unchanged; quantization is applied to inference cache.
- Includes bf16 and quantized launch scripts for three long-video / streaming generation repos.
- Targets memory-heavy long-context settings where KV cache dominates peak usage.

<a id="installation"></a>

## 📦 Installation

```bash
conda create -n qvg python=3.12.9 -y
conda activate qvg

pip install uv

# Everything, recommended for reproducing all experiments.
uv pip install -e ".[all]"

# Or install only one experiment extra.
uv pip install -e ".[longcat]"
uv pip install -e ".[selfforcing]"
uv pip install -e ".[hyworldplay]"

# Flash Attention, CUDA 12 / torch 2.8 wheel.
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

<a id="download-models"></a>

## ⬇️ Download Models

Download model checkpoints before running experiments.

```bash
# LongCat-Video
hf download meituan-longcat/LongCat-Video --local-dir ckpts/LongCat-Video

# Self-Forcing
bash scripts/Self-Forcing/download_models.sh

# HY-WorldPlay
bash scripts/HY-WorldPlay/download_models.sh
```

<a id="quick-start"></a>

## 💥 Quick Start

Each integration provides a bf16 baseline script and a quantized script. The quantized scripts currently use `triton-nstages-kmeans-int2` with block size 64 and 256 K/V centroids by default.

```bash
# LongCat-Video
bash scripts/LongCat/run_bf16.sh
bash scripts/LongCat/run_qvg.sh

# Self-Forcing
bash scripts/Self-Forcing/run_bf16.sh
bash scripts/Self-Forcing/run_qvg.sh

# HY-WorldPlay
bash scripts/HY-WorldPlay/run_bf16.sh
bash scripts/HY-WorldPlay/run_qvg.sh
```

Outputs are written under `results/`. Quantization options can be changed directly in the corresponding `run_qvg.sh` script.

<a id="memory-results"></a>

## 📊 Memory Results

The table below reports KV-cache memory for the provided scripts. Numbers are in MB.

| Model | Precision | QVG | Per Layer KV | Total KV Cache | Compression Rate |
| --- | ---: | :---: | ---: | ---: | ---: |
| LongCat-Video | BF16 | ✗ | 464.00 | 22272.00 | 1.00x |
| LongCat-Video | INT2 | ✓ | 67.32 | 3231.28 | 6.89x |
| Self-Forcing | BF16 | ✗ | 1535.76 | 46072.88 | 1.00x |
| Self-Forcing | INT2 | ✓ | 220.45 | 6613.59 | 6.97x |
| HY-WorldPlay | BF16 | ✗ | 990.00 | 29700.00 | 1.00x |
| HY-WorldPlay | INT2 | ✓ | 141.18 | 4235.45 | 7.01x |

Across these runs, QuantVideoGen reduces total KV-cache memory by about 85%.

<a id="citation"></a>

## ✏️ Citation

```bibtex
@article{xi2026quant,
  title={Quant VideoGen: Auto-Regressive Long Video Generation via 2-Bit KV-Cache Quantization},
  author={Xi, Haocheng and Yang, Shuo and Zhao, Yilong and Li, Muyang and Cai, Han and Li, Xingyang and Lin, Yujun and Zhang, Zhuoyang and Zhang, Jintao and Li, Xiuyu and others},
  journal={arXiv preprint arXiv:2602.02958},
  year={2026}
}
```

