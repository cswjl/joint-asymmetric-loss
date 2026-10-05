<div align="center">

# JAL

### Joint Asymmetric Loss for Learning with Noisy Labels

**Official PyTorch Implementation · ICCV 2025**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2507.17692)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)

[Overview](#overview) · [Poster](#poster) · [Usage](#usage) · [Examples](#examples) · [Citation](#citation) · [Contact](#contact)

</div>

---

<a id="overview"></a>

## ✨ Overview

This repository provides training scripts for benchmark and real-world noisy-label learning.

- **Losses:** Joint Asymmetric Loss (JAL), combining NCE or NFL with Asymmetric Mean Square Error (AMSE).
- **Noise types:** symmetric, asymmetric, instance-dependent, human, and real-world.
- **Datasets:** CIFAR-10, CIFAR-100, CIFAR-N, WebVision, and Clothing1M.

> [!NOTE]
> In the code, **JAL-CE** and **JAL-FL** are named `NCEandAMSE` and `NFLandAMSE`, respectively.

<a id="poster"></a>

## 🖼️ Poster

![Joint Asymmetric Loss poster](poster.png)

<a id="usage"></a>

## 🛠️ Usage

```bash
git clone https://github.com/cswjl/joint-asymmetric-loss.git && cd joint-asymmetric-loss
```

| Setting | Entry point | Datasets | Noise types |
| :--- | :--- | :--- | :--- |
| Benchmark | [`main.py`](main.py) | `cifar10`, `cifar100` | `symmetric`, `asymmetric`, `dependent`, `human` |
| Real-world | [`main_real_world.py`](main_real_world.py) | `webvision`, `clothing1m` | Real-world label noise |

| Argument | Description | Examples |
| :--- | :--- | :--- |
| `--dataset` | Dataset | `cifar10`, `cifar100`, `webvision`, `clothing1m` |
| `--loss` | Loss function | `NCEandAMSE`, `NFLandAMSE`, `CE`, `GCE` |
| `--noise_type` | Noise type (benchmark) | `symmetric`, `asymmetric`, `dependent`, `human` |
| `--noise_rate` | Noise rate, or CIFAR-N label set with `human` (benchmark) | `0.8`, `worst`, `noisy100` |
| `--root` | Dataset root directory | `../data` (default) |


<details>
<summary><strong>📂 Repository structure</strong></summary>

```text
joint-asymmetric-loss/
├── main.py                 # Benchmark training
├── main_real_world.py      # Real-world dataset training
├── losses.py               # Loss function implementations
├── config.py               # Loss and regularization configurations
├── models.py               # Network architectures
├── utils.py                # Training and evaluation utilities
└── datasets/               # Data loaders, noise annotations, and dataset lists
```

</details>

<a id="examples"></a>

## 🚀 Examples

```bash
# CIFAR-10 · 80% symmetric noise · JAL-CE
python3 main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8 --loss NCEandAMSE

# CIFAR-10N · worst human labels · JAL-FL
python3 main.py --dataset cifar10 --noise_type human --noise_rate worst --loss NFLandAMSE

# WebVision · real-world noise · JAL-CE
python3 main_real_world.py --dataset webvision --loss NCEandAMSE
```

<a id="citation"></a>

## 🎓 Citation

If you find this work useful, please cite our [paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Joint_Asymmetric_Loss_for_Learning_with_Noisy_Labels_ICCV_2025_paper.html):

```bibtex
@inproceedings{wang2025joint,
  title={Joint asymmetric loss for learning with noisy labels},
  author={Wang, Jialiang and Liu, Xianming and Zhou, Xiong and Hu, Gangfeng and Zhai, Deming and Jiang, Junjun and Ji, Xiangyang},
  booktitle={2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={1947--1956},
  year={2025},
  organization={IEEE}
}
```

<a id="contact"></a>

## 📬 Contact

Questions about the paper or code? Contact **Jialiang Wang** at [cswjl@stu.hit.edu.cn](mailto:cswjl@stu.hit.edu.cn).

---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

</div>
