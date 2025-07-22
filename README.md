# Joint Asymmetric Loss for Learning with Noisy Labels

This repository is the official pytorch code of the **Joint Asymmetric Loss (JAL)**. 


## How to use
<font color=blue>**We use NCEandAMSE as JAL-CE and NFLandAMSE as JAL-FL in the code.**</font>

**Benchmark Datasets:** The running file is `main.py`.
* --dataset: cifar10 | cifar100.
* --loss: NCEandAMSE, NFLandAMSE, CE, GCE, etc.
* --noise_type: symmetric | asymmetric | dependent (instance-dependent
noise) | human (cifar-n).


**Real-World Datasets:** The running file is `main_real_world.py`.
* --dataset: webvision | clothing1m.
* --loss: NCEandAMSE, NFLandAMSE, CE, GCE, etc.

## Examples

NCEandAMSE for cifar10 with 0.8 symmetric noise:
```console
python main.py --dataset cifar10 --noise_type symmetric --noise_rate 0.8 --loss NCEandAMSE    
```

NCEandAMSE for webvision:
```console
python main_real_world.py --dataset webvision --loss NCEandAMSE
```


<!-- ## Reference
For details, please check the paper. If you have used our method or code in your own, please consider citing:

```bibtex
@inproceedings{wang2024epsilonsoftmax,
  title={$\epsilon$-Softmax: Approximating One-Hot Vectors for Mitigating Label Noise},
  author={Jialiang, Wang and Xiong, Zhou and Deming, Zhai and Junjun, Jiang and Xiangyang, Ji and Xianming, Liu},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
``` -->

If you have any question, you can contact cswjl@stu.hit.edu.cn