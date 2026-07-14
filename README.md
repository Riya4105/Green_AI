# Green AI Approaches: A Comparative Study of Pruning, Quantization, and Knowledge Distillation for Energy-Efficient Deep Learning

**Published — IEEE ICEI 2026 | Student Member, IEEE | Sole Author**
[![IEEE Xplore](https://img.shields.io/badge/IEEE%20Xplore-10.1109%2FICEI64468.2025.11447688-1a5276)](https://ieeexplore.ieee.org/document/11447688)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lF3Roaeu6qADrKDL-vVPnL4keUYgluvt?usp=sharing)

## Abstract

The rapid growth of AI has raised serious sustainability concerns due to the massive energy demands of training and deploying deep learning models. This paper investigates strategies to reduce energy consumption and CO2 emissions while maintaining competitive performance, comparing three optimization techniques — **pruning, quantization, and knowledge distillation** — against baseline models across multiple architectures. Experiments are tracked with **CodeCarbon** for real-time CO2 estimation.

Our findings show **ResNet-18 achieved the highest accuracy (72.07%)**, **quantized models achieved the lowest energy consumption (0.000290 kWh) and lowest emissions (0.000137 kg CO2)**, and **pruned models achieved the best Sustainability-Adjusted Accuracy (66.40)** — indicating the optimal balance between performance and environmental impact. We introduce hybrid optimization strategies, a novel **Sustainability-Adjusted Accuracy (SAA)** metric, and provide a fully open-source, reproducible implementation.

## Key Contributions

- A comprehensive comparative analysis of pruning, quantization, and knowledge distillation across multiple architectures (Baseline CNN, ResNet-18, MobileNetV2)
- A real-world energy-tracking implementation using **CodeCarbon**
- A novel **Sustainability-Adjusted Accuracy (SAA)** metric for Green AI evaluation, with comprehensive sensitivity analysis across lambda in [0, 1]
- Analysis of **hybrid optimization strategies** (Pruned+Quantized, Pruned+Knowledge Distillation) and their interactions
- A full **lifecycle emissions analysis** (training, inference, storage) and discussion of fairness/ethical trade-offs in model compression
- An open-source, fully reproducible implementation

## Methodology Overview

| Stage | Description |
|---|---|
| 1. Baseline training | Train Baseline CNN, ResNet-18, and MobileNetV2 on CIFAR-10 |
| 2. Pruning | Apply 30% unstructured pruning (empirically optimal sparsity level) |
| 3. Quantization | Dynamic post-training quantization to 8-bit integers |
| 4. Knowledge distillation | Baseline CNN (teacher) to Student CNN, temperature T = 2.0, alpha = 0.7 |
| 5. Hybrid strategies | Pruned->Quantized and Pruned->Distilled sequences, with order-sensitivity analysis |
| 6. Energy instrumentation | Measure energy draw per configuration using CodeCarbon |
| 7. SAA scoring | `SAA = Accuracy / (Energy(kWh) + lambda * CO2 emissions)`, sensitivity-tested across lambda in [0,1] |
| 8. Lifecycle estimation | Extend beyond training to inference (at scale) and storage emissions |

**Sustainability-Adjusted Accuracy (SAA):**

```
SAA = Accuracy / (Energy(kWh) + lambda * CO2 emissions)
```

Lambda controls the weighting between energy and emissions; lambda = 0.5 was validated as the recommended default (ranking stays consistent for lambda in [0.3, 0.7]).

## Results

**Table II — Model Performance Comparison with Sustainability Metrics**

| Model | Accuracy (%) | Energy (kWh) | CO2 (kg) | CO2 per 1% Acc (kg) | SAA |
|---|---|---|---|---|---|
| Baseline CNN | 65.55 | 0.000487 | 0.000229 | 3.49e-06 | 58.92 |
| **Pruned** | 66.12 | 0.000362 | 0.000170 | 2.57e-06 | **66.40 (best)** |
| **Quantized** | 64.89 | **0.000290 (lowest)** | **0.000137 (lowest)** | **2.11e-06 (best)** | 63.45 |
| KD Student | 62.34 | 0.000415 | 0.000195 | 3.13e-06 | 55.18 |
| ResNet-18 | **72.07 (highest)** | 0.001394 | 0.000655 | 9.09e-06 | 52.89 |
| MobileNetV2 | 53.16 | 0.000937 | 0.000440 | 8.28e-06 | 38.45 |
| Pruned+Quantized | 64.12 | 0.000328 | 0.000154 | 2.40e-06 | 62.78 |
| Pruned+KD | 52.62 | 0.000572 | 0.000269 | 5.11e-06 | 42.10 |

**Headline findings:**
- **Pruning** (30% sparsity): 66.12% accuracy, a **25.7% energy reduction** vs. baseline (0.000487 -> 0.000362 kWh), and the best overall SAA (66.40) — the optimal accuracy/environment balance.
- **Quantization** (8-bit dynamic): lowest absolute energy (0.000290 kWh) and lowest emissions (0.000137 kg CO2) of any single technique, at a competitive 64.89% accuracy — the preferred choice when environmental impact is the dominant constraint.
- **Knowledge distillation**: 62.34% student accuracy (a 4.21% drop from teacher), moderate efficiency gains, sensitive to temperature and architecture matching.
- **Pruned + Quantized** hybrid: 64.12% accuracy with 0.000328 kWh — cumulative, synergistic benefits with minimal additional accuracy loss.
- **Pruned + KD** hybrid: underperformed (52.62% accuracy, SAA 42.10) — over-pruning the teacher degrades distillation quality; a Distill->Prune order performed better (58.34% vs. 52.62%) than Prune->Distill.
- **Pruning sensitivity**: 30% sparsity was optimal (66.12% accuracy); accuracy degraded to 61.23% at 50% sparsity and 52.18% at 70% sparsity.
- **SAA sensitivity**: lambda = 0.0 favors ResNet-18 (raw accuracy); lambda = 0.5 favors Pruned (balanced); lambda = 1.0 favors Quantized (environment-first). Ranking is stable across lambda in [0.3, 0.7], validating lambda = 0.5 as the default.
- **Lifecycle emissions**: for models deployed at scale, **inference accounts for 85–95% of total lifecycle emissions** (training is typically ~1%), meaning techniques that reduce inference cost (like quantization) matter more long-term than training-time savings alone.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 (50,000 train / 10,000 test, 10 classes, 32x32x3) |
| Batch size | 128 |
| Base / KD epochs | 6 |
| Optimizer | SGD, momentum = 0.9, weight decay = 1e-4 |
| Learning rate | 0.1, cosine annealing |
| Pruning sparsity | 30% (unstructured) |
| Quantization | 8-bit dynamic, post-training |
| KD temperature / alpha | T = 2.0, alpha = 0.7 |
| Random seed | 42 |

**Hardware:** NVIDIA Tesla T4 (16GB VRAM), Intel Xeon @ 2.3GHz, 12.7GB RAM (Google Colab Pro)
**Software:** PyTorch 1.12.1+cu113, torchvision 0.13.1+cu113, CUDA 11.3, Python 3.7.13, CodeCarbon 2.1.4, NumPy 1.21.6, Ubuntu 18.04.6 LTS

## Repository Contents

```
Green_AI/
|-- Final_Green_AI.ipynb   # Full pipeline: baseline training, pruning, quantization,
                            # knowledge distillation, hybrid strategies, CodeCarbon
                            # instrumentation, SAA scoring, and all figures/dashboards
```

## Running This Project

1. Open the notebook: [Open in Colab](https://colab.research.google.com/drive/1lF3Roaeu6qADrKDL-vVPnL4keUYgluvt?usp=sharing)
2. Run all cells in order — CIFAR-10 and CodeCarbon are installed/downloaded automatically.
3. Runs with GPU support out of the box; no additional setup required.

## Limitations

As noted in the paper: CodeCarbon's regional-average emissions estimates may not reflect real-time grid conditions; the shared Colab environment introduces measurement noise; CIFAR-10 scale may not fully represent production-scenario trade-offs; results are specific to the NVIDIA T4 GPU and may not generalize to other hardware; and fairness/subgroup-performance analysis was limited in scope.

## Citation

```bibtex
@inproceedings{joshi2026greenai,
  author    = {Riya Joshi},
  title     = {Green AI Approaches: A Comparative Study of Pruning, Quantization,
               and Knowledge Distillation for Energy-Efficient Deep Learning},
  booktitle = {IEEE International Conference on Emerging Innovations (ICEI 2026)},
  year      = {2026},
  doi       = {10.1109/ICEI64468.2025.11447688}
}
```

## Author

**Riya Joshi** — Student Member, IEEE. B.E. Computer Engineering, Pune Institute of Computer Technology
[GitHub](https://github.com/Riya4105) - [LinkedIn](https://www.linkedin.com/in/riya-joshi-0b36423a5) - riyajoshi4105@gmail.com
