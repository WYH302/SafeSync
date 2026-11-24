<div align="center">

<img src="assets/safesync_github_banner.png" alt="SafeSync Overview Banner" width="92%"/>

# SafeSync

**SafeSync: Zero-Footprint Certified State Synchronization and Open-World Semantic Routing for Bandwidth-Constrained LEO Earth Observation**  
**面向带宽受限低轨对地观测的零内存足迹认证状态同步与开放世界语义路由框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](pyproject.toml)
[![Sender: CPU-Only](https://img.shields.io/badge/Pixel_Sender-CPU_Only_(117.5ms)-2EA44F.svg)](#computational-cost--reproducibility--算力开销与复现环境)
[![Benchmarks](https://img.shields.io/badge/Datasets-SpaceNet_7_%7C_SECOND-F59E0B.svg)](#key-empirical-findings--核心实验结果)

[**English**](#english-overview) · [**简体中文**](#中文说明) · [**Editable Fig. 1 (`PPTX` / `SVG`)**](assets/layered_state.pptx) · [**Quickstart / 快速上手**](#quickstart--快速开始) · [**Citation / 引用**](#citation--引用说明)

</div>

---

## English Overview

**SafeSync** is an open-source framework for **asymmetric space–ground state synchronization** in high-revisit Low-Earth-Orbit (LEO) Earth observation. Ground stations retain persistent high-resolution world-state archives and abundant compute, whereas small satellites operate under strict onboard memory, radiation-tolerant CPU, and per-pass downlink limits.

Instead of re-encoding static terrain on every pass or uplinking historical imagery, **SafeSync** decouples an immutable ground archive from a zero-footprint onboard state engine (`12.3 KB/area`, `0` neural parameters, `8.51 FPS` on CPU) with mathematical zero-omission safety bounds (`Proposition 1` + anytime-valid infinite-horizon `Corollary 1`), paired with a codec-agnostic open-world semantic routing layer (`US-BRS` + rate–distortion 0-1 knapsack `TC-PCR`) that elevates held-out change coverage across every base codec up to $U_{30}=0.8336$.

<div align="center">
<img src="assets/layered_state.png" alt="SafeSync Dual-Tier Architecture" width="98%"/>
<p><em>Fig. 1. Overview of the SafeSync dual-tier space-ground state synchronization and co-construction architecture.</em></p>
</div>

### Core Technical Modules

1. **Asymmetric State Factorization (`ASF`) & Adaptive-History Reuse Certificate**
   - Separates an immutable ground archive of acknowledged absolute image blocks from a pass-specific $4\times 4$ coarse illumination field (`LB01`), committing updates via a **60-byte uplink-free ACK**.
   - Restricts adaptive archive histories to a finite candidate set, enabling a union-bound reuse certificate (`Proposition 1`) with **0 false-stable block violations** across 118,317 evaluated blocks across all three caps, plus an anytime-valid log-polynomial extension (`Corollary 1`) that remains bounded through $t=1{,}000$ passes without expiration.
2. **Zero-Footprint Seeded Sketching (`ZFSS`) & Dual-Gate Asymmetric Decoupling (`DGAD`)**
   - Streams row-orthonormal Gaussian projections in 512-coordinate chunks directly from a 32-bit seed stored in the 24-byte `SC02` header (**0 bytes** persistent matrix memory), keeping persistent historical state at **12,328 bytes per $1024\times 1024$ area** ($99.09\%$ smaller than lossless PNG history even at $N=1$).
   - Decouples conservative $\chi^2_{24}(\alpha)$ safety certification (`STABLE`) from unbiased point-estimate candidate ranking (`UPDATED`).
3. **Unit-Sphere Bounded Residual Synchronization (`US-BRS`) & Prototype-Calibrated Routing (`TC-PCR`)**
   - Exploits unit-sphere token concentration on frozen `DINOv2-S/14` features ($\max|\Delta q|=41 \le 127$) for **100% lossless closed-loop `int8` residuals** ($20.62\%\text{--}35.37\%$ byte savings over open-loop delta coding) alongside NACK-triggered `Risk Anchor` erasure recovery.
   - Combines active-support packet selection, sender-visible 30-dB threshold-crossing prioritization, Ledoit–Wolf shrinkage-calibrated prototype distances, and per-cell rate–distortion 0-1 knapsack routing over `WebP75` residuals for open-world semantic change discovery (`TE01/RE01`).

---

## 中文说明

**SafeSync** 是面向低轨（LEO）高重访对地观测星座的开源**非对称星地状态同步与开放世界语义路由框架**：地面站维护高分辨率持久化世界状态档案并承担重型计算，而星载端在无 GPU、无历史参考图上注以及极严单帧下行带宽（如 32 KiB / 8 KiB）约束下运行。

SafeSync 从底层彻底打破传统无状态压缩与朴素历史缩略图的结构性瓶颈：像素层将星上历史状态压缩至 **12.3 KB/区域（降幅 99.09%）** 并提供 **0 漏报数学证明（Proposition 1 + 无限轨道寿命 Corollary 1）**，纯 CPU 端以 **8.51 FPS** 在 SpaceNet 7 三档码率全面超越 4.18 GB GPU 神经编解码器 `LIC-TCM`；语义层通过即插即用 0-1 背包 `TC-PCR` 路由在 SECOND 开放世界基准上全面拔高各类基础编解码器的未知目标覆盖率至 **$U_{30}=0.8336$**（精细 ROI F1 达 **0.8717**、三类宏平均 $U_{30}$ 与 FNR 双第一）。

### 方法组成概览

- **第一层：像素档案层同步（`ASF` + `ZFSS` + `DGAD`）**
  - **非对称状态分解（ASF）**：地面端将已确认的绝对像素块档案 $a_t$ 与当轨 $4\times 4$ 粗粒度光照场 $c_t$（`LB01`）解耦，仅通过 **60 字节免图像上注确认帧（ACK）** 保持星地状态一致，避免闭环渲染回写导致的误差累积，并通过有限候选集联合界（Proposition 1）保证自适应历史选择下的安全复用（三档码率实测 118,317 块中伪稳定违约数为 **0**）。
  - **零持久矩阵种子流式草图（ZFSS）**：从 `SC02` 摘要头部的 32-bit 种子按 512 维分块流式生成正交高斯投影行，持久投影矩阵开销为 **0 字节**，单区域（$N=1$）与多区域（$N=8$）历史状态均压缩至 **12.3 KB/区域**（较无损 PNG 历史档案节省 **99.09%**）。
  - **双门限非对称解耦（DGAD）**：将保守的 $\chi^2_{24}(\alpha)$ 安全认证门限与无偏点估计候选排序解耦，在保持 0 伪稳定违约的同时提升低码率建筑分割与抗轨间辐射漂移能力。
- **第二层：语义特征与开放世界共建层（`US-BRS` + `TC-PCR`）**
  - **单位球有界残差同步（US-BRS）**：利用归一化 `DINOv2-S/14` 特征在单位球面上的坐标集中特性，实现零额外失真的闭环 `int8` 特征残差传输（节省 $20.62\%\text{--}35.37\%$ 字节），并支持丢包场景下的 NACK 风险锚点（`Risk Anchor`）快速重同步。
  - **跨阈原型校准路由（TC-PCR）**：结合有效支撑集包选择、发送端可见 30 dB 跨阈指示以及 Ledoit–Wolf 协方差收缩校准原型距离，在严格 8 KiB 预算下保护开放世界新类别局部证据。

---

## Key Empirical Findings / 核心实验结果

### 1. SpaceNet 7 Multi-Temporal Urban Monitoring (8 Confirmation AOIs, 178 Monthly Updates)

<div align="center">
<img src="assets/sn7_decision_curves.png" alt="SpaceNet 7 Author-OOF Rate-Utility Curves" width="95%"/>
<p><em>Fig. 2. Four-fold cross-validated Author-OOF rate–utility curves across 32, 64, and 128-KiB update caps on SpaceNet 7.</em></p>
</div>

| Method / 配置 | Onboard State ($N=1$) | Model / Matrix Weights | Sender Time | 32-KiB B-F1 $\uparrow$ | 64-KiB B-F1 $\uparrow$ | 128-KiB B-F1 $\uparrow$ | 32-KiB E-F1 $\uparrow$ | False-Stable Viol. $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SafeSync-ASF+ (Ours)** | **12.3 KB** | **0 B (streamed)** | **117.5 ms (4-thr CPU)** | **0.4760** | **0.4964** | **0.5216** | 0.1899 | **0 / 118,317 (0%)** |
| SafeSync-DGAD (Legacy Block) | 12.3 KB | 0 B (streamed) | 254.7 ms (4-thr CPU) | 0.4250 | 0.4377 | 0.4431 | 0.1337 | 0 (0.00%) |
| SafeSync-ZFSS (Legacy Coupled) | 12.3 KB | 0 B (streamed) | 254.7 ms (4-thr CPU) | 0.3943 | 0.4122 | 0.4308 | 0.1418 | 0 (0.00%) |
| Deterministic DCT Bound | 13.9 KB | 0 B | 291.2 ms (4-thr CPU) | 0.3852 | 0.3975 | 0.4316 | 0.1050 | 0 (0.00%) |
| Low-Res $4\times$ WebP Thumbnail | 10.7 KB | 0 B | 226.4 ms (4-thr CPU) | 0.4390* | 0.4475* | 0.4526* | 0.1386 | 1,495 (21.79%) |
| Stateless Full WebP | 241 B (meta) | 0 B | 2,063.1 ms search (CPU) | 0.2794 | 0.3788 | 0.4698 | 0.1020 | N/A (stateless) |
| Scale-Hyperprior (`q1..8`) | 241 B (meta) | 61.1 MB | 11.54 s GPU / 36.4 s CPU | 0.3513 | 0.4623 | 0.5181 | 0.1693 | N/A (stateless) |
| LIC-TCM (35 candidates) | 241 B (meta) | 4,179.9 MB | 30.88 s GPU search | 0.4374 | 0.4957 | 0.5076 | **0.2255** | N/A (stateless) |

> *Note / 说明*：`SafeSync-ASF+` ranks **#1 in four-fold Author-OOF building F1 across all three caps (32, 64, 128 KiB)** on CPU with 0 B codec weights and 0 false-stable violations across all 118,317 blocks ($263\times$ faster than GPU LIC-TCM). Heavy GPU codecs (`LIC-TCM` / `Hyperprior`) retain an event-F1 edge at 64–128 KiB. `4x WebP` reaches 0.4390* on clean sequences only when granted SafeSync's `LB01` coarse restoration while committing 1,495 false-stable violations; without `LB01` under pass-to-pass orbital radiometric drift, direct `4x WebP` suffers **79.80%** background false triggers (vs. **19.37%** for SafeSync).

<div align="center">
<img src="assets/receiver_crops.png" alt="SpaceNet 7 Decoded Native Crops at 32 KiB" width="95%"/>
<p><em>Fig. 3. Native-resolution decoded crops at the 32-KiB cap on SpaceNet 7 (cyan outlines indicate ground-truth new-track buildings).</em></p>
</div>

---

### 2. State Scaling, Drift Immunity & Universal Plug-and-Play TC-PCR on SECOND (Cohort 3, 256 Pairs, 8-KiB Cap)

<div align="center">
<img src="assets/storage_utility_tradeoff.png" alt="Multi-Dimensional State, Safety, Drift and Open-World Results" width="95%"/>
<p><em>Fig. 4. (a) Multi-AOI persistent state scaling; (b) reuse safety violations and orbital-drift false-trigger rates; (c) SpaceNet 7 state footprint vs. 32-KiB Author-OOF building F1; (d) universal codec-agnostic TC-PCR across WebP, Hyperprior, and LIC-TCM on SECOND Cohort 3 at 8 KiB.</em></p>
</div>

| SECOND Cohort 3 Method (8-KiB Cap, Building Held-Out) | Base Codec Weights | Mean Payload | Held-Out $U_{30}$ $\uparrow$ | Gain vs. Standalone | Held-Out FNR $\downarrow$ | Global / Refined Later F1 $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Stateless Full WebP | 0 B (+DINO task) | 7.91 KiB | 0.5380 | — | 0.1699 | 0.8332 / 0.8332 |
| **SafeSync TC-PCR (WebP Base)** | **0 B (+DINO task)** | 7.86 KiB | **0.7117** | **+17.37 pp** ($>$ Hyperprior!) | 0.1747 (**0.0997** ref.) | 0.8209 / **0.8717** |
| Full Scale-Hyperprior | 61.1 MB | 7.36 KiB | 0.6379 | — | 0.1965 | 0.8212 / 0.8212 |
| **SafeSync TC-PCR (Hyperprior Base)** | **61.1 MB** | 7.73 KiB | **0.7682** | **+13.03 pp** (-36.0% UMR) | 0.1776 (0.1017 ref.) | 0.8133 / 0.8635 |
| Full LIC-TCM (35-cand. GPU SOTA) | 4,179.9 MB | **7.31 KiB** | 0.7688 | — | 0.1891 | 0.8268 / 0.8268 |
| **SafeSync TC-PCR (LIC-TCM Base)** | **4,179.9 MB** | 7.57 KiB | **0.8336** | **+6.48 pp** (-28.0% UMR) | 0.1693 (0.1053 ref.) | 0.8237 / 0.8633 |
| **SafeSync TC-PCR (LIC-TCM Raw)** | **4,179.9 MB** | 7.86 KiB | 0.8116 | +4.29 pp | **0.1523** | 0.8239 / — |
| Post-Hoc Hyperprior GT Oracle | 61.1 MB | 7.80 KiB | 0.8163 | Offline GT ceiling | 0.1576 | 0.8214 / — |

---

## Repository Structure / 目录结构

```text
SafeSync/
├── assets/                  # Architecture diagrams and benchmark result figures / 架构图与实验结果图
├── benchmarks/              # Summary CSV/JSON tables for quick inspection / 核心实验统计汇总表
├── configs/                 # Deterministic protocol and experiment configs / 协议与实验配置
├── docs/                    # Protocol specifications & reproducibility notes / 协议说明与复现文档
├── safesync/                # Core Python library (ASF, ZFSS, DGAD, US-BRS, TC-PCR) / 核心算法库
├── scripts/                 # Evaluation, profiling, plotting & verification scripts / 实验与绘图脚本
├── tests/                   # Unit & wire-protocol contract tests (pytest) / 单元测试与协议契约测试
├── LICENSE                  # MIT License & third-party data/model notices / 开源许可与第三方声明
├── pyproject.toml           # Package metadata & dependencies / 项目元数据与依赖配置
└── requirements.txt         # Pinned lightweight Python dependencies / 依赖列表
```

---

## Quickstart / 快速开始

### 1. Environment Setup / 安装环境

```bash
git clone https://github.com/WYH302/SafeSync.git
cd SafeSync
pip install -r requirements.txt
```

### 2. Run Protocol & Unit Tests / 运行协议与单元测试（纯 CPU，仅需数秒）

```bash
pytest -q
```

This verifies `SC02` zero-footprint seeded chunk streaming (`ZFSS`), `LB01/LA01` asymmetric state factorization (`ASF`), dual-gate decoupling (`DGAD`), `US-BRS` lossless `int8` residual bounds, and `TC-PCR` prototype routing contracts on CPU without downloading multi-gigabyte datasets.

### 3. Regenerate Figures & Summary Tables / 重新生成论文图表

```bash
python scripts/draw_layered_method.py
```

For full dataset preparation (`SpaceNet 7` and `SECOND`) and end-to-end neural baseline reproduction (`CompressAI` and `LIC-TCM`), please see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

## Computational Cost & Reproducibility / 算力开销与复现环境

All benchmark experiments in our study were conducted without any model training or fine-tuning on a single workstation:

- **Hardware**: AMD Ryzen 7 5700X CPU (pinned to 4 threads for sender benchmarks) + NVIDIA GeForce RTX 3090 GPU (24 GB VRAM)
- **Software**: Python 3.11 / 3.12, PyTorch `2.8.0+cu128`, CUDA `12.8`, NumPy `2.4`, Pillow `12.2`, OpenJPEG `2.5.4`
- **Total Campaign Compute**:
  - **8.42 RTX 3090 GPU-hours** in total across all neural baselines, downstream evaluations, and supplementary audits (`1.70 h` SpaceNet 7 `LIC-TCM`, `1.69 h` SECOND `LIC-TCM`, `0.82 h` SpaceNet 7 `Scale-Hyperprior` under a 1.8 GB VRAM cap, `1.39 h` SECOND `Hyperprior`/`DINOv2-S/14`/`TC-PCR v2`, `1.08 h` primary `U-Net` evaluation, and `1.74 h` supplementary `exp0200`/orbital-drift/refined-ROI/cross-architecture audits; **0.00 training hours**).
  - **0.00 GPU-hours / 0 MB VRAM / 0 neural params / 0.157 GFLOPs per 1024² frame** for **SafeSync's pixel-tier sender** (`117.5 ms` / **`8.51 FPS`** at 32 KiB, `136.7 ms` / `7.32 FPS` at 64 KiB, `173.7 ms` / `5.76 FPS` at 128 KiB on 4 CPU threads; `20.9 s` per 178-update pass).
  - **5.32 CPU-hours** across all stateful protocol transactions, orbital-drift tests, archive profiling, and 10,000-resample paired geographic bootstrap intervals.

---

## Design Reflections, Pitfalls & Empirical Observations / 真实思考、踩坑复盘与独特观察

Building **SafeSync** under strict zero-training, hard byte-cap, and frozen-evaluator constraints uncovered four non-obvious engineering lessons that directly shaped the final architecture:

1. **Why our early block-piecewise `ASF` lost `-8.18 pp` to our own `SafeSync-ASF+` (`0.3943 -> 0.4760`)**: Our first working prototype used $16\times 16$ nearest-neighbor coarse tiling (`LB01` JPEG) plus quality-60 block JPEG. Mathematically, it satisfied Proposition 1 with 0 violations—yet its Author-OOF building F1 trailed GPU `LIC-TCM` by `-4.31 pp`! Visualizing decoded crops immediately exposed the culprit: $16\times 16$ staircase seams across reused blocks injected fake grid edges into frozen EfficientNet-B3 early conv layers, while JPEG coarse encoding silently ate `8.40 KiB` of a `32 KiB` cap. Switching to WebP85 coarse (`1.33 KiB`), smooth bilinear transfer $\widetilde U$, $r=7$-px feathering, and calibration-locked position-invariant JPEG 4:4:4 preserved Proposition 1 unchanged while flipping CPU SafeSync to **#1 across all three caps**.
2. **The hidden `2.36 MB` projection-matrix trap at $N=1$**: Classic random-projection sketching looks tiny on paper (`12.3 KB/area`), until you count the $24\times 12{,}288$ `float64` matrix `A` itself (`2.36 MB`—bigger than a lossless PNG archive for a single area!). Streaming orthonormal rows in 512-coordinate chunks from a 32-bit `SC02` header seed eliminated all `2.36 MB` of persistent matrix RAM while remaining bit-identical across all 47,616 blocks (max float diff $5.55\times 10^{-16}$).
3. **Why matched-storage `4x WebP` thumbnails deceive clean benchmarks yet fail in orbit**: Downsampling a reference frame $4\times$ fits inside `10.7 KB`, but averages away 4-pixel rooftop construction edges (**1,495 false-stable violations / 21.79%**) and collapses under pass-to-pass solar/radiometric shift (**79.80%** background false triggers vs. **19.14%–19.37%** under SafeSync's orthogonal `ASF` projection).
4. **Why rate–distortion SSIM / DINOv2 cosine diverges from native 4-m task utility—and how active-support voting saved `TC-PCR`**: Across 2,136 decoded SpaceNet 7 frames, stateless codecs achieve higher current-frame SSIM (`0.9602` vs. `0.8414`) because they re-encode pass-$t$ seasonal shading, yet blur tiny 4-px buildings by up to `-19.66 pp` F1. Similarly, on SECOND Cohort 3, naive full-image residual routing got outvoted by ~53 static background cells until we restricted packet selection strictly to active changed support and prioritized 30-dB threshold crossings.

> **中文复盘精华**：
> 1. **早期分块 `ASF` 为什么差点输给神经编解码器（`-8.18 pp` 踩坑）**：初版 $16\times 16$ 最近邻块拼接虽满足 0 违约证明，但会在复用区引入棋盘格伪边缘干扰冻结 U-Net，且粗光照场 JPEG 偷走了 `8.40 KiB` 带宽；改为 WebP85（`1.33 KiB`）+ 双线性平滑对齐 + 位置无关 JPEG 4:4:4 后，在 **0 权重、不破坏 Proposition 1** 的前提下实现纯 CPU 三档全第一。
> 2. **单区域 `2.36 MB` 投影矩阵陷阱**：若把 $24\times 12288$ 双精度矩阵常驻内存，单区域（$N=1$）反而比无损 PNG 历史更占空间；通过 `SC02` 头部 32-bit 种子按 512 维流式重生成正交行，才真正做到单/多区域均稳定减容 **99.09%**。
> 3. **缩略图参考的“干净假象”**：$4\times$ WebP 缩略图在无漂移时看似够用，但下采样会抹平 4 像素新建屋顶（导致 **1,495 次伪稳定漏报**），在真实轨间辐射漂移下背景误触发飙升至 **79.80%**（SafeSync 仅 **19.14%–19.37%**）。
> 4. **为何像素指标（PSNR/SSIM）与 4 米建筑分割背离**：无状态编解码器重编码当轨季节色差而拿到更高全局 SSIM，却糊掉了小目标屋顶；在 SECOND 开放世界路由中，同样只有把数据包裁决限定在**有效变化支撑集**并显式奖励 **30 dB 跨阈块**，才能避免 53 个静态背景块“票决淹没”真正的未知目标增益。

---

## Citation / 引用说明

If you find this repository or the experimental protocols helpful in your research, please consider citing:

```bibtex
@article{wu2026safesync,
  author  = {Wu, Yonghao},
  title   = {{SafeSync}: Zero-Footprint Certified State Synchronization and Open-World Semantic Routing for Bandwidth-Constrained {LEO} {Earth} Observation},
  journal = {Advances in Space Research},
  year    = {2026},
  note    = {Code available at https://github.com/WYH302/SafeSync}
}
```

## Acknowledgments / 致谢

We sincerely thank the teams behind [SpaceNet 7](https://spacenet.ai/sn7-challenge/), [SECOND](https://captain-whu.github.io/SCD/), [CompressAI](https://github.com/InterDigitalInc/CompressAI), [LIC-TCM](https://github.com/jmliu206/LIC_TCM), and [DINOv2](https://github.com/facebookresearch/dinov2) for sharing their public datasets, codebases, and pretrained checkpoints with the research community. Feedback, issues, and pull requests are always welcome.
