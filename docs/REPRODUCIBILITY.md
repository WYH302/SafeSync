# SafeSync local reproducibility guide / 本地复现说明

## 当前研究范围与历史快照

2026-11-04继续开展原科研目标的实测。第三组256对冻结确认已完成29,952条实际观测，七项联合条件通过五项。现代TCM比较仍在运行。当前PDF与新研究快照会在该批比较结束后绑定到最终构建清单。原科研总验收仍未通过。

`safesync-review-FINAL-02.zip`固定保存2026-10-29收缩技术稿，当时主稿9页、补充6页。它未包含本轮新增研究，历史包的完整性通过不能代表原科研目标通过。后续新增包使用RESEARCH/DRAFT名称并单独报告未完成项。下文FINAL-02命令仅复查这份历史快照。

新增研究可重跑命令保留在各驱动脚本。第三组结果的离线资产生成使用以下命令。

```powershell
.venv-safesync\Scripts\python.exe scripts/analyze_residual_confirmation.py
.venv-safesync\Scripts\python.exe scripts/build_residual_confirmation_assets.py
```

TCM使用作者固定commit的模型与七组公开权重，部署时作为双方共有资源，下载地址与SHA256见resources/modern_codec/weights/registry.json。原模型算子保持作者实现，官方CompressAI原生rANS通过本地WSL桥调用。流式解码接口使用精确前缀重放，会增加实现相关耗时。原128倍数规则增加最小256边长的兼容修订，原计划及修订单独保留。来源、菜单检查和实际码流均独立记录。

在原CUDA环境与官方原生熵桥准备完毕后，新增TCM依赖安装到独立目录，命令如下。

```powershell
uv pip install --python .venv/Scripts/python.exe --no-deps --target resources/modern_codec/.deps einops==0.6.1 timm==0.6.13
.venv\Scripts\python.exe scripts/fetch_tcm_weights.py
.venv\Scripts\python.exe scripts/check_tcm_menu.py
```

同一工作区中，下面两条分别运行SECOND和SN7发送端。SECOND默认三个互斥worker，需要分别传入--worker-index 0、1和2。实验完成后运行对应--finalize-only，再执行分离地面评价与分析。运行中的续跑会核对已有锁与结果哈希。

```powershell
.venv\Scripts\python.exe scripts/run_tcm_control.py --workers 3 --worker-index 0
.venv\Scripts\python.exe scripts/run_sn7_tcm_control.py --workers 1 --worker-index 0
.venv\Scripts\python.exe scripts/evaluate_tcm_control.py
.venv\Scripts\python.exe scripts/evaluate_sn7_tcm_control.py
.venv-safesync\Scripts\python.exe scripts/analyze_tcm_controls.py second
.venv-safesync\Scripts\python.exe scripts/analyze_tcm_controls.py sn7
```

This package supports two different checks: verifying the retained evidence and
recomputing results from images and public models. The first is small, CPU-only,
and offline. The second requires separately obtained data, model weights and the
disclosed numerical environments. A successful evidence check does not claim that
the neural inference or every packet was rerun.

The archive is a **local reviewer package**, not an external release. Author names,
affiliations, acknowledgments and the SafeSync source license await the authors.
Third-party source notices are preserved; no new license is asserted for their
weights or data. See `reproduction/THIRD_PARTY_NOTICES.md` inside the archive.

## 1. Verify the historical package / 历史包快速验收

历史技术包为 **`safesync-review-FINAL-02.zip`**。先解压到一个新目录，再在该解压
目录中运行以下命令；只需要 Python 3.11 或更新版本的标准库，不安装任何依赖：

```powershell
python -I -S scripts/verify_reproduction_bundle.py --root . --require-complete --report ../safesync-review-FINAL-02.verify.json
```

结果应为 `status: passed`、`package_mode: final`、`complete_evidence: true`，
且报告的 `pending` 为空。报告放在解压目录外，避免下次完整性检查把它当成未列出的
新增文件。`FINAL-01` 是被替代的旧快照；下面的 DRAFT 命令只供开发阶段另建包使用。

The historical delivery is `safesync-review-FINAL-02.zip`. Extract it to a new
directory and run the standard-library command above from that extracted root.
Expect a passed final package with complete evidence and no pending items.
From the complete project workspace, the equivalent archive check is:

```powershell
python -I -S scripts/verify_reproduction_bundle.py --archive reproduction/safesync-review-FINAL-02.zip --require-complete --report reproduction/safesync-review-FINAL-02.offline.json
```

### Building a separate development draft

To build another development snapshot from a complete project source workspace:

```powershell
python scripts/build_reproduction_bundle.py --output reproduction/review-DRAFT-01.zip
python scripts/verify_reproduction_bundle.py --archive reproduction/review-DRAFT-01.zip --report reproduction/review-DRAFT-01.check.json
```

These commands use only the Python standard library. They do not access the
network, install packages, load Torch, use the GPU, or change the frozen runs.
An existing archive is never overwritten: choose a new filename for each build.
The builder verifies a freshly extracted copy of the ZIP, then creates a SHA256
sidecar and a JSON verification report alongside it.

`BUNDLE_MANIFEST.json` lists the relative path, byte length and SHA256 of every
payload file. Its own hash is recorded in the external verification report; the
ZIP hash also binds the manifest. This is an integrity receipt, not a digital
signature or independent attestation of scientific correctness.

The allowlist is explicit in `scripts/build_reproduction_bundle.py`. It contains
the manuscript/supplement, relevant implementation and launch scripts, frozen
source snapshots, raw metric CSV/JSON files, summary/specification/completion
receipts, data manifests, download recipes, and public model URLs and hashes.
It excludes raw imagery/labels, packet and ACK binaries, decoded image sequences,
probability NPZ files, model weights, dependency wheels, virtual environments,
logs and credentials. Some paper figures contain illustrative image crops.
The compressed-archive control also retains 24 final historical PNG/JSON pairs
(one per AOI/cap), with SpaceNet/Planet attribution; full image sequences remain
excluded. See `docs/COMPRESSED_ARCHIVE_CONTROL.md` for this memory sensitivity.
Paths to excluded objects in the original receipts remain useful provenance;
their presence in a receipt does not mean the object is in this compact archive.

The default checker independently recomputes the eight-AOI/186-month/178-update
cohort, the 5,874 main and 534 gate30 transport row counts and caps, task count
aggregation, and the 27-row exp0200 paper table's building F1, event F1 and actual
mean update bytes. It pools TP/FP/FN within each AOI and then gives each AOI equal
weight. It does not average per-frame F1. It also checks completion and hash
bindings for the additional OOF and full-current/old-only controls when present.
For the 4,806-row OOF core it reuses `validation_report.json` only after checking
the verifier source hash and every required delivered artifact hash. That report
already covers the full model/fold/threshold identities, count-first aggregation
and all 12 paired bootstrap contrasts; packaging does not repeat that analysis.

To run the existing asset builder's hash check and rebuild its numeric tables in
a disposable copy, add `--check-assets`. This needs NumPy and Matplotlib, but no
images, models, CUDA or internet:

```powershell
.venv-safesync\Scripts\python.exe scripts/verify_reproduction_bundle.py --archive reproduction/review-DRAFT-01.zip --check-assets --report reproduction/review-DRAFT-01.tables.json
```

The corresponding commands inside an already extracted **disposable copy** are:

```powershell
python scripts/build_task_paper_assets.py --check
python scripts/build_task_paper_assets.py
```

`--check` alone verifies existing bound file hashes; it does not recompute the
numbers. The wrapper performs both steps and compares the regenerated table
values. Figure PDF/SVG byte identity is not required across fonts or renderer
versions. The default archive verifier can also run on an extracted directory
with `--root PATH`; put its `--report` outside that directory to avoid an unlisted
file on the next integrity check.

## 2. Completion is explicit

A DRAFT may contain only completed portions of a running experiment. Its report
and `reproduction/COMPLETION_STATUS.json` enumerate every pending item.
`status: passed` on a DRAFT verification means its stated scope passed, not that
all experiments have finished. Final assembly is separate:

```powershell
.venv-safesync\Scripts\python.exe scripts/build_reproduction_bundle.py --final --check-assets --output reproduction/safesync-review-FINAL-02.zip
python scripts/verify_reproduction_bundle.py --archive reproduction/safesync-review-FINAL-02.zip --require-complete
```

Final mode refuses missing/unfinished evidence, including:

| Evidence | Required outputs |
|---|---|
| Original transport | Eight area markers, 5,874 rows |
| exp0200 tasks | 3,738 main + 534 gate30 + 534 hyperprior rows |
| Image quality | 5,874 main records; 534 learned-codec records |
| Learned transport | 534 packets and 534 separately replayed outputs |
| Author OOF sensitivity | `resources/pretrained_ground_crossfit/receipts_v1/complete.json`: 4,806 outputs, eight area markers; `validation_report.json`: passed |
| exp0200 full-current / old-only | `results/lossless_control/progress.json`: 356 outputs |
| OOF full-current / old-only | `resources/pretrained_ground_crossfit/lossless_controls_v1/complete.json`: 356 outputs, eight AOIs |
| CPU learned profile | Two completed full-candidate-search frames |
| Compressed historical archive | 558 exact PNG snapshot checks, including 534 updates and 24 bootstraps |
| Paper artifacts | `paper/manuscript.pdf`, `paper/supplement/supplementary.pdf` and their TeX sources |
| Paper technical receipt | `paper/technical_checks.json`: PASS, binding both final PDF hashes and the checker source hash |

Final mode refers to evidence completeness for local review. It does not supply
author approval, a release license, submission authorization or external hosting.

## 3. Environments: keep the three execution paths separate

`reproduction/environment/observed.json` and the adjacent `*.observed.txt` files
are generated from project-local distribution metadata at packaging time.
The original base locks are preserved unchanged. They omit the overlays below;
neither base lock alone describes the full ground evaluator. Observed inventories
are not a guarantee that modern package resolvers can recreate this integration.

| Path | Recorded execution environment / additional requirements |
|---|---|
| Offline hash/count checks | Python 3.11+ standard library only |
| Tables and layered CPU runs | `.venv-safesync`: Python 3.12.14, NumPy 2.4.3, Pillow 12.2.0; tables use Matplotlib 3.10.9. `requirements-analysis.lock.txt` records the broader environment, including packages not needed for tables |
| GeoTIFF/GeoJSON preparation | `data/spacenet7_pilot/.deps`: Rasterio 1.5.2/GDAL 3.12.2 plus affine, attrs, click, click-plugins, cligj, certifi, pyparsing; exact versions in `geo-overlay.observed.txt` |
| Ground task inference | `.venv`: Python 3.11.15, Torch 2.8.0+cu128, NumPy 2.4.6, Pillow 12.3.0; base lock `requirements-cuda.lock.txt` |
| Ground import overlays | `resources/pretrained_ground/.deps`: torchvision 0.23.0+cu128, OpenCV-headless 4.10.0.84, timm 0.1.20, efficientnet-pytorch 0.7.0, pretrainedmodels 0.7.4 |
| Legacy author imports | `resources/pretrained_ground/.legacy_deps`: timm 0.1.20, efficientnet-pytorch 0.7.0, pretrainedmodels 0.7.4, munch 4.0.0, six 1.17.0, tqdm 4.67.1; author SMP 0.1.0 source in `resources/pretrained_ground/model_source` |
| Task metric overlay | `results/layered_tasks/.deps`: SciPy 1.17.1 |
| Learned entropy bridge | Windows Torch network plus original CompressAI 1.2.8 C++ extensions under local WSL Python 3.10.12; separate process and numerical flags |

For a new machine, create independent environments with the specified Python
versions. Tables need only NumPy/Matplotlib and their normal dependencies.
Full inference needs the base CUDA environment and overlays. The following are
**optional network installation recipes**, not steps performed by the bundler:

```powershell
# Run with the selected Python environments, after creating them in a NEW work root.
.venv-safesync\Scripts\python.exe -m pip install -r requirements-analysis.lock.txt
.venv\Scripts\python.exe -m pip install -r requirements-cuda.lock.txt
.venv-safesync\Scripts\python.exe -m pip install --no-deps --target data/spacenet7_pilot/.deps -r reproduction/environment/geo-overlay.observed.txt
.venv\Scripts\python.exe -m pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cu128 --target resources/pretrained_ground/.deps -r reproduction/environment/ground-overlay.observed.txt
.venv\Scripts\python.exe -m pip install --no-deps --target resources/pretrained_ground/.legacy_deps -r reproduction/environment/ground-legacy-overlay.observed.txt
.venv\Scripts\python.exe -m pip install --no-deps --target results/layered_tasks/.deps -r reproduction/environment/metrics-overlay.observed.txt
```

The explicit `--target` paths are part of the frozen import behavior. Some tools
put `.legacy_deps` before `.deps`, and the author model source before both.
Do not replace the author fork with a current PyPI SMP package. Its fixed commit
is `e590675a2c23e2fb68a37541e88fccd57d8075e5`; the original source and license are
included. `safesync/pretrained_ground.py` contains the disclosed `torch._six`
compatibility alias for legacy timm. These frozen sources are not patched by this
packaging work. CPU execution is supported by that model adapter, but the recorded
task results used the CUDA environment and flags in their receipts.

The learned-codec integration is **not a standard CompressAI pip installation**.
Its NumPy 2.4.6 is outside upstream CompressAI 1.2.8's declared `numpy<2` bound;
the tested source-module/entropy-bridge path is the stated scope. The bridge in
`safesync/entropy_bridge.py` explicitly invokes the WSL distribution
`Astra-Ubuntu-22.04`. Recreating this original path requires a local distribution
of that name with Python 3.10 and compatible Linux C++ runtime. A renamed distro,
Linux-native implementation, or different software stack is an adaptation and
must receive a new run identity; it is not silently substituted into a frozen run.

The original native-extension wheel is:

```text
resources/learned_codec/wheelhouse/compressai-1.2.8-cp310-cp310-manylinux_2_34_x86_64.whl
bytes: 442184
SHA256: 7ba289e3f8188845f8c02b6852ab3364e26c6d27d699f9cbd4e9a1cec8c486ca
```

It is excluded from the bundle. Obtain the matching official PyPI release wheel,
verify this hash, and extract it to `resources/learned_codec/linux_wheel/` in the
new work root. Fetch the official source tarball using the included URL and hash
in `resources/learned_codec/source/compressai-1.2.8.tar.download.json`; the included
source under `source/vendor/compressai/` is the recorded version. The two `.so`
hashes are in `native_extensions_probe.json`. Run the included
`probe_native_extensions.py` and `smoke_entropy_bridge.py` before any full codec
run. Their checks are limited to the original ScaleHyperprior/rANS path.
The ground evaluator never imports the learned codec: run them in separate
processes so their Torch numerical flags cannot leak between tasks.

## 4. Data and public model reconstruction

Never run download or inference commands in the original confirmation folders.
Keep the verified archive/extracted evidence unchanged. Make a **new work root**,
for example `reproduction/runs/recompute-01/`, containing copies of the code,
data/source manifests, resource provenance and third-party source. Do not copy
completed output folders or probability caches into this new work root. Preserve
the top-level helper scripts under `results/layered_tasks/`. Create its own
environments and run every command below from that new root. Hardcoded relative
paths then write new results without overwriting the original evidence.

The data downloader retains the frozen selection order. Twelve downloaded AOIs
comprise the four original calibration/development AOIs and the next eight
confirmation AOIs. Missing months are not interpolated. Run the plan first and
inspect its byte total; set `--max-download-bytes` to an intentional cap if the
default is insufficient:

```powershell
.venv-safesync\Scripts\python.exe scripts/fetch_spacenet7.py --aois 12 --plan-only
.venv-safesync\Scripts\python.exe scripts/fetch_spacenet7.py --aois 12 --workers 4
.venv-safesync\Scripts\python.exe scripts/fetch_spacenet7.py --validate-only
.venv\Scripts\python.exe scripts/fetch_sn7_evaluator.py --fetch-weight
.venv-safesync\Scripts\python.exe resources/pretrained_ground_crossfit/prepare_crossfit.py --download
.venv\Scripts\python.exe resources/pretrained_ground_crossfit/prepare_crossfit.py --inspect-only
.venv\Scripts\python.exe resources/learned_codec/fetch_public_assets.py
```

The package already contains the fixed official solution source and author SMP
fork. `--fetch-model-source` is available in `fetch_sn7_evaluator.py` if that
source must be fetched again. Keep `source_identity.json` to preserve the recorded
official commit rather than resolving a new repository head. The data download
recreates `manifest.json`; do not replace the frozen `development_manifest_v1.json`
or `confirmation_manifest_v1.json` with that combined manifest.

All original object names, byte counts, ETags and SHA256 records are in the data
manifests/listings. Source objects and regenerated lossless derivatives should be
checked against the frozen manifest identities. The exp0200 URL/hash is in
`resources/pretrained_ground/weight_receipt.json`; OOF weights 0201–0204 are in
`download_receipts.json` and `model_catalog.json`; eight hyperprior weights are in
`resources/learned_codec/asset_manifest.json`. Dataset, source-code, model and
calibration hashes identify different objects and must not be interchanged.

## 5. Full rerun sequence in the new work root

Use the original 11 methods and all three caps. A short development sanity run
can precede the full run; do not shorten a confirmation sequence. Each area is
frozen before its pixels are read:

```powershell
foreach ($area in 0..7) {
  .venv-safesync\Scripts\python.exe scripts/run_layered_sequences.py --confirmation --manifest data/spacenet7_pilot/confirmation_manifest_v1.json --area-indices $area --out "results/layered/confirmation_v1/area_$area" --freeze-only
  .venv-safesync\Scripts\python.exe scripts/run_layered_sequences.py --confirmation --manifest data/spacenet7_pilot/confirmation_manifest_v1.json --area-indices $area --out "results/layered/confirmation_v1/area_$area"
  .venv-safesync\Scripts\python.exe scripts/run_full_cache_controls.py --confirmation --manifest data/spacenet7_pilot/confirmation_manifest_v1.json --area-indices $area --out "results/full_cache_controls/gate30/area_$area" --freeze-only
  .venv-safesync\Scripts\python.exe scripts/run_full_cache_controls.py --confirmation --manifest data/spacenet7_pilot/confirmation_manifest_v1.json --area-indices $area --out "results/full_cache_controls/gate30/area_$area"
}
.venv-safesync\Scripts\python.exe scripts/summarize_layered_quality.py --scope confirmation
.venv\Scripts\python.exe results/layered_tasks/run_confirmation.py
.venv\Scripts\python.exe results/layered_tasks/summarize_confirmation.py
.venv\Scripts\python.exe results/layered_tasks/make_confirmation_report.py
foreach ($area in 0..7) {
  .venv\Scripts\python.exe scripts/evaluate_full_cache_controls.py --area-index $area
}
.venv\Scripts\python.exe scripts/evaluate_full_cache_controls.py --summarize-only
```

The learned-codec sequence follows installation and successful entropy-bridge
sanity checks. Replays and ground task inference are separate commands:

```powershell
.venv\Scripts\python.exe scripts/run_learned_codec.py --confirmation --manifest data/spacenet7_pilot/confirmation_manifest_v1.json --area-indices 0 1 2 3 4 5 6 7 --limit-months 0 --out results/learned_codec/confirmation_q1_8_v1 --entropy-bridge
.venv\Scripts\python.exe resources/learned_codec/replay_packets.py --run results/learned_codec/confirmation_q1_8_v1
.venv\Scripts\python.exe scripts/evaluate_learned_receipts.py
.venv-safesync\Scripts\python.exe scripts/evaluate_learned_quality.py
.venv\Scripts\python.exe resources/pretrained_ground_crossfit/run_receipts.py --check-inputs
.venv\Scripts\python.exe resources/pretrained_ground_crossfit/run_receipts.py
.venv-safesync\Scripts\python.exe resources/pretrained_ground_crossfit/verify_completed_receipts.py
.venv\Scripts\python.exe scripts/run_lossless_control.py
.venv\Scripts\python.exe resources/pretrained_ground_crossfit/run_lossless_controls.py
```

The included frozen calibration files are the reproduction inputs. Repeating
calibration is a separate provenance check: use another fresh work root and the
original two calibration AOIs, never tune on the eight confirmation AOIs.
`scripts/calibrate_crossfit_ground.py` is retained for this purpose. OOF means
author validation-best folds, not an untouched test cohort; see the dedicated
OOF documentation for the model-selection boundary.

All commands fail on conflicting frozen specifications. Do not bypass a failed
identity check or edit its stored hash. Record the reason and create a new run
identity for a documented software adaptation. Recomputed timestamps and runtime
measurements will differ. Different codec, Pillow, BLAS, GPU, CUDA or Torch
versions can alter packet/reconstruction/probability bytes and even selection near
thresholds; bitwise identity across such differences is not promised. The retained
metric-table path is separately testable without these software differences.

## 中文运行要点

1. **快速验收不需要原图、权重或 GPU。** 先解压 `safesync-review-FINAL-02.zip`，
   用第 1 节置顶命令验证最终包；只用 Python 标准库。DRAFT 命令用于另建开发快照。
   `--check-assets` 另外需要 NumPy 和
   Matplotlib，会在临时副本中校验现有哈希、重建数值表并比较，原文件不被改写。
2. **DRAFT 通过不代表实验全部结束。** 检查报告的 `complete_evidence` 和
   `pending`。`--final` 必须等 4,806 条 OOF、两套各 356 条对照、全部地区完成标记
   和论文 PDF 到齐才会成功。最终包仍是本地审阅包，作者、授权和公开发布另行处理。
3. **完整复跑必须新建工作根目录。** 保留原始包和确认输出；在例如
   `reproduction/runs/recompute-01/` 下复制代码、固定清单、资源身份文件和第三方
   源码，建立独立环境，再按第 4、5 节下载和执行。不要把完成的结果目录或概率缓存
   拷入新运行目录；保留 `results/layered_tasks/` 顶层的启动/汇总脚本。
4. **CPU、CUDA 地面模型、WSL 熵编码是三条不同环境路径。** 根目录两个 lock 文件
   不包含所有 `.deps`；第 3 节给出实际导入目录和固定版本。WSL 分发名
   `Astra-Ubuntu-22.04` 写在冻结桥接脚本中，不具备完全跨平台即插即用性。
   如需适配，另建运行身份，不能修改原确认代码和哈希后宣称同一冻结运行。
5. **下载的是公开来源，清单记录身份。** 包内不含原图、GT、权重、码流、NPZ、
   Python 环境；按清单的 URL/对象键和 SHA256 重建。数字验收从逐月计数按 AOI
   汇总，再八区等权，不把月份、seed 或像素当作独立地理样本。
6. **证据边界不混淆。** 离线表格复算通过不等于模型重新推理通过；原 exp0200
   结果与 author-OOF 敏感性分析分别保留。软件和硬件变化可能改变逐字节输出，
   不承诺跨版本 bitwise 一致；时间、字体渲染和 PDF 元数据也可能变化。
