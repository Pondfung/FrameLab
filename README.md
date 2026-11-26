# 帧工坊 FrameLab

Windows 本地视频超分与插帧工具。日常导出选 NVIDIA VFG High 插帧、VSR Ultra 超分，可单独使用，也可先插帧再超分。输入支持逐行、方形像素、BT.709 SDR、恒定帧率视频。每段素材可单独设置输出并在分块边界续跑。

[开发交接](docs/FrameLab-开发交接.md)记录本机最新测试构建、验收样本与待办。

## 选择版本

- **日常导出**用本机单文件版 `dist\FrameLab-NVIDIA.exe`。在配有兼容 NVIDIA 显卡和驱动的 Windows 机器上双击启动，首次启动会将随包运行时解压到 `%LOCALAPPDATA%\FrameLab NVIDIA\bundles`。只做超分时关闭插帧，只做插帧时关闭超分。串联快速导出选择先插帧、后超分和 MP4。
- **模型评估与开发**用完整版 `.runtime\framelab.exe` 或 CLI。完整版菜单保留全部引擎。两个版本的全新配置均选择 VFG High，新导入的素材继承界面中的默认配置。

截至 2026-11-26，桌面导出默认使用 SDK 自动切镜，CLI 提供独立自动切点试验选项。1126 录屏的 60→120 片尾时间戳兼容修复已经验证。从当前源码构建或使用交接文档中的测试版时，导入视频或选中已完成的作业后，在主播放器左上角点「复核模式」，播放或逐帧定位，再标记当前帧。生成修复预览后可在同一时间轴切换「修复前／修复后」。 [VFG 错帧优化与验收](docs/VFG-错帧排查与测试备选.md)

## 完整版安装与自检

在仓库根目录的 PowerShell 中执行。需要 Python 3.13、Git、FFmpeg 和 ffprobe。GPU 环境及权重保存在 `.runtime` 中。

```powershell
python setup_runtime.py deps
python setup_runtime.py nvidia-vfx
.\run_video_lab.ps1 doctor
```

`deps --torch-wheel <路径>` 可复用官方 CUDA Torch wheel，`nvidia-vfx --wheel <路径>` 可复用 NVIDIA VFX wheel。启动脚本为中文路径下的 NVRTC 提供 ASCII 路径别名。

评估其他模型时，`python setup_runtime.py weights` 准备 GIMM-VFI 和 SeedVR2 3B 权重，可按 `weights gimm-vfi` 或 `weights seedvr2` 单独安装。7B FP16 使用 `python setup_runtime.py seedvr2-7b`。

BasicVSR++ 和 RIFE 的 `run` 作业要求权重已就位。新装环境可以先用同一引擎的 `render` 跑一段两帧短样，自动下载所选权重。以下示例使用一个符合输入要求的 `input.mkv`。

```powershell
.\run_video_lab.ps1 render .\input.mkv .\basic-preview.mkv --engine basicvsrpp --size 1280x720 --frames 2
.\run_video_lab.ps1 render .\input.mkv .\rife-preview.mkv --engine rife --frames 2
```

## 桌面界面

从源码运行完整版桌面界面时，在准备好本机环境并安装 Rust 后执行以下命令。

```powershell
cargo run --release --locked --manifest-path desktop/Cargo.toml
```

单文件版也从 `desktop/` 构建。安装本页的完整版运行环境、Rust 和 FFmpeg 后，从仓库根目录执行 `python scripts/build_nvidia.py`，产物为 `dist\FrameLab-NVIDIA.exe`。这一步打包两版共用的 Python 作业、引擎和切点实现。

左侧预览原片。右上为当前素材配置超分、插帧和技术路径，右下导入多段素材、设置输出位置并启动批量导出。作业串行运行，每段素材可使用自己的模型、分辨率、帧率、格式和输出目录。分辨率可选 1080P、2K（1440P）、4K（2160P）或自定义，预设按原片宽高比换算。帧率可选 24、30、60、120 或自定义。素材卡片显示原片帧率、导出进度与完成用时。已完成或中断的素材可点「新建导出」重新配置。界面范围见 [Rust 桌面前端开发计划](docs/Rust桌面前端开发计划.md)。

## 处理与续跑

```powershell
.\run_video_lab.ps1 probe .\input.mkv
# 示例输入为 1080p30，目标为 1440p60
.\run_video_lab.ps1 run .\input.mkv .\output.mp4 --interpolate nvidia-vfg --upscale nvidia-vsr --size 2560x1440 --fps 60
.\run_video_lab.ps1 status .\output.mp4.job
# 在另一终端暂停，当前分块完成后停下
.\run_video_lab.ps1 pause .\output.mp4.job
.\run_video_lab.ps1 resume .\output.mp4.job
```

- `run` 可只做超分或插帧。串联默认先插帧，`--order upscale-first` 可反转。`--cut-frame` 使用从零开始的源帧号，可重复指定。`--scene-threshold` 会将 FFmpeg 场景分数过阈值的帧直接加入切点清单。该参数独立于 VFG 的 SDK 自动切镜开关。切点前和末帧区间延展原帧。
- 评估 VFG 切镜时可用 `--no-vfg-sdk-shot-detection` 关闭 SDK 自动切镜，配合 `--cut-frame` 保留已核对的硬切。开关和切点保存在作业清单，续跑沿用该配置。 [VFG 对照方法](docs/VFG-错帧排查与测试备选.md)
- `vfg-cuts .\input.mp4` 只扫描并输出候选切点及判定理由。`run` 的 `--vfg-auto-cuts` 先扫描原片、合并自动切点和已有 `--cut-frame`，再关闭 SDK 自动切镜运行 VFG。候选、来源、有效切点及算法版本记录在 `.job/job.json` 的 `cut_detection` 中，续跑复用列表。 [自动切点实验](docs/VFG-错帧排查与测试备选.md)
- `run --vfg-review` 可记录 VFG 疑似复制、连续停帧和回跳。新版报告用 `anchor_echo` 标注“前一画面 → 变化 → 回到前一画面”的复制帧。导出完成后在 `.job/vfg-review.json` 查看，暂停时可查看已完成分段的回执。参数适用于纯 VFG 和串联导出，可与切点选项一起使用。 [复检方法与实测](docs/VFG-错帧排查与测试备选.md)
- `run --vfg-retry` 启用试验性倒序重算，并生成复检报告。只对接近复制原帧的槽位尝试一次。报告的 `retries` 记录是否采用和依据。`--vfg-auto-cuts` 可同时使用。 [实测与局限](docs/VFG-错帧排查与测试备选.md)
- `--fps` 指定目标恒定帧率，接受 1–240 范围内的小数或有理数。GIMM-VFI 和 RIFE 每轮补中间帧，必要时多轮处理，再按目标时间网格取样。先插帧后超分时，超分读取阶段按目标帧率取样，减少需超分的帧数。目标帧率不高于原片时直接取样。省略 `--fps` 沿用插帧 2× 的命令行行为。
- 完整作业复制容器兼容的音轨和字幕。`--start-frame` 与 `--frames` 可输出分段短样。`.mkv` 输出 FFV1，`.mp4` 输出 H.264。含 MP4 不支持的轨道时选 MKV。`render` 为单引擎短样入口，输出 FFV1 MKV。
- 保留 `.job` 目录才能续跑。可用 `--job-dir` 指定位置。作业记录源文件、权重和分块的 SHA-256。`run`、`resume`、`pause`、`status` 的标准输出是逐行 JSON，模型和 FFmpeg 诊断走标准错误。内存交接路径只保留编码分段，其余串联可直接读取上游已校验的无损分块。旧作业仍按原清单续跑。
- `--upscale nvidia-vsr` 选择 VSR Ultra，目标尺寸须大于原片。`--interpolate nvidia-vfg` 选择 VFG High，可直接生成指定帧率所需的帧。实际运行 NVIDIA VFX 的 MP4 作业使用 H.264 NVENC。
- `--upscale seedvr2` 选择 3B INT8。`--upscale seedvr2-7b` 选择 7B FP16，目标需与源宽高比相同。7B 可先用 `--chunk 5 --overlap 0` 试跑。`render --engine` 同样提供这两个选项。

新建 MP4 作业在以下配置下自动走快速导出。

- 纯插帧选 NVIDIA VFG High，目标帧率高于源帧率。
- 纯超分选 NVIDIA VSR Ultra、BasicVSR++ 或 SeedVR2 3B／7B，保持源帧率。桌面端关闭插帧即可。
- 纯插帧选 GIMM-VFI 或 RIFE，目标帧率高于源帧率。多轮插帧在内存中交接 RGB48，最后一轮按目标帧率直接编码。
- 串联选 NVIDIA VFG High、NVIDIA VSR Ultra、先插帧后超分，目标帧率高于源帧率。VFG 帧在显存中交给 VSR。
- 串联选 RIFE、NVIDIA VSR Ultra、先插帧后超分，目标帧率高于源帧率。RIFE 帧通过内存交给 VSR，需要取样时先落到目标帧率再超分。

最终帧经 FFmpeg 送入 H.264 NVENC。最后一轮保存可续跑的已编码 MP4 分段，成片通过流复制拼接并封装音轨。暂停在当前分段提交后生效。编码使用 P5、CQ 18，每段从关键帧开始，关闭 B 帧以按精确的 CFR 时间轴拼接。GIMM-VFI、RIFE、BasicVSR++、SeedVR2 和 RIFE → VSR 会在规划时用 FFmpeg 检查 NVENC 输出规格，检查失败时沿用原导出流程。其他引擎组合、纯超分同时改帧率、逆序、MKV 和既有作业清单沿用各自的处理路径。

参数见 `python video_lab.py run --help` 和 `python video_lab.py render --help`。

## 测试与选型

`prepare <片源目录> <测试目录>` 用于本项目的五段测试素材，生成 `samples.json` 和三个无损对照输入。`score <候选> <参考>` 比较同尺寸、同帧率的视频。用 `--reference-start-frame` 对齐起点，2× 对照中，`--inserted-only` 只取两端有锚点的奇数位插入帧。3× 和 4× 按源帧及子帧位置另行对齐。`lanczos` 和 `hold` 可作为基线。

模型保留范围、真值素材和超分对照标准见 [选型与开发计划](docs/离线视频超分与插帧-选型与开发计划.md)。

截至 2026-11-24，RTX 5070 已通过 CUDA 自检。白鹤整段经 GIMM-VFI 输出 1080p48、482 帧，436 个 AAC 包的内容哈希一致，最大 PTS 差低于 0.5 ms。虞姬已跑通双顺序 GPU 串联、4K 硬切插帧和两个 SeedVR2 版本。强制退出后续跑与 10 分钟低分辨率合成片的暂停、分数帧率也已通过。样片在本机测试目录的 `kernel-validation` 和 `seedvr2-7b-test` 下。

NVIDIA 单项与串联的整片用时、分段写入量、桌面批量恢复以及 FFV1 格式对照见 [快速导出实测](docs/VSR与VFG-快速导出开发计划.md)。
