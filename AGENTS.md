# FrameLab 项目约定

FrameLab 的常规导出使用 NVIDIA VFG High 和 VSR Ultra。保留其他模型的本地权重和评估入口，评价画质时检查原片轮廓、纹理和运动连续性。模型分工见 [选型与开发计划](docs/离线视频超分与插帧-选型与开发计划.md)。

- 从仓库根目录在 PowerShell 运行 `.\run_video_lab.ps1 doctor` 检查 GPU。环境准备和作业命令见 [README](README.md)。GPU 命令使用启动脚本，以兼容中文路径下的 NVRTC。
- 完整版 NVIDIA 张量使用 PyTorch，单 EXE 使用 CuPy。两版共用 `engines.py`、`jobs.py`、`video_lab.py` 和 `desktop/`，用 `python scripts/build_nvidia.py` 构建单 EXE。修改帧序、切镜或分块时，同一短片要走两个版本的实际导出入口。编码前优先在显存或内存交接，保留可续跑的成片分段。
- `video_lab.py` 是 CLI，`jobs.py` 管作业、分块、续跑和导出，`engines.py` 是模型适配层，`setup_runtime.py` 安装依赖和权重。`.runtime` 本地保存环境和权重，测试视频与作业缓存留在仓库外。
- 完整版桌面前端通过启动脚本调用 `run`、`pause` 和 `resume`，通过逐行 JSON 更新作业状态。从源帧号传入切点，保留音画时间关系和可恢复分块。
- VFG 正式导出默认用 SDK 自动切镜。CLI 可用 `--no-vfg-sdk-shot-detection` 做对照，用 `--vfg-review` 在放大或编码前记录疑似异常。复检结果随分段回执提交，续跑沿用。`--vfg-retry` 显式启用倒序重算并复核候选。复核规则以 [VFG 错帧优化计划](docs/VFG-错帧排查与测试备选.md) 为准。测试构建、样本和接手顺序见 [开发交接](docs/FrameLab-开发交接.md)。
