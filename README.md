# 视频超分与插帧

面向 Windows 和 RTX 5070 的本地视频评估工具。BasicVSR++、GIMM-VFI-R-P 是核心引擎。SeedVR2 3B INT8、RIFE 4.25 是对照和备用引擎。当前仓库用于命令行和样片验证。

## 测试素材

建议从仓库根目录在 PowerShell 运行命令。需要 Python 3.13、FFmpeg、ffprobe 和 Git。

```powershell
python video_lab.py prepare 'H:\video test' 'H:\video test\超分插帧测试'
```

`samples.json` 记录片源路径、逐帧探测数据和已人工核对的切点。 `inputs` 内含虞姬和陈汤的 540p 无损派生片，以及从白鹤抽帧得到的 12 fps 派生片。原始视频留在原目录。重复运行时检查已有派生片的尺寸和帧数。

## CUDA 环境

```powershell
python setup_runtime.py deps
python setup_runtime.py weights
.\run_video_lab.ps1 doctor
```

依赖、模型和第三方代码均在仓库内的 `.runtime`。若已经下载了对应的官方 CUDA Torch wheel，可以用 `python setup_runtime.py deps --torch-wheel <wheel-path>`。 `weights gimm-vfi` 或 `weights seedvr2` 可单独准备一个引擎的权重。BasicVSR++ 和 RIFE 在首次调用时下载相应权重。运行 GPU 命令时用 `run_video_lab.ps1`，它在当前用户的 LocalAppData 下创建指向现有虚拟环境的英文目录别名，供 Windows NVRTC 编译 CuPy 内核。仓库和视频继续使用原路径。

## 单引擎试跑

```powershell
$testDir = 'H:\video test\超分插帧测试'
$inputFile = Join-Path $testDir 'inputs\yuji-540p.mkv'
.\run_video_lab.ps1 render $inputFile (Join-Path $testDir 'yuji-basicvsrpp.mkv') --engine basicvsrpp --frames 8 --tile 192
$inputFile = Join-Path 'H:\video test' '湖面如镜，白鹤伴飞2-许负1级.mp4'
.\run_video_lab.ps1 render $inputFile (Join-Path $testDir 'cranes-gimm.mkv') --engine gimm-vfi --frames 6 --motion-scale 0.5
```

`--cut-frame` 是从零开始的源帧索引，可重复传入。插帧在每镜头内做两倍帧率，到切点前和末帧区间延展原帧。超分只在镜头内用重叠上下文。 `--start-frame` 和 `--frames` 用于输出无声短样片。完整视频从零时间戳开始时复制源音轨和字幕。输出使用 FFV1 MKV，提前校验帧数、帧率和宽高。

`lanczos` 和 `hold` 是两个基线，可在有 NumPy 的 Python 环境里独立使用。 `seedvr2` 与 `lanczos` 均需指定 `--size`。同尺寸参考可通过 `score` 计算 PSNR 和 SSIM。插帧回测用 `--inserted-only`，末尾缺少右侧锚点的帧自动排除。

```powershell
$candidate = Join-Path $testDir 'cranes-hold-preview.mkv'
$reference = Join-Path 'H:\video test' '湖面如镜，白鹤伴飞2-许负1级.mp4'
python video_lab.py score $candidate $reference --reference-start-frame 60 --inserted-only
```

可用选项见 `python video_lab.py render --help`。开发范围与画质评估标准见 [选型与开发计划](docs/离线视频超分与插帧-选型与开发计划.md)。

## 当前验证结果

- 用户的五段视频均已探测并生成三个无损对照输入。
- 白鹤整段的原帧延展基线已输出 1920×1080、48 fps、482 帧。输出 AAC 与原 AAC 流的 SHA-256 一致。
- 白鹤 12→24 的原帧延展基线按 35 个缺失帧评分为 27.85 dB，SSIM 0.9349。评分使用当前 CLI 的统一 RGB/YUV 转换。
- 通过合成硬切测试验证插帧任务分块不会把切点两侧的帧交给同一次模型推理。该测试使用替代推理函数检查作业控制层。
- 本机独立环境已通过依赖检查，PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128 和 CuPy 14.2.0 在 RTX 5070 上通过 CUDA 张量、可变形卷积与 NVRTC 自检。
- BasicVSR++ 已完成两帧 540p→4K 自检。GIMM-VFI 与 RIFE 已用两帧源画面验证 1080p 24→48。SeedVR2 3B INT8 已完成两帧 540p→720p 自检。产物在测试目录的 `environment-smoke` 内，画质、显存上限和长片处理待评估。

## SeedVR2 画质对照

保留 3B INT8 和 7B FP16。3B INT8 用于常规 SeedVR2 处理，7B FP16 用于逐镜头对照。虞姬同一组 5 帧在 720p 下，7B 的 PSNR 和 SSIM 高于 3B，但人工看片发现 7B 补出了较多不符合原片的细节，这个镜头选 3B。对照时先看纹理、轮廓和材质是否忠于原片，再结合客观指标。

7B FP16 通过独立的 NumZ 后端完成 720p 和 1080p 的各 5 帧试跑，使用 36 层 CPU BlockSwap 和 512 像素 VAE 分块。7B 权重位于 `.runtime/seedvr2-7b-weights`，样片位于测试目录的 `seedvr2-7b-test`。长片稳定性和跨段一致性待评估。

## 可恢复作业

`run` 支持只超分、只插帧或串联。默认先插帧，再超分。可用 `--order upscale-first` 反转。插帧为原 fps 的两倍。`--cut-frame` 使用零基源帧号，可重复指定。`--scene-threshold` 可添加自动候选切点。短样用 `--start-frame` 和 `--frames`，完整作业复制原音轨与字幕。需要保留 MP4 不支持的轨道时选 MKV。

```powershell
$test = 'H:\video test\超分插帧测试'
$inputFile = Join-Path $test 'inputs\yuji-540p.mkv'
$output = Join-Path $test 'yuji-720p48.mkv'
.\run_video_lab.ps1 run $inputFile $output --interpolate gimm-vfi --upscale basicvsrpp --size 1280x720 --motion-scale 0.5
.\run_video_lab.ps1 status "$output.job"
# 从另一终端请求暂停，当前分块完成后停下
.\run_video_lab.ps1 pause "$output.job"
.\run_video_lab.ps1 resume "$output.job"
```

`.mkv` 输出 FFV1，`.mp4` 输出 H.264。`--job-dir` 可指定分块目录，默认用输出文件名加 `.job`。作业目录记录输入和权重的 SHA-256、完整处理参数、切点和分块校验值。续跑需要保留该目录。每个推理阶段使用单独进程。`run`、`resume`、`pause` 和 `status` 的标准输出是逐行 JSON，模型和 FFmpeg 的诊断输出写到标准错误。

`--upscale seedvr2` 选择 3B INT8，`--upscale seedvr2-7b` 选择 7B FP16。需要在新机器上安装 7B 时，先准备基础环境，再执行 `python setup_runtime.py seedvr2-7b`。7B 当前适合用同宽高比的目标尺寸，首轮可用 `--size 1280x720 --chunk 5 --overlap 0`。两套模型也可在 `render --engine` 中选择。

### 本机作业验证

- 白鹤整段经 GIMM-VFI 输出 1920×1080、48 fps、482 帧。436 个 AAC 包的内容哈希与原片一致，封装后的包 PTS 差值小于 0.5 ms。
- 虞姬已验证 BasicVSR++ 与 GIMM-VFI 的两种串联顺序，以及原生 4K 切镜插帧。SeedVR2 3B、SeedVR2 7B 和 RIFE 4.25 已通过同一 `run` 入口输出短样片。
- 强制退出后成功续跑两阶段作业并导出 MP4。10 分钟合成低分辨率素材已通过主动暂停、续跑和分数帧率检查。延迟音轨与字幕使用独立合成素材核对。

样片和作业目录在测试目录的 `kernel-validation` 下。实际长片 GPU 压力、接缝画质及不同素材的模型优选仍待验证。
