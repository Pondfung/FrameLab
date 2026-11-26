# VSR 与 VFG：NVIDIA 视频增强组件的获取与接入

> LiveCard 历史参考。本文的安装、接口和数据对应实时采集项目。FrameLab 当前的引擎和用法见 [选型计划](离线视频超分与插帧-选型与开发计划.md) 与 [README](../README.md)。

> 2026-11-24。来源是把这两个组件真正接进 LiveCard（Rust + wgpu/D3D12 的采集卡查看器）
> 并跑满 60 fps 的过程。所有耗时都是 RTX 5060 Ti + 驱动 610.88 实测，不是官方标称值。
> 目标读者：要在自己的视频应用里加超分 / 插帧的人，尤其是**生成式视频**这种
> 「低分辨率、低帧率、帧间不连续」的素材。

## 零、选型速查

| | **VSR**（超分） | **VFG**（插帧） |
| --- | --- | --- |
| 解决什么 | 1080p 源上 1440p/4K 屏 | 24/30 fps 源补到 60/120 |
| 硬件门槛 | RTX 20 系及以上（Turing） | **Ada / Blackwell，即 RTX 40 系起** |
| 获取方式 | developer.nvidia.com 注册下载 | `pip install nvidia-vfx`（公开，无需账号） |
| 交付形态 | SDK：头文件 + 示例 + `nvngx_vsr.dll` | 只有运行时 dll，**没有头文件** |
| 数据面 | GPU 纹理进、GPU 纹理出，零 CPU 往返 | 我们这条线是 CPU NV12 往返，也可走 D3D11 零拷贝 |
| 1080p 单帧耗时 | 3.2 ms（档 2）/ 9.0 ms（档 4） | 5.4 ms（Medium）/ 23.5 ms（High） |
| 分发 | 可随应用打包，不可单独分发 | 模型可再分发（附协议副本） |

一句话：**VSR 是正经 SDK，VFG 是从 pip 包里扒出来的运行时**——后者要自己声明函数签名，
遇到文档查不到的东西只能靠反推 dll 字符串。

---

## 一、VSR（RTX Video Super Resolution）

### 1.1 获取

```
https://developer.nvidia.com/rtx-video-sdk
```

注册 NVIDIA Developer Program（免费，不需要企业账号、不需要 NGC、不需要 AI Enterprise），
下载 **RTX Video SDK 1.1.0**。解压后需要的东西很少：

| 路径（相对 SDK 根） | 用途 |
| --- | --- |
| `include/` 或 `api/` 下的 `rtx_video_api.h` 等 | 接口声明 |
| `bin/Windows/x64/rel/nvngx_vsr.dll` | 实际干活的 dll（19 MB） |
| `samples/` | D3D11 / D3D12 / Vulkan 三套示例，**照着改比自己摸快得多** |

SDK 自带 `rtx_video_api_dx12_impl.cpp` 这类实现文件，也可以直接编进自己的工程。

### 1.2 许可

`nvngx_vsr.dll` 走 **NVIDIA RTX SDKs LICENSE**：

- 可以随你自己的应用一起分发（作为应用的一部分）；
- **不可以**单独分发、销售、再许可这个 dll 本身；
- 不可以逆向。

这和纯自用没区别，商用也宽松（有开源项目 HiResLoupe 就是这么发的：MIT 代码 +
单独 NOTICE 说明 dll 的许可）。对比之下，Maxine/VFX SDK 那条线连付费应用都不许打包。

### 1.3 硬件与驱动

- **RTX 20 系及以上**（Turing 起，需要 Tensor Core）。GTX 16 系没有 Tensor Core，不支持。
- 驱动 522.25 以上（本机 610.88）。
- 初始化后可以查 `VSR.Available`，返回 1 才真正可用——**别假设有卡就能跑**，老驱动会返回 0。

### 1.4 代码注意事项

以下每条都是踩出来的，不是抄文档。

**① 传设备指针要传接口本身，不是 `&interface`**

在 Rust 里用 `windows` crate 时，`ID3D12Device` 是 `#[repr(transparent)] struct(ID3D12Device(*mut c_void))`，
所以 `&ID3D12Device` 的地址是**指向接口指针的指针**。直接 `as *mut c_void` 传过去会段错误：

```rust
let raw: &ID3D12Device = hal_dev.raw_device();
let dev_raw = unsafe { raw.as_raw() };   // 正确：拿到接口指针本尊
```

**② D3D12 会出现「输出全黑」，根因是 fence 值被 SDK 内部推进**

表现为：调用成功、没有报错，但输出纹理全黑。排查时先排除这几项（都已验证不是原因）：
fence 初值、资源状态、格式、档位、驱动版本。

真正的原因是 SDK 内部会推进你自己传进去的 fence 值，导致你以为的「等到 N」其实早就过了，
等于假同步。修复办法是**用自己的 fence 对象 + 每次重新取当前值**，不要复用同一个计数器。
完整排查过程见 `docs/RTX-VSR-实测报告.md` 第三节。

**③ 输出纹理不要带 UAV 标志**

VSR 的输出纹理加 `ALLOW_UNORDERED_ACCESS` 会失败或黑屏。但**输入**纹理（作为渲染目标的那个）
必须带 `ALLOW_RENDER_TARGET`，漏了整个设备实例会挂（`0x887A0005`，不是普通报错）。

**④ 导入外部纹理时初始状态用 `UNKNOWN`**

wgpu 的 `create_texture_from_hal` 第三个参数表示「这张纹理现在处于什么状态」。按惯例写
`COPY_DST | COPY_SRC` 会出错——VSR 处理完之后实际状态已经不是它以为的了，下一次
`copy_texture_to_buffer` 会插入错误的 barrier，直接 `BufferAsyncError`。
填 `TextureUses::UNKNOWN` 让 wgpu 在首次使用前插无条件 barrier，问题消失。

**⑤ 必须延后一帧**

VSR 读的是「上一帧渲染完成」的纹理。所以每帧开头要先 `poll(Wait)` 等上一帧提交完，
再调 `vsr_process`，再渲染当前帧。代价是画面比声音晚一帧（30 fps 下 33 ms）。
想省掉这一帧延迟就得双缓冲 + 自己管同步，不值得。

**⑥ 会话和共享纹理按尺寸/档位缓存，变了就重建**

`prepare_sr()` 那种模式：尺寸或档位变化时销毁重建，失败只记错误、不 panic。
程序能跑但超分开关无效，比崩溃好。

**⑦ dll 路径别用相对路径**

从计划任务、快捷方式、服务启动时 cwd 不是仓库根。相对路径会解析失败，报一个含义不明的
`err=-2`。做法是以某个**只存在于 bin 目录**的 dll 为基准做探测，或直接存绝对路径。

### 1.5 性能

1080p 输入，含提交与同步（fence 实测）：

| 输出 | 档 1 | 档 2 | 档 3 | 档 4（Ultra） |
| --- | --- | --- | --- | --- |
| 2560×1440 | 2.59 ms | 3.17 ms | 6.75 ms | 9.03 ms |
| 3840×2160 | 2.57 ms | — | — | 8.97 ms |

对比 60 fps 的 16.7 ms 预算，档 2 只用掉五分之一。官方给的 Ultra 2.01 ms 是 4090 的数字，
入门卡上别按那个估。

---

## 二、VFG（Video Frame Generation）

### 2.1 获取 —— 这条渠道是重点

官方路子在 NGC：`ngc registry resource download-version nvidia/maxine/vfx_sdk_core`，
**个人账号返回 403**，需要企业授权。所以很多人以为这个组件拿不到。

但 NVIDIA 把运行时和模型直接发在了 pip 上：

```bash
pip install nvidia-vfx --index-url https://pypi.nvidia.com
```

- 包 `nvidia-vfx`（import 名 `nvvfx`），Windows wheel **435.8 MB**
- **公开下载，不需要 API key、不需要组织权限**
- 装完在 `site-packages/nvvfx/libs/`

| 文件 | 大小 | 内容 |
| --- | --- | --- |
| `nvVFXVideoFrameGeneration.dll` | 197.7 MB | **VFG，Low/Medium/High 三档模型全部内嵌** |
| `nvngx_vsr.dll` | 56.4 MB | 超分（和 RTX Video SDK 里那个是同一个东西） |
| `nvngx_truehdr.dll` | 3.5 MB | SDR→HDR |
| `NVVideoEffects.dll` | 0.1 MB | 统一 C API 入口（`NvVFX_CreateEffect` 等） |
| `NVCVImage.dll` | 2.6 MB | `NvCVImage` 图像容器 |
| `npp*.dll`、`cudart64_12.dll` | ~270 MB | NPP 与 CUDA 运行时 |

**模型不是独立文件，全部编进 dll 里**——这一点和 VFX SDK 的「core + install_feature 拉模型」
结构完全不同，不需要跑安装脚本。

> 所以「VFX SDK 个人拿不到」要修正为：**头文件和示例拿不到，运行时不缺**。
> C++ 侧 `LoadLibrary` + `GetProcAddress` 自己声明签名即可。

### 2.2 许可

wheel 里带两份 PDF：`NVIDIA-Software-License-Agreement` 与 `NVIDIA-Open-Model-License-Agreements`。

- **core 包本体**（dll/lib/头文件）走 SDK License：不可再分发，只能作为自己应用的一部分
- **模型**走 Community Model License 1.2(c)：允许 `distribute NVIDIA Models and Derivative
  Models with a copy of this Agreement`，例外条款覆盖「RTX 本机非商业多用户使用」

实践结论：**声明依赖让用户自己 `pip install`，不要把 dll 塞进你的发布包**。

### 2.3 硬件

**Ada / Blackwell（RTX 40 系、50 系）**。Turing / Ampere 不在支持列表——它跑的是 TensorRT
engine，dll 里能看到 `model_medium_fp8` 这类 fp8 变体，fp8 需要 Ada 以上。

### 2.4 C API 调用序列

没有头文件，以下全部是从 dll 导出表和报错字符串反推出来的。

**入口**：`NVVideoEffects.dll`

```c
NvVFX_CreateEffect("VideoFrameGeneration", &handle);
NvVFX_SetU32(handle, "InputWidth",  w);
NvVFX_SetU32(handle, "InputHeight", h);
NvVFX_SetU32(handle, "Mode", 1);              // 0=Low 1=Medium 2=High
NvVFX_SetU32(handle, "FrameMultiplier", 2);   // 2..8，0 表示用 Timestep
NvVFX_SetU32(handle, "FrameIndex", 1);        // 1..M-1
NvVFX_SetImage(handle, "SrcImage0", prev);    // 上一帧
NvVFX_SetImage(handle, "SrcImage1", cur);     // 当前帧
NvVFX_SetImage(handle, "DstImage0", mid);     // 输出
NvVFX_Load(handle);
NvVFX_Run(handle, 0);
```

**selector 名是坑**：文档/示例里写的 `NVVFX_INPUT_IMAGE_0` 是宏名，直接传给
`NvVFX_SetImage` 一律返回 `-5`（selector 无效）。真正认的字符串是
**`SrcImage0` / `SrcImage1` / `DstImage0`**。这是从 dll 里的报错格式串
`%s image must be (RGBA|BGRA)+U8 or RGB10`、`SrcImage0 (NvCVImage*) - previous frame`
反推出来的。

其余 selector：`Timestep`（0–1 开区间）、`ShotChange`、
`AutomaticShotChangeDetectionEnabled`（默认开，检出切镜就直通复制）。

**像素格式枚举**（用 `NvCVImage_FromD3DFormat` 反查）：

| 格式 | pixelFormat | componentType |
| --- | --- | --- |
| NV12 | 10 | 1 |
| RGBA U8 | 6 | 1 |
| BGRA U8 | 7 | 1 |
| R16G16B16A16_FLOAT | 6 | 4 |

VFG **只吃 `(RGBA|BGRA)+U8` 或 `RGB10A2`，且必须在显存里**，不接受 NV12。

### 2.5 代码注意事项

**① `NvCVImage_Alloc` 是 8 个参数**

```c
NvCVImage_Alloc(im, width, height, pixelFormat, componentType, lineAlign, hostBuffer, flags);
```

少传 `hostBuffer` 会让后面整体错位。而且**第 7 个参数必须传 1**（要 CPU 可访问的紧凑缓冲），
传 0 会分到 device 侧，pixels 直接是空的，等到 `SetImage` 才报 `-9`。

`lineAlign` **必须 ≥ 4**。传 1 能分配成功但 pitch 是错的，同样到 `SetImage` 才炸。

**② 不要自己算 pixels 偏移**

`NvCVImage` 结构里的 `gpuMem`（+32）、`pixels`（+48）两个指针位置随分配方式变，
按固定偏移 memcpy 会崩。正确做法是全走库函数：

```
CPU 缓冲 --NvCVImage_Init--> NvCVImage --Transfer--> GPU NV12
         --Transfer--> RGBA --Run--> RGBA --Transfer--> GPU NV12 --Transfer--> CPU
```

**③ 色度不能信模型输出**

用纯色图实测：模型输出 Y 是对的，但 **UV 只有输入的 52% 饱和**（红→蓝插值，
输入饱和度 138/123，输出 68.5）。纯色块不存在降采样损失，这是模型自己的行为。
后果是插值帧发灰，和原始帧交替显示就是「饱和度呼吸」。

处置：**亮度用模型结果，UV 直接抄当前帧**。色度是 4:2:0、分辨率只有亮度四分之一，
超前半帧肉眼看不出来。

**④ full / limited range 必须自己映射**

`NvCVImage` 把 NV12 **一律按 limited range (16..235)** 处理，而 MJPG/JPEG 解码出来的是
full range (0..255)。不映射就会被静默截断：

| 输入 Y | 不映射的输出 |
| --- | --- |
| 0 | 16 |
| 8 | 16 |
| 235 | 235 |
| 255 | 235 |

暗部一片死黑、高光削平，在深色场景和高光物体上表现为闪烁。进出各查一次表即可，
代价约 0.5 ms。

**⑤ 推帧顺序决定观感**

插值帧要排在「上一帧」和「当前帧」之间。两种写法：

- 先放上一帧、再放插值帧（延迟一帧，但要注意别重复 push）
- **先放插值帧、再放当前帧**（延迟最低，推荐）

写错的表现是每帧显示间隔变成 1-2-1-2，看着像规律的抽动。

**⑥ 会话按尺寸重建**

输入尺寸变了必须销毁重建（VFG 和 VSR 都是）。失败要降级直通、定时重试，
**不能让画面停在 0 fps**。

### 2.6 性能

C API 实测（含 GPU↔CPU 往返与 stream sync）：

| 分辨率 | Low | Medium | High |
| --- | --- | --- | --- |
| 1280×720 | 1.69 ms | 4.50 ms | 22.00 ms |
| 1920×1080 | 2.69 ms | **5.39 ms** | 23.49 ms |
| 2560×1440 | 4.10 ms | 6.81 ms | 25.02 ms |

`Load`（首次初始化）29–312 ms，不要放在渲染循环里。

倍率 2 时每输入帧产出一帧中间帧，等效吞吐 = 1/(耗时)。1080p Medium 5.4 ms 对应
约 185 fps 的上限，30→60 完全无压力；**High 档 23.5 ms 上不了 60 fps**，只能离线用。

---

## 三、两个一起用

### 3.1 串接顺序

```
采集/解码 → [VFG 插帧] → [VSR 超分] → 显示
```

**先插帧后超分**，不要反过来：

- 超分在前会让 VFG 处理更大的画面，耗时线性上升；
- VFG 在低分辨率上做运动估计更稳（大位移的相对像素少，容易估）；
- 超分对「已经是合成帧」做锐化，能把插值伪影一起放大（见 3.2）。

### 3.2 互相放大的伪影

实测过一组数据（同一场景，量画面局部高频能量）：

| 配置 | 地板区域 | 顶部暗部 |
| --- | --- | --- |
| 超分关 + 插帧开 | 2.29 | 0.71 |
| 超分档 4 + 插帧关 | 3.61 | 1.65 |
| 超分档 4 + 插帧开 | 3.51 | 1.61 |

**插帧开或关，高频几乎不变**；打开超分，高频翻倍。也就是说超分会把源本身的压缩噪声
（8×8 DCT 块效应）锐化成肉眼可见的点阵。这个锅不在插帧，但两个一起用会显得更明显。

对策：源本身有压缩噪声时，超分档位往下调；或者接受「超分只对干净源有效」这个事实。

---

## 四、给生成式视频的额外提醒

生成模型的输出和采集卡画面有两点本质区别，直接套用会踩坑。

**① 帧率通常很低，VFG 的倍率用得上**

AI 视频生成常见 8 fps / 16 fps / 24 fps 输出。VFG 的 `FrameMultiplier` 支持 2–8，
理论上 8 fps 可以补到 64 fps。但每多一帧都要跑一次 `Run`，耗时线性增长：
8→64 是 8 倍，1080p Medium 就是 43 ms/输入帧，得算清楚预算。

**② 帧间连续性可能很差，VFG 会放大它**

`AutomaticShotChangeDetectionEnabled` 默认开，检出切镜会直通复制当前帧——这对真视频是好事，
对生成视频可能**误判**（生成片段常有局部形变、无中生有），导致该插的地方不插。

更重要的是：VFG 假设前后两帧是同一个场景的连续运动。生成视频如果帧间闪烁、物件跳变，
插值结果会把这种不稳定放大成「抖动的重影」，比直接显示更难看。

**建议**：先不开插帧跑一遍，确认生成结果本身时域稳定；稳定之后再上 VFG，并且把
`Mode` 从 Low 试起（Low 的模型简单、对异常帧更宽容）。

**③ 色彩空间要自己确认**

生成模型常输出 linear / float（`R16G16B16A16_FLOAT` 是支持的，见 2.4 表），
但 VFG 的输入输出要求一致。从 float 转到 `RGBA+U8` 时注意 gamma：
在 linear 空间插值再转 sRGB，和先转 sRGB 再插值，观感不同（前者物理正确，后者更符合直觉）。
这一条我们没实测过，标为**未验证**。

---

## 五、参考实现索引

LiveCard 里的位置（Rust + wgpu）：

| 文件 | 内容 |
| --- | --- |
| `native/vsr/vsr_ffi.cpp` | VSR 的 C ABI 封装（`vsr_create` / `vsr_process` / `vsr_destroy`） |
| `native/vfg/vfg_ffi.cpp` | VFG 的 C ABI 封装，含 `RangeLut` 与色度旁路 |
| `livecard/src/vsr.rs` | Rust 侧加载与调用 |
| `livecard/src/vfg.rs` | 同上，外加 `nvvfx/libs` 定位逻辑 |
| `livecard/src/interp.rs` | 插帧工作线程：引擎选择、倍率计算、会话重建、失败直通 |
| `livecard/src/render.rs` | 超分三段式管线（上传 → 离屏渲染 → VSR → 采样上屏） |
| `livecard/src/bin/vsrprobe.rs` | 超分端到端探针（读回采样校验） |
| `docs/RTX-VSR-实测报告.md` | VSR 完整排查记录，D3D12 黑屏那段值得看 |
| `docs/插帧-VFG-方案.md` | VFG 探通过程与文件清单 |

两个原生模块都编成独立 dll，主程序通过 `LoadLibraryExW` + `LOAD_WITH_ALTERED_SEARCH_PATH`
加载（中文路径下要先 `canonicalize`，否则找不到同目录依赖，报 `0x8007007E`）。

## 六、没有验证的部分

诚实标注，别当成已知：

- **VFG 在 30 系上的可用性**：官方只列 Ada/Blackwell，本机没有 30 系可试。
- **VFG 长跑稳定性**：目前只跑过数分钟量级，没有几小时压测。FRUC 有过长跑崩溃历史。
- **VFG 自定义 CUDA stream**：`NvVFX_SetCudaStream` 的正确用法没摸出来，现在用默认 stream。
- **VFG 在生成式素材上的实际效果**：第四节②的推断基于插值原理，没有真实素材验证。
- **VSR 与 VFG 同时开时的总延迟**：分别测过，串起来的端到端延迟没单独量。
