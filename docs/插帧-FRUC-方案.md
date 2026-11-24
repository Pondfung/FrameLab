# 插帧 · NVOFA FRUC 方案（速度优先引擎）

更新：2026-11-23　|　代码：`native/fruc/fruc_ffi.cpp` + `livecard/src/fruc.rs` + `livecard/src/interp.rs`

RIFE 之外的第二套插帧引擎。RIFE 走 ncnn + Vulkan、画质好但慢（1080p 24.8 ms）；FRUC 走
NVIDIA 的光流硬件（OFA），1080p 只要 6.4 ms，代价是运动估计精度不如 RIFE，快速运动的大位移
场景容易出现边缘伪影。

---

## 一、为什么是它

调研过一圈，能落地的只剩这一条：

| 方案 | 淘汰原因 |
| --- | --- |
| DLSS Frame Generation / FSR3 FG / XeSS FG | 需要引擎提供运动矢量与深度缓冲，采集卡画面没有这些 |
| NVIDIA Smooth Motion | 驱动级，只对 DX/Vulkan 全屏游戏生效，抓不到采集卡窗口 |
| Lossless Scaling | 抓窗口与 VSR 抢同一块 GPU 资源，延迟 8–22 ms 不可控 |
| SVP 4 | CPU 光流，延迟 40–80 ms，音画同步直接崩 |
| IFRNet / EMA-VFI / VFIMamba | 纯模型推理，1080p 都超过 30 ms，没有实用价值 |
| **NVOFA FRUC** | **硬件光流，NV12 原生输入，OFA 独立于 CUDA/Tensor Core，1080p 6.4 ms** |

关键点：OFA（Optical Flow Accelerator）是独立硬件单元，跑 FRUC 时不会和 VSR 抢 SM/Tensor Core，
两者可以同时开。

## 二、怎么装

SDK：`https://developer.nvidia.com/optical-flow-sdk`（要 NVIDIA 账号登录，约 239 MB）。
解压后放在本仓库的同级目录，`native/build.bat` 会自动找 `Optical_Flow_SDK_*`。

**不需要 CUDA Toolkit**。`NvOFFRUC.dll` 是预编译的，只额外依赖 `cudart64_110.dll`（SDK 里自带）。
这两个 dll 属 NVIDIA 二进制，不进仓库，首次使用自己拷到 `native\bin\`。

## 三、架构

SDK 只支持 D3D11 / CUDA 的资源注册，不支持 D3D12，所以走的是 D3D11 中转：

```
NV12 字节 → staging 上传 → input0 / input1 (D3D11 NV12)
                                    ↓
                          NvOFFRUCProcess (OFA 硬件)
                                    ↓
                            output (D3D11 NV12)
                                    ↓
                          staging 读回 → NV12 字节
```

纹理创建时带 `D3D11_RESOURCE_MISC_SHARED | SHARED_NTHANDLE`，为将来做零拷贝留口子。
目前还是 CPU 往返，1080p 一次往返约 1.5 ms，占总耗时 6.4 ms 的四分之一。

## 四、三个必踩的坑

1. **`NvOFFRUCCreate` 的 `pDevice` 要传 `ID3D11Device5`**，不是 `ID3D11Device`。
   传错不报错，后续 `RegisterResource` 才失败。
2. **`CreateFence` 必须带 `D3D11_FENCE_FLAG_SHARED`**。
   用 `NONE` 会返回 `OUT_OF_SYSTEM_MEMORY`（4），指向完全不相干的方向。
3. **注册资源数照 sample 走 2 进 1 出**。1080p 下注册 2 进 2 出会返回 `status=4`；
   720p 下 2 进 2 出没事，所以这个坑只在切到 1080p 时才炸，很容易误判成别的问题。

还有一条不是坑但容易写错：NV12 staging 读回时 UV 平面偏移 = `RowPitch * height`，
**不要用 `DepthPitch`**（两者在 1080p/720p 下数值相同，换尺寸就可能不同）。

## 五、1080p 的两个固有缺陷

`NvOFFRUC` 在高度不是 16 倍数的尺寸上（1920×1080 正好踩中：1080 = 16 × 67.5）行为不对：

| 症状 | 现象 | 处置 |
| --- | --- | --- |
| 色度平面不写 | UV 平面 99% 是 0 → 整屏荧光绿 | 插值帧的色度整个改用上一帧 |
| 色度平面乱写 | UV 在 0/255 之间跳 → 绿紫大块 | 同上 |
| 资源注册失败 | `RegisterResource status=4` | 输出纹理只留 1 个 |

亮度平面一直是正确的，所以处置方式统一是「色度用上一帧、亮度用 FRUC 的结果」。
1080 % 16 = 8，720 % 16 = 0，所以 720p 一直没暴露这些问题。

## 五之二、切换分辨率：会话必须整条重建（2026-11-24）

换档位或拔插重连时，采集尺寸会变。此时**旧会话必须先彻底销毁再建新的**，否则
`NvOFFRUCCreate` 返回 `NvOFFRUCRegisterResource status=4`，或 `fruc_process` 返回 `-99`。
原因是 `fruc_destroy` 之后 GPU 侧的纹理回收有延迟，两套会话短时并存会撞注册上限。

落地做法（`interp.rs`）：

1. `app.rs::start()` 直接丢弃插帧线程；`Interp::Drop` 关通道并等线程退出（最多 1.5 s），
   确保 `fruc_destroy` 执行完。
2. 线程内检测到尺寸变化时，先 `fruc = None` 再 `Fruc::new`，不并存。
3. 创建失败不放弃：`dims` 置空 + 300 ms 节流重试，期间**直通当前帧**。
   实测第二次尝试基本都成功。
4. 会话没建起来时绝不能 `continue` 丢帧——那会让画面停在 0 FPS，
   外部表现就是「必须手动关插帧再开」。

验证：1080P ↔ 720P 连续切换 4 轮，每轮插帧自动恢复（720p 119 fps / 1080p 61 fps），
无 `-99`、无 `status=4`、无 0 FPS。

## 六、性能实测

RTX 5060 Ti（Blackwell cc 12.0）、驱动 610.88：

| 分辨率 | 均值 | 中位 | P95 | 理论上限 |
| --- | --- | --- | --- | --- |
| 1280×720 | 5.6 ms | — | 6.2 ms | 178 fps |
| 1920×1080 | 6.4 ms | 6.4 ms | 6.9 ms | 149–156 fps |

对比 RIFE：720p 9.5–11.4 ms、1080p 24.8 ms（都含 NV12↔RGB 转换）。

实机组合：1080P 采集 + FRUC 插帧 = 61 FPS、6.4 ms/帧、GPU 40%；
720P 采集 + FRUC + VSR 档 2 = 120 FPS。

## 七、验收方法

```bat
cd livecard
cargo run --release --bin frucprobe -- 1920 1080 40
```

输出均值/中位/P95 耗时，同时把连续帧写进 `fruc_seq.nv12`（原始 NV12 交错序列），
可以转图看插值帧正确性。注意探针会独占采集卡，**跑之前先关掉主程序**。

稳定性：`fruc_process` 外面套了 `__try/__except`（SEH），异常返回 -99，上层收到后重建会话。
这是兜底不是修复——真遇到 -99 说明 NVIDIA 库内部出了问题，值得记录当时的尺寸和调用序列。
