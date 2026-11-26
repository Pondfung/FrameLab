use std::{
    collections::{HashSet, VecDeque},
    ffi::OsString,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Stdio,
    sync::mpsc::Sender,
    thread,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::media;

#[cfg(not(feature = "portable"))]
pub const UPSCALERS: [(&str, &str); 6] = [
    ("", "关闭"),
    ("basicvsrpp", "BasicVSR++"),
    ("seedvr2", "SeedVR2 3B INT8"),
    ("seedvr2-7b", "SeedVR2 7B FP16"),
    ("nvidia-vsr", "NVIDIA VSR Ultra"),
    ("lanczos", "Lanczos"),
];
#[cfg(not(feature = "portable"))]
pub const INTERPOLATORS: [(&str, &str); 5] = [
    ("", "关闭"),
    ("gimm-vfi", "GIMM-VFI"),
    ("rife", "RIFE 4.25"),
    ("nvidia-vfg", "NVIDIA VFG High"),
    ("hold", "原帧延展"),
];

#[cfg(feature = "portable")]
pub const UPSCALERS: [(&str, &str); 2] = [("", "关闭"), ("nvidia-vsr", "NVIDIA VSR Ultra")];
#[cfg(feature = "portable")]
pub const INTERPOLATORS: [(&str, &str); 2] = [("", "关闭"), ("nvidia-vfg", "NVIDIA VFG High")];

#[derive(Clone, Serialize, Deserialize)]
pub struct Settings {
    pub upscale: String,
    pub interpolate: String,
    pub height: u32,
    #[serde(default)]
    pub custom_size: bool,
    #[serde(default)]
    pub width: u32,
    #[serde(default = "default_fps")]
    pub fps: f64,
    #[serde(default)]
    pub custom_fps: bool,
    pub upscale_first: bool,
    pub mp4: bool,
    pub scale: String,
    pub directory: Option<PathBuf>,
}

fn default_fps() -> f64 {
    60.0
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            upscale: "".into(),
            interpolate: "nvidia-vfg".into(),
            height: 1080,
            custom_size: false,
            width: 0,
            fps: default_fps(),
            custom_fps: false,
            upscale_first: false,
            mp4: true,
            scale: "0.5".into(),
            directory: None,
        }
    }
}

impl Settings {
    pub fn size(&self, video: &media::Video) -> (u32, u32) {
        let ratio = f64::from(video.width) / f64::from(video.height);
        let height = self.height.max(2) & !1;
        if self.custom_size && self.width > 0 {
            return (self.width.max(2) & !1, height);
        }
        (
            ((f64::from(height) * ratio / 2.0).round() as u32).max(1) * 2,
            height,
        )
    }

    pub fn args(&self, video: &media::Video) -> Result<Vec<OsString>, String> {
        if self.upscale.is_empty() && self.interpolate.is_empty() {
            return Err("请至少启用超分或插帧".into());
        }
        let mut args = Vec::new();
        if !self.upscale.is_empty() {
            let (w, h) = self.size(video);
            if self.upscale == "nvidia-vsr"
                && ((w, h) == (video.width, video.height) || w < video.width || h < video.height)
            {
                return Err("NVIDIA VSR Ultra 需要比原片更大的输出尺寸".into());
            }
            if self.upscale == "seedvr2-7b"
                && u64::from(w) * u64::from(video.height) != u64::from(h) * u64::from(video.width)
            {
                return Err("7B 要求输出宽高比与源素材完全一致，请调整目标高度".into());
            }
            args.extend([
                "--upscale".into(),
                self.upscale.clone().into(),
                "--size".into(),
                format!("{w}x{h}").into(),
            ]);
            if self.upscale == "seedvr2-7b" {
                args.extend(["--chunk", "5", "--overlap", "0"].map(OsString::from));
            }
        }
        if !self.interpolate.is_empty() {
            args.extend([
                "--interpolate".into(),
                self.interpolate.clone().into(),
                "--motion-scale".into(),
                self.scale.clone().into(),
            ]);
            if !self.fps.is_finite() || !(1.0..=240.0).contains(&self.fps) {
                return Err("目标帧率须在 1–240 fps 之间".into());
            }
            let fps = format!("{:.6}", self.fps);
            if self.interpolate == "nvidia-vfg"
                && self.mp4
                && !self.upscale_first
                && matches!(self.upscale.as_str(), "" | "nvidia-vsr")
                && video
                    .reported_fps()
                    .is_some_and(|source| fps.parse::<f64>().is_ok_and(|target| target > source))
            {
                args.push("--vfg-review".into());
            }
            args.extend(["--fps".into(), fps.into()]);
        }
        if self.upscale_first {
            args.extend(["--order", "upscale-first"].map(OsString::from));
        }
        if (self.upscale == "nvidia-vsr" || self.interpolate == "nvidia-vfg")
            && matches!(self.upscale.as_str(), "" | "nvidia-vsr")
            && matches!(self.interpolate.as_str(), "" | "nvidia-vfg")
        {
            args.extend(["--chunk", "24"].map(OsString::from));
        }
        Ok(args)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Task {
    pub output: PathBuf,
    pub job: PathBuf,
}

impl Task {
    pub fn planned_fps(&self) -> Option<f64> {
        let manifest = std::fs::File::open(self.job.join("job.json")).ok()?;
        let plan: Value = serde_json::from_reader(BufReader::new(manifest)).ok()?;
        let rate = plan["output_rate"]
            .as_str()
            .or_else(|| plan["stages"].as_array()?.last()?["rate"].as_str())?;
        if let Some((numerator, denominator)) = rate.split_once('/') {
            Some(numerator.parse::<f64>().ok()? / denominator.parse::<f64>().ok()?)
        } else {
            rate.parse().ok()
        }
    }
}

pub fn allocate(input: &Path, directory: &Path, mp4: bool, assigned: &HashSet<PathBuf>) -> Task {
    let ext = if mp4 { "mp4" } else { "mkv" };
    let mut index = 1;
    loop {
        let mut name = input.file_stem().unwrap_or_default().to_os_string();
        name.push(if index == 1 {
            format!("-FrameLab.{ext}")
        } else {
            format!("-FrameLab-{index}.{ext}")
        });
        let output = directory.join(name);
        let mut job = output.as_os_str().to_os_string();
        job.push(".job");
        let job = PathBuf::from(job);
        if !output.exists() && !job.exists() && !assigned.contains(&output) {
            return Task { output, job };
        }
        index += 1;
    }
}

pub enum Message {
    Progress {
        label: String,
        fraction: Option<f32>,
    },
    Finished(Result<bool, String>),
    PauseError(String),
}

fn read_progress(event: &Value, stages: &mut f32) -> Option<Message> {
    let kind = event["event"].as_str()?;
    let stage = event["stage"].as_f64().unwrap_or(1.0).max(1.0) as f32;
    let total = event["total"].as_f64().unwrap_or(1.0).max(1.0) as f32;
    let stage_fraction = |completed: &Value| {
        let blocks = completed.as_f64().unwrap_or(0.0) as f32;
        0.90 * (stage - 1.0 + (blocks / total).clamp(0.0, 1.0)) / *stages
    };
    let (label, fraction) = match kind {
        "plan" => {
            *stages = event["stages"].as_f64().unwrap_or(1.0).max(1.0) as f32;
            return None;
        }
        "stage" => (
            format!(
                "阶段 {} · 已有 {}/{} 块",
                event["stage"], event["reused"], event["total"]
            ),
            Some(stage_fraction(&event["reused"])),
        ),
        "block" => {
            let committed = event.get("committed").unwrap_or(&event["block"]);
            (
                format!(
                    "阶段 {} · {committed}/{} 块",
                    event["stage"], event["total"]
                ),
                Some(stage_fraction(committed)),
            )
        }
        "phase" => {
            let (status, fraction) = match event["phase"].as_str()? {
                "prepare" => ("分析源素材和模型", None),
                "validate" => ("校验续跑素材和模型", None),
                "cache" => ("校验分块缓存", None),
                "probe" => ("读取源视频时间戳", None),
                "load" => ("初始化模型", None),
                "infer_nvenc" => ("VFG → VSR → NVENC", None),
                "join_nvenc" => ("拼接已编码分段", Some(0.94)),
                "infer" => ("推理并写入分块", None),
                "assemble" => ("合并旧作业分块", None),
                "encode" => (
                    "编码成片",
                    Some(
                        0.90 + 0.07
                            * (event["done"].as_f64().unwrap_or(0.0) as f32 / total)
                                .clamp(0.0, 1.0),
                    ),
                ),
                "verify" => ("校验视频", Some(0.97)),
                "mux" => ("封装音轨和字幕", Some(0.98)),
                "verify_mux" => ("校验封装", Some(0.985)),
                "publish" => ("写入成片", Some(0.99)),
                _ => return None,
            };
            let label = if event.get("stage").is_some() {
                format!("阶段 {} · {status}", event["stage"])
            } else {
                status.into()
            };
            (label, fraction)
        }
        _ => return None,
    };
    Some(Message::Progress { label, fraction })
}

#[cfg(not(feature = "portable"))]
pub(crate) fn launcher(repo: &Path) -> std::process::Command {
    let mut command = media::command("powershell.exe");
    command
        .current_dir(repo)
        .env("PYTHONUTF8", "1")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
        .arg(repo.join("run_video_lab.ps1"));
    command
}

#[cfg(feature = "portable")]
pub(crate) fn launcher(repo: &Path) -> std::process::Command {
    let mut command = media::command(repo.join("python").join("python.exe"));
    command
        .current_dir(repo)
        .args(["-X", "utf8"])
        .arg(repo.join("video_lab.py"));
    command
}

pub fn start(repo: PathBuf, input: PathBuf, task: Task, args: Vec<OsString>, tx: Sender<Message>) {
    thread::spawn(move || {
        let result = (|| {
            let mut command = launcher(&repo);
            if task.job.join("job.json").is_file() {
                command.arg("resume").arg(&task.job);
            } else {
                command
                    .arg("run")
                    .arg(&input)
                    .arg(&task.output)
                    .arg("--job-dir")
                    .arg(&task.job)
                    .args(args);
            }
            let mut child = command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| e.to_string())?;
            let stderr = child.stderr.take().expect("piped stderr");
            let diagnostics = thread::spawn(move || {
                let mut recent = VecDeque::with_capacity(20);
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    if recent.len() == 20 {
                        recent.pop_front();
                    }
                    recent.push_back(line);
                }
                recent.into_iter().collect::<Vec<_>>().join("\n")
            });
            let mut terminal = None;
            let mut stages = 1.0;
            for line in BufReader::new(child.stdout.take().expect("piped stdout"))
                .lines()
                .map_while(Result::ok)
            {
                let Ok(event) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                match event["event"].as_str() {
                    Some("complete") => terminal = Some(true),
                    Some("paused") => terminal = Some(false),
                    _ => {
                        if let Some(progress) = read_progress(&event, &mut stages) {
                            let _ = tx.send(progress);
                        }
                    }
                }
            }
            let status = child.wait().map_err(|e| e.to_string())?;
            let log = diagnostics.join().unwrap_or_default();
            if status.success() {
                terminal.ok_or_else(|| format!("内核未返回作业结果\n{log}"))
            } else {
                Err(format!("导出失败（{status}）\n{log}"))
            }
        })();
        let _ = tx.send(Message::Finished(result));
    });
}

pub fn pause(repo: PathBuf, job: PathBuf, tx: Sender<Message>) {
    thread::spawn(
        move || match launcher(&repo).arg("pause").arg(job).output() {
            Ok(output) if output.status.success() => {}
            Ok(output) => {
                let _ = tx.send(Message::PauseError(
                    String::from_utf8_lossy(&output.stderr).into_owned(),
                ));
            }
            Err(error) => {
                let _ = tx.send(Message::PauseError(error.to_string()));
            }
        },
    );
}
