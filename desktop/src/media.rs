use std::{
    io::Read,
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    thread,
};

use serde::Deserialize;

#[derive(Clone, Deserialize)]
pub struct Video {
    pub width: u32,
    pub height: u32,
    avg_frame_rate: String,
    duration: Option<String>,
    nb_frames: Option<String>,
}

impl Video {
    pub fn reported_fps(&self) -> Option<f64> {
        self.avg_frame_rate
            .split_once('/')
            .and_then(|(a, b)| Some(a.parse::<f64>().ok()? / b.parse::<f64>().ok()?))
            .filter(|r| r.is_finite() && *r > 0.0)
    }

    pub fn fps(&self) -> f64 {
        self.reported_fps().unwrap_or(24.0)
    }

    pub fn frame_count(&self, duration: f64) -> usize {
        self.nb_frames
            .as_deref()
            .and_then(|n| n.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| (duration * self.fps()).round() as usize)
    }
}

pub fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let program = program.as_ref();
    let bundled = match program.to_str() {
        Some("ffmpeg") | Some("ffprobe") => std::env::var_os("FRAMELAB_BUNDLED_BIN").map(|bin| {
            std::path::PathBuf::from(bin)
                .join(program)
                .with_extension("exe")
        }),
        _ => None,
    };
    let mut command = match bundled {
        Some(path) => Command::new(path),
        None => Command::new(program),
    };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command
}

pub fn probe(path: &Path) -> Result<(Video, f64), String> {
    let output = command("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "V:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,duration,nb_frames:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .map_err(|e| format!("无法启动 ffprobe：{e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    #[derive(Deserialize)]
    struct Format {
        duration: Option<String>,
    }
    #[derive(Deserialize)]
    struct Probe {
        streams: Vec<Video>,
        format: Format,
    }
    let probe: Probe = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    let video = probe
        .streams
        .into_iter()
        .find(|v| v.width > 0 && v.height > 0)
        .ok_or("文件中没有可预览的视频流")?;
    let duration = [video.duration.as_ref(), probe.format.duration.as_ref()]
        .into_iter()
        .flatten()
        .filter_map(|s| s.parse::<f64>().ok())
        .find(|d| d.is_finite() && *d > 0.0)
        .unwrap_or(0.0);
    Ok((video, duration))
}

pub struct Frame {
    pub size: [usize; 2],
    pub pixels: Vec<u8>,
    pub time: f64,
}

pub fn decode(
    path: &Path,
    video: &Video,
    start: f64,
    limit: Option<usize>,
    exact: bool,
) -> (mpsc::Receiver<Result<Frame, String>>, f64) {
    let (tx, rx) = mpsc::sync_channel(3);
    let path = path.to_owned();
    let ratio = (960.0 / f64::from(video.width))
        .min(540.0 / f64::from(video.height))
        .min(1.0);
    let size = [
        (f64::from(video.width) * ratio).round().max(1.0) as usize,
        (f64::from(video.height) * ratio).round().max(1.0) as usize,
    ];
    let fps = if limit.is_some() || exact {
        video.fps()
    } else {
        video.fps().min(30.0)
    };
    thread::spawn(move || {
        let filter = format!(
            "scale={}:{}:flags=bilinear{}",
            size[0],
            size[1],
            if limit.is_some() || exact {
                String::new()
            } else {
                format!(",fps={fps}")
            }
        );
        let mut cmd = command("ffmpeg");
        // Review exports are CFR. Seek inside the previous tick so the desired
        // frame remains the first timestamp at or after the seek point.
        let seek = if exact {
            (start - 0.25 / fps).max(0.0)
        } else {
            start
        };
        cmd.args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            &format!("{seek:.8}"),
        ])
        .arg("-i")
        .arg(&path)
        .args(["-map", "0:V:0", "-an", "-sn", "-vf", &filter]);
        if exact {
            cmd.args(["-fps_mode", "passthrough"]);
        }
        if let Some(limit) = limit {
            cmd.args(["-frames:v", &limit.to_string()]);
        }
        let child = cmd
            .args([
                "-threads:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match child {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(Err(format!("无法启动 FFmpeg：{error}")));
                return;
            }
        };
        let mut stdout = child.stdout.take().expect("piped stdout");
        let mut index = 0;
        loop {
            let mut pixels = vec![0; size[0] * size[1] * 3];
            if stdout.read_exact(&mut pixels).is_err() {
                break;
            }
            if tx
                .send(Ok(Frame {
                    size,
                    pixels,
                    time: start + f64::from(index) / fps,
                }))
                .is_err()
            {
                let _ = child.kill();
                break;
            }
            index += 1;
        }
        let success = child.wait().is_ok_and(|status| status.success());
        if !success || index == 0 {
            let _ = tx.send(Err("FFmpeg 未能读取此位置的视频画面".into()));
        }
    });
    (rx, fps)
}
