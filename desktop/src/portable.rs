use std::{
    env, fs,
    io::{self, Read, Seek, SeekFrom},
    os::windows::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{SystemTime, UNIX_EPOCH},
};

const MAGIC: &[u8; 16] = b"FRAMELAB-NVIDIA!";
const FOOTER_BYTES: u64 = 40;
const HIDDEN_PROCESS: u32 = 0x0800_0000;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn has_runtime(path: &Path) -> bool {
    [
        "python/python.exe",
        "video_lab.py",
        "jobs.py",
        "engines.py",
        "scene_detection.py",
        "vfg_review.py",
        "vfg_retry.py",
        "python/Lib/site-packages/cv2/cv2.pyd",
        "bin/ffmpeg.exe",
        "bin/ffprobe.exe",
        ".runtime/vfx-deps/nvvfx/_ext.pyd",
    ]
    .iter()
    .all(|name| path.join(name).is_file())
}

fn extract(exe: &Path, staging: &Path) -> io::Result<()> {
    let system = env::var_os("SystemRoot").ok_or_else(|| invalid("SystemRoot is unavailable"))?;
    // A seekable read uses the ZIP directory. Windows tar can skip initial
    // members when it examines the same archive through a pipe.
    let result = Command::new(PathBuf::from(system).join("System32").join("tar.exe"))
        .arg("-xf")
        .arg(exe)
        .arg("-C")
        .arg(staging)
        .stdin(Stdio::null())
        .creation_flags(HIDDEN_PROCESS)
        .output()?;
    if !result.status.success() {
        return Err(invalid(&format!(
            "Unable to extract NVIDIA runtime: {}",
            String::from_utf8_lossy(&result.stderr)
        )));
    }
    if !has_runtime(staging) {
        return Err(invalid("Unable to unpack the bundled NVIDIA runtime"));
    }
    fs::write(staging.join(".ready"), b"FrameLab NVIDIA")
}

fn install() -> io::Result<PathBuf> {
    let executable = env::current_exe()?;
    let mut exe = fs::File::open(&executable)?;
    let size = exe.metadata()?.len();
    exe.seek(SeekFrom::End(-(FOOTER_BYTES as i64)))?;
    let mut footer = [0u8; FOOTER_BYTES as usize];
    exe.read_exact(&mut footer)?;
    if &footer[..16] != MAGIC {
        return Err(invalid("The NVIDIA runtime archive is missing"));
    }
    let length = u64::from_le_bytes(footer[16..24].try_into().unwrap());
    size.checked_sub(FOOTER_BYTES)
        .and_then(|offset| offset.checked_sub(length))
        .filter(|offset| *offset > 0)
        .ok_or_else(|| invalid("Invalid NVIDIA runtime archive size"))?;
    let version: String = footer[24..]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let local =
        env::var_os("LOCALAPPDATA").ok_or_else(|| invalid("LOCALAPPDATA is unavailable"))?;
    let base = PathBuf::from(local).join("FrameLab NVIDIA").join("bundles");
    fs::create_dir_all(&base)?;
    let destination = base.join(version);
    let ready = |path: &Path| path.join(".ready").is_file() && has_runtime(path);
    if ready(&destination) {
        return Ok(destination);
    }
    if destination.exists() {
        return Err(invalid("Installed runtime files are missing. Delete the damaged runtime folder and launch again."));
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging = base.join(format!(".unpack-{}-{stamp}", std::process::id()));
    fs::create_dir(&staging)?;
    let result = (|| {
        extract(&executable, &staging)?;
        match fs::rename(&staging, &destination) {
            Ok(()) => Ok(()),
            Err(_) if ready(&destination) => Ok(()),
            Err(error) => Err(error),
        }
    })();
    if staging.is_dir() && staging.canonicalize()?.parent() == Some(base.canonicalize()?.as_path())
    {
        let _ = fs::remove_dir_all(&staging);
    }
    result?;
    Ok(destination)
}

pub fn prepare() -> io::Result<PathBuf> {
    let root = install()?;
    configure(&root)?;
    Ok(root)
}

pub fn configure(root: &Path) -> io::Result<()> {
    let bin = root.join("bin");
    let cuda = root.join("cuda");
    let inherited = env::var_os("PATH").unwrap_or_default();
    let path = env::join_paths(
        [bin.clone(), cuda.join("bin")]
            .into_iter()
            .chain(env::split_paths(&inherited)),
    )
    .map_err(|error| invalid(&error.to_string()))?;
    env::set_var("PATH", path);
    env::set_var("CUDA_PATH", cuda);
    env::set_var("CUPY_CACHE_DIR", root.join(".runtime").join("cupy-cache"));
    env::set_var("FRAMELAB_BUNDLED_BIN", bin);
    env::remove_var("PYTHONHOME");
    env::remove_var("PYTHONPATH");
    env::set_var("PYTHONNOUSERSITE", "1");
    env::set_var("PYTHONDONTWRITEBYTECODE", "1");
    Ok(())
}

pub fn show_error(detail: &str) {
    #[link(name = "user32")]
    extern "system" {
        fn MessageBoxW(window: isize, text: *const u16, title: *const u16, flags: u32) -> i32;
    }
    let text: Vec<_> = format!("FrameLab NVIDIA 无法启动。\n\n{detail}")
        .encode_utf16()
        .chain([0])
        .collect();
    let title: Vec<_> = "FrameLab NVIDIA".encode_utf16().chain([0]).collect();
    unsafe { MessageBoxW(0, text.as_ptr(), title.as_ptr(), 0x10) };
}
