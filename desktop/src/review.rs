use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::{self, File},
    io::BufReader,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use eframe::egui;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{export, media};

#[derive(Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
enum Fault {
    #[default]
    Unset,
    Echo,
    Freeze,
    Ghosting,
}

impl Fault {
    const OPTIONS: [(Self, &'static str, Option<&'static str>); 4] = [
        (Self::Unset, "选择坏帧类型", None),
        (Self::Echo, "ABA 回跳", Some("echo")),
        (Self::Freeze, "连续静止", Some("freeze")),
        (Self::Ghosting, "严重拖影", Some("ghosting")),
    ];

    fn option(self) -> (&'static str, Option<&'static str>) {
        Self::OPTIONS
            .iter()
            .find_map(|&(fault, name, key)| (self == fault).then_some((name, key)))
            .unwrap()
    }
}

#[derive(Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
enum Motion {
    #[default]
    Unset,
    Normal,
    Fast,
}

impl Motion {
    const OPTIONS: [(Self, &'static str, Option<&'static str>); 3] = [
        (Self::Unset, "未指定", None),
        (Self::Normal, "普通镜头", Some("normal")),
        (Self::Fast, "高速镜头", Some("fast")),
    ];

    fn option(self) -> (&'static str, Option<&'static str>) {
        Self::OPTIONS
            .iter()
            .find_map(|&(motion, name, key)| (self == motion).then_some((name, key)))
            .unwrap()
    }
}

#[derive(Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct Mark {
    fault: Fault,
    motion: Motion,
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Reverse,
    Rife,
    Gimm,
}

impl Method {
    const OPTIONS: [(Self, &'static str, &'static str); 3] = [
        (Self::Reverse, "VFG 倒序", "reverse"),
        (Self::Rife, "RIFE 4.25", "rife"),
        (Self::Gimm, "GIMM-VFI", "gimm-vfi"),
    ];

    fn option(self) -> (&'static str, &'static str) {
        Self::OPTIONS
            .iter()
            .find_map(|&(method, name, key)| (self == method).then_some((name, key)))
            .unwrap()
    }

    fn recommended(mark: Mark) -> Self {
        if cfg!(feature = "portable") || mark.fault != Fault::Ghosting {
            Self::Reverse
        } else if mark.motion == Motion::Fast {
            Self::Gimm
        } else {
            Self::Rife
        }
    }
}

pub enum Action {
    None,
    Reload,
    Seek(usize),
}

#[derive(Deserialize)]
struct RepairedSlot {
    frame: usize,
    role: String,
}

#[derive(Deserialize)]
struct Selection {
    slots: Vec<RepairedSlot>,
}

#[derive(Deserialize, Serialize)]
struct Comparison {
    output: PathBuf,
    frames: Vec<usize>,
}

type RepairResult = Result<(usize, Option<Comparison>), String>;

pub struct Review {
    task: export::Task,
    from_job: bool,
    pub video: media::Video,
    count: usize,
    suspects: Vec<usize>,
    marks: BTreeMap<usize, Mark>,
    selected: Option<usize>,
    method: Method,
    comparisons: BTreeMap<usize, Comparison>,
    show_repaired: bool,
    work: Option<Receiver<RepairResult>>,
    message: String,
}

impl Review {
    pub fn open(output: PathBuf, job: Option<PathBuf>) -> Result<Self, String> {
        let file = output.file_name().ok_or("无法识别视频文件名")?;
        let sidecar = |suffix: &str| {
            let mut name = file.to_os_string();
            name.push(suffix);
            output.with_file_name(name)
        };
        let job = job.unwrap_or_else(|| sidecar(".job"));
        let read = |name: &str| -> Option<Value> {
            serde_json::from_reader(BufReader::new(File::open(job.join(name)).ok()?)).ok()
        };
        let (video, duration) = media::probe(&output)?;
        let plan_count = read("job.json").and_then(|plan| {
            let stages = plan["stages"].as_array()?;
            let original = Path::new(plan["output"].as_str()?).canonicalize().ok()?;
            if output.canonicalize().ok()? != original || !job.join("result.json").is_file()
                || !matches!(stages.as_slice(), [stage] if stage["nvenc_blocks"] == true
                    && stage["options"]["engine"] == "nvidia-vfg" && stage.get("reverse_slots").is_none()
                    && stage.get("repair_slots").is_none())
            {
                return None;
            }
            plan["output_frames"].as_u64().filter(|&n| n > 0).map(|n| n as usize)
        });
        let count = plan_count.unwrap_or_else(|| video.frame_count(duration));
        if count == 0 {
            return Err("无法确定视频帧数".into());
        }
        let suspects = plan_count
            .and_then(|_| read("vfg-review.json"))
            .and_then(|report| report["suspects"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(|entry| entry["output_frame"].as_u64().map(|n| n as usize))
            .filter(|&n| n < count)
            .collect();
        let storage = if plan_count.is_some() {
            job
        } else {
            sidecar(".review")
        };
        let marks: BTreeMap<usize, Mark> = match fs::read(storage.join("review-marks.json")) {
            Ok(bytes) => {
                serde_json::from_slice(&bytes).map_err(|e| format!("无法读取复核标记：{e}"))?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => BTreeMap::new(),
            Err(error) => return Err(format!("无法读取复核标记：{error}")),
        };
        let selected = marks.keys().next().copied();
        let method = selected.map_or(Method::Reverse, |frame| Method::recommended(marks[&frame]));
        let mut comparisons: BTreeMap<usize, Comparison> =
            fs::read(storage.join("review-previews.json"))
                .ok()
                .and_then(|data| serde_json::from_slice(&data).ok())
                .unwrap_or_default();
        comparisons
            .retain(|frame, comparison| marks.contains_key(frame) && comparison.output.is_file());
        Ok(Self {
            task: export::Task {
                output,
                job: storage,
            },
            from_job: plan_count.is_some(),
            video,
            count,
            suspects,
            marks,
            selected,
            method,
            comparisons,
            show_repaired: false,
            work: None,
            message: String::new(),
        })
    }

    pub fn busy(&self) -> bool {
        self.work.is_some()
    }

    pub fn duration(&self) -> f64 {
        self.count as f64 / self.video.fps()
    }

    pub fn path(&self) -> &Path {
        if self.show_repaired {
            if let Some(comparison) = self.comparison() {
                return &comparison.output;
            }
        }
        &self.task.output
    }

    fn comparison(&self) -> Option<&Comparison> {
        self.selected.and_then(|frame| self.comparisons.get(&frame))
    }

    pub fn paint_marks(&self, ui: &egui::Ui, rect: egui::Rect) {
        let track = rect.shrink2(egui::vec2(4.0, 0.0));
        let painter = ui.painter();
        let tick = |frame: usize, row: f32, color| {
            let x = egui::lerp(track.x_range(), frame as f32 / self.count as f32);
            painter.line_segment(
                [egui::pos2(x, row), egui::pos2(x, row + 4.0)],
                egui::Stroke::new(2.0_f32, color),
            );
        };
        for &frame in self.marks.keys() {
            let color = if self.comparisons.contains_key(&frame) {
                egui::Color32::from_rgb(76, 191, 138)
            } else {
                egui::Color32::from_rgb(245, 204, 63)
            };
            tick(frame, track.top(), color);
        }
        if let Some(comparison) = self.comparison() {
            for &frame in &comparison.frames {
                if !self.marks.contains_key(&frame) {
                    tick(
                        frame,
                        track.bottom() - 5.0,
                        egui::Color32::from_rgb(76, 191, 138),
                    );
                }
            }
        }
    }

    pub fn controls(
        &mut self,
        ui: &mut egui::Ui,
        repo: &Path,
        displayed_time: Option<f64>,
    ) -> Action {
        let updated = match self.work.as_ref().map(Receiver::try_recv) {
            Some(Ok(result)) => Some(result),
            Some(Err(TryRecvError::Disconnected)) => Some(Err("复核进程提前结束".into())),
            _ => None,
        };
        let mut action = Action::None;
        if let Some(result) = updated {
            self.work = None;
            match result {
                Ok((marked, Some(comparison))) => {
                    self.message = format!("已生成 {} 帧的修复候选", comparison.frames.len());
                    self.comparisons.insert(marked, comparison);
                    self.save_previews();
                    self.show_repaired = true;
                    action = Action::Reload;
                }
                Ok((marked, None)) => {
                    self.message = "已记录保留修复前结果".into();
                    self.comparisons.remove(&marked);
                    self.save_previews();
                    if self.show_repaired {
                        self.show_repaired = false;
                        action = Action::Reload;
                    }
                }
                Err(error) => self.message = error,
            }
        }
        if self.busy() {
            ui.ctx().request_repaint_after(Duration::from_millis(75));
        }
        let frame = displayed_time
            .map(|time| ((time * self.video.fps()).round() as usize).min(self.count - 1));
        egui::ScrollArea::horizontal()
            .id_salt("review-mark-row")
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        frame
                            .map_or_else(|| "复核成片".into(), |n| format!("复核成片 · 第 {n} 帧")),
                    );
                    if ui
                        .add_enabled(
                            frame.is_some() && !self.busy(),
                            egui::Button::new("标记当前帧"),
                        )
                        .clicked()
                    {
                        if let Some(frame) = frame {
                            let reload = self.show_repaired && self.selected != Some(frame);
                            self.marks.entry(frame).or_default();
                            self.select(frame);
                            self.save();
                            if reload {
                                action = Action::Reload;
                            }
                        }
                    }
                    if !self.marks.is_empty() {
                        let mut target = None;
                        ui.add_enabled_ui(!self.busy(), |ui| {
                            egui::ComboBox::from_id_salt("manual-review-marks")
                                .width(92.0)
                                .selected_text(
                                    self.selected
                                        .map_or_else(|| "坏帧标记".into(), |n| format!("标记 {n}")),
                                )
                                .show_ui(ui, |ui| {
                                    for (&n, mark) in &self.marks {
                                        if ui
                                            .selectable_label(
                                                self.selected == Some(n),
                                                format!("{n}  {}", mark.fault.option().0),
                                            )
                                            .clicked()
                                        {
                                            target = Some(n);
                                        }
                                    }
                                })
                        });
                        if let Some(n) = target {
                            self.select(n);
                            action = Action::Seek(n);
                        }
                    }
                    if let Some(selected) = self.selected {
                        let reload = ui
                            .add_enabled_ui(!self.busy(), |ui| self.edit_mark(ui, selected))
                            .inner;
                        if reload {
                            action = Action::Reload;
                        }
                    }
                    if !self.suspects.is_empty() {
                        egui::ComboBox::from_id_salt("review-report")
                            .selected_text("复检候选")
                            .show_ui(ui, |ui| {
                                for &n in &self.suspects {
                                    if ui.button(n.to_string()).clicked() {
                                        action = Action::Seek(n);
                                        ui.close_menu();
                                    }
                                }
                            });
                    }
                })
            });
        ui.horizontal_wrapped(|ui| {
            if self.comparison().is_some() {
                let before = self.show_repaired;
                ui.selectable_value(&mut self.show_repaired, false, "修复前");
                ui.selectable_value(&mut self.show_repaired, true, "修复后");
                if before != self.show_repaired {
                    action = Action::Reload;
                }
            }
            if let Some(selected) = self.selected {
                ui.add_enabled_ui(!self.busy(), |ui| self.model_controls(ui, repo, selected));
            }
        });
        if !self.message.is_empty() {
            ui.add(egui::Label::new(&self.message).truncate());
        }
        action
    }

    fn select(&mut self, frame: usize) {
        if self.selected != Some(frame) {
            self.show_repaired = false;
        }
        self.selected = Some(frame);
        self.method = Method::recommended(self.marks[&frame]);
    }

    fn edit_mark(&mut self, ui: &mut egui::Ui, frame: usize) -> bool {
        let mut mark = self.marks[&frame];
        egui::ComboBox::from_id_salt("review-fault")
            .width(112.0)
            .selected_text(mark.fault.option().0)
            .show_ui(ui, |ui| {
                for (fault, name, _) in Fault::OPTIONS {
                    ui.selectable_value(&mut mark.fault, fault, name);
                }
            });
        egui::ComboBox::from_id_salt("review-motion")
            .width(82.0)
            .selected_text(mark.motion.option().0)
            .show_ui(ui, |ui| {
                for (motion, name, _) in Motion::OPTIONS {
                    ui.selectable_value(&mut mark.motion, motion, name);
                }
            });
        let edited = self.marks[&frame] != mark;
        let removed = ui.button("移除此标记").clicked();
        if edited {
            self.marks.insert(frame, mark);
            self.method = Method::recommended(mark);
            self.save();
        }
        if removed {
            self.marks.remove(&frame);
            self.selected = None;
            self.save();
        }
        if edited || removed {
            if self.comparisons.remove(&frame).is_some() {
                self.save_previews();
            }
            return std::mem::take(&mut self.show_repaired);
        }
        false
    }

    fn model_controls(&mut self, ui: &mut egui::Ui, repo: &Path, frame: usize) {
        let mark = self.marks[&frame];
        egui::ComboBox::from_id_salt("candidate-method")
            .selected_text(self.method.option().0)
            .show_ui(ui, |ui| {
                for (method, name, _) in Method::OPTIONS {
                    let enabled = !cfg!(feature = "portable") || method == Method::Reverse;
                    if ui
                        .add_enabled(
                            enabled,
                            egui::Button::new(name).selected(self.method == method),
                        )
                        .on_disabled_hover_text("此模型需要完整版运行环境")
                        .clicked()
                    {
                        self.method = method;
                    }
                }
            });
        if ui
            .add_enabled(
                mark.fault != Fault::Unset,
                egui::Button::new("生成修复预览"),
            )
            .clicked()
        {
            self.submit(repo, frame, mark, self.method.option().1);
        }
        if ui.button("保留修复前").clicked() {
            self.submit(repo, frame, mark, "keep");
        }
    }

    fn save(&mut self) {
        if let Err(error) = save_json(&self.task.job, "review-marks.json", &self.marks) {
            self.message = format!("保存标记失败：{error}");
        }
    }

    fn save_previews(&mut self) {
        if let Err(error) = save_json(&self.task.job, "review-previews.json", &self.comparisons) {
            self.message = format!("保存预览状态失败：{error}");
        }
    }

    fn submit(&mut self, repo: &Path, frame: usize, mark: Mark, method: &'static str) {
        let repo = repo.to_owned();
        let output = self
            .task
            .job
            .join(format!("review-{frame}-{method}-{}.mp4", nonce()));
        let (command, input) = if self.from_job {
            ("review-repair", &self.task.job)
        } else {
            ("review-import", &self.task.output)
        };
        let mut args: Vec<OsString> = vec![
            command.into(),
            input.clone().into(),
            frame.to_string().into(),
            output.clone().into(),
            "--choice".into(),
            method.into(),
        ];
        if self.from_job {
            args.push("--infer-range".into());
        }
        if let Some(label) = mark.fault.option().1 {
            args.extend([OsString::from("--label"), label.into()]);
        }
        if let Some(label) = mark.motion.option().1 {
            args.extend([OsString::from("--motion"), label.into()]);
        }
        self.message = if method == "keep" {
            "正在记录选择…"
        } else {
            "正在判断范围并生成修复预览…"
        }
        .into();
        let (tx, rx) = mpsc::channel();
        self.work = Some(rx);
        thread::spawn(move || {
            let result =
                run_candidate(&repo, args, output, method).map(|candidate| (frame, candidate));
            let _ = tx.send(result);
        });
    }
}

fn save_json(storage: &Path, name: &str, value: &impl Serialize) -> std::io::Result<()> {
    fs::create_dir_all(storage)?;
    let temporary = storage.join(format!("review-{}.tmp", nonce()));
    let result = serde_json::to_vec_pretty(value)
        .map_err(std::io::Error::other)
        .and_then(|data| fs::write(&temporary, data))
        .and_then(|()| fs::rename(&temporary, storage.join(name)));
    if result.is_err() {
        let _ = fs::remove_file(temporary);
    }
    result
}

fn nonce() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn run_candidate(
    repo: &Path,
    args: Vec<OsString>,
    output: PathBuf,
    method: &str,
) -> Result<Option<Comparison>, String> {
    let result = export::launcher(repo)
        .args(args)
        .output()
        .map_err(|error| error.to_string())?;
    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(stderr
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("复核作业失败")
            .into());
    }
    if method == "keep" {
        return Ok(None);
    }
    let directory = output.with_extension("mp4.job");
    let file =
        File::open(directory.join("review-selection.json")).map_err(|error| error.to_string())?;
    let choice: Selection =
        serde_json::from_reader(BufReader::new(file)).map_err(|error| error.to_string())?;
    let frames = choice
        .slots
        .into_iter()
        .filter(|slot| matches!(slot.role.as_str(), "generated" | "replacement"))
        .map(|slot| slot.frame)
        .collect();
    Ok(Some(Comparison { output, frames }))
}
