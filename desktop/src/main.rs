#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod export;
mod media;
#[cfg(feature = "portable")]
mod portable;
mod review;

use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    ops::Range,
    path::{Path, PathBuf},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

use eframe::egui::{self, Color32, RichText};
use export::{Settings, Task};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Item {
    path: PathBuf,
    settings: Settings,
    task: Option<Task>,
    #[serde(skip)]
    info: Option<(media::Video, f64)>,
    #[serde(skip)]
    probe: Option<mpsc::Receiver<ProbeResult>>,
    #[serde(skip)]
    state: String,
    #[serde(skip)]
    planned_fps: Option<f64>,
    #[serde(default)]
    progress: f32,
    #[serde(default)]
    elapsed_ms: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct Saved {
    items: Vec<Item>,
    directory: Option<PathBuf>,
    #[serde(default)]
    export_defaults: Settings,
}

#[derive(Default)]
struct Preview {
    position: f64,
    texture: Option<egui::TextureHandle>,
    frames: Option<mpsc::Receiver<Result<media::Frame, String>>>,
    pending: Option<media::Frame>,
    clock: Option<(Instant, f64)>,
    seeking: bool,
    error: Option<String>,
    displayed_time: Option<f64>,
    steps: Option<StepCache>,
    step_repeat: Option<(i8, Instant)>,
}

struct StepCache {
    range: Range<usize>,
    fps: f64,
    frames: BTreeMap<usize, media::Frame>,
    reader: Option<mpsc::Receiver<Result<media::Frame, String>>>,
    target: Option<usize>,
}

impl Preview {
    fn request(&mut self, item: &Item, play: bool) {
        let Some((video, duration)) = &item.info else {
            return;
        };
        self.request_media(&item.path, video, *duration, play, false);
    }

    fn request_media(
        &mut self,
        path: &Path,
        video: &media::Video,
        duration: f64,
        play: bool,
        exact: bool,
    ) {
        let last = (duration - 1.0 / video.fps()).max(0.0);
        self.position = self.position.clamp(0.0, last);
        if exact {
            self.position = (self.position * video.fps()).round() / video.fps();
        }
        self.steps = None;
        self.frames =
            Some(media::decode(path, video, self.position, (!play).then_some(1), exact).0);
        self.displayed_time = None;
        self.pending = None;
        self.clock = play.then(|| (Instant::now(), self.position));
        self.error = None;
    }

    fn stop(&mut self) {
        self.frames = None;
        self.pending = None;
        self.clock = None;
    }

    fn step(&mut self, path: &Path, video: &media::Video, duration: f64, direction: i8) {
        let fps = video.fps();
        let count = (duration * fps).round().max(1.0) as usize;
        let current = (self.position * fps).round() as usize;
        let target = current
            .saturating_add_signed(isize::from(direction))
            .min(count - 1);
        self.stop();
        if target == current && self.displayed_time.is_some() {
            return;
        }
        self.position = target as f64 / fps;
        self.displayed_time = None;
        let refill = self
            .steps
            .as_ref()
            .is_none_or(|cache| !cache.range.contains(&target));
        if refill {
            let (before, after) = if direction < 0 { (22, 4) } else { (4, 22) };
            let range = target.saturating_sub(before)..(target + after + 1).min(count);
            let (reader, _) = media::decode(
                path,
                video,
                range.start as f64 / fps,
                Some(range.len()),
                true,
            );
            self.steps = Some(StepCache {
                range,
                fps,
                frames: BTreeMap::new(),
                reader: Some(reader),
                target: Some(target),
            });
            self.error = None;
        } else if let Some(cache) = &mut self.steps {
            cache.target = Some(target);
        }
    }

    fn step_direction(&mut self, ctx: &egui::Context, held: i8, clicked: i8, now: Instant) -> i8 {
        let previous = self.step_repeat;
        if held == 0 {
            self.step_repeat = None;
            return if previous.is_none() { clicked } else { 0 };
        }
        match previous {
            Some((direction, due)) if direction == held => {
                if now >= due
                    && self
                        .steps
                        .as_ref()
                        .is_none_or(|cache| cache.target.is_none())
                {
                    self.step_repeat = Some((held, now + Duration::from_millis(100)));
                    ctx.request_repaint_after(Duration::from_millis(100));
                    held
                } else {
                    ctx.request_repaint_after(
                        due.saturating_duration_since(now)
                            .max(Duration::from_millis(5)),
                    );
                    0
                }
            }
            _ => {
                self.step_repeat = Some((held, now + Duration::from_millis(250)));
                ctx.request_repaint_after(Duration::from_millis(250));
                held
            }
        }
    }

    fn upload(
        texture: &mut Option<egui::TextureHandle>,
        ctx: &egui::Context,
        frame: &media::Frame,
    ) {
        let image = egui::ColorImage::from_rgb(frame.size, &frame.pixels);
        if let Some(texture) = texture {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            *texture = Some(ctx.load_texture("preview", image, egui::TextureOptions::LINEAR));
        }
    }

    fn tick(&mut self, ctx: &egui::Context) {
        if let Some(cache) = &mut self.steps {
            while let Some(reader) = &cache.reader {
                match reader.try_recv() {
                    Ok(Ok(frame)) => {
                        cache
                            .frames
                            .insert((frame.time * cache.fps).round() as usize, frame);
                    }
                    Ok(Err(error)) => {
                        self.error = Some(error);
                        cache.reader = None;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => cache.reader = None,
                    Err(mpsc::TryRecvError::Empty) => break,
                }
            }
            if let Some(target) = cache.target {
                if let Some(frame) = cache.frames.get(&target) {
                    Self::upload(&mut self.texture, ctx, frame);
                    self.displayed_time = Some(frame.time);
                    cache.target = None;
                } else if cache.reader.is_none() {
                    cache.target = None;
                    self.error = Some("无法读取该帧".into());
                }
            }
            if cache.reader.is_some() {
                ctx.request_repaint_after(Duration::from_millis(5));
            }
        }
        let mut due = self.clock.map_or(f64::INFINITY, |(start, offset)| {
            offset + start.elapsed().as_secs_f64()
        });
        if let Some(rx) = &self.frames {
            loop {
                if self.pending.is_none() {
                    match rx.try_recv() {
                        Ok(Ok(frame)) => {
                            if self.displayed_time.is_none() && self.clock.is_some() {
                                self.clock = Some((Instant::now(), frame.time));
                                due = frame.time;
                            }
                            self.pending = Some(frame);
                        }
                        Ok(Err(error)) => {
                            self.error = Some(error);
                            self.stop();
                            break;
                        }
                        Err(mpsc::TryRecvError::Disconnected) => {
                            self.stop();
                            break;
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                    }
                }
                if self.pending.as_ref().is_some_and(|frame| frame.time > due) {
                    break;
                }
                if let Some(frame) = self.pending.take() {
                    Self::upload(&mut self.texture, ctx, &frame);
                    self.position = frame.time;
                    self.displayed_time = Some(frame.time);
                }
            }
        }
        if self.frames.is_some() {
            ctx.request_repaint_after(Duration::from_millis(5));
        }
    }
}

type ProbeResult = Result<(media::Video, f64), String>;

struct App {
    repo: PathBuf,
    saved: Saved,
    selected: Option<usize>,
    preview: Preview,
    queue: VecDeque<usize>,
    active: Option<usize>,
    tick: Instant,
    worker: Option<mpsc::Receiver<export::Message>>,
    pause_tx: Option<mpsc::Sender<export::Message>>,
    notice: String,
    details: String,
    stopping: bool,
    review: Option<review::Review>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, repo: PathBuf, initial: Vec<PathBuf>) -> Self {
        style(&cc.egui_ctx);
        let saved: Saved = cc
            .storage
            .and_then(|storage| eframe::get_value(storage, "framelab"))
            .unwrap_or_default();
        let mut app = Self {
            repo,
            saved,
            selected: None,
            preview: Preview::default(),
            queue: VecDeque::new(),
            active: None,
            tick: Instant::now(),
            worker: None,
            pause_tx: None,
            notice: String::new(),
            details: String::new(),
            stopping: false,
            review: None,
        };
        for item in &mut app.saved.items {
            item.state = "读取中".into();
            item.probe = Some(probe_async(item.path.clone()));
        }
        app.select((!app.saved.items.is_empty()).then_some(0));
        app.add(initial);
        app
    }

    fn add(&mut self, paths: Vec<PathBuf>) {
        let mut first = None;
        for path in paths {
            let Ok(path) = path.canonicalize().map(|p| windows_path(&p)) else {
                continue;
            };
            if !path.is_file()
                || self
                    .saved
                    .items
                    .iter()
                    .any(|item| item.path == path && task_state(item.task.as_ref()) != "已导出")
            {
                continue;
            }
            let probe = probe_async(path.clone());
            first.get_or_insert(self.saved.items.len());
            self.saved.items.push(Item {
                path,
                settings: self.saved.export_defaults.clone(),
                task: None,
                info: None,
                probe: Some(probe),
                state: "读取中".into(),
                planned_fps: None,
                progress: 0.0,
                elapsed_ms: 0,
            });
        }
        if first.is_some() {
            self.select(first);
        }
    }

    fn select(&mut self, index: Option<usize>) {
        if self.review.as_ref().is_some_and(review::Review::busy) {
            return;
        }
        self.review = None;
        self.selected = index;
        self.preview = Preview::default();
        if let Some(item) = index.and_then(|i| self.saved.items.get(i)) {
            self.preview.request(item, false);
        }
    }

    fn begin(&mut self, only: Option<usize>) {
        if self.review.as_ref().is_some_and(review::Review::busy) {
            self.notice = "请等待成片复核结束。".into();
            return;
        }
        self.notice.clear();
        self.details.clear();
        let mut assigned: HashSet<_> = self
            .saved
            .items
            .iter()
            .filter_map(|i| i.task.as_ref().map(|t| t.output.clone()))
            .collect();
        for (index, item) in self.saved.items.iter_mut().enumerate() {
            if only.is_some_and(|i| i != index) || task_state(item.task.as_ref()) == "已导出" {
                continue;
            }
            let Some((video, _)) = &item.info else {
                self.notice = format!("{} · 无法读取素材，已跳过", filename(&item.path));
                continue;
            };
            if item
                .task
                .as_ref()
                .is_none_or(|t| !t.job.join("job.json").is_file())
            {
                if let Err(error) = item.settings.args(video) {
                    item.state = error;
                    continue;
                }
                let dir = item
                    .settings
                    .directory
                    .as_deref()
                    .or(self.saved.directory.as_deref())
                    .or_else(|| item.path.parent());
                let Some(dir) = dir else { continue };
                let task = export::allocate(&item.path, dir, item.settings.mp4, &assigned);
                assigned.insert(task.output.clone());
                item.task = Some(task);
                item.planned_fps = None;
                item.progress = 0.0;
                item.elapsed_ms = 0;
            }
            item.state = "等待导出".into();
            self.queue.push_back(index);
        }
        if self.queue.is_empty() && self.notice.is_empty() {
            self.notice = "没有可导出的素材，请检查每项设置。".into();
        }
        self.next_job();
    }

    fn next_job(&mut self) {
        while let Some(index) = self.queue.pop_front() {
            let item = &mut self.saved.items[index];
            let task = item.task.as_ref().unwrap();
            let args = if task.job.join("job.json").is_file() {
                Ok(Vec::new())
            } else {
                item.settings.args(&item.info.as_ref().unwrap().0)
            };
            match args {
                Ok(args) => {
                    let (tx, rx) = mpsc::channel();
                    export::start(
                        self.repo.clone(),
                        item.path.clone(),
                        task.clone(),
                        args,
                        tx.clone(),
                    );
                    self.worker = Some(rx);
                    self.pause_tx = Some(tx);
                    self.active = Some(index);
                    self.tick = Instant::now();
                    item.state = "准备导出".into();
                    return;
                }
                Err(error) => item.state = error,
            }
        }
    }

    fn drain(&mut self, ctx: &egui::Context) {
        let mut pending = false;
        for (index, item) in self.saved.items.iter_mut().enumerate() {
            let Some(probe) = &item.probe else { continue };
            let info = match probe.try_recv() {
                Ok(info) => info,
                Err(mpsc::TryRecvError::Empty) => continue,
                Err(mpsc::TryRecvError::Disconnected) => Err("素材读取意外中断".into()),
            };
            item.probe = None;
            match info {
                Ok(info) => {
                    item.info = Some(info);
                    item.planned_fps = item.task.as_ref().and_then(Task::planned_fps);
                    item.state = task_state(item.task.as_ref()).into();
                    if self.selected == Some(index) && self.review.is_none() {
                        self.preview.request(item, false);
                    }
                }
                Err(error) => {
                    item.state = "读取失败".into();
                    self.notice = error;
                }
            }
        }
        if let (Some(index), Some(rx)) = (self.active, &self.worker) {
            let now = Instant::now();
            let elapsed = now.duration_since(self.tick).as_millis();
            self.saved.items[index].elapsed_ms += elapsed as u64;
            self.tick += Duration::from_millis(elapsed as u64);
            let mut done = None;
            for message in rx.try_iter() {
                match message {
                    export::Message::Progress { label, fraction } => {
                        let item = &mut self.saved.items[index];
                        item.state = label;
                        if let Some(fraction) = fraction {
                            item.progress = item.progress.max(fraction);
                        }
                    }
                    export::Message::PauseError(error) => {
                        self.notice = format!("暂停失败 · {error}");
                        self.stopping = false;
                    }
                    export::Message::Finished(result) => done = Some(result),
                }
            }
            if let Some(result) = done {
                let paused = matches!(result, Ok(false));
                self.saved.items[index].state = match result {
                    Ok(true) => {
                        self.saved.items[index].progress = 1.0;
                        "已导出".into()
                    }
                    Ok(false) => "已暂停 · 可续跑".into(),
                    Err(error) => {
                        self.details = error;
                        "导出失败".into()
                    }
                };
                if paused || self.stopping {
                    for index in self.queue.drain(..) {
                        self.saved.items[index].state =
                            task_state(self.saved.items[index].task.as_ref()).into();
                    }
                }
                self.active = None;
                self.worker = None;
                self.pause_tx = None;
                self.stopping = false;
                self.next_job();
            }
            pending = true;
        }
        if pending || self.saved.items.iter().any(|item| item.probe.is_some()) {
            ctx.request_repaint_after(Duration::from_millis(80));
        }
        self.preview.tick(ctx);
    }

    fn request_pause(&mut self) {
        let (Some(index), Some(tx)) = (self.active, &self.pause_tx) else {
            return;
        };
        let job = self.saved.items[index].task.as_ref().unwrap().job.clone();
        if !job.join("job.json").is_file() {
            self.notice = "作业正在初始化，稍后可以暂停。".into();
            return;
        }
        self.stopping = true;
        self.notice = "当前分块完成后暂停批量导出。".into();
        export::pause(self.repo.clone(), job, tx.clone());
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        ui.add_space(9.0);
        ui.heading("导出配置");
        let Some(index) = self.selected else {
            ui.add_space(4.0);
            ui.weak("默认配置 · 用于新导入素材");
            ui.add_space(9.0);
            export_controls(ui, &mut self.saved.export_defaults, None, None, false);
            return;
        };
        let item = &mut self.saved.items[index];
        ui.add_space(4.0);
        ui.label(RichText::new(filename(&item.path)).color(Color32::from_rgb(164, 180, 199)));
        ui.add_space(9.0);
        let busy = self.active.is_some() || self.review.as_ref().is_some_and(review::Review::busy);
        let resumable = item
            .task
            .as_ref()
            .is_some_and(|t| t.job.join("job.json").is_file())
            && task_state(item.task.as_ref()) != "已导出";
        let finished = task_state(item.task.as_ref()) == "已导出";
        ui.add_enabled_ui(!busy && !resumable && !finished, |ui| {
            export_controls(
                ui,
                &mut item.settings,
                item.info.as_ref().map(|(video, _)| video),
                item.planned_fps,
                true,
            );
        });
        if !busy && (resumable || finished) {
            ui.add_space(8.0);
            let (resume, reset) = ui
                .horizontal(|ui| {
                    let resume = resumable && ui.button("续跑此素材").clicked();
                    let reset = ui
                        .button("新建导出")
                        .on_hover_text("下次导出使用新的文件名和作业目录")
                        .clicked();
                    (resume, reset)
                })
                .inner;
            if reset {
                self.review = None;
                self.preview = Preview::default();
                self.preview.request(item, false);
                item.task = None;
                item.state = "待导出".into();
                item.planned_fps = None;
                item.progress = 0.0;
                item.elapsed_ms = 0;
            }
            if resume {
                self.begin(Some(index));
            }
        }
    }

    fn video(&mut self, ui: &mut egui::Ui, height: f32, keys: (i8, i8, bool)) {
        let (key_held, key_tap, toggle) = keys;
        let rect = ui
            .allocate_exact_size(
                egui::vec2(ui.available_width(), height),
                egui::Sense::hover(),
            )
            .0;
        ui.painter()
            .rect_filled(rect, 8.0, Color32::from_rgb(9, 12, 17));
        if let Some(texture) = &self.preview.texture {
            let size = texture.size_vec2();
            let scale = (rect.width() / size.x).min(rect.height() / size.y);
            let area = egui::Rect::from_center_size(rect.center(), size * scale);
            ui.painter().image(
                texture.id(),
                area,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                Color32::WHITE,
            );
        } else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                self.preview
                    .error
                    .as_deref()
                    .unwrap_or(if self.selected.is_some() {
                        "正在读取预览…"
                    } else {
                        "导入或拖入视频以预览"
                    }),
                egui::FontId::proportional(15.0),
                Color32::from_rgb(141, 157, 177),
            );
        }

        let review = self.review.as_ref();
        let (path, video, duration) = if let Some(review) = review {
            (review.path(), &review.video, review.duration())
        } else {
            let Some(item) = self.selected.and_then(|i| self.saved.items.get(i)) else {
                return;
            };
            let Some((video, duration)) = &item.info else {
                return;
            };
            (item.path.as_path(), video, *duration)
        };
        let exact = review.is_some();
        let request =
            |preview: &mut Preview, play| preview.request_media(path, video, duration, play, exact);
        ui.add_space(5.0);
        let mut time = self.preview.position;
        let seek = ui
            .scope(|ui| {
                ui.spacing_mut().slider_width = ui.available_width();
                ui.add(egui::Slider::new(&mut time, 0.0..=duration).show_value(false))
            })
            .inner;
        if seek.changed() {
            self.preview.stop();
            self.preview.position = time;
            self.preview.seeking = seek.dragged();
            if !self.preview.seeking {
                request(&mut self.preview, false);
            }
        }
        if seek.drag_stopped() && self.preview.seeking {
            self.preview.seeking = false;
            request(&mut self.preview, false);
        }
        if let Some(review) = review {
            review.paint_marks(ui, seek.rect);
        }
        ui.horizontal(|ui| {
            if ui.button("首帧").clicked() {
                self.preview.position = 0.0;
                request(&mut self.preview, false);
            }
            let back = ui
                .button("上一帧")
                .on_hover_text("←  ·  长按每秒步进 10 帧");
            let playing = self.preview.clock.is_some();
            let play = ui
                .button(if playing { "暂停" } else { "播放" })
                .on_hover_text("空格");
            let next = ui
                .button("下一帧")
                .on_hover_text("→  ·  长按每秒步进 10 帧");
            if play.clicked() || toggle {
                if playing {
                    self.preview.stop();
                } else {
                    if self.preview.position >= duration - 1.0 / video.fps() {
                        self.preview.position = 0.0;
                    }
                    request(&mut self.preview, true);
                }
            }
            let mouse_held = i8::from(next.is_pointer_button_down_on())
                - i8::from(back.is_pointer_button_down_on());
            let mouse_click = i8::from(next.clicked()) - i8::from(back.clicked());
            let held = if mouse_held != 0 {
                mouse_held
            } else {
                key_held
            };
            let direction = self.preview.step_direction(
                ui.ctx(),
                held,
                if mouse_click != 0 {
                    mouse_click
                } else {
                    key_tap
                },
                Instant::now(),
            );
            if direction != 0 {
                self.preview.step(path, video, duration, direction);
                ui.ctx().request_repaint();
            }
            ui.weak(format!(
                "{} / {}",
                timestamp(self.preview.position),
                timestamp(duration)
            ));
        });
    }

    fn review_toolbar(&mut self, ui: &mut egui::Ui) {
        if let Some(review) = &self.review {
            if ui
                .add_enabled(
                    !review.busy(),
                    egui::Button::new("退出复核").min_size(egui::vec2(84.0, 0.0)),
                )
                .clicked()
            {
                self.select(self.selected);
            }
            return;
        }
        let Some(item) = self
            .selected
            .and_then(|i| self.saved.items.get(i))
            .filter(|item| item.info.is_some())
        else {
            return;
        };
        let task = item
            .task
            .as_ref()
            .filter(|_| task_state(item.task.as_ref()) == "已导出");
        let mut open = None;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.active.is_none(),
                    egui::Button::new("复核模式").min_size(egui::vec2(84.0, 0.0)),
                )
                .clicked()
            {
                open = Some(task.map_or_else(
                    || (item.path.clone(), None),
                    |task| (task.output.clone(), Some(task.job.clone())),
                ));
            }
            if task.is_some() && self.active.is_none() && ui.button("复核导入视频").clicked()
            {
                open = Some((item.path.clone(), None));
            }
        });
        if let Some((path, job)) = open {
            match review::Review::open(path, job) {
                Ok(review) => {
                    self.preview = Preview::default();
                    self.preview.request_media(
                        review.path(),
                        &review.video,
                        review.duration(),
                        false,
                        true,
                    );
                    self.review = Some(review);
                }
                Err(error) => self.notice = error,
            }
        }
    }

    fn list(&mut self, ui: &mut egui::Ui) {
        let busy = self.active.is_some() || self.review.as_ref().is_some_and(review::Review::busy);
        let mut add = false;
        let mut remove = false;
        ui.horizontal(|ui| {
            ui.heading("素材");
            ui.weak(self.saved.items.len().to_string());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                add = ui
                    .add_enabled(!busy, egui::Button::new("添加素材"))
                    .clicked();
                remove = ui
                    .add_enabled(!busy && self.selected.is_some(), egui::Button::new("移除"))
                    .clicked();
            });
        });
        if add {
            if let Some(files) = rfd::FileDialog::new()
                .add_filter("视频", &["mp4", "mkv", "mov", "avi", "webm", "m4v", "ts"])
                .pick_files()
            {
                self.add(files);
            }
        }
        if remove {
            if let Some(index) = self.selected {
                self.saved.items.remove(index);
                let new = self
                    .saved
                    .items
                    .len()
                    .checked_sub(1)
                    .map(|end| index.min(end));
                self.select(new);
            }
        }
        ui.add_space(5.0);
        let mut selected = None;
        egui::ScrollArea::vertical()
            .id_salt("items")
            .show(ui, |ui| {
                for (index, item) in self.saved.items.iter().enumerate() {
                    let active = self.active == Some(index);
                    if material_card(ui, item, self.selected == Some(index), active).clicked() {
                        selected = Some(index);
                    }
                }
            });
        if selected.is_some() && selected != self.selected {
            self.select(selected);
        }
    }

    fn footer(&mut self, ui: &mut egui::Ui) {
        let busy = self.active.is_some();
        ui.separator();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("导出位置"))
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                    self.saved.directory = Some(path);
                }
            }
            if let Some(path) = &self.saved.directory {
                if ui.button("打开目录").clicked() {
                    let _ = media::command("explorer.exe").arg(path).spawn();
                }
            }
            let directory = self
                .saved
                .directory
                .as_ref()
                .map_or("各素材所在目录".into(), |p| p.display().to_string());
            ui.add(egui::Label::new(RichText::new(directory).weak()).truncate());
        });
        ui.horizontal(|ui| {
            let width = if busy { 175.0 } else { ui.available_width() };
            if ui
                .add_enabled_ui(!busy && !self.saved.items.is_empty(), |ui| {
                    ui.add_sized(
                        [width, 31.0],
                        egui::Button::new(RichText::new("批量导出").strong())
                            .fill(Color32::from_rgb(39, 88, 163)),
                    )
                })
                .inner
                .clicked()
            {
                self.begin(None);
            }
            if busy
                && ui
                    .add_enabled(!self.stopping, egui::Button::new("暂停导出"))
                    .clicked()
            {
                self.request_pause();
            }
        });
        let status = self
            .active
            .map(|index| {
                let item = &self.saved.items[index];
                let status = if self.stopping {
                    "当前分块完成后暂停"
                } else {
                    &item.state
                };
                format!("{status} · 用时 {}", elapsed_label(item.elapsed_ms))
            })
            .unwrap_or_else(|| self.notice.clone());
        ui.add(egui::Label::new(RichText::new(&status).weak()).truncate())
            .on_hover_text(&status);
    }
}

fn windows_path(path: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(unc) = text.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{unc}"));
    }
    PathBuf::from(text.strip_prefix(r"\\?\").unwrap_or(&text))
}

fn filename(path: &Path) -> String {
    path.file_name()
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned()
}

fn task_state(task: Option<&Task>) -> &'static str {
    match task {
        Some(task) if task.output.is_file() && task.job.join("result.json").is_file() => "已导出",
        Some(task) if task.job.join("job.json").is_file() => "可续跑",
        _ => "待导出",
    }
}

fn timestamp(seconds: f64) -> String {
    let ms = (seconds.max(0.0) * 1000.0) as u64;
    format!("{:02}:{:02}.{:03}", ms / 60_000, ms / 1000 % 60, ms % 1000)
}

fn fps_label(fps: f64) -> String {
    if fps.fract() == 0.0 {
        format!("{fps:.0} fps")
    } else {
        format!("{fps:.3} fps")
    }
}

fn elapsed_label(ms: u64) -> String {
    let seconds = ms / 1000;
    if seconds >= 3600 {
        format!(
            "{:02}:{:02}:{:02}",
            seconds / 3600,
            seconds / 60 % 60,
            seconds % 60
        )
    } else {
        format!("{:02}:{:02}", seconds / 60, seconds % 60)
    }
}

fn material_card(ui: &mut egui::Ui, item: &Item, selected: bool, active: bool) -> egui::Response {
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 43.0), egui::Sense::click());
    let painter = ui.painter();
    let visuals = ui.visuals();
    let background = if selected {
        visuals.selection.bg_fill
    } else if response.hovered() {
        visuals.widgets.hovered.bg_fill
    } else {
        visuals.widgets.inactive.bg_fill
    };
    painter.rect_filled(rect, 4, background);
    let fraction = item.progress.clamp(0.0, 1.0);
    if active && fraction > 0.0 {
        let fill =
            egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * fraction, rect.height()));
        painter
            .with_clip_rect(fill)
            .rect_filled(rect, 4, Color32::from_rgb(27, 105, 57));
    }
    if selected {
        painter.rect_stroke(rect, 4, visuals.selection.stroke, egui::StrokeKind::Inside);
    }
    let name = filename(&item.path);
    let caption = if active {
        format!("{}%", (fraction * 100.0).floor() as u32)
    } else if item.state == "已导出" && item.elapsed_ms > 0 {
        format!("总用时 {}", elapsed_label(item.elapsed_ms))
    } else {
        String::new()
    };
    let text_area = rect.shrink2(egui::vec2(9.0, 3.0));
    let font = egui::FontId::proportional(14.0);
    let title_y = if active {
        text_area.center().y
    } else {
        text_area.top() + 8.0
    };
    let caption_rect = painter.text(
        egui::pos2(text_area.right(), title_y),
        egui::Align2::RIGHT_CENTER,
        &caption,
        font.clone(),
        Color32::WHITE,
    );
    let right = if caption.is_empty() {
        text_area.right()
    } else {
        caption_rect.left() - 10.0
    };
    let clip = egui::Rect::from_min_max(text_area.min, egui::pos2(right, text_area.bottom()));
    painter.with_clip_rect(clip).text(
        egui::pos2(text_area.left(), title_y),
        egui::Align2::LEFT_CENTER,
        name,
        font,
        Color32::WHITE,
    );
    if !active {
        let meta = item.info.as_ref().map_or_else(
            || item.state.clone(),
            |(video, duration)| {
                format!(
                    "{}×{} · {} · {} · {}",
                    video.width,
                    video.height,
                    video
                        .reported_fps()
                        .map_or_else(|| "帧率未知".into(), fps_label),
                    timestamp(*duration),
                    item.state
                )
            },
        );
        painter.with_clip_rect(text_area).text(
            text_area.left_bottom(),
            egui::Align2::LEFT_BOTTOM,
            meta,
            egui::FontId::proportional(12.0),
            Color32::from_rgb(198, 211, 229),
        );
    }
    response.on_hover_text(item.path.display().to_string())
}

fn probe_async(path: PathBuf) -> mpsc::Receiver<ProbeResult> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(media::probe(&path));
    });
    rx
}

fn chooser(ui: &mut egui::Ui, key: &str, selected: &mut String, values: &[(&str, &str)]) {
    let name = values
        .iter()
        .find(|(value, _)| *value == selected)
        .map_or("选择", |(_, label)| label);
    egui::ComboBox::from_id_salt(key)
        .width(230.0)
        .selected_text(name)
        .show_ui(ui, |ui| {
            for &(value, label) in values {
                ui.selectable_value(selected, value.into(), label);
            }
        });
}

fn style(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    for file in ["msyh.ttc", "simhei.ttf"] {
        if let Ok(bytes) = std::fs::read(Path::new(r"C:\Windows\Fonts").join(file)) {
            fonts
                .font_data
                .insert("chinese".into(), egui::FontData::from_owned(bytes).into());
            for names in fonts.families.values_mut() {
                names.push("chinese".into());
            }
            break;
        }
    }
    ctx.set_fonts(fonts);
    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::dark();
    style.visuals.panel_fill = Color32::from_rgb(22, 27, 35);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(34, 43, 57);
    style.visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(34, 43, 57);
    style.visuals.widgets.inactive.fg_stroke.color = Color32::from_rgb(223, 231, 244);
    style.visuals.widgets.noninteractive.fg_stroke.color = Color32::from_rgb(214, 224, 238);
    style.visuals.override_text_color = Some(Color32::from_rgb(225, 232, 243));
    style.spacing.interact_size.y = 26.0;
    style.visuals.selection.bg_fill = Color32::from_rgb(44, 78, 129);
    style.spacing.item_spacing = egui::vec2(9.0, 7.0);
    ctx.set_style(style);
}

fn export_controls(
    ui: &mut egui::Ui,
    settings: &mut Settings,
    video: Option<&media::Video>,
    planned_fps: Option<f64>,
    per_item: bool,
) {
    ui.heading("超分");
    chooser(ui, "upscale", &mut settings.upscale, &export::UPSCALERS);
    ui.add_enabled_ui(!settings.upscale.is_empty(), |ui| {
        let mut resolution = if settings.custom_size {
            0
        } else {
            settings.height
        };
        let label = match resolution {
            1080 => "1080P",
            1440 => "2K · 1440P",
            2160 => "4K · 2160P",
            _ => "自定义",
        };
        egui::ComboBox::from_id_salt("resolution")
            .width(230.0)
            .selected_text(label)
            .show_ui(ui, |ui| {
                for (height, label) in [
                    (1080, "1080P"),
                    (1440, "2K · 1440P"),
                    (2160, "4K · 2160P"),
                    (0, "自定义"),
                ] {
                    ui.selectable_value(&mut resolution, height, label);
                }
            });
        settings.custom_size = !matches!(resolution, 1080 | 1440 | 2160);
        if settings.custom_size {
            ui.horizontal(|ui| {
                ui.label("宽");
                ui.add(
                    egui::DragValue::new(&mut settings.width)
                        .range(0..=8192)
                        .speed(2),
                );
                ui.label("高");
                ui.add(
                    egui::DragValue::new(&mut settings.height)
                        .range(2..=4320)
                        .speed(2),
                );
            });
            ui.weak("宽度设为 0 时按原片比例计算");
        } else {
            settings.height = resolution;
        }
        if let Some(video) = video {
            let (w, h) = settings.size(video);
            ui.weak(format!("{w} × {h}"));
        }
    });
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(4.0);
    ui.heading("插帧");
    chooser(
        ui,
        "interpolate",
        &mut settings.interpolate,
        &export::INTERPOLATORS,
    );
    if let Some(rate) = planned_fps {
        ui.weak(format!("导出帧率 · {}", fps_label(rate)));
    } else if !settings.interpolate.is_empty() {
        let mut fps = if settings.custom_fps {
            0.0
        } else {
            settings.fps
        };
        let label = if fps == 0.0 {
            "自定义帧率".into()
        } else {
            fps_label(fps)
        };
        egui::ComboBox::from_id_salt("target-fps")
            .width(230.0)
            .selected_text(label)
            .show_ui(ui, |ui| {
                for rate in [24.0, 30.0, 60.0, 120.0] {
                    ui.selectable_value(&mut fps, rate, fps_label(rate));
                }
                ui.selectable_value(&mut fps, 0.0, "自定义帧率");
            });
        settings.custom_fps = fps == 0.0;
        if settings.custom_fps {
            ui.add(
                egui::DragValue::new(&mut settings.fps)
                    .range(1.0..=240.0)
                    .max_decimals(6)
                    .speed(0.1)
                    .suffix(" fps"),
            );
        } else {
            settings.fps = fps;
        }
    }

    if matches!(settings.interpolate.as_str(), "gimm-vfi" | "rife") {
        ui.horizontal(|ui| {
            ui.label("光流尺度");
            chooser(
                ui,
                "scale",
                &mut settings.scale,
                &[("0.25", "0.25×"), ("0.5", "0.5×"), ("1.0", "1.0×")],
            );
        });
    }
    ui.add_space(6.0);
    ui.separator();
    ui.add_space(4.0);
    ui.heading("技术路径");
    ui.add_enabled_ui(
        !settings.upscale.is_empty() && !settings.interpolate.is_empty(),
        |ui| {
            egui::ComboBox::from_id_salt("order")
                .width(230.0)
                .selected_text(if settings.upscale_first {
                    "先超分 → 后插帧"
                } else {
                    "先插帧 → 后超分"
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut settings.upscale_first, false, "先插帧 → 后超分");
                    ui.selectable_value(&mut settings.upscale_first, true, "先超分 → 后插帧");
                });
        },
    );
    ui.label("输出格式");
    egui::ComboBox::from_id_salt("container")
        .width(230.0)
        .selected_text(if settings.mp4 {
            "H.264 · MP4"
        } else {
            "FFV1 · MKV"
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut settings.mp4, true, "H.264 · MP4");
            ui.selectable_value(&mut settings.mp4, false, "FFV1 · MKV");
        });
    if per_item {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            if ui.button("单独设置目录").clicked() {
                if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                    settings.directory = Some(dir);
                }
            }
            if settings.directory.is_some() && ui.button("使用批量目录").clicked() {
                settings.directory = None;
            }
        });
        if let Some(dir) = &settings.directory {
            ui.weak(dir.display().to_string());
        }
    }
}

fn player_keys(ctx: &egui::Context) -> (i8, i8, bool) {
    let keyboard = !ctx.memory(|m| m.any_popup_open())
        && ctx
            .memory(|m| m.focused())
            .is_none_or(|id| egui::TextEdit::load_state(ctx, id).is_none());
    if keyboard {
        ctx.input_mut(|input| {
            if !input.focused || !input.modifiers.is_none() {
                return (0, 0, false);
            }
            let held = i8::from(input.key_down(egui::Key::ArrowRight))
                - i8::from(input.key_down(egui::Key::ArrowLeft));
            let mut tap = 0;
            let mut toggle = false;
            input.events.retain(|event| match event {
                egui::Event::Key {
                    key,
                    pressed: true,
                    repeat,
                    ..
                } => match key {
                    egui::Key::ArrowLeft => {
                        if !repeat {
                            tap = -1;
                        }
                        false
                    }
                    egui::Key::ArrowRight => {
                        if !repeat {
                            tap = 1;
                        }
                        false
                    }
                    egui::Key::Space => {
                        toggle |= !repeat;
                        false
                    }
                    _ => true,
                },
                _ => true,
            });
            (held, tap, toggle)
        })
    } else {
        (0, 0, false)
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "framelab", &self.saved);
    }
    fn auto_save_interval(&self) -> Duration {
        Duration::from_secs(1)
    }
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.drain(ctx);
        let keys = if self.review.is_some()
            || self
                .selected
                .and_then(|i| self.saved.items.get(i))
                .is_some_and(|item| item.info.is_some())
        {
            player_keys(ctx)
        } else {
            (0, 0, false)
        };
        if self.review.as_ref().is_some_and(review::Review::busy)
            && ctx.input(|i| i.viewport().close_requested())
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.notice = "请等待成片复核结束。".into();
        }
        if self.active.is_some() && ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.notice = "请先暂停导出，当前分块完成后即可关闭。".into();
        }
        if self.active.is_none() && !self.review.as_ref().is_some_and(review::Review::busy) {
            let dropped = ctx.input(|i| {
                i.raw
                    .dropped_files
                    .iter()
                    .filter_map(|f| f.path.clone())
                    .collect::<Vec<_>>()
            });
            self.add(dropped);
        }
        egui::SidePanel::right("configuration")
            .resizable(false)
            .exact_width(345.0)
            .show(ctx, |ui| {
                egui::TopBottomPanel::bottom("export")
                    .exact_height(100.0)
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| self.footer(ui));
                let settings_height = ui.available_height() * 0.68;
                egui::TopBottomPanel::top("settings")
                    .exact_height(settings_height)
                    .frame(egui::Frame::NONE)
                    .show_inside(ui, |ui| {
                        egui::ScrollArea::vertical()
                            .id_salt("settings-scroll")
                            .show(ui, |ui| {
                                if !self.details.is_empty() {
                                    egui::CollapsingHeader::new("导出诊断").show(ui, |ui| {
                                        ui.monospace(&self.details);
                                    });
                                }
                                self.settings(ui);
                            });
                    });
                ui.separator();
                self.list(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.review_toolbar(ui);
            let reserve = if self.review.is_some() { 143.0 } else { 74.0 };
            self.video(ui, (ui.available_height() - reserve).max(120.0), keys);
            if let Some(review) = &mut self.review {
                let action = review.controls(ui, &self.repo, self.preview.displayed_time);
                match action {
                    review::Action::None => {}
                    review::Action::Reload => {
                        let playing = self.preview.clock.is_some();
                        self.preview.request_media(
                            review.path(),
                            &review.video,
                            review.duration(),
                            playing,
                            true,
                        );
                    }
                    review::Action::Seek(frame) => {
                        self.preview.position = frame as f64 / review.video.fps();
                        self.preview.request_media(
                            review.path(),
                            &review.video,
                            review.duration(),
                            false,
                            true,
                        );
                    }
                }
            }
        });
    }
}

#[cfg(not(feature = "portable"))]
fn repository() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("FRAMELAB_ROOT").map(PathBuf::from) {
        if path.join("run_video_lab.ps1").is_file() {
            return Some(path);
        }
    }
    let cwd = std::env::current_dir().ok()?;
    let exe = std::env::current_exe().ok()?;
    let build = Path::new(env!("CARGO_MANIFEST_DIR"));
    cwd.ancestors()
        .chain(exe.ancestors())
        .chain(build.ancestors())
        .find(|path| path.join("run_video_lab.ps1").is_file())
        .map(Path::to_owned)
}

fn main() -> eframe::Result {
    #[cfg(not(feature = "portable"))]
    let repo = repository().unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    #[cfg(feature = "portable")]
    let repo = {
        let verify = std::env::args_os()
            .nth(1)
            .is_some_and(|arg| arg == "--verify-runtime");
        match portable::prepare() {
            Ok(_) if verify => return Ok(()),
            Ok(path) => path,
            Err(error) => {
                if !verify {
                    portable::show_error(&error.to_string());
                }
                std::process::exit(1);
            }
        }
    };
    let (id, title) = if cfg!(feature = "portable") {
        ("FrameLab NVIDIA Edition", "帧工坊 FrameLab · NVIDIA")
    } else {
        ("FrameLab", "帧工坊 FrameLab")
    };
    let inputs = std::env::args_os().skip(1).map(PathBuf::from).collect();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_app_id(id)
            .with_inner_size([1220.0, 780.0])
            .with_min_inner_size([850.0, 610.0]),
        ..Default::default()
    };
    eframe::run_native(
        title,
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc, repo, inputs)))),
    )
}

#[cfg(test)]
mod default_settings_test {
    use super::*;

    #[test]
    fn playback_shortcuts_consume_the_keys_before_widgets() {
        let ctx = egui::Context::default();
        let keys = [egui::Key::Space, egui::Key::ArrowRight];
        let raw = egui::RawInput {
            focused: true,
            events: keys
                .into_iter()
                .map(|key| egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                })
                .collect(),
            ..Default::default()
        };
        let _ = ctx.run(raw, |ctx| {
            assert_eq!(player_keys(ctx), (1, 1, true));
            assert!(!ctx.input(|input| input.key_pressed(egui::Key::Space)));
            assert!(!ctx.input(|input| input.key_pressed(egui::Key::ArrowRight)));
        });
    }

    #[test]
    fn held_step_repeats_every_tenth_of_a_second_without_release_step() {
        let ctx = egui::Context::default();
        let mut preview = Preview::default();
        let start = Instant::now();
        let mut step = |ms, held, clicked| {
            preview.step_direction(&ctx, held, clicked, start + Duration::from_millis(ms))
        };
        assert_eq!(step(0, 1, 0), 1);
        assert_eq!(step(249, 1, 0), 0);
        assert_eq!(step(250, 1, 0), 1);
        assert_eq!(step(349, 1, 0), 0);
        assert_eq!(step(350, 1, 0), 1);
        assert_eq!(step(360, 0, 1), 0);
        assert_eq!(step(370, 0, -1), -1);
        assert_eq!(step(380, -1, 0), -1);
    }

    #[test]
    fn empty_queue_draws_default_export_controls() {
        let saved: Saved = serde_json::from_str(r#"{"items":[],"directory":null}"#).unwrap();
        assert_eq!(saved.export_defaults.interpolate, "nvidia-vfg");
        let mut app = App {
            repo: PathBuf::new(),
            saved,
            selected: None,
            preview: Preview::default(),
            queue: VecDeque::new(),
            active: None,
            tick: Instant::now(),
            worker: None,
            pause_tx: None,
            notice: String::new(),
            details: String::new(),
            stopping: false,
            review: None,
        };
        let ctx = egui::Context::default();
        let result = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| app.settings(ui));
        });
        let labels: Vec<_> = result
            .shapes
            .iter()
            .filter_map(|item| match &item.shape {
                egui::Shape::Text(text) => Some(text.galley.job.text.as_str()),
                _ => None,
            })
            .collect();
        for required in [
            "导出配置",
            "超分",
            "1080P",
            "插帧",
            "NVIDIA VFG High",
            "60 fps",
            "技术路径",
            "先插帧 → 后超分",
            "H.264 · MP4",
        ] {
            assert!(labels.contains(&required), "missing {required}: {labels:?}");
        }
    }
}

#[cfg(test)]
mod export_smoke_test {
    use super::*;

    #[test]
    #[ignore = "requires an NVIDIA GPU, FRAMELAB_QA_ROOT, FRAMELAB_QA_INPUT and FRAMELAB_QA_OUTPUT"]
    fn exports_nvidia_modes() {
        let path = |key| PathBuf::from(std::env::var_os(key).expect(key));
        let root = path("FRAMELAB_QA_ROOT");
        let source = path("FRAMELAB_QA_INPUT");
        let output = path("FRAMELAB_QA_OUTPUT");
        #[cfg(feature = "portable")]
        portable::configure(&root).expect("bundled environment");
        std::fs::create_dir_all(&output).unwrap();
        let (video, _) = media::probe(&source).expect("probe QA input");
        for (stem, upscaler, interpolator) in [
            ("vfg", "", "nvidia-vfg"),
            ("vsr", "nvidia-vsr", ""),
            ("chain", "nvidia-vsr", "nvidia-vfg"),
        ] {
            let task = Task {
                output: output.join(format!("{stem}.mp4")),
                job: output.join(format!("{stem}.mp4.job")),
            };
            let settings = Settings {
                upscale: upscaler.into(),
                interpolate: interpolator.into(),
                height: (video.height * 4 / 3) & !1,
                fps: video.fps() * 2.0,
                ..Default::default()
            };
            let (sender, receiver) = mpsc::channel();
            export::start(
                root.clone(),
                source.clone(),
                task.clone(),
                settings.args(&video).unwrap(),
                sender,
            );
            loop {
                match receiver
                    .recv_timeout(Duration::from_secs(90))
                    .expect("export activity")
                {
                    export::Message::Finished(result) => {
                        assert!(result.unwrap_or_else(|err| panic!("{stem}: {err}")));
                        break;
                    }
                    export::Message::Progress { .. } => {}
                    export::Message::PauseError(error) => panic!("{error}"),
                }
            }
            assert!(task.output.is_file());
            println!("Export validated: {stem}");
        }
    }
}
