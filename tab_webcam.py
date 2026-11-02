"""Module Webcam của 3G: tách nền trực tiếp, chụp PNG và ghi MP4.
Camera chạy trong trình duyệt; xử lý ảnh bằng U2Net-P trên CPU.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import cv2
import gradio as gr
from ui_3g import intro, step, help_box, note, flow
import numpy as np

from image_matting import composite_background, resize_for_demo
from processing import guided_filter_mask


# Thư mục lưu ảnh và video Webcam; danh sách hiệu ứng nền.
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "webcam"
MODES = [
    ("Làm mờ nền", "blur"),
    ("Đổi nền màu", "color"),
    ("Thay ảnh nền", "image"),
    ("Mask trắng/đen", "mask"),
    ("Ảnh gốc (đối chiếu)", "original"),
]


# Chuẩn hóa màu do ColorPicker trả về thành mã HEX.
def _color_hex(value: Any) -> str:
    """ColorPicker Gradio 5 returns hex or CSS rgba(float channels)."""
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        components = [float(v) for v in value[:3]]
    elif isinstance(value, str):
        v = value.strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            return v.upper()
        if re.fullmatch(r"#[0-9a-fA-F]{3}", v):
            return "#" + "".join(k * 2 for k in v[1:]).upper()
        match = re.fullmatch(r"rgba?\s*\(([^)]+)\)", v, re.I)
        if not match:
            return "#E6F0FA"
        raw = [x.strip() for x in match.group(1).split(",")]
        if len(raw) < 3:
            return "#E6F0FA"
        try:
            components = [float(k) for k in raw[:3]]
        except ValueError:
            return "#E6F0FA"
    else:
        return "#E6F0FA"
    if not all(math.isfinite(x) for x in components):
        return "#E6F0FA"
    return "#" + "".join(f"{int(np.clip(round(x), 0, 255)):02X}" for x in components)


# Tăng tương phản cho ảnh đầu vào AI khi webcam thiếu sáng.
def _make_ai_frame(frame_bgr: np.ndarray, adjust_dark: bool) -> tuple[np.ndarray, bool]:
    if not adjust_dark:
        return frame_bgr, False
    # Only brighten the frame passed into AI. Output keeps original colours.
    small = cv2.resize(frame_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    mean_luma = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())
    if mean_luma >= 82.0:
        return frame_bgr, False
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced, True


# Record processed preview frames, not the raw webcam. Keep cv2.VideoWriter objects
# in the server registry: they cannot be safely stored inside Gradio State.
# Quản lý các phiên ghi và bộ nhớ khung hình đã xử lý.
MAX_RECORD_SECONDS = 180.0
_RECORDINGS: dict[str, "_WebcamRecording"] = {}
_LIVE_FRAMES: dict[str, tuple[float, np.ndarray]] = {}
_RECORDING_LOCK = threading.RLock()


# Quản lý một phiên ghi video Webcam đã qua xử lý AI.
class _WebcamRecording:
    # Tạo phiên ghi video và chuẩn bị các biến lưu khung hình.
    def __init__(self, fps: int):
        self.fps = max(5, min(20, int(fps)))
        self.token = uuid.uuid4().hex[:12]
        self.raw_path = OUT_DIR / f"webcam_{self.token}_raw.mp4"
        self.final_path = OUT_DIR / f"webcam_{self.token}.mp4"
        self.start_time = time.monotonic()
        self.first_frame_time: float | None = None
        self.writer: cv2.VideoWriter | None = None
        self.last_bgr: np.ndarray | None = None
        self.written = 0
        self.captured = 0
        self.capped = False

    # Ghi khung hình đã xử lý, bổ sung khung lặp nếu cần giữ thời lượng video.
    def push(self, image_rgb: np.ndarray, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        if now - self.start_time >= MAX_RECORD_SECONDS:
            self.capped = True
            return
        bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
        if self.writer is None:
            h, w = bgr.shape[:2]
            # H.264 yuv420p needs even dimensions.
            w -= w % 2
            h -= h % 2
            if min(w, h) < 2:
                raise RuntimeError("Kich thuoc webcam qua nho de ghi video.")
            bgr = bgr[:h, :w].copy()
            self.writer = cv2.VideoWriter(str(self.raw_path),
                                          cv2.VideoWriter_fourcc(*"mp4v"),
                                          float(self.fps), (w, h))
            if not self.writer.isOpened():
                self.writer = None
                raise RuntimeError("Khong mo duoc VideoWriter MP4. Kiem tra OpenCV.")
            self.first_frame_time = now
        else:
            target_h, target_w = self.last_bgr.shape[:2]
            if bgr.shape[:2] != (target_h, target_w):
                # Changing the webcam resolution mid-recording must not corrupt MP4.
                bgr = _resize_letterbox(bgr, target_w, target_h)
        assert self.writer is not None
        assert self.first_frame_time is not None
        target_count = int((now - self.first_frame_time) * self.fps) + 1
        # Bound a long browser stall: a brief freeze is preferable to writing
        # thousands of duplicate frames after a tab sleeps for several minutes.
        target_count = min(target_count, self.written + 2 * self.fps + 1)
        while self.last_bgr is not None and self.written < target_count - 1:
            self.writer.write(self.last_bgr)
            self.written += 1
        self.writer.write(bgr)
        self.written += 1
        self.captured += 1
        self.last_bgr = bgr

    # Kết thúc ghi MP4 và chuyển sang H.264 nếu có FFmpeg.
    def finish(self) -> tuple[Path | None, str]:
        if self.writer is None or self.written == 0:
            return None, "Chưa nhận được khung hình đã xử lý; hãy bật Webcam rồi ghi lại."
        self.writer.release()
        self.writer = None
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except (ImportError, OSError, RuntimeError):
                pass
        encoded_path = self.raw_path
        codec_name = "mp4v (cần FFmpeg để phát trên Chrome)"
        if ffmpeg_exe:
            command = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-y",
                       "-i", str(self.raw_path), "-an", "-c:v", "libx264",
                       "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", str(self.final_path)]
            try:
                proc = subprocess.run(command, capture_output=True, text=True, timeout=180)
                if proc.returncode == 0 and self.final_path.is_file() and self.final_path.stat().st_size > 500:
                    encoded_path = self.final_path
                    codec_name = "H.264"
                    self.raw_path.unlink(missing_ok=True)
                else:
                    self.final_path.unlink(missing_ok=True)
            except (OSError, subprocess.TimeoutExpired):
                self.final_path.unlink(missing_ok=True)
        duration = self.written / self.fps
        status = (f"✅ Đã ghi **{self.captured} khung Webcam thực nhận**, lưu **{self.written} "
                  f"khung** ở {self.fps} FPS (bù khung để giữ thời lượng), "
                  f"**{duration:.1f} giây**, codec **{codec_name}**.")
        if self.capped:
            status += " Đã chạm giới hạn 180 giây."
        if codec_name != "H.264":
            status += (" Để xuất H.264, chạy một lần: "
                       r"`.\.venv\Scripts\python.exe -m pip install imageio-ffmpeg==0.6.0`.")
        return encoded_path, status


# Đổi kích thước khung hình giữ tỷ lệ để không làm méo video.
def _resize_letterbox(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    old_h, old_w = frame.shape[:2]
    scale = min(w / old_w, h / old_h)
    new_w, new_h = max(1, int(old_w * scale)), max(1, int(old_h * scale))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    top, left = (h - new_h) // 2, (w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = small
    return canvas


# Lấy mã phiên Webcam để quản lý ghi hình riêng cho từng người dùng.
def _request_key(request: gr.Request | None = None, session_id: str | None = None) -> str:
    # An explicit per-browser gr.State ID prevents mismatched Gradio request
    # session hashes between webcam.stream and record button callbacks.
    if isinstance(session_id, str) and len(session_id) >= 10:
        return session_id
    key = getattr(request, "session_hash", None)
    if not key:
        raise gr.Error("Không tìm được phiên Webcam. Hãy tải lại trang và thử lại.")
    return str(key)


# Bắt đầu ghi MP4 từ khung hình đã được AI xử lý.
def start_webcam_recording(fps: int, session_id: str, request: gr.Request | None = None):
    """Start recording only when an actual processed webcam frame is available."""
    key = _request_key(request, session_id)
    with _RECORDING_LOCK:
        if key in _RECORDINGS:
            return (None, "🔴 **Đang ghi.** Hãy dùng nút **Dừng ghi và lưu MP4**.",
                    gr.update(interactive=False), gr.update(interactive=True))
        live = _LIVE_FRAMES.get(key)
        if live is None or time.monotonic() - live[0] > 8.0:
            return (None, "⚠️ **Chưa nhận được ảnh đã tách nền từ webcam.** "
                    "Hãy bật camera (nút Start trong khung bên trái), đợi hình hiện ở bên phải "
                    "rồi bấm **Bắt đầu ghi MP4**.",
                    gr.update(interactive=True), gr.update(interactive=False))
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        rec = _WebcamRecording(int(fps))
        try:
            # Start immediately with the last processed frame, even when the
            # browser's next frame is delayed by a slow CPU model.
            rec.push(live[1])
        except Exception as exc:
            return (None, f"⚠️ Không khởi tạo ghi MP4: {type(exc).__name__}: {exc}",
                    gr.update(interactive=True), gr.update(interactive=False))
        _RECORDINGS[key] = rec
        print(f"[3G Webcam] RECORD START key={key[:8]} fps={rec.fps}", flush=True)
    return (None, "🔴 **ĐANG GHI VIDEO ĐÃ TÁCH NỀN.** "
            "Giữ camera bật, sau đó nhấn **Dừng ghi và lưu MP4** (không nhấn Stop camera).",
            gr.update(interactive=False), gr.update(interactive=True))


# Dừng ghi và trả đường dẫn MP4 để xem hoặc tải xuống.
def stop_webcam_recording(session_id: str, request: gr.Request | None = None):
    key = _request_key(request, session_id)
    with _RECORDING_LOCK:
        rec = _RECORDINGS.pop(key, None)
    if rec is None:
        return (None, "⚠️ Chưa bắt đầu ghi. Hãy nhấn **Bắt đầu ghi MP4** trước.",
                gr.update(interactive=True), gr.update(interactive=False))
    try:
        path, message = rec.finish()
        print(f"[3G Webcam] RECORD STOP key={key[:8]} frames={rec.written} path={path}", flush=True)
    except Exception as exc:
        return (None, f"⚠️ Không kết thúc được video: {type(exc).__name__}: {exc}",
                gr.update(interactive=True), gr.update(interactive=False))
    if path is None:
        return (None, message, gr.update(interactive=True), gr.update(interactive=False))
    return (str(path), message + " **MP4 đã xuất hiện ở bên dưới để tải về.**",
            gr.update(interactive=True), gr.update(interactive=False))


# Lưu khung Webcam mới nhất và đưa vào bản ghi nếu đang ghi.
def _write_record_frame(session_id: str, frame: np.ndarray) -> str:
    if not isinstance(session_id, str) or len(session_id) < 10:
        return ""
    with _RECORDING_LOCK:
        now = time.monotonic()
        _LIVE_FRAMES[session_id] = (now, np.ascontiguousarray(frame).copy())
        # Remove abandoned webcam sessions after one hour.
        if len(_LIVE_FRAMES) > 30:
            stale = [sid for sid, (stamp, _) in _LIVE_FRAMES.items()
                     if now - stamp > 3600 and sid not in _RECORDINGS]
            for sid in stale:
                del _LIVE_FRAMES[sid]
        rec = _RECORDINGS.get(session_id)
        if rec is None:
            return ""
        try:
            rec.push(frame, now=now)
        except Exception as exc:
            return f" ⚠️ **Lỗi ghi video:** {type(exc).__name__}: {exc}"
        return (f" · 🔴 **ĐANG GHI** {rec.written / rec.fps:.1f}s "
                f"({rec.captured} khung nhận)"
                + (" · Đã đạt giới hạn" if rec.capped else ""))


# Dùng Optical Flow ước lượng vị trí mask ở khung hình tiếp theo.
def _flow_warp_mask(prev_mask: np.ndarray, prev_gray: np.ndarray,
                    current_gray: np.ndarray, out_shape: tuple[int, int],
                    *, return_quality: bool = False):
    """Warp old alpha to current frame, with cheap photometric-flow confidence.

    The flow maps each current pixel to its previous-frame location. Black
    padding is intentional: replicated-border alpha can cause a subject to
    smear across the frame edge when leaving the camera view.
    """
    small_h, small_w = current_gray.shape
    backward = cv2.calcOpticalFlowFarneback(
        current_gray, prev_gray, None, pyr_scale=0.5, levels=2,
        winsize=13, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
    yy, xx = np.mgrid[0:small_h, 0:small_w].astype(np.float32)
    map_x = xx + backward[:, :, 0]
    map_y = yy + backward[:, :, 1]
    prev_small = cv2.resize(np.asarray(prev_mask, dtype=np.float32),
                            (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    warped = cv2.remap(prev_small, map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)
    h, w = out_shape
    output = np.clip(cv2.resize(warped, (w, h),
                                interpolation=cv2.INTER_LINEAR), 0, 1)
    if not return_quality:
        return output
    prev_aligned = cv2.remap(prev_gray, map_x, map_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Compare in the old subject area only; the entire image may move.
    subject_pixels = warped > 0.22
    if int(subject_pixels.sum()) >= 30:
        flow_error = float(np.mean(cv2.absdiff(prev_aligned, current_gray)[subject_pixels]))
        flow_motion = float(np.percentile(
            np.sqrt(backward[:, :, 0][subject_pixels] ** 2 +
                    backward[:, :, 1][subject_pixels] ** 2), 90))
    else:
        flow_error, flow_motion = 100.0, 100.0
    old_area = max(1e-4, float(np.mean(prev_small > 0.5)))
    new_area = float(np.mean(warped > 0.5))
    area_change = abs(new_area - old_area) / old_area
    return output, {"residual": flow_error,
                    "motion_p90": flow_motion, "area_change": area_change}


# Giảm quầng mờ ở viền mà vẫn giữ chuyển tiếp alpha mềm.
def _refine_webcam_edge(mask: np.ndarray, strength: int) -> np.ndarray:
    """Reduce the faint, broad transparent halo without hard-binarizing hair.

    Remapping is deliberately applied only to the displayed alpha. Keeping
    the original soft mask in state avoids progressively eroding hair with
    each consecutive Optical Flow frame.
    """
    amount = float(np.clip(int(strength), 0, 80)) / 80.0
    if amount <= 0:
        return np.asarray(mask, dtype=np.float32)
    src = np.clip(mask.astype(np.float32), 0, 1)
    low = 0.02 + amount * 0.11
    high = 0.98 - amount * 0.09
    stretched = np.clip((src - low) / (high - low), 0, 1)
    # Smoothstep retains soft transitions while suppressing background spill.
    smoothstep = stretched * stretched * (3.0 - 2.0 * stretched)
    return np.clip((1 - 0.55 * amount) * src + 0.55 * amount * smoothstep,
                   0, 1).astype(np.float32)


# Xử lý một frame: quyết định chạy AI/Flow, tinh chỉnh mask và ghép nền.
def process_webcam_frame(
    webcam_rgb: np.ndarray | None,
    mode: str,
    max_side: int,
    guided: bool,
    smooth: bool,
    low_light: bool,
    color_value: Any,
    bg_rgb: np.ndarray | None,
    blur_strength: int,
    paused: bool,
    state: dict | None,
    ai_stride: int = 2,
    motion_ai: bool = True,
    edge_refine: bool = True,
    edge_strength: int = 45,
    *,
    get_model: Callable[[], Any],
    model_lock: Any,
) -> tuple[np.ndarray | None, str, dict | None]:
    """Pure-ish per-frame callback; state is per Gradio browser session."""
    if webcam_rgb is None:
        return None, "Chưa có hình từ webcam. Hãy cấp quyền camera trong Chrome.", None
    if paused:
        output = state.get("last_rgb") if isinstance(state, dict) else webcam_rgb
        return output, "⏸ Đang tạm dừng xử lý (camera trên trình duyệt vẫn có thể đang mở).", state

    started = time.perf_counter()
    image = np.asarray(webcam_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise gr.Error("Webcam cần hình RGB hợp lệ.")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    bgr = resize_for_demo(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), max_side=int(max_side))
    h, w = bgr.shape[:2]
    if mode == "original":
        result_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elapsed = time.perf_counter() - started
        return result_rgb, f"Ảnh gốc {w}×{h}; không suy luận AI. {elapsed * 1000:.1f} ms/frame.", {
            "prev_mask": None,
            "last_rgb": result_rgb,
            "shape": (h, w),
            "mode": mode,
            "frames": 0,
        }

    # Tối ưu CPU: xen kẽ dự đoán AI và theo dõi mask bằng Optical Flow.
    stride = max(1, min(4, int(ai_stride)))
    old_state = state if isinstance(state, dict) else {}
    prev = old_state.get("prev_mask")
    prev_gray = old_state.get("prev_gray")
    # Smaller motion image keeps CPU flow cheap compared with AI inference.
    small_scale = min(1.0, 224.0 / max(w, h))
    gray = cv2.cvtColor(
        cv2.resize(bgr, (max(64, round(w * small_scale)),
                         max(64, round(h * small_scale))),
                   interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    seen = int(old_state.get("frames", 0)) + 1
    has_prev = (isinstance(prev, np.ndarray) and prev.shape == (h, w)
                and isinstance(prev_gray, np.ndarray) and prev_gray.shape == gray.shape
                and old_state.get("mode") == mode)
    # If background illumination changes dramatically, refresh the AI too.
    change = float(cv2.absdiff(gray, prev_gray).mean()) if has_prev else 255.0
    active_fraction = (float(np.mean(cv2.absdiff(gray, prev_gray) > 26))
                       if has_prev else 1.0)
    high_motion = bool(motion_ai) and has_prev and (change >= 16.0 or active_fraction >= 0.30)
    scene_change = has_prev and (change >= 44.0 or active_fraction >= 0.69)
    need_ai = (not has_prev or stride == 1 or (seen - 1) % stride == 0
               or high_motion or scene_change)
    boosted = False
    flow_quality = None
    flow_rejected = False
    if not need_ai:
        try:
            mask, flow_quality = _flow_warp_mask(
                prev, prev_gray, gray, (h, w), return_quality=True)
            # Reject obvious optical-flow ghosts: blurry / untrackable
            # silhouettes should be replaced with a fresh AI prediction.
            if bool(motion_ai) and (flow_quality["residual"] > 24.0
                                    or flow_quality["motion_p90"] > 24.0
                                    or flow_quality["area_change"] > 0.27):
                need_ai = True
                flow_rejected = True
        except (cv2.error, ValueError, TypeError):
            need_ai = True
            flow_rejected = True
    # Khi có chuyển động mạnh hoặc Flow không tin cậy, chạy lại U2Net-P.
    if need_ai:
        ai_bgr, boosted = _make_ai_frame(bgr, bool(low_light))
        try:
            with model_lock:
                predicted = get_model().predict_mask(ai_bgr)
        except Exception as exc:
            raise gr.Error(f"Webcam: không chạy được AI: {type(exc).__name__}: {exc}") from exc
        mask = np.clip(np.asarray(predicted, dtype=np.float32), 0, 1)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if guided:
            # Small radius avoids excessive feathering/halos.
            mask = guided_filter_mask(bgr, mask, radius=3, eps=0.0007)
        mask = np.clip(mask, 0, 1).astype(np.float32)
        # Do not blend an unaligned old mask into a moving subject. This was
        # the main source of ghosting in v4.8.
        if (smooth and has_prev and not high_motion and not scene_change
                and change < 5.5):
            disagreement = float(np.mean(np.abs(mask - prev.astype(np.float32))))
            if disagreement < 0.10:
                mask = np.clip(0.92 * mask + 0.08 * prev.astype(np.float32), 0, 1)

    # Giữ mask gốc cho lần theo dõi tiếp theo, chỉ tinh chỉnh viền trên ảnh hiển thị.
    # Preserve this UN-SHARPENED alpha for tracking. Otherwise repeated
    # edge-remapping during skipped AI frames slowly deletes thin details.
    tracking_mask = np.asarray(mask, dtype=np.float32)
    display_mask = (_refine_webcam_edge(tracking_mask, edge_strength)
                    if edge_refine else tracking_mask)

    if mode == "mask":
        mask_u8 = np.uint8(np.clip(np.rint(display_mask * 255), 0, 255))
        out_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    else:
        kwargs: dict[str, Any] = {}
        if mode == "color":
            kwargs["bg_color"] = _color_hex(color_value)
        elif mode == "image":
            if bg_rgb is None:
                # Non-error fallback: user can upload an image later.
                kwargs["bg_color"] = "#E6F0FA"
                mode_for_comp = "color"
            else:
                bg_np = np.asarray(bg_rgb)
                if bg_np.dtype != np.uint8:
                    bg_np = np.clip(bg_np, 0, 255).astype(np.uint8)
                if bg_np.ndim == 2:
                    bg_np = cv2.cvtColor(bg_np, cv2.COLOR_GRAY2RGB)
                if bg_np.shape[2] == 4:
                    bg_np = cv2.cvtColor(bg_np, cv2.COLOR_RGBA2RGB)
                kwargs["bg_image_bgr"] = cv2.cvtColor(bg_np, cv2.COLOR_RGB2BGR)
                mode_for_comp = "image"
        elif mode == "blur":
            kwargs["blur_sigma"] = float(blur_strength)
        if mode != "image":
            mode_for_comp = mode
        out_bgr = composite_background(bgr, display_mask, mode=mode_for_comp, **kwargs)

    result_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
    dt = time.perf_counter() - started
    ended = time.perf_counter()
    prev_end = old_state.get("last_end", None)
    fps_actual = (1.0 / max(ended - prev_end, 1e-4)) if isinstance(prev_end, (int, float)) else 0.0
    fps_rolling = (0.25 * fps_actual + 0.75 * float(old_state.get("fps_rolling", fps_actual))
                   if fps_actual else 0.0)
    label = "Có (AI tăng sáng)" if boosted else "Không"
    extra = " · Hãy tải ảnh nền bên phải" if mode == "image" and bg_rgb is None else ""
    if scene_change:
        mode_ai = "AI (đổi cảnh)"
    elif high_motion:
        mode_ai = "AI (chuyển động nhanh)"
    elif flow_rejected:
        mode_ai = "AI (Flow kém tin cậy)"
    else:
        mode_ai = "AI" if need_ai else "Optical Flow"
    info = (f"**Webcam CPU v4.9:** {w}×{h} · **{dt*1000:.0f} ms/frame** "
            f"· **~{fps_rolling:.1f} FPS nhận thực tế** (sau lần đầu) "
            f"· Frame {seen} ({mode_ai}) · AI đã chạy: "
            f"{int(old_state.get('ai_frames', 0)) + int(need_ai)} "
            f"· Chuyển động Δ: {change:.1f} · Tăng sáng: {label}{extra}")
    new_state = {
        "prev_mask": tracking_mask.astype(np.float16),
        "prev_gray": gray,
        "last_rgb": result_rgb,
        "shape": (h, w),
        "mode": mode,
        "frames": seen,
        "ai_frames": int(old_state.get("ai_frames", 0)) + int(need_ai),
        "last_end": ended,
        "fps_rolling": fps_rolling,
    }
    return result_rgb, info, new_state


# Lưu kết quả Webcam hiện tại thành ảnh PNG.
def save_webcam_photo(state: dict | None) -> tuple[str | None, str]:
    if not isinstance(state, dict) or not isinstance(state.get("last_rgb"), np.ndarray):
        return None, "⚠️ Chưa có kết quả. Hãy bật webcam và xử lý ít nhất một khung hình."
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    photo = state["last_rgb"]
    out = OUT_DIR / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.png"
    ok = cv2.imwrite(str(out), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))
    if not ok:
        return None, "⚠️ Không ghi được ảnh PNG."
    return str(out), f"✅ Đã lưu ảnh: `{out.name}`"


# Tạo giao diện Webcam, liên kết camera với các nút chụp/ghi video.
def add_webcam_tab(*, get_model: Callable[[], Any], model_lock: Any) -> None:
    # V5.2 chi cai thien bo cuc va dau ra ghi hinh; thuat toan v4.9 giu nguyen.
    with gr.Column(visible=False,elem_classes=['g3-page'],elem_id='g3-webcam-page') as webcam_page:
        gr.HTML(intro('Webcam — Trực tiếp & ghi hình',
            'Xem ảnh gốc và kết quả cạnh nhau. Ghi MP4 hoặc chụp PNG ngay dưới camera.'))
        session = gr.State(None)
        record_key = gr.State(value=lambda: uuid.uuid4().hex)

        # Các tùy chọn hiệu ứng, độ phân giải và tần suất suy luận AI.
        with gr.Group(elem_classes=['g3-card']):
          with gr.Row():
            mode = gr.Dropdown(choices=MODES,value='blur',label='Hiệu ứng nền',scale=3)
            resolution = gr.Dropdown(
                choices=[('320 · nhẹ CPU',320),('480 · cân bằng',480),('640 · rõ hơn',640)],
                value=480,label='Độ phân giải',scale=2)
            ai_stride = gr.Dropdown(
                choices=[('AI mỗi khung',1),('AI mỗi 2 khung',2),('AI mỗi 3 khung',3)],
                value=2,label='Tần suất AI',scale=2)
            paused = gr.Checkbox(label='Tạm dừng AI',value=False,scale=1)
        with gr.Row():
            with gr.Column(visible=False) as color_column:
                color = gr.ColorPicker(label='Màu nền',value='#E6F0FA')
            with gr.Column(visible=False) as bg_column:
                background = gr.Image(label='Chọn ảnh nền',sources=['upload'],type='numpy',image_mode='RGB',height=160)
            with gr.Column(visible=True) as blur_column:
                blur = gr.Slider(2,35,value=14,step=1,label='Độ mờ nền')
        with gr.Accordion('Tùy chọn chất lượng (chỉ chỉnh khi cần)',open=False,elem_classes=['aetm-accordion']):
            with gr.Row():
                guided = gr.Checkbox(label='Guided Filter',value=True)
                smooth = gr.Checkbox(label='Ổn định mask',value=True)
                low_light = gr.Checkbox(label='Xử lý phòng tối',value=True)
            with gr.Row():
                motion_ai = gr.Checkbox(label='AI cập nhật khi chuyển động nhanh',value=True)
                edge_refine = gr.Checkbox(label='Giảm quầng viền',value=True)
            edge_strength = gr.Slider(0,80,value=45,step=5,label='Mức giảm quầng viền')

        gr.HTML('<div class="aetm-section-sep" id="aetm-webcam-live">Camera &amp; kết quả trực tiếp</div>')
        gr.HTML(note('Trong khung Camera, nút Record/Stop của trình duyệt chỉ BẬT/TẮT CAMERA. Ghi file MP4 bằng hai nút màu bên dưới — không cần dùng nút Stop của camera để lưu.'))
        with gr.Row(equal_height=False,elem_classes=['g3-two-up']):
            webcam = gr.Image(label='Camera gốc · bật camera tại đây',sources=['webcam'],
                streaming=True,type='numpy',image_mode='RGB',height=285)
            output = gr.Image(label='Kết quả AI trực tiếp',type='numpy',
                image_mode='RGB',interactive=False,height=285)
        status = gr.Markdown('**Camera chưa bật.** Cho phép sử dụng camera khi Chrome hỏi.',
                             elem_classes=['aetm-recording-status'])

        # Các nút chụp PNG, bắt đầu/dừng ghi video MP4.
        with gr.Group(elem_classes=['aetm-highlight']):
            gr.HTML('<div class="aetm-section-sep">Ghi MP4 / Chụp ảnh</div>')
            with gr.Row():
                record_start = gr.Button('●  BẮT ĐẦU GHI MP4',variant='primary',scale=4,
                                          elem_classes=['aetm-record'])
                record_stop = gr.Button('■  DỪNG GHI & XEM LẠI',interactive=False,scale=4,
                                         elem_classes=['aetm-record'])
                photo = gr.Button('📷  Chụp PNG',scale=2,elem_classes=['aetm-secondary'])
                reset = gr.Button('↻  Reset mask',scale=1,elem_classes=['aetm-secondary'])
            with gr.Row():
                record_fps = gr.Dropdown(choices=[8,10,12,15],value=10,
                    label='FPS phát file MP4 · không phải FPS AI',scale=2)
                record_status = gr.Markdown('**Sẵn sàng.** Bật camera, xem khung đã tách rồi nhấn Bắt đầu ghi.',
                                            elem_classes=['aetm-recording-status'])
            # Khong tao khung video rong truoc khi ghi: chi hien sau khi co MP4.
            recorded_video = gr.Video(label='Xem lại & tải video MP4 đã ghi',format='mp4',
                                       height=290,visible=False,elem_id='aetm-webcam-recorded')
            with gr.Row():
                snapshot = gr.File(label='Tải PNG đã chụp',visible=False)
                photo_info = gr.Markdown('')
            gr.HTML(note('Video chỉ ghi các khung đã tách nền và chưa thu âm. Nếu CPU chậm, một số khung có thể được lặp để giữ thời lượng phát đúng.'))

        # Chỉ hiển thị điều khiển phù hợp với hiệu ứng nền được chọn.
        def update_controls(selected: str):
            return (gr.update(visible=selected == 'color'),
                    gr.update(visible=selected == 'image'),
                    gr.update(visible=selected == 'blur'))
        mode.change(update_controls,inputs=mode,outputs=[color_column,bg_column,blur_column],queue=False)
        inputs = [webcam,mode,resolution,guided,smooth,low_light,
                  color,background,blur,paused,session,ai_stride,
                  motion_ai,edge_refine,edge_strength,record_key]

        # Kết nối luồng webcam đến AI và hệ thống ghi hình.
        def callback(webcam_rgb, selected_mode, max_side, use_guided, use_smooth,
                     use_low_light, bg_color, bg_image, blur_strength, is_paused,
                     state_value, stride_value, use_motion_ai, use_edge_refine,
                     edge_strength_value, session_id: str):
            frame, info, next_state = process_webcam_frame(
                webcam_rgb, selected_mode, max_side, use_guided, use_smooth,
                use_low_light, bg_color, bg_image, blur_strength, is_paused,
                state_value, stride_value, use_motion_ai, use_edge_refine,
                edge_strength_value, get_model=get_model, model_lock=model_lock)
            if isinstance(frame, np.ndarray):
                info += _write_record_frame(session_id, frame)
            return frame, info, next_state

        # Liên tục nhận khung hình Camera và gọi hàm xử lý.
        webcam.stream(callback,inputs=inputs,outputs=[output,status,session],
            stream_every=.20,concurrency_limit=1,show_progress='hidden')

        # Cập nhật trạng thái nút khi bắt đầu ghi MP4.
        def start_record_ui(fps:int,session_id:str,request:gr.Request|None=None):
            _, msg, start_btn, stop_btn = start_webcam_recording(fps,session_id,request)
            return gr.update(value=None,visible=False),msg,start_btn,stop_btn

        # Cập nhật giao diện và hiển thị MP4 sau khi dừng ghi.
        def stop_record_ui(session_id:str,request:gr.Request|None=None):
            path,msg,start_btn,stop_btn = stop_webcam_recording(session_id,request)
            return gr.update(value=path,visible=bool(path)),msg,start_btn,stop_btn

        # Dừng và lưu bản ghi nếu camera bị tắt trong lúc đang ghi.
        def camera_cleared(session_id:str):
            with _RECORDING_LOCK:
                _LIVE_FRAMES.pop(session_id,None)
                recording = session_id in _RECORDINGS
            if recording:
                path,info,start_button,stop_button = stop_webcam_recording(session_id)
                info += ' Camera đã dừng, bản ghi được lưu tự động.'
            else:
                path,info = None,'Camera đã dừng. Muốn ghi lần tiếp theo, bật camera rồi nhấn Bắt đầu ghi MP4.'
                start_button,stop_button=gr.update(interactive=True),gr.update(interactive=False)
            return (None,'Camera đã dừng.',None,
                    (gr.update(value=path,visible=True) if path else gr.update()),info,start_button,stop_button)

        webcam.clear(camera_cleared,inputs=[record_key],
            outputs=[output,status,session,recorded_video,record_status,record_start,record_stop],queue=False)
        reset.click(lambda:(None,'Đã đặt lại mask; tiếp tục dùng camera để nhận ảnh mới.'),
                    outputs=[session,status],queue=False)
        # Gọi hàm chụp PNG và cập nhật ô tải ảnh.
        def photo_ui(state):
            path,info=save_webcam_photo(state)
            return gr.update(value=path,visible=bool(path)),info
        photo.click(photo_ui,inputs=session,outputs=[snapshot,photo_info],queue=False)
        record_start.click(start_record_ui,inputs=[record_fps,record_key],
                           outputs=[recorded_video,record_status,record_start,record_stop],queue=False)
        stop_evt=record_stop.click(stop_record_ui,inputs=[record_key],
                           outputs=[recorded_video,record_status,record_start,record_stop],queue=False)
        stop_evt.then(fn=None,inputs=[],outputs=[],queue=False,
            js="() => { setTimeout(() => document.getElementById('aetm-webcam-recorded')?.scrollIntoView({behavior:'smooth',block:'start'}), 160); }")
        return webcam_page
