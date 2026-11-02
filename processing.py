"""processing.py - Cac ham xu ly cho du an 3G.

"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


# =========================
# Kiem tra du lieu dau vao
# =========================

# Kiểm tra ảnh đầu vào có đúng định dạng BGR 3 kênh hay không.
def _check_frame(frame_bgr: np.ndarray) -> None:
    """Kiem tra anh BGR truoc khi xu ly."""
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or min(frame_bgr.shape[:2]) < 2
    ):
        raise ValueError("Can anh BGR uint8, shape (H,W,3), moi chieu >= 2.")


# Kiểm tra kích thước mask và chuẩn hóa giá trị alpha trong khoảng 0–1.
def _check_mask(mask: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
    """Tra mask 2D float32, cung chieu va nam trong [0,1]."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask phai la mang 2 chieu (H,W).")
    if mask.shape != frame_bgr.shape[:2]:
        raise ValueError(f"Mask {mask.shape} khac kich thuoc anh {frame_bgr.shape[:2]}.")
    mask = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(mask).all():
        raise ValueError("Mask chua NaN hoac Inf.")
    return np.clip(mask, 0.0, 1.0)


# =========================
# Guided Filter
# =========================

# Tự tính Guided Filter trên ảnh xám khi OpenCV thiếu ximgproc.
def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Ban du phong cho anh xam neu OpenCV thieu ximgproc."""
    win = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, -1, win)
    mean_p = cv2.boxFilter(src, -1, win)
    var_i = cv2.boxFilter(guide * guide, -1, win) - mean_i * mean_i
    cov_ip = cv2.boxFilter(guide * src, -1, win) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, -1, win) * guide + cv2.boxFilter(b, -1, win)


# Làm mượt mask nhưng vẫn bám theo đường viền của ảnh gốc.
def guided_filter_mask(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
    eps: float = 0.001,
    color_guide: bool = False,
) -> np.ndarray:
    """Lam min mask theo bien anh; khong nguong nhi phan alpha.

    radius: ban kinh Guided Filter (nen thu 3..12 cho demo)
    eps: he so dieu hoa, guide duoc chuan hoa ve [0,1]
    color_guide: True -> guide 3 kenh, False -> guide xam
    """
    _check_frame(frame_bgr)
    src = _check_mask(mask, frame_bgr)
    if not 1 <= radius <= 100:
        raise ValueError("radius phai trong [1, 100].")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps phai > 0 va huu han.")

    guide = (
        frame_bgr.astype(np.float32) / 255.0
        if color_guide
        else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    )

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        result = cv2.ximgproc.guidedFilter(guide, src, radius, float(eps))
    else:
        if color_guide:
            warnings.warn(
                "Thieu cv2.ximgproc: tam dung guide xam thay guide mau.",
                RuntimeWarning,
                stacklevel=2,
            )
            guide = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        result = _guided_filter_gray(guide, src, radius, float(eps))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


# =========================
# Cai tien cho ANH TINH
# =========================
# Tập hợp các tham số tiền xử lý và tinh chỉnh mask ảnh tĩnh.
@dataclass
class ImageRefineConfig:
    """Tham so cai tien uu tien cho anh tinh."""

    # Tien xu ly truoc khi dua vao model
    use_clahe: bool = True
    use_gamma: bool = True
    gamma: float = 1.15
    fusion_mode: str = "meanmax"   # mean | max | meanmax

    # Hau xu ly mask
    use_guided_filter: bool = True
    radius: int = 6
    eps: float = 0.001
    color_guide: bool = False
    keep_largest: bool = True
    close_kernel: int = 7
    open_kernel: int = 3
    median_ksize: int = 5
    fg_threshold: float = 0.75
    bg_threshold: float = 0.08
    edge_width: int = 14
    alpha_gamma: float = 0.95
    hard_fg_boost: bool = True

    # Kiểm tra các tham số cấu hình để tránh giá trị không hợp lệ.
    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError("gamma phai > 0.")
        if self.fusion_mode not in {"mean", "max", "meanmax"}:
            raise ValueError("fusion_mode phai la mean, max hoac meanmax.")
        if self.close_kernel < 0 or self.open_kernel < 0 or self.median_ksize < 0:
            raise ValueError("Cac kernel morphology phai >= 0.")
        if self.median_ksize not in {0, 1} and self.median_ksize % 2 == 0:
            raise ValueError("median_ksize phai le hoac 0 de bo qua.")
        if not (0 <= self.bg_threshold < self.fg_threshold <= 1):
            raise ValueError("Can 0 <= bg_threshold < fg_threshold <= 1.")
        if self.edge_width < 1:
            raise ValueError("edge_width phai >= 1.")
        if self.alpha_gamma <= 0:
            raise ValueError("alpha_gamma phai > 0.")


# Đảm bảo kích thước bộ lọc là số lẻ.
def _odd(n: int) -> int:
    """Doi ve so le >= 1 neu n > 0."""
    if n <= 0:
        return 0
    return n if n % 2 == 1 else n + 1


# Tạo kernel hình elip dùng cho các phép toán hình thái học.
def _kernel(size: int) -> np.ndarray | None:
    size = _odd(size)
    if size <= 1:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


# Tăng tương phản cục bộ để làm nổi rõ chi tiết trong ảnh.
def apply_clahe_bgr(frame_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Tang tuong phan cuc bo tren kenh L cua LAB de de nhan biet doi tuong hon."""
    _check_frame(frame_bgr)
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# Điều chỉnh độ sáng ảnh bằng phép biến đổi Gamma.
def apply_gamma_correction(frame_bgr: np.ndarray, gamma: float = 1.15) -> np.ndarray:
    """Lam sang anh toi. gamma > 1 thuong lam anh sang hon trong cong thuc nay."""
    _check_frame(frame_bgr)
    if gamma <= 0:
        raise ValueError("gamma phai > 0.")
    values = np.arange(256, dtype=np.float32) / 255.0
    table = np.clip(np.power(values, 1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(frame_bgr, table)


# Chuẩn bị cả ảnh gốc và ảnh đã tăng cường để đưa vào AI.
def enhance_frame_for_model(
    frame_bgr: np.ndarray,
    *,
    use_clahe: bool = True,
    use_gamma: bool = True,
    gamma: float = 1.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Tra 2 phien ban anh de dua vao model: goc va anh da duoc ho tro.

    - anh goc giu nguyen mau sac tu nhien.
    - anh enhanced giup model nhin ro hon trong dieu kien toi / tuong phan thap.
    """
    _check_frame(frame_bgr)
    enhanced = frame_bgr.copy()
    if use_clahe:
        enhanced = apply_clahe_bgr(enhanced)
    if use_gamma:
        enhanced = apply_gamma_correction(enhanced, gamma=gamma)
    return frame_bgr, enhanced


# Kết hợp hai mask dự đoán để hạn chế bỏ sót vùng đối tượng.
def fuse_masks(mask_a: np.ndarray, mask_b: np.ndarray, mode: str = "meanmax") -> np.ndarray:
    """Tron 2 mask, uu tien giu tien canh nhung van tranh no mask qua muc."""
    mask_a = np.asarray(mask_a, dtype=np.float32)
    mask_b = np.asarray(mask_b, dtype=np.float32)
    if mask_a.shape != mask_b.shape:
        raise ValueError("Hai mask phai cung kich thuoc de fusion.")
    if mode == "mean":
        fused = 0.5 * (mask_a + mask_b)
    elif mode == "max":
        fused = np.maximum(mask_a, mask_b)
    else:  # meanmax
        fused = 0.35 * ((mask_a + mask_b) * 0.5) + 0.65 * np.maximum(mask_a, mask_b)
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


# Chạy AI trên hai phiên bản ảnh và tạo mask kết hợp.
def predict_enhanced_mask(
    model: Any,
    frame_bgr: np.ndarray,
    config: ImageRefineConfig | None = None,
) -> dict[str, np.ndarray]:
    """Chay model tren anh goc va anh da preprocess, sau do fusion.

    Tra ve dict gom:
        raw_original, raw_enhanced, raw_fused
    """
    if not hasattr(model, "predict_mask"):
        raise TypeError("model phai co ham predict_mask(frame_bgr).")
    cfg = config if config is not None else ImageRefineConfig()
    original, enhanced = enhance_frame_for_model(
        frame_bgr,
        use_clahe=cfg.use_clahe,
        use_gamma=cfg.use_gamma,
        gamma=cfg.gamma,
    )
    raw_original = _check_mask(model.predict_mask(original), original)
    raw_enhanced = _check_mask(model.predict_mask(enhanced), enhanced)
    raw_fused = fuse_masks(raw_original, raw_enhanced, mode=cfg.fusion_mode)
    return {
        "raw_original": raw_original,
        "raw_enhanced": raw_enhanced,
        "raw_fused": raw_fused,
    }


# Chỉ giữ vùng đối tượng liên thông lớn nhất trong mask.
def keep_largest_component(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Chi giu connected component lon nhat; hop cho anh 1 doi tuong chinh."""
    binary = (np.asarray(mask, dtype=np.float32) >= float(threshold)).astype(np.uint8)
    if binary.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(np.float32)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    kept = (labels == largest).astype(np.float32)
    return kept


# Dùng morphology để lấp lỗ nhỏ và loại bỏ nhiễu rời rạc.
def remove_small_holes_and_noise(mask: np.ndarray, close_kernel: int = 7, open_kernel: int = 3) -> np.ndarray:
    """Dong lo nho roi mo nhiu nho tren mask nhi phan."""
    binary = np.asarray(mask, dtype=np.float32)
    binary = (binary > 0.5).astype(np.uint8)
    ker_close = _kernel(close_kernel)
    ker_open = _kernel(open_kernel)
    if ker_close is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ker_close)
    if ker_open is not None:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker_open)
    return binary.astype(np.float32)


# Xác định dải viền cần được tinh chỉnh quanh đối tượng.
def build_edge_band(mask: np.ndarray, width: int = 14) -> np.ndarray:
    """Tao vung bien de chi refine manh quanh ranh gioi doi tuong."""
    width = max(1, int(width))
    alpha_u8 = np.clip(np.rint(np.asarray(mask, dtype=np.float32) * 255.0), 0, 255).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(width), _odd(width)))
    dilated = cv2.dilate(alpha_u8, ker, iterations=1)
    eroded = cv2.erode(alpha_u8, ker, iterations=1)
    band = ((dilated.astype(np.int16) - eroded.astype(np.int16)) > 0).astype(np.float32)
    return band


# Điều chỉnh alpha để nền rõ hơn và mép đối tượng vẫn mềm.
def alpha_remap(
    mask: np.ndarray,
    *,
    bg_threshold: float = 0.08,
    fg_threshold: float = 0.75,
    gamma: float = 0.95,
    hard_fg_boost: bool = True,
) -> np.ndarray:
    """Nen 0/1 ro hon, con vung giua van giu chuyen tiep mem cho ghep nen.

    - <= bg_threshold  -> 0
    - >= fg_threshold  -> 1
    - o giua -> scale ve [0,1], co them gamma de dieu khien alpha
    """
    alpha = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if not 0 <= bg_threshold < fg_threshold <= 1:
        raise ValueError("Can 0 <= bg_threshold < fg_threshold <= 1.")
    denom = max(fg_threshold - bg_threshold, 1e-6)
    alpha = (alpha - bg_threshold) / denom
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha = np.power(alpha, float(gamma))
    if hard_fg_boost:
        alpha = np.where(alpha > 0.985, 1.0, alpha)
        alpha = np.where(alpha < 0.01, 0.0, alpha)
    return alpha.astype(np.float32)


# Thực hiện chuỗi lọc, làm sạch và tinh chỉnh mask ảnh tĩnh.
def refine_mask_image(
    frame_bgr: np.ndarray,
    raw_mask: np.ndarray,
    config: ImageRefineConfig | None = None,
) -> dict[str, np.ndarray]:
    """Pipeline refine mask uu tien cho anh tinh.

    Thu tu thuc hien:
        1) Loc median nhe (neu bat)
        2) Guided Filter toan cuc
        3) Tao ban nhi phan tam thoi -> morphology clean
        4) Giu component lon nhat (tuy chon)
        5) Guided Filter lai o vung bien
        6) Alpha remap de ghep nen dep hon

    Tra ve dict de de danh gia:
        raw_mask, guided_mask, cleaned_binary, edge_band, alpha
    """
    _check_frame(frame_bgr)
    raw = _check_mask(raw_mask, frame_bgr)
    cfg = config if config is not None else ImageRefineConfig()

    work = raw.copy()
    if _odd(cfg.median_ksize) >= 3:
        work = cv2.medianBlur(np.rint(work * 255).astype(np.uint8), _odd(cfg.median_ksize)).astype(np.float32) / 255.0

    # Guided Filter lan 1: lam muot nhung van bam theo bien anh.
    guided_1 = guided_filter_mask(
        frame_bgr,
        work,
        radius=cfg.radius,
        eps=cfg.eps,
        color_guide=cfg.color_guide,
    ) if cfg.use_guided_filter else work

    # Tao binary tam thoi de lap lo va bo nhiu.
    binary_seed = (guided_1 >= max(cfg.bg_threshold + 0.15, 0.5)).astype(np.float32)
    cleaned_binary = remove_small_holes_and_noise(binary_seed, cfg.close_kernel, cfg.open_kernel)
    if cfg.keep_largest:
        cleaned_binary = keep_largest_component(cleaned_binary, threshold=0.5)

    # Bien de triet de refine khu vuc can canh, khong can tac dong manh len toan anh.
    edge_band = build_edge_band(cleaned_binary, cfg.edge_width)

    # Guided Filter lan 2: chi tang cuong o vung bien.
    if cfg.use_guided_filter:
        guided_2 = guided_filter_mask(
            frame_bgr,
            guided_1,
            radius=max(2, cfg.radius),
            eps=cfg.eps,
            color_guide=cfg.color_guide,
        )
        mixed = guided_1 * (1.0 - edge_band) + guided_2 * edge_band
    else:
        mixed = guided_1

    # Ghep them thong tin binary da duoc clean de bo sung than/vai bi thieu nhe.
    combined = np.maximum(mixed, cleaned_binary * 0.98)

    alpha = alpha_remap(
        combined,
        bg_threshold=cfg.bg_threshold,
        fg_threshold=cfg.fg_threshold,
        gamma=cfg.alpha_gamma,
        hard_fg_boost=cfg.hard_fg_boost,
    )

    # Sau remap, van ton trong binary sach de tranh sot than/vai trong anh 1 doi tuong.
    alpha = np.maximum(alpha, cleaned_binary * 0.92)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return {
        "raw_mask": raw,
        "guided_mask": guided_1.astype(np.float32),
        "cleaned_binary": cleaned_binary.astype(np.float32),
        "edge_band": edge_band.astype(np.float32),
        "alpha": alpha,
    }


# Gộp tiền xử lý, suy luận AI và tinh chỉnh thành một quy trình.
def process_image_mask_enhanced(
    frame_bgr: np.ndarray,
    model: Any,
    config: ImageRefineConfig | None = None,
) -> dict[str, np.ndarray]:
    """Ham tien loi: tu anh -> model -> fusion -> refine -> alpha cuoi.

    Ham nay rat hop de cap nhat image_matting.py o buoc sau.
    """
    cfg = config if config is not None else ImageRefineConfig()
    predicted = predict_enhanced_mask(model, frame_bgr, cfg)
    refined = refine_mask_image(frame_bgr, predicted["raw_fused"], cfg)
    return {**predicted, **refined}


# =========================
# Temporal consistency cho VIDEO (giu tuong thich)
# =========================
# Cấu hình EMA, Optical Flow, Guided Filter và khoảng cách keyframe.
@dataclass
class TemporalConfig:
    """Tham so co the bat/tat doc lap de chay ablation A/B/C/D."""
    use_ema: bool = False
    use_keyframe_flow: bool = False
    use_guided_filter: bool = False
    key_every: int = 4
    radius: int = 5
    eps: float = 0.001
    color_guide: bool = False
    ema_base: float = 0.55
    ema_motion_gain: float = 8.0
    flow_max_side: int = 256
    scene_threshold: float = 0.32

    # Kiểm tra cấu hình xử lý liên tiếp các khung hình.
    def __post_init__(self) -> None:
        if self.key_every < 1:
            raise ValueError("key_every phai >= 1.")
        if self.flow_max_side < 32:
            raise ValueError("flow_max_side phai >= 32.")
        if not 0 <= self.ema_base <= 1 or self.ema_motion_gain < 0:
            raise ValueError("ema_base phai trong [0,1] va ema_motion_gain >= 0.")
        if not 0 < self.scene_threshold <= 1:
            raise ValueError("scene_threshold phai trong (0,1].")
        if self.radius < 1 or self.eps <= 0:
            raise ValueError("radius phai >=1 va eps phai >0.")


# Quản lý mask qua nhiều frame để giảm nhấp nháy khi xử lý video.
class TemporalProcessor:
    """Xu ly tung frame theo thu tu; KHONG chia se giua cac video/webcam.

    model can co phuong thuc predict_mask(frame_bgr) -> mask float32 (H,W).
    Su dung: mask, info = processor.process_frame(frame_bgr)
    """

    # Nhận mô hình và cấu hình rồi khởi tạo trạng thái xử lý.
    def __init__(self, model: Any, config: TemporalConfig | None = None) -> None:
        self.model = model
        self.config = config if config is not None else TemporalConfig()
        self.reset()

    # Xóa trạng thái frame cũ khi bắt đầu video hoặc nguồn mới.
    def reset(self) -> None:
        """Goi khi mo video moi hoac nguoi dung doi nguon webcam."""
        self.frame_index = 0
        self._prev_gray: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None
        self._prev_shape: tuple[int, int] | None = None

    # Thu nhỏ frame và chuyển sang ảnh xám để tính chuyển động nhanh hơn.
    def _small_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.config.flow_max_side / max(h, w))
        size = (max(16, round(w * scale)), max(16, round(h * scale)))
        small = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Phát hiện chuyển cảnh lớn để tránh dùng lại mask không phù hợp.
    def _scene_cut(self, old_gray: np.ndarray, new_gray: np.ndarray) -> bool:
        mean_diff = float(np.mean(cv2.absdiff(old_gray, new_gray))) / 255.0
        hist_old = cv2.calcHist([old_gray], [0], None, [32], [0, 256])
        hist_new = cv2.calcHist([new_gray], [0], None, [32], [0, 256])
        cv2.normalize(hist_old, hist_old, 1.0, 0, cv2.NORM_L1)
        cv2.normalize(hist_new, hist_new, 1.0, 0, cv2.NORM_L1)
        distance = float(cv2.compareHist(hist_old, hist_new, cv2.HISTCMP_BHATTACHARYYA))
        return (mean_diff >= self.config.scene_threshold and distance >= 0.35) or mean_diff >= max(0.60, self.config.scene_threshold * 1.8)

    # Dùng Optical Flow để dịch chuyển mask của frame trước sang frame mới.
    def _warp_previous(self, new_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._prev_gray is not None and self._prev_mask is not None
        flow = cv2.calcOpticalFlowFarneback(
            new_gray, self._prev_gray, None,
            0.5, 2, 15, 2, 5, 1.1, 0,
        )
        h, w = new_gray.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        low_mask = cv2.resize(self._prev_mask, (w, h), interpolation=cv2.INTER_AREA)
        warped_small = cv2.remap(
            low_mask,
            xx + flow[..., 0],
            yy + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        full_h, full_w = self._prev_shape[0], self._prev_shape[1]
        warped = cv2.resize(warped_small, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        motion = magnitude / max(h, w)
        motion = cv2.resize(motion, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(warped, 0, 1), motion

    # Xử lý một frame: chọn AI/Flow, ổn định EMA và tinh chỉnh mask.
    def process_frame(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        _check_frame(frame_bgr)
        start = time.perf_counter()
        small_gray = self._small_gray(frame_bgr)
        shape = frame_bgr.shape[:2]
        scene_cut = False
        first = self._prev_gray is None or self._prev_shape != shape
        if not first:
            scene_cut = self._scene_cut(self._prev_gray, small_gray)
        must_reset = first or scene_cut
        needs_ai = must_reset or not self.config.use_keyframe_flow or self.frame_index % self.config.key_every == 0
        motion_mean = 0.0
        warped = None
        motion = None

        if not must_reset and self.config.use_keyframe_flow:
            warped, motion = self._warp_previous(small_gray)
            motion_mean = float(np.mean(motion))

        if needs_ai:
            fresh = _check_mask(self.model.predict_mask(frame_bgr), frame_bgr)
            if self.config.use_ema and not must_reset:
                if self.config.use_keyframe_flow:
                    previous = warped
                    weight = np.clip(self.config.ema_base + self.config.ema_motion_gain * motion, 0.0, 1.0)
                else:
                    previous = self._prev_mask
                    diff = float(np.mean(cv2.absdiff(self._prev_gray, small_gray))) / 255.0
                    motion_mean = diff
                    weight = float(np.clip(self.config.ema_base + self.config.ema_motion_gain * diff, 0.0, 1.0))
                current = fresh * weight + previous * (1.0 - weight)
            else:
                current = fresh
        else:
            current = warped

        current = np.clip(current, 0.0, 1.0).astype(np.float32)
        self._prev_mask = current
        self._prev_gray = small_gray
        self._prev_shape = shape

        if self.config.use_guided_filter:
            output = guided_filter_mask(
                frame_bgr,
                current,
                self.config.radius,
                self.config.eps,
                self.config.color_guide,
            )
        else:
            output = current.copy()

        self.frame_index += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        info = {
            "frame_index": self.frame_index - 1,
            "keyframe": bool(needs_ai),
            "scene_cut": bool(scene_cut),
            "motion_mean": round(motion_mean, 6),
            "processing_ms": round(elapsed_ms, 3),
            "processing_fps": round(1000.0 / max(elapsed_ms, 1e-6), 2),
            "backend": getattr(self.model, "backend", "unknown"),
        }
        return output, info


# =========================
# Tien ich CLI de test processing.py rieng
# =========================
# Quy đổi đường dẫn tương đối về thư mục chứa chương trình.
def _path_in_project(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p)


# Đọc ảnh đầu vào để kiểm thử phần xử lý mask.
def read_image(path: str | Path) -> np.ndarray:
    p = _path_in_project(path)
    if not p.is_file():
        raise FileNotFoundError(f"Khong tim thay anh: {p}")
    buffer = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Khong doc duoc anh: {p}")
    return image


# Đọc ảnh mask xám và đưa về giá trị 0–1.
def read_mask_gray(path: str | Path) -> np.ndarray:
    p = _path_in_project(path)
    if not p.is_file():
        raise FileNotFoundError(f"Khong tim thay mask: {p}")
    buffer = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Khong doc duoc mask: {p}")
    return image.astype(np.float32) / 255.0


# Lưu mask dưới dạng ảnh xám để kiểm tra kết quả.
def save_gray(path: str | Path, image: np.ndarray) -> Path:
    p = _path_in_project(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(np.clip(image, 0, 1) * 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(p.suffix if p.suffix else ".png", arr)
    if not ok:
        raise RuntimeError(f"Khong ma hoa duoc anh: {p}")
    encoded.tofile(str(p))
    return p


# Nhận tham số Terminal, chạy thử tinh chỉnh mask và lưu kết quả.
def main() -> None:
    parser = argparse.ArgumentParser(description="Test cac buoc refine mask cho anh tinh")
    parser.add_argument("--image", required=True, help="Vi du: samples/images/test.jpg")
    parser.add_argument("--mask", required=True, help="Vi du: outputs/test_mask.png")
    parser.add_argument("--radius", type=int, default=6)
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--color", action="store_true", help="Dung guide mau")
    parser.add_argument("--keep-largest", action="store_true")
    parser.add_argument("--fg-thresh", type=float, default=0.75)
    parser.add_argument("--bg-thresh", type=float, default=0.08)
    parser.add_argument("--alpha-gamma", type=float, default=0.95)
    parser.add_argument("--prefix", default=None, help="Tien to file xuat; mac dinh outputs/<ten>_refine")
    args = parser.parse_args()

    frame = read_image(args.image)
    mask = read_mask_gray(args.mask)
    started = time.perf_counter()
    cfg = ImageRefineConfig(
        radius=args.radius,
        eps=args.eps,
        color_guide=args.color,
        keep_largest=args.keep_largest,
        fg_threshold=args.fg_thresh,
        bg_threshold=args.bg_thresh,
        alpha_gamma=args.alpha_gamma,
    )
    refined = refine_mask_image(frame, mask, cfg)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    image_path = _path_in_project(args.image)
    prefix = _path_in_project(args.prefix) if args.prefix else (ROOT / "outputs" / f"{image_path.stem}_refine")
    files = {
        "guided": save_gray(f"{prefix}_guided.png", refined["guided_mask"]),
        "cleaned": save_gray(f"{prefix}_cleaned.png", refined["cleaned_binary"]),
        "edge": save_gray(f"{prefix}_edge.png", refined["edge_band"]),
        "alpha": save_gray(f"{prefix}_alpha.png", refined["alpha"]),
    }
    print("[OK] Da refine mask cho anh tinh:")
    for k, v in files.items():
        print(f"  {k}: {v}")
    print(
        f"[REFINE] radius={cfg.radius}, eps={cfg.eps}, color={cfg.color_guide}, "
        f"keep_largest={cfg.keep_largest}, fg={cfg.fg_threshold}, bg={cfg.bg_threshold}, "
        f"alpha_gamma={cfg.alpha_gamma}"
    )
    print(f"[THOI GIAN] {elapsed_ms:.2f} ms; khong bao gom inference AI.")


if __name__ == "__main__":
    main()
