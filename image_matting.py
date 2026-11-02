"""image_matting.py - Xu ly anh tinh cho 3G.

Tuong thich voi:
    model_loader.py: U2NetPModel
    processing.py: guided_filter_mask, enhance_frame_for_model, fuse_masks,
                   alpha_remap, refine_mask_image, ImageRefineConfig
    video_matting.py: read_image, composite_background

Co 3 che do:
    - Mac dinh: 1 lan AI + Guided Filter (giu nguyen cach cu).
    - --enhanced: AI tren anh goc va anh tang tuong phan; fusion 2 mask;
                   refine MEM (uu tien giu toc va alpha mem).
    - --enhanced --strong-refine: them morphology, giu component lon nhat
                   neu duoc chon; CO THE lam mat toc manh, can so sanh.

"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Nhập các hàm từ module nạp AI và tinh chỉnh mask.
from model_loader import PROJECT_DIR, U2NetPModel
from processing import (
    ImageRefineConfig,
    alpha_remap,
    enhance_frame_for_model,
    fuse_masks,
    guided_filter_mask,
    refine_mask_image,
)

# Thư mục lưu ảnh sau khi tách nền.
OUTPUT_DIR = PROJECT_DIR / "outputs"



# ===== 1. ĐỌC / GHI FILE VÀ KIỂM TRA DỮ LIỆU =====
# Quy đổi đường dẫn về thư mục dự án 3G.
def _path_in_project(path: str | Path) -> Path:
    """Giai duong dan tuong doi dua tren thu muc chua code."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_DIR / p


# Đọc ảnh bằng OpenCV, hỗ trợ đường dẫn có dấu trên Windows.
def read_image(path: str | Path) -> np.ndarray:
    """Doc anh mau BGR, ho tro duong dan co dau tren Windows."""
    p = _path_in_project(path)
    if not p.is_file():
        raise FileNotFoundError(f"Khong tim thay anh: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Khong doc duoc anh hop le: {p}")
    return image


# Lưu ảnh ra ổ đĩa, bao gồm PNG có kênh trong suốt (alpha).
def save_image(path: str | Path, image: np.ndarray) -> Path:
    """Luu anh khong bi loi duong dan Unicode, giu alpha khi la PNG."""
    p = _path_in_project(path)
    ext = p.suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Dinh dang dau ra phai la .png, .jpg, .jpeg hoac .webp.")
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Anh dau ra khong hop le.")
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Khong ma hoa duoc anh: {p}")
    buffer.tofile(str(p))
    return p


# Kiểm tra kích thước, kiểu dữ liệu và giá trị của ảnh, mask.
def _validate_inputs(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Dinh dang chung cho anh va mask alpha [0..1]."""
    if (
        not isinstance(frame_bgr, np.ndarray)
        or frame_bgr.dtype != np.uint8
        or frame_bgr.ndim != 3
        or frame_bgr.shape[2] != 3
        or min(frame_bgr.shape[:2]) < 2
    ):
        raise ValueError("Anh phai la BGR uint8, shape (H,W,3), moi chieu >=2.")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("Mask phai co shape (H,W).")
    if mask.shape != frame_bgr.shape[:2]:
        raise ValueError(f"Mask {mask.shape} khong khop anh {frame_bgr.shape[:2]}.")
    alpha = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(alpha).all():
        raise ValueError("Mask co NaN/Inf.")
    return np.clip(alpha, 0.0, 1.0)


# Chuyển mã màu HEX sang BGR cho OpenCV.
def _hex_to_bgr(color: str | tuple[int, int, int]) -> tuple[int, int, int]:
    """Chuoi RGB '#RRGGBB' -> tuple BGR cho OpenCV."""
    if isinstance(color, tuple) and len(color) == 3:
        if all(isinstance(v, (int, np.integer)) and 0 <= v <= 255 for v in color):
            return tuple(int(v) for v in color)
        raise ValueError("Tuple mau BGR phai co 3 so nguyen trong [0,255].")
    if not isinstance(color, str):
        raise ValueError("Mau phai la chuoi #RRGGBB hoac tuple BGR.")
    color = color.strip().lstrip("#")
    if len(color) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in color):
        raise ValueError("Mau nen phai co dang #RRGGBB, vi du #FFFFFF.")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


# Co giãn và cắt ảnh nền vừa khung mà không làm méo.
def _resize_cover(background_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Phong anh nen theo cover roi crop giua, khong keo meo anh."""
    if (
        not isinstance(background_bgr, np.ndarray)
        or background_bgr.ndim != 3
        or background_bgr.shape[2] != 3
        or background_bgr.dtype != np.uint8
        or min(background_bgr.shape[:2]) < 2
    ):
        raise ValueError("Anh nen phai la BGR uint8 hop le.")
    old_h, old_w = background_bgr.shape[:2]
    scale = max(width / old_w, height / old_h)
    new_w = max(width, int(np.ceil(old_w * scale)))
    new_h = max(height, int(np.ceil(old_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(background_bgr, (new_w, new_h), interpolation=interpolation)
    left, top = (new_w - width) // 2, (new_h - height) // 2
    return resized[top : top + height, left : left + width]



# ===== 2. TẠO ẢNH TRONG SUỐT VÀ THAY NỀN =====
# Đưa mask vào kênh alpha để tạo ảnh PNG trong suốt.
def make_transparent(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tra BGRA uint8 de ghi PNG trong suot (alpha that, khong phai nen trang)."""
    alpha = _validate_inputs(frame_bgr, mask)
    image_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    image_bgra[:, :, 3] = np.rint(alpha * 255).astype(np.uint8)
    return image_bgra


# Ghép đối tượng với nền màu, ảnh khác hoặc nền làm mờ.
def composite_background(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    mode: str = "color",
    *,
    bg_color: str | tuple[int, int, int] = "#F1F5F9",
    bg_image_bgr: np.ndarray | None = None,
    blur_sigma: float = 12.0,
) -> np.ndarray:
    """Ghep nen mau / nen anh / nen mo. Ham nay duoc video_matting.py dung lai."""
    alpha = _validate_inputs(frame_bgr, mask)
    h, w = frame_bgr.shape[:2]
    if mode == "color":
        background = np.empty_like(frame_bgr)
        background[:] = _hex_to_bgr(bg_color)
    elif mode == "image":
        if bg_image_bgr is None:
            raise ValueError("Che do image can bg_image_bgr.")
        background = _resize_cover(bg_image_bgr, h, w)
    elif mode == "blur":
        if not np.isfinite(blur_sigma) or not (0 < blur_sigma <= 100):
            raise ValueError("blur_sigma phai trong (0,100].")
        background = cv2.GaussianBlur(frame_bgr, (0, 0), sigmaX=float(blur_sigma))
    else:
        raise ValueError("mode phai la color, image hoac blur.")
    a = alpha[..., None]
    output = frame_bgr.astype(np.float32) * a + background.astype(np.float32) * (1.0 - a)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


# Thu nhỏ ảnh lớn để giảm thời gian xử lý và dùng ít RAM.
def resize_for_demo(frame_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    """Thu nho anh qua lon de tiet kiem RAM. max_side=0 de giu nguyen."""
    if max_side < 0:
        raise ValueError("max_side phai >= 0.")
    if max_side == 0 or max(frame_bgr.shape[:2]) <= max_side:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    scale = max_side / max(h, w)
    return cv2.resize(
        frame_bgr,
        (max(2, round(w * scale)), max(2, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )



# ===== 3. QUY TRÌNH TÁCH NỀN ẢNH BẰNG AI =====
# Tách nền bằng AI, tinh chỉnh mask, ghép nền và đo thời gian.
def process_image(
    frame_bgr: np.ndarray,
    model: Any,
    *,
    use_guided_filter: bool = True,
    radius: int = 5,
    eps: float = 0.001,
    color_guide: bool = False,
    bg_color: str = "#F1F5F9",
    bg_image_bgr: np.ndarray | None = None,
    blur_sigma: float = 12.0,
    enhanced: bool = False,
    strong_refine: bool = False,
    refine_config: ImageRefineConfig | None = None,
) -> dict[str, Any]:
    """Tach nen anh. Mac dinh la pipeline cu de de so sanh.

    enhanced=True: chay AI tren anh goc va anh tang sang/tuong phan, sau do fusion.
    strong_refine=True: cleanup nhi phan (anh 1 doi tuong), de mat toc manh hon.
    refine_config: cau hinh chi tiet cho enhanced mode.

    Cac khoa tra ve cu duoc giu nguyen de app.py co the dung lai:
        raw_mask, mask, mask_u8, transparent_bgra, replaced_bgr, blurred_bgr,
        backend, inference_ms, guided_ms, total_ms, fps_equivalent.
    """
    if not hasattr(model, "predict_mask"):
        raise TypeError("Model phai co predict_mask(frame_bgr).")
    if strong_refine and not enhanced:
        raise ValueError("--strong-refine chi dung cung --enhanced.")
    _validate_inputs(frame_bgr, np.zeros(frame_bgr.shape[:2], dtype=np.float32))
    started = time.perf_counter()

    cfg = refine_config or ImageRefineConfig(
        use_guided_filter=use_guided_filter,
        radius=radius,
        eps=eps,
        color_guide=color_guide,
    )
    # Flag ngoai ham luon co hieu luc, khong de config bat nguoc.
    cfg.use_guided_filter = use_guided_filter
    cfg.radius = radius
    cfg.eps = eps
    cfg.color_guide = color_guide

    # Bước 1: Dùng AI dự đoán mask của ảnh gốc.
    raw_mask = _validate_inputs(frame_bgr, model.predict_mask(frame_bgr))
    inference_ms = float(getattr(model, "last_inference_ms", 0.0))
    enhanced_mask: np.ndarray | None = None
    fused_mask: np.ndarray | None = None
    diagnostics: dict[str, np.ndarray] = {}

    # Bước 2: Nếu bật Enhanced, dự đoán thêm trên ảnh tăng cường và gộp mask.
    if enhanced:
        _, enhanced_frame = enhance_frame_for_model(
            frame_bgr,
            use_clahe=cfg.use_clahe,
            use_gamma=cfg.use_gamma,
            gamma=cfg.gamma,
        )
        if np.array_equal(frame_bgr, enhanced_frame):
            enhanced_mask = raw_mask.copy()
        else:
            enhanced_mask = _validate_inputs(frame_bgr, model.predict_mask(enhanced_frame))
            inference_ms += float(getattr(model, "last_inference_ms", 0.0))
        fused_mask = fuse_masks(raw_mask, enhanced_mask, mode=cfg.fusion_mode)
        seed = fused_mask
    else:
        seed = raw_mask
    after_ai = time.perf_counter()

    # Bước 3: Tinh chỉnh mask theo chế độ cơ bản, mềm hoặc mạnh.
    if enhanced and strong_refine:
        refined = refine_mask_image(frame_bgr, seed, cfg)
        mask = refined["alpha"]
        diagnostics = {
            "guided_mask": refined["guided_mask"],
            "cleaned_binary": refined["cleaned_binary"],
            "edge_band": refined["edge_band"],
        }
    elif enhanced:
        # Refine mem: khong ep mask thanh nhi phan, ton trong toc va ria manh.
        guided = (
            guided_filter_mask(frame_bgr, seed, radius, eps, color_guide)
            if use_guided_filter else seed.copy()
        )
        mask = alpha_remap(
            guided,
            bg_threshold=cfg.bg_threshold,
            fg_threshold=cfg.fg_threshold,
            gamma=cfg.alpha_gamma,
            hard_fg_boost=False,
        )
        diagnostics = {"guided_mask": guided}
    else:
        # Baseline cu: chi chay 1 lan model, tuy chon Guided Filter.
        mask = (
            guided_filter_mask(frame_bgr, seed, radius, eps, color_guide)
            if use_guided_filter else seed.copy()
        )
    mask = _validate_inputs(frame_bgr, mask)
    after_refine = time.perf_counter()

    # Bước 4: Tạo các dạng ảnh đầu ra và thống kê thời gian.
    transparent = make_transparent(frame_bgr, mask)
    replaced = composite_background(
        frame_bgr, mask,
        mode="image" if bg_image_bgr is not None else "color",
        bg_color=bg_color,
        bg_image_bgr=bg_image_bgr,
    )
    blurred = composite_background(frame_bgr, mask, mode="blur", blur_sigma=blur_sigma)
    end = time.perf_counter()
    total_ms = (end - started) * 1000.0
    return {
        "raw_mask": raw_mask,
        "enhanced_mask": enhanced_mask,
        "fused_mask": fused_mask,
        "diagnostics": diagnostics,
        "mask": mask,
        "mask_u8": np.rint(mask * 255).astype(np.uint8),
        "transparent_bgra": transparent,
        "replaced_bgr": replaced,
        "blurred_bgr": blurred,
        "backend": getattr(model, "backend", "unknown"),
        "inference_ms": inference_ms,
        "guided_ms": (after_refine - after_ai) * 1000.0 if use_guided_filter else 0.0,
        "refine_ms": (after_refine - after_ai) * 1000.0,
        "total_ms": total_ms,
        "fps_equivalent": 1000.0 / max(total_ms, 1e-6),
    }



# ===== 4. CHẠY ĐỘC LẬP BẰNG DÒNG LỆNH =====
# Nhận tham số từ Terminal, chạy xử lý ảnh và lưu kết quả.
def main() -> None:
    # Khai báo các tùy chọn nhập từ dòng lệnh.
    parser = argparse.ArgumentParser(description="U2Net-P: baseline va cai tien tach nen anh AETM")
    parser.add_argument("--image", required=True, help="Duong dan anh, vd samples/images/test.jpg")
    parser.add_argument("--backend", choices=["cpu", "openvino_cpu", "openvino_gpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-guided-filter", action="store_true", help="Tat Guided Filter de doi chung")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--color-guide", action="store_true", help="Dung anh mau de Guided Filter")
    parser.add_argument("--bg-color", default="#F1F5F9", help="Mau nen #RRGGBB")
    parser.add_argument("--bg-image", default=None, help="Thay nen bang anh khac")
    parser.add_argument("--blur-sigma", type=float, default=12.0)
    parser.add_argument("--max-side", type=int, default=1600, help="0 = giu kich thuoc goc")
    parser.add_argument("--output-prefix", default=None, help="Tien to dau ra")

    # Them moi: chi anh tinh. Phai bat --enhanced thi cac tuy chon nay moi hoat dong.
    parser.add_argument("--enhanced", action="store_true", help="Chay 2 du doan AI + fusion mask + refine mem")
    parser.add_argument("--strong-refine", action="store_true", help="Cleanup binary manh, co the lam mat toc manh")
    parser.add_argument("--no-clahe", action="store_true", help="Tat tang tuong phan")
    parser.add_argument("--no-gamma", action="store_true", help="Tat tang sang")
    parser.add_argument("--gamma", type=float, default=1.15, help="Gamma cho tien xu ly")
    parser.add_argument("--fusion-mode", choices=["mean", "max", "meanmax"], default="mean")
    parser.add_argument("--keep-largest", action="store_true", help="Chi giu 1 doi tuong lon nhat (strong refine)")
    parser.add_argument("--fg-thresh", type=float, default=0.98, help="Nguong chac chan tien canh trong soft refine")
    parser.add_argument("--bg-thresh", type=float, default=0.02, help="Nguong chac chan nen trong soft refine")
    parser.add_argument("--alpha-gamma", type=float, default=1.0, help="Gamma cua alpha cuoi")
    parser.add_argument("--close-kernel", type=int, default=5)
    parser.add_argument("--open-kernel", type=int, default=3)
    args = parser.parse_args()

    if args.keep_largest and not args.strong_refine:
        parser.error("--keep-largest yeu cau them --enhanced --strong-refine")
    if args.strong_refine and not args.enhanced:
        parser.error("--strong-refine yeu cau --enhanced")

    # Đọc và điều chỉnh kích thước ảnh trước khi chạy AI.
    image_path = _path_in_project(args.image)
    frame = read_image(image_path)
    old_h, old_w = frame.shape[:2]
    frame = resize_for_demo(frame, args.max_side)
    if frame.shape[:2] != (old_h, old_w):
        print(f"[ANH] Thu nho anh {old_w}x{old_h} -> {frame.shape[1]}x{frame.shape[0]} de giam RAM.")

    background = read_image(args.bg_image) if args.bg_image else None
    # Tạo cấu hình tinh chỉnh và khởi tạo mô hình AI.
    cfg = ImageRefineConfig(
        use_clahe=not args.no_clahe,
        use_gamma=not args.no_gamma,
        gamma=args.gamma,
        fusion_mode=args.fusion_mode,
        use_guided_filter=not args.no_guided_filter,
        radius=args.radius,
        eps=args.eps,
        color_guide=args.color_guide,
        keep_largest=args.keep_largest,
        close_kernel=args.close_kernel,
        open_kernel=args.open_kernel,
        fg_threshold=args.fg_thresh,
        bg_threshold=args.bg_thresh,
        alpha_gamma=args.alpha_gamma,
    )

    model = U2NetPModel(backend=args.backend, num_threads=args.threads)
    result = process_image(
        frame,
        model,
        use_guided_filter=not args.no_guided_filter,
        radius=args.radius,
        eps=args.eps,
        color_guide=args.color_guide,
        bg_color=args.bg_color,
        bg_image_bgr=background,
        blur_sigma=args.blur_sigma,
        enhanced=args.enhanced,
        strong_refine=args.strong_refine,
        refine_config=cfg,
    )

    # Lưu mask và các loại ảnh kết quả vào thư mục outputs.
    suffix = "enhanced_strong" if args.strong_refine else ("enhanced" if args.enhanced else "aetm")
    prefix = (_path_in_project(args.output_prefix) if args.output_prefix
              else OUTPUT_DIR / f"{image_path.stem}_{suffix}")
    files = {
        "mask": save_image(str(prefix) + "_mask.png", result["mask_u8"]),
        "transparent": save_image(str(prefix) + "_transparent.png", result["transparent_bgra"]),
        "replaced": save_image(str(prefix) + "_replaced.png", result["replaced_bgr"]),
        "blurred": save_image(str(prefix) + "_blurred.png", result["blurred_bgr"]),
    }
    if args.enhanced:
        files["raw_mask"] = save_image(str(prefix) + "_raw_mask.png", np.rint(result["raw_mask"] * 255).astype(np.uint8))
        files["enhanced_mask"] = save_image(str(prefix) + "_enhanced_mask.png", np.rint(result["enhanced_mask"] * 255).astype(np.uint8))
        files["fused_mask"] = save_image(str(prefix) + "_fused_mask.png", np.rint(result["fused_mask"] * 255).astype(np.uint8))
        for key, value in result["diagnostics"].items():
            files[key] = save_image(str(prefix) + f"_{key}.png", np.rint(value * 255).astype(np.uint8))

    # In đường dẫn và thông số xử lý để theo dõi kết quả.
    print("[OK] Xu ly anh thanh cong:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    print(f"[CHE DO] {'enhanced strong' if args.strong_refine else 'enhanced soft' if args.enhanced else 'mac dinh'}")
    print(f"[MODEL] Backend: {result['backend']}")
    print(f"[THOI GIAN] Inference AI tong: {result['inference_ms']:.2f} ms")
    print(f"[THOI GIAN] Xu ly mask: {result['refine_ms']:.2f} ms")
    print(f"[THOI GIAN] Toan pipeline: {result['total_ms']:.2f} ms")
    print(f"[TOC DO] {result['fps_equivalent']:.2f} FPS tuong duong cho 1 anh, khong phai benchmark video")


if __name__ == "__main__":
    main()
