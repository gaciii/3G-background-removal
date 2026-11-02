
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
matplotlib.use("Agg")  # Khong mo cua so Figure khi chay Windows/VS Code.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from processing import TemporalConfig, TemporalProcessor, guided_filter_mask
# Cấu hình thư mục đầu ra và các định dạng ảnh/video hỗ trợ.

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "evaluation"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----- Ho tro duong dan chua dau tren Windows -----
# Chuẩn hóa đường dẫn file/thư mục; hỗ trợ đường dẫn tương đối và tuyệt đối.
def to_path(path: str | Path) -> Path:
    obj = Path(path).expanduser()
    return (obj if obj.is_absolute() else ROOT / obj).resolve()


# Đọc ảnh bằng OpenCV, hỗ trợ cả đường dẫn Windows có dấu tiếng Việt.
def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError(f"File anh rong: {path}")
    mode = cv2.IMREAD_UNCHANGED if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(raw, mode)
    if img is None:
        raise ValueError(f"Khong doc duoc anh: {path}")
    return img


# Tìm các video cần đánh giá từ tham số --video hoặc --videos-dir.
def find_videos(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.video or []:
        path = to_path(item)
        if not path.is_file():
            raise FileNotFoundError(f"Khong co video: {path}")
        paths.append(path)
    if args.videos_dir:
        directory = to_path(args.videos_dir)
        if directory.is_dir():
            paths.extend(sorted(p for p in directory.iterdir()
                                if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES))
        else:
            print(f"[CANH BAO] Khong tim thay thu muc video: {directory}")
    # Loai file trung de khong tinh 2 lan neu --video + --videos-dir trung nhau.
    return list(dict.fromkeys(paths))


# Tạo 4 cấu hình A/B/C/D để so sánh tác dụng của EMA, Optical Flow và Guided Filter.
def ablation_configs(key_every: int, radius: int, eps: float) -> dict[str, TemporalConfig]:
    common = dict(key_every=key_every, radius=radius, eps=eps,
                  ema_base=0.55, ema_motion_gain=8.0,
                  flow_max_side=256, scene_threshold=0.32)
    return {
        "A": TemporalConfig(use_ema=False, use_keyframe_flow=False,
                            use_guided_filter=False, **common),
        "B": TemporalConfig(use_ema=True, use_keyframe_flow=False,
                            use_guided_filter=False, **common),
        "C": TemporalConfig(use_ema=True, use_keyframe_flow=True,
                            use_guided_filter=False, **common),
        "D": TemporalConfig(use_ema=True, use_keyframe_flow=True,
                            use_guided_filter=True, **common),
    }


# Thu nhỏ khung hình lớn, giữ tỉ lệ để giảm chi phí xử lý.
def resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    if max(h, w) <= max_side:
        return frame
    scale = max_side / max(h, w)
    nw = max(2, 2 * (int(round(w * scale)) // 2))
    nh = max(2, 2 * (int(round(h * scale)) // 2))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


# Ghép đối tượng với nền xám cố định để đo FPS công bằng giữa các cấu hình.
def composite_gray_background(frame: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Ghep nen mau xam, tinh chi phi ghep nen trong FPS cua pipeline."""
    a = alpha[..., None]
    # Co dinh cung 1 mau nen cho moi config, khong anh huong so sanh.
    bg = np.array([240, 245, 249], dtype=np.float32)
    return np.clip(frame.astype(np.float32) * a + bg * (1.0 - a), 0, 255).astype(np.uint8)


# Đo mức nhấp nháy (flicker) của mask ở những pixel gần như đứng yên.
def flicker_contribution(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_alpha: np.ndarray,
    current_alpha: np.ndarray,
    pixel_threshold: int,
) -> tuple[float, int, int]:
    """Tinh tong flicker alpha tren diem anh it thay doi mau va so pixel hop le.

    Tao mask 'dung yen' bang |gray_t-gray_(t-1)| <= pixel_threshold.
    Day la proxy don gian, thich hop camera co dinh, khong phai flow-compensated GT.
    """
    static = cv2.absdiff(previous_gray, current_gray) <= pixel_threshold
    count = int(np.count_nonzero(static))
    if count == 0:
        return 0.0, 0, int(static.size)
    # float 0..1 => flicker [% alpha]
    delta = np.abs(current_alpha - previous_alpha)
    return float(np.sum(delta[static], dtype=np.float64)), count, int(static.size)


# Đọc lần lượt từng frame, chạy một cấu hình và thống kê FPS, flicker, số keyframe.
def evaluate_video(
    video_path: Path,
    label: str,
    config: TemporalConfig,
    model: Any,
    *,
    max_side: int,
    max_frames: int,
    static_threshold: int,
) -> dict[str, Any]:
    """Chay 1 cau hinh / 1 video, doc frame theo luong de tiet kiem RAM."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Khong mo duoc video: {video_path}")
    processor = TemporalProcessor(model, config)
    # Chuẩn bị các biến thống kê trước khi duyệt video.

    frames = 0
    keyframes = 0
    scene_cuts = 0
    inference_ms: list[float] = []
    frame_times_ms: list[float] = []
    flicker_sum = 0.0
    static_pixels = 0
    tested_pixels = 0
    previous_gray: np.ndarray | None = None
    previous_mask: np.ndarray | None = None
    in_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Xử lý lần lượt từng frame và tính thời gian thực thi.
    try:
        begin = time.perf_counter()
        while True:
            if max_frames > 0 and frames >= max_frames:
                break
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame = resize_frame(frame, max_side)
            mask, info = processor.process_frame(frame)
            # Phan cong viec ghep nen duoc tinh vao FPS.
            _ = composite_gray_background(frame, mask)
            frame_elapsed_ms = (time.perf_counter() - frame_started) * 1000.0
            frame_times_ms.append(frame_elapsed_ms)
            frames += 1

            if info["keyframe"]:
                keyframes += 1
                inference_ms.append(float(getattr(model, "last_inference_ms", math.nan)))
            if info["scene_cut"]:
                scene_cuts += 1

            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Bo qua canh chuyen va frame dau de khong coi scene cut la flicker.
            if previous_mask is not None and not info["scene_cut"]:
                val, count, total = flicker_contribution(
                    previous_gray, current_gray, previous_mask, mask,
                    static_threshold,
                )
                flicker_sum += val
                static_pixels += count
                tested_pixels += total
            previous_gray = current_gray
            previous_mask = mask
        elapsed = time.perf_counter() - begin
    finally:
        capture.release()

    if frames == 0:
        raise ValueError(f"Video khong co frame hop le: {video_path}")
    measured_fps = frames / max(elapsed, 1e-9)
    valid_ms = [x for x in inference_ms if math.isfinite(x)]
    # Tổng hợp số liệu đo được để ghi báo cáo CSV.
    return {
        "video": video_path.name,
        "config": label,
        "use_ema": int(config.use_ema),
        "use_keyframe_flow": int(config.use_keyframe_flow),
        "use_guided_filter": int(config.use_guided_filter),
        "key_every": config.key_every,
        "backend_requested": getattr(model, "requested_backend", "cpu"),
        "backend_actual": getattr(model, "backend", "cpu"),
        "frames": frames,
        "width": int(previous_mask.shape[1]),
        "height": int(previous_mask.shape[0]),
        "source_width": in_width,
        "source_height": in_height,
        "fps_pipeline_no_encode": round(measured_fps, 3),
        "processing_ms_mean": round(float(np.mean(frame_times_ms)), 3),
        "processing_ms_median": round(float(np.median(frame_times_ms)), 3),
        "inference_ms_keyframe_mean": round(float(np.mean(valid_ms)), 3) if valid_ms else math.nan,
        "flicker_static_percent": round(100.0 * flicker_sum / static_pixels, 5) if static_pixels else math.nan,
        "static_pixel_fraction": round(static_pixels / tested_pixels, 5) if tested_pixels else math.nan,
        "keyframes": keyframes,
        "scene_cuts": scene_cuts,
        "elapsed_s": round(elapsed, 3),
    }


# Tìm mask nhãn chuẩn (ground truth) có tên tương ứng với ảnh đầu vào.
def lookup_gt(image_path: Path, masks_dir: Path) -> Path | None:
    """Ghep anh/GT theo stem (anh.jpg -> anh.png, anh_mask.png, anh_gt.png)."""
    candidates: list[Path] = []
    for suffix in ("", "_mask", "_gt", "_mask_gt"):
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            candidate = masks_dir / f"{image_path.stem}{suffix}{ext}"
            if candidate.is_file():
                candidates.append(candidate)
    if not candidates:
        return None
    # Uu tien mat na PNG hon JPG de tranh artefact lossy.
    candidates.sort(key=lambda p: (p.suffix.lower() != ".png", len(p.stem)))
    return candidates[0]


# Đọc và chuẩn hóa mask ground truth về khoảng 0–1 để so sánh với AI.
def read_gt_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    img = read_image(path, grayscale=True)
    if img.ndim == 2:
        gray = img
    elif img.shape[2] == 4:
        # Neu alpha khong phai tat ca 255, dung alpha cua PNG gán nhan.
        alpha = img[:, :, 3]
        gray = alpha if np.any(alpha != alpha.max()) or np.any(alpha != 255) else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Mask GT khong hop le: {path}")
    if gray.shape[:2] != expected_shape:
        raise ValueError(f"Mask GT {path.name} co size {gray.shape[:2]} khac anh {expected_shape}; "
                         "can can chinh anh va mask truoc khi tinh MAE/IoU.")
    if np.issubdtype(gray.dtype, np.integer):
        scale = np.iinfo(gray.dtype).max
    else:
        scale = 1.0 if float(np.max(gray)) <= 1.0 else 255.0
    return np.clip(gray.astype(np.float32) / float(scale), 0.0, 1.0)


# Tính hai chỉ số chất lượng mask: MAE (càng nhỏ càng tốt) và IoU (càng lớn càng tốt).
def metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    if pred.shape != gt.shape:
        raise ValueError("Kich thuoc mask pred va GT khac nhau")
    error = float(np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32))))
    predicted = pred >= 0.5
    actual = gt >= 0.5
    union = int(np.count_nonzero(predicted | actual))
    intersection = int(np.count_nonzero(predicted & actual))
    iou = intersection / union if union else 1.0
    return error, float(iou)


# Đánh giá ảnh tĩnh bằng cách so mask AI trước/sau Guided Filter với mask nhãn chuẩn.
def evaluate_images(
    images_dir: Path,
    masks_dir: Path,
    model: Any,
    radius: int,
    eps: float,
) -> list[dict[str, Any]]:
    """Anh tinh khong co thoi gian: chi so sanh A (AI) va D (AI + GF)."""
    results: list[dict[str, Any]] = []
    if not images_dir.is_dir() or not masks_dir.is_dir():
        print("[ANH] Chua co du thu muc anh/mask GT; bo qua MAE/IoU.")
        return results
    images = sorted(p for p in images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    pairs = [(image, lookup_gt(image, masks_dir)) for image in images]
    pairs = [(image, gt) for image, gt in pairs if gt is not None]
    if not pairs:
        print("[ANH] Khong tim thay cap anh + mask GT trung ten; bo qua MAE/IoU.")
        return results
    print(f"[ANH] Co {len(pairs)} cap anh / GT")
    # Duyệt từng cặp ảnh/nhãn chuẩn và so sánh hai cách xử lý.

    for index, (image_path, mask_path) in enumerate(pairs, 1):
        try:
            image = read_image(image_path)
            gt = read_gt_mask(mask_path, image.shape[:2])
            initial = time.perf_counter()
            original = model.predict_mask(image)
            base_ms = (time.perf_counter() - initial) * 1000
            after = time.perf_counter()
            guided = guided_filter_mask(image, original, radius=radius, eps=eps)
            guided_ms = (time.perf_counter() - after) * 1000
            for label, prediction, elapsed in (
                ("A", original, base_ms),
                ("D_GF_only", guided, base_ms + guided_ms),
            ):
                mae, iou = metrics(prediction, gt)
                results.append({
                    "image": image_path.name,
                    "gt_mask": mask_path.name,
                    "config": label,
                    "mae": round(mae, 6),
                    "iou": round(iou, 6),
                    "elapsed_ms": round(elapsed, 3),
                    "backend_actual": getattr(model, "backend", "cpu"),
                })
            print(f"[ANH {index}/{len(pairs)}] {image_path.name}: "
                  f"IoU A={results[-2]['iou']:.3f}, GF={results[-1]['iou']:.3f}")
        except (ValueError, OSError) as exc:
            print(f"[CANH BAO] Bo qua {image_path.name}: {exc}")
    return results


# Vẽ biểu đồ cột và lưu thành ảnh PNG để đưa vào báo cáo thực nghiệm.
def plot_bar(
    summary: pd.DataFrame,
    data_column: str,
    ylabel: str,
    path: Path,
    *,
    low_is_better: bool = False,
) -> None:
    if summary.empty or summary[data_column].isna().all():
        return
    labels = summary["config"].tolist()
    values = summary[data_column].to_numpy(dtype=float)
    plt.figure(figsize=(7.6, 4.5))
    plt.bar(labels, values)
    plt.title("AETM ablation: " + ylabel)
    plt.xlabel("Cau hinh A / B / C / D")
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.2)
    for i, v in enumerate(values):
        if math.isfinite(v):
            plt.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# Đo hiệu năng các backend CPU/OpenVINO và ghi đúng backend thực tế được dùng.
def run_backend_benchmarks(
    sample_frame: np.ndarray,
    output_dir: Path,
    threads: int,
    repeats: int,
) -> None:
    # Mo model rieng tung backend; khong cho phep fallback am tham bi coi la GPU.
    from model_loader import U2NetPModel

    rows = []
    for requested in ("cpu", "openvino_cpu", "openvino_gpu"):
        try:
            model = U2NetPModel(backend=requested, num_threads=threads)
            result = model.benchmark(sample_frame, warmup=3, repeats=repeats)
            row = {"requested": requested, "actual": model.backend, **result}
            row["true_backend_test"] = model.backend == requested
            rows.append(row)
            print(f"[BACKEND] {requested} -> {model.backend}: {result['pipeline_fps']} FPS")
        except Exception as exc:
            rows.append({"requested": requested, "actual": "ERROR", "error": str(exc),
                         "true_backend_test": False})
            print(f"[CANH BAO] Backend {requested}: {exc}")
    pd.DataFrame(rows).to_csv(output_dir / "backend_benchmark.csv", index=False, encoding="utf-8-sig")


# Lấy một frame mẫu từ video hoặc ảnh để khởi động/đánh giá mô hình.
def sample_frame_from_video_or_image(videos: list[Path], images_dir: Path) -> np.ndarray | None:
    for path in videos:
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
            if ok and frame is not None:
                return resize_frame(frame, 720)
        finally:
            capture.release()
    if images_dir.is_dir():
        for path in sorted(images_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                return resize_frame(read_image(path), 720)
    return None


# Tự kiểm thử các hàm xử lý và chỉ số bằng model giả, không cần nạp AI thật.
def self_test() -> None:
    """Kiem tra code tuong thich TemporalProcessor, KHONG dung model that."""
    class FakeModel:
        backend = "fake"
        requested_backend = "fake"
        last_inference_ms = 0.0

        def predict_mask(self, image: np.ndarray) -> np.ndarray:
            h, w = image.shape[:2]
            yy, xx = np.ogrid[:h, :w]
            return ((xx - w * 0.5)**2 + (yy - h * 0.5)**2 < (min(w, h) * 0.25)**2).astype(np.float32)

    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    frame[18:70, 23:95] = [70, 160, 220]
    for label, cfg in ablation_configs(2, 5, 0.001).items():
        processor = TemporalProcessor(FakeModel(), cfg)
        mask1, _ = processor.process_frame(frame)
        mask2, _ = processor.process_frame(frame)
        assert mask1.shape == mask2.shape == (96, 128)
        assert 0 <= mask1.min() and mask1.max() <= 1
        value, count, total = flicker_contribution(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                   mask1, mask2, 6)
        assert count == total == 96 * 128 and value >= 0
        print(f"[TEST] {label}: OK")
    assert metrics(np.ones((3, 4), dtype=np.float32), np.ones((3, 4), dtype=np.float32)) == (0.0, 1.0)
    print("[OK] Self-test: Temporal, metric va flicker khong co loi.")


# Khai báo các tham số dòng lệnh để tùy chọn nguồn dữ liệu và cấu hình đánh giá.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AETM: do FPS, flicker, MAE, IoU A/B/C/D")
    parser.add_argument("--video", action="append", help="Duong dan mot video; lap lai de danh gia nhieu video")
    parser.add_argument("--videos-dir", default=None, help="Thu muc video; vi du samples/videos")
    parser.add_argument("--images-dir", default="samples/images")
    parser.add_argument("--masks-dir", default="samples/masks_gt")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--backend", choices=["cpu", "openvino_cpu", "openvino_gpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--key-every", type=int, default=2)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--max-side", type=int, default=720,
                        help="Canh toi da xu ly, giong video_matting.py")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 = tat ca frame; VD 40 de thu nhanh")
    parser.add_argument("--static-threshold", type=int, default=6,
                        help="Nguong chenh lech anh xam de coi pixel dung yen (0..255)")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--compare-backends", action="store_true",
                        help="Benchmark CPU/OpenVINO va ghi backend thuc te")
    parser.add_argument("--bench-repeats", type=int, default=10)
    parser.add_argument("--self-test", action="store_true", help="Kiem tra code khong can model")
    return parser.parse_args()


# Điều phối toàn bộ quá trình: nạp mô hình, chạy thử nghiệm, lưu CSV và biểu đồ.
def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.threads < 1 or args.key_every < 1 or args.max_side < 0 or args.max_frames < 0:
        raise ValueError("Kiem tra --threads, --key-every, --max-side, --max-frames")
    if not 0 <= args.static_threshold <= 255:
        raise ValueError("--static-threshold phai trong [0,255]")
    if args.bench_repeats < 1:
        raise ValueError("--bench-repeats phai >= 1")

    videos = find_videos(args)
    images_dir = to_path(args.images_dir)
    masks_dir = to_path(args.masks_dir)
    output_dir = to_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_images and args.skip_videos and not args.compare_backends:
        raise ValueError("Khong co tac vu nao duoc chon")
    if not videos and not args.skip_videos:
        print("[CANH BAO] Chua chi dinh video. Dung --video hoac --videos-dir de do A/B/C/D.")

    # Nạp mô hình và chạy thử vài lần trước khi đo để ổn định kết quả.
    from model_loader import U2NetPModel
    model = U2NetPModel(backend=args.backend, num_threads=args.threads)
    # Warm-up truoc khi tinh; khong dung thoi gian tai model.
    example = sample_frame_from_video_or_image(videos, images_dir)
    if example is not None:
        print("[WARMUP] Chay model 3 lan de on dinh cache va tai nguyen.")
        for _ in range(3):
            model.predict_mask(example)

    # Thực nghiệm Video: so sánh bốn cấu hình A/B/C/D.
    if not args.skip_videos:
        rows: list[dict[str, Any]] = []
        settings = ablation_configs(args.key_every, args.radius, args.eps)
        for video_path in videos:
            print(f"\n[VIDEO] {video_path.name}")
            for label, config in settings.items():
                try:
                    row = evaluate_video(
                        video_path, label, config, model,
                        max_side=args.max_side, max_frames=args.max_frames,
                        static_threshold=args.static_threshold,
                    )
                    rows.append(row)
                    print(f"  {label}: FPS={row['fps_pipeline_no_encode']:.2f}; "
                          f"Flicker={row['flicker_static_percent']:.4f}%; "
                          f"AI={row['keyframes']}/{row['frames']}")
                except (ValueError, cv2.error, RuntimeError) as exc:
                    print(f"  [CANH BAO] {label} khong chay duoc: {exc}")

        if rows:
            details = pd.DataFrame(rows)
            details.to_csv(output_dir / "ablation_video.csv", index=False, encoding="utf-8-sig")
            summary = (details.groupby("config", as_index=False, sort=True)
                       .agg(videos=("video", "count"),
                            fps_mean=("fps_pipeline_no_encode", "mean"),
                            fps_min=("fps_pipeline_no_encode", "min"),
                            flicker_mean=("flicker_static_percent", "mean"),
                            static_fraction_mean=("static_pixel_fraction", "mean"),
                            total_keyframes=("keyframes", "sum"),
                            total_frames=("frames", "sum")))
            summary = summary.round(4)
            summary.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
            print("\n[BANG TONG HOP VIDEO]")
            print(summary.to_string(index=False))
            plot_bar(summary, "fps_mean", "FPS (khong bao gom ghi/ma hoa video)",
                     output_dir / "fps_ablation.png")
            plot_bar(summary, "flicker_mean", "Flicker tren vung dung yen (% alpha)",
                     output_dir / "flicker_ablation.png", low_is_better=True)

    # Thực nghiệm Ảnh: tính MAE/IoU khi có nhãn chuẩn độc lập.
    if not args.skip_images:
        image_rows = evaluate_images(images_dir, masks_dir, model, args.radius, args.eps)
        if image_rows:
            pd.DataFrame(image_rows).to_csv(output_dir / "ablation_image.csv", index=False, encoding="utf-8-sig")
            image_summary = pd.DataFrame(image_rows).groupby("config")[["mae", "iou"]].mean().round(4)
            print("\n[BANG TONG HOP ANH (MAE NHO / IOU LON LA TOT)]")
            print(image_summary.to_string())

    # Thực nghiệm backend: so sánh tốc độ CPU/OpenVINO khi được yêu cầu.
    if args.compare_backends:
        if example is None:
            print("[CANH BAO] Can it nhat 1 anh hoac video de benchmark backend.")
        else:
            run_backend_benchmarks(example, output_dir, args.threads, args.bench_repeats)

    print(f"\n[HOAN THANH] Ket qua do that o: {output_dir}")
    print("[GHI CHU] Flicker la proxy tren pixel dung yen, khong phai chi so video GT chuan.")
    print("[GHI CHU] Khong duoc ket luan AETM tot hon neu so lieu khong ho tro.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"[LOI] {exc}", file=sys.stderr)
        sys.exit(1)
