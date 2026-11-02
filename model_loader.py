"""
model_loader.py - Nap va chay U2Net-P (ONNX)

- Mac dinh ONNX Runtime CPU; khong can CUDA.
- Co the thu OpenVINO CPU/GPU neu da cai 'openvino'.
- Neu OpenVINO bi loi, tu chuyen ve CPU.
- predict_mask(anh_bgr) -> mask float32 [0, 1], cung kich thuoc anh goc.
- Model duoc tai DUY NHAT lan dau, luu trong models/u2netp.onnx.

"""

# ===== THƯ VIỆN =====
from __future__ import annotations

import argparse
import hashlib
import shutil
import statistics
import time
import urllib.request
import warnings
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


# ===== THƯ MỤC MÔ HÌNH VÀ THÔNG SỐ CHUẨN HÓA =====
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "u2netp.onnx"
FULL_MODEL_PATH = MODEL_DIR / "u2net.onnx"
OUTPUT_DIR = PROJECT_DIR / "outputs"

# URL va MD5 theo ma nguon rembg (u2netp.py).
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
MODEL_MD5 = "8e83ca70e441ab06c318d82300c84806"
FULL_MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
FULL_MODEL_MD5 = "60024c5c889badc19c04ad937298a77b"
INPUT_SIZE = (320, 320)  # (width, height)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===== KIỂM TRA VÀ TẢI MÔ HÌNH =====
# Loại ngoại lệ riêng để báo lỗi tải, nạp hoặc chạy mô hình AI.
class ModelError(RuntimeError):
    """Loi lien quan den viec tai, nap hoac chay model."""


# Tính mã MD5 để kiểm tra tính toàn vẹn của tệp mô hình.
def _md5_file(path: Path) -> str:
    """Tinh MD5 theo tung khoi de khong doc ca model vao RAM."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Kiểm tra mô hình đã có; nếu thiếu thì tìm trong cache hoặc tải về và xác minh MD5.
def ensure_model(model_path: str | Path = MODEL_PATH) -> Path:
    """Tai model u2netp hoac u2net; doi chieu MD5 va luu de tai su dung."""
    model_path = Path(model_path).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.name == 'u2netp.onnx':
        model_url, expected_md5 = MODEL_URL, MODEL_MD5
    elif model_path.name == 'u2net.onnx':
        model_url, expected_md5 = FULL_MODEL_URL, FULL_MODEL_MD5
    else:
        raise ModelError(f"Khong ho tro model: {model_path.name}")

    if model_path.is_file():
        if _md5_file(model_path) != expected_md5:
            raise ModelError(f"Model sai checksum: {model_path}. Hay xoa file hong va tai lai.")
        print(f"[MODEL] Da co, khong tai lai: {model_path}")
        return model_path

    cache_paths = [
        Path.home() / '.u2net' / model_path.name,
        Path.home() / '.rembg' / 'models' / model_path.stem / model_path.name,
        Path.home() / '.rembg' / model_path.name,
    ]
    for cached in cache_paths:
        if cached.is_file() and _md5_file(cached) == expected_md5:
            shutil.copy2(cached, model_path)
            print(f"[MODEL] Da sao chep model tu cache: {cached}")
            return model_path

    print(f"[MODEL] Dang tai lan dau: {model_path.name} (se chi tai mot lan)")
    print(f"[MODEL] Nguon: {model_url}")
    temporary_path = model_path.with_suffix('.onnx.part')
    try:
        request = urllib.request.Request(
            model_url, headers={'User-Agent': 'AETM/2.8'}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary_path.open('wb') as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
        if _md5_file(temporary_path) != expected_md5:
            raise ModelError('File model tai ve sai checksum MD5.')
        temporary_path.replace(model_path)
        print(f"[MODEL] Tai thanh cong: {model_path}")
        return model_path
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise ModelError(
            f"Khong tai duoc {model_path.name}. Kiem tra Internet/GitHub. "
            f"Tai thu cong: {model_url}; luu vao: {model_path}. Chi tiet: {exc}"
        ) from exc


# ===== NẠP MÔ HÌNH VÀ TẠO MASK =====
# Lớp U2Net-P: khởi tạo mô hình, tiền xử lý, suy luận và tạo mask.
class U2NetPModel:
    """Boc model U2Net-P, cho phep thay doi backend ma khong doi API."""

    # Chọn CPU/OpenVINO, nạp mô hình; nếu OpenVINO lỗi thì chuyển sang CPU.
    def __init__(
        self,
        backend: str = "cpu",
        num_threads: int = 4,
        model_path: str | Path = MODEL_PATH,
    ) -> None:
        if backend not in ("cpu", "openvino_cpu", "openvino_gpu"):
            raise ValueError("backend phai la cpu, openvino_cpu hoac openvino_gpu")
        if num_threads < 1:
            raise ValueError("num_threads phai >= 1")

        self.model_path = ensure_model(model_path)
        self.requested_backend = backend
        self.backend = "cpu"
        self.num_threads = num_threads
        self.last_inference_ms = 0.0
        self.last_total_ms = 0.0

        self._ort_session = None
        self._ort_input_name = None
        self._ov_request = None

        if backend == "cpu":
            self._init_cpu()
        else:
            try:
                self._init_openvino(backend)
                # Kiem tra suy luan that, vi viec nhan GPU khong dam bao model chay.
                dummy = np.zeros((1, 3, 320, 320), dtype=np.float32)
                self._run_raw(dummy)
            except Exception as exc:
                warnings.warn(
                    f"OpenVINO khong san sang ({exc}). Chuyen ve ONNX Runtime CPU.",
                    RuntimeWarning,
                )
                self._ov_request = None
                self._init_cpu()

        print(f"[MODEL] Backend dang su dung: {self.backend}")

    # Thiết lập ONNX Runtime để chạy mô hình bằng CPU.
    def _init_cpu(self) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.num_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._ort_input_name = session.get_inputs()[0].name
            self._ort_session = session
            self.backend = "cpu"
        except Exception as exc:
            raise ModelError(f"Khong nap duoc ONNX Runtime CPU: {exc}") from exc

    # Nạp mô hình vào OpenVINO CPU/GPU khi máy hỗ trợ.
    def _init_openvino(self, backend: str) -> None:
        try:
            import openvino as ov  # Thu vien tuy chon, khong co trong requirements co ban.
        except ImportError as exc:
            raise ModelError("Chua cai openvino: python -m pip install openvino") from exc

        device = "GPU" if backend == "openvino_gpu" else "CPU"
        core = ov.Core()
        if device not in core.available_devices:
            raise ModelError(f"OpenVINO khong tim thay {device}: {core.available_devices}")
        compiled = core.compile_model(str(self.model_path), device)
        self._ov_request = compiled.create_infer_request()
        self._ort_session = None
        self.backend = backend

    # Chạy mô hình trên backend hiện tại và lấy kết quả dự đoán thô.
    def _run_raw(self, tensor: np.ndarray) -> np.ndarray:
        """Chay inference va tra ve output dau tien cua model."""
        if self.backend == "cpu":
            return self._ort_session.run(None, {self._ort_input_name: tensor})[0]
        # OpenVINO chap nhan index 0 la input dau tien.
        self._ov_request.infer({0: tensor})
        return np.array(self._ov_request.get_output_tensor(0).data, copy=True)

    # Chuyển ảnh BGR thành tensor RGB đã chuẩn hóa, kích thước [1, 3, 320, 320].
    @staticmethod
    def preprocess(image_bgr: np.ndarray) -> np.ndarray:
        """BGR uint8 -> [1, 3, 320, 320] float32, theo rembg U2Net-P."""
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.dtype != np.uint8
            or image_bgr.shape[0] == 0
            or image_bgr.shape[1] == 0
        ):
            raise ValueError("Can anh BGR uint8, shape (H,W,3), khong rong.")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = Image.fromarray(rgb).resize(INPUT_SIZE, Image.Resampling.LANCZOS)
        data = np.asarray(resized, dtype=np.float32)

        # Giong rembg: chia cho max cua anh sau resize, khong phai chia co dinh 255.
        data = data / max(float(data.max()), 1e-6)
        data = (data - MEAN) / STD
        data = np.transpose(data, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(data, dtype=np.float32)

    # Đưa kết quả dự đoán về mask alpha [0, 1] có kích thước bằng ảnh đầu vào.
    @staticmethod
    def postprocess(output: np.ndarray, height: int, width: int) -> np.ndarray:
        """Output model -> mask float32 [0,1] cung kich thuoc anh goc."""
        raw = np.asarray(output, dtype=np.float32)
        # U2Net-P output dau tien thuong co shape [1,1,320,320].
        mask = np.squeeze(raw)
        if mask.ndim != 2:
            raise ModelError(f"Output model khong co dang mask 2D: {raw.shape}")

        minimum = float(mask.min())
        maximum = float(mask.max())
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ModelError("Output model co gia tri NaN/Inf.")
        if maximum - minimum < 1e-8:
            mask = np.zeros_like(mask, dtype=np.float32)
        else:
            mask = (mask - minimum) / (maximum - minimum)

        # Phong mask ve do phan giai goc de file tiep theo lam Guided Filter.
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_CUBIC)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    # Tạo mask: tiền xử lý ảnh → chạy AI → hậu xử lý; đồng thời đo thời gian.
    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        """Tra mask float32 (H,W) trong khoang [0,1].

        Input: khung hinh BGR tu OpenCV, shape (H,W,3), dtype uint8.
        Khong nguong hoa mask o day: giu alpha mem de refine ve sau.
        """
        t0 = time.perf_counter()
        tensor = self.preprocess(image_bgr)
        t1 = time.perf_counter()
        try:
            raw = self._run_raw(tensor)
        except Exception as exc:
            if self.backend == "cpu":
                raise ModelError(f"Suy luan CPU bi loi: {exc}") from exc
            warnings.warn(
                f"Suy luan OpenVINO bi loi ({exc}). Thu lai bang CPU.",
                RuntimeWarning,
            )
            self._ov_request = None
            self._init_cpu()
            raw = self._run_raw(tensor)
        t2 = time.perf_counter()
        mask = self.postprocess(raw, image_bgr.shape[0], image_bgr.shape[1])
        t3 = time.perf_counter()
        self.last_inference_ms = (t2 - t1) * 1000.0
        self.last_total_ms = (t3 - t0) * 1000.0
        return mask

    # Chạy thử nhiều lần trên một ảnh để đo thời gian và FPS xử lý.
    def benchmark(
        self,
        image_bgr: np.ndarray,
        warmup: int = 3,
        repeats: int = 10,
    ) -> dict:
        """Do FPS pipeline (preprocess + inference + postprocess) tren anh cung cap.

        Chi so do tren chinh may chay, khong phai con so uoc tinh.
        """
        if warmup < 0 or repeats < 1:
            raise ValueError("warmup phai >= 0 va repeats phai >= 1")
        for _ in range(warmup):
            self.predict_mask(image_bgr)

        total_times = []
        infer_times = []
        for _ in range(repeats):
            self.predict_mask(image_bgr)
            total_times.append(self.last_total_ms)
            infer_times.append(self.last_inference_ms)

        avg_ms = statistics.mean(total_times)
        return {
            "backend": self.backend,
            "repeats": repeats,
            "pipeline_fps": round(1000.0 / avg_ms, 2),
            "pipeline_ms_mean": round(avg_ms, 2),
            "pipeline_ms_median": round(statistics.median(total_times), 2),
            "inference_ms_mean": round(statistics.mean(infer_times), 2),
        }



# ===== MÔ HÌNH U2NET ĐẦY ĐỦ =====
# Lớp U2Net đầy đủ: kế thừa cách xử lý nhưng dùng trọng số mô hình khác.
class U2NetFullModel(U2NetPModel):
    """U2Net ONNX day du cho anh/video kho. Van 320x320; cham hon u2netp."""

    # Khởi tạo U2Net đầy đủ từ tệp trong thư mục models.
    def __init__(self, backend: str = 'cpu', num_threads: int = 4):
        super().__init__(backend=backend, num_threads=num_threads,
                         model_path=FULL_MODEL_PATH)

# ===== CHẠY THỬ RIÊNG FILE MODEL_LOADER.PY =====
# Đọc ảnh bằng OpenCV, hỗ trợ đường dẫn Windows có dấu tiếng Việt.
def _read_image_unicode(path: Path) -> np.ndarray:
    """Doc anh ho tro duong dan co dau tieng Viet tren Windows."""
    if not path.is_file():
        raise FileNotFoundError(f"Khong tim thay anh: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV khong doc duoc anh: {path}")
    return image


# Nhận lệnh Terminal, nạp mô hình, thử tạo mask và tùy chọn đo hiệu suất.
def main() -> None:
    parser = argparse.ArgumentParser(description="Thu U2Net-P tren CPU/OpenVINO")
    parser.add_argument("--image", type=str, help="Duong dan anh de thu, vi du samples/images/test.jpg")
    parser.add_argument(
        "--backend",
        choices=["cpu", "openvino_cpu", "openvino_gpu"],
        default="cpu",
    )
    parser.add_argument("--threads", type=int, default=4, help="So luong CPU thread")
    parser.add_argument("--benchmark", action="store_true", help="Do FPS sau warm-up")
    args = parser.parse_args()

    model = U2NetPModel(backend=args.backend, num_threads=args.threads)
    if not args.image:
        print("[OK] Model da nap. De xuat mask, chay:")
        print("python model_loader.py --image samples/images/test.jpg --benchmark")
        return

    image_path = Path(args.image).expanduser()
    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path
    image = _read_image_unicode(image_path)
    mask = model.predict_mask(image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"{image_path.stem}_mask.png"
    success, buffer = cv2.imencode(".png", (mask * 255).astype(np.uint8))
    if not success:
        raise ModelError("Khong ma hoa duoc PNG mask.")
    buffer.tofile(str(save_path))

    print(f"[OK] Da luu mask: {save_path}")
    print(f"[THOI GIAN] Inference: {model.last_inference_ms:.1f} ms")
    print(f"[THOI GIAN] Toan pipeline: {model.last_total_ms:.1f} ms")
    if args.benchmark:
        print("[BENCHMARK]", model.benchmark(image))


if __name__ == "__main__":
    main()
