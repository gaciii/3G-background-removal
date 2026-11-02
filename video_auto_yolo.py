
from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import time
from typing import Any, Callable
import uuid

import cv2
import numpy as np

# ĐƯỜNG DẪN VÀ TÊN MÔ HÌNH AI
ROOT = Path(__file__).resolve().parent
MODEL_FILENAME = 'yolo26n-seg.pt'


# CẤU HÌNH XỬ LÝ VIDEO
@dataclass
# Lưu các tham số chạy YOLO và tùy chọn xử lý video.
class AutoConfig:
    video_path: str
    output_path: str
    mode: str = 'blur'  # mask, blur, color, image
    blur_strength: int = 19
    bg_color: str = '#DDEBFF'
    background_path: str | None = None
    max_side: int = 640
    imgsz: int = 640
    confidence: float = .15
    safe_fallback: bool = True
    preserve_audio: bool = False
    max_seconds: float = 30.0
    retry_hard_frames: bool = True
    refine_edges: bool = True
    temporal_rescue: bool = True
    max_rescue_gap: int = 16
    stabilize_masks: bool = True
    quality_guard: bool = True


# ĐỌC VIDEO VÀ CÁC HÀM HỖ TRỢ
# Đọc thông tin video: FPS, số frame, độ phân giải và thời lượng.
def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Không đọc được video. Hãy dùng MP4 hợp lệ.')
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 1 or h <= 1:
            raise ValueError('Video không có kích thước hợp lệ.')
        return dict(fps=fps, frames=n, width=w, height=h,
                    duration=n / fps if n > 0 else 0)
    finally:
        cap.release()


# Thu nhỏ video theo kích thước tối đa và bảo đảm chiều rộng/cao là số chẵn.
def dimensions(w: int, h: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, float(max_side) / max(w, h))
    return (max(2, int(round(w * scale)) // 2 * 2),
            max(2, int(round(h * scale)) // 2 * 2))


# Chuyển dữ liệu tensor của mô hình AI thành mảng NumPy.
def _as_numpy(tensor: Any) -> np.ndarray:
    for method in ('detach', 'cpu'):
        if hasattr(tensor, method):
            tensor = getattr(tensor, method)()
    return np.asarray(tensor)


# Tính IoU giữa hai hộp bao để đánh giá mức độ trùng nhau.
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    intersection = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0,a[2]-a[0]) * max(0.0,a[3]-a[1])
    ab = max(0.0,b[2]-b[0]) * max(0.0,b[3]-b[1])
    return float(intersection / max(aa+ab-intersection, 1e-6))


# Tìm mã lớp "cat" trong danh sách nhãn của YOLO.
def _cat_id(names) -> int:
    if isinstance(names, dict):
        items = names.items()
    else:
        items = enumerate(names)
    for idx, name in items:
        if str(name).strip().lower() == 'cat':
            return int(idx)
    raise ValueError('Mô hình không có lớp "cat". Hãy dùng yolo26n-seg.pt huấn luyện COCO.')


# NHẬN DIỆN VÀ TÁCH MASK BẰNG YOLO
# Lớp phát hiện và tách mask con mèo trên từng khung hình bằng YOLO-seg.
class CatSegmenter:
    # Khởi tạo mô hình, kích thước ảnh AI và ngưỡng tin cậy.
    def __init__(self, model: Any, imgsz: int = 640, confidence: float = .15):
        self.model = model
        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.cat_id = _cat_id(model.names)

    # Dự đoán mask mèo, ưu tiên đối tượng liên tục giữa các frame.
    def predict(self, frame: np.ndarray, previous_box: np.ndarray | None = None,
                *, imgsz: int | None = None, confidence: float | None = None,
                pad_ratio: float = 0.0):
        """Infer cat mask, optionally padded for close-up retry; crop back exactly."""
        h, w = frame.shape[:2]
        px, py = int(round(w*pad_ratio)), int(round(h*pad_ratio))
        if px or py:
            source=cv2.copyMakeBorder(frame,py,py,px,px,
                                     cv2.BORDER_CONSTANT,value=(114,114,114))
        else:
            source=frame
        result = self.model.predict(source=source, device='cpu',
                                    imgsz=int(imgsz or self.imgsz),
                                    conf=float(confidence if confidence is not None else self.confidence),
                                    classes=[self.cat_id], retina_masks=True,
                                    max_det=8, verbose=False)[0]
        if result.boxes is None or result.masks is None:
            return None, None, 0.0
        boxes = _as_numpy(result.boxes.xyxy)
        class_ids = _as_numpy(result.boxes.cls).reshape(-1)
        scores = _as_numpy(result.boxes.conf).reshape(-1)
        masks = _as_numpy(result.masks.data)
        if masks.ndim == 2:
            masks = masks[None]
        opts=[]
        for i in range(min(len(boxes),len(scores),len(class_ids),len(masks))):
            if int(class_ids[i]) != self.cat_id:
                continue
            box=np.asarray(boxes[i],dtype=np.float32).copy()
            box[0::2]-=px;box[1::2]-=py
            box[0::2]=np.clip(box[0::2],0,w)
            box[1::2]=np.clip(box[1::2],0,h)
            area=max(0.,float((box[2]-box[0])*(box[3]-box[1])/(w*h)))
            if area<.003:
                continue
            continuity=_iou(box,previous_box)
            weight=float(scores[i])*(.6+.4*np.sqrt(min(area,1.)))+.25*continuity
            opts.append((weight,i,box))
        if not opts:
            return None, None, 0.0
        _,i,box=max(opts,key=lambda x:x[0])
        mask=np.asarray(masks[i],dtype=np.float32)
        ph,pw=source.shape[:2]
        if mask.shape!=(ph,pw):
            mask=cv2.resize(mask,(pw,ph),interpolation=cv2.INTER_LINEAR)
        if px or py:
            mask=mask[py:py+h,px:px+w]
        mask=np.clip(np.nan_to_num(mask,nan=0.0,posinf=0.0,neginf=0.0),0,1)
        binary=(mask>.47).astype(np.uint8)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        mask=cv2.GaussianBlur(binary.astype(np.float32),(0,0),1.0)
        return mask,box,float(scores[i])


# Tải mô hình YOLO trong môi trường riêng và cấu hình số luồng CPU.
def load_model():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Chưa cài AI tự động. Chạy setup_yolo.bat một lần trong D:\\TGMT.') from exc
    try:
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 2))
    except ImportError:
        pass
    cv2.setNumThreads(2)
    model_dir=ROOT/'models'
    model_dir.mkdir(exist_ok=True,parents=True)
    old_dir=Path.cwd()
    try:
        os.chdir(model_dir)
        # The official pretrained model downloads on first use, subsequent calls are local.
        model=YOLO(MODEL_FILENAME)
    finally:
        os.chdir(old_dir)
    return model


# KIỂM TRA MASK VÀ TINH CHỈNH VIỀN
# Kiểm tra mask bất thường; bỏ qua kết quả quá nhỏ hoặc kém tin cậy.
def safe_mask(mask: np.ndarray | None, score: float,
              recent_areas: deque, *, safe: bool = True) -> tuple[np.ndarray | None,str]:
    if not safe:
        return mask, ('detected' if mask is not None else 'undetected')
    if mask is None:
        return None, 'no_cat_detected'
    a = float(np.mean(mask > .5))
    if a < .008:
        return None, 'mask_too_small'
    if recent_areas:
        typical=float(np.median(recent_areas))
        # A collapse after previously tracking a full animal is unsafe.
        if typical > .07 and a < max(.014, typical*.27):
            return None, 'sudden_mask_loss'
    if score < .12:
        return None, 'low_confidence'
    return mask, 'detected'


# Tinh chỉnh viền mask theo cạnh ảnh, hạn chế quầng mờ quanh đối tượng.
def refine_mask_edge(frame: np.ndarray, mask: np.ndarray, radius: int = 4,
                     eps: float = .012) -> np.ndarray:
    """Edge-aware guided refinement without opencv-contrib or CUDA.

    Does not hallucinate whiskers or missing parts; it only avoids the worst
    halo from uniform Gaussian blur by using intensity edges in the input.
    """
    if mask is None:
        return None
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    p=np.clip(mask.astype(np.float32),0,1)
    k=(2*radius+1,2*radius+1)
    filt=lambda x:cv2.boxFilter(x,cv2.CV_32F,k,normalize=True,
                               borderType=cv2.BORDER_REFLECT)
    mi=filt(image);mp=filt(p)
    cov=filt(image*p)-mi*mp
    var=filt(image*image)-mi*mi
    a=cov/(var+eps)
    b=mp-a*mi
    q=filt(a)*image+filt(b)
    q=np.clip(q,0,1)
    # Never let the edge filter punch a hole inside sure foreground.
    core=cv2.erode((p>.9).astype(np.uint8),np.ones((3,3),np.uint8))>0
    q[core]=1.0
    return q.astype(np.float32)


# Tạo một dòng số liệu đánh giá cho mỗi frame video.
def diagnostic_row(index:int,fps:float,status:str,score:float,mask:np.ndarray|None,
                   retry:bool,first_issue:str) -> dict:
    return {'frame':index,'seconds':round(index/fps,3),'status':status,
            'confidence':round(float(score),4),
            'mask_area_percent':round(100*float(np.mean(mask>.5)),3)
            if mask is not None else 0.0,
            'retry_used':bool(retry),'first_pass_status':first_issue}


# GHÉP NỀN VÀ XUẤT VIDEO
# Đổi mã màu HEX thành BGR để dùng với OpenCV.
def color_bgr(hexcolor: str) -> tuple[int,int,int]:
    text=str(hexcolor).lstrip('#')
    if len(text)==3:
        text=''.join(x*2 for x in text)
    if len(text)!=6:
        raise ValueError('Màu nền phải có dạng #RRGGBB')
    try:
        rgb=tuple(int(text[i:i+2],16) for i in (0,2,4))
    except ValueError as exc:
        raise ValueError('Mã màu không hợp lệ') from exc
    return (rgb[2],rgb[1],rgb[0])


# Ghép đối tượng với nền mờ, nền màu, nền ảnh hoặc xuất mask.
def render(frame: np.ndarray, mask: np.ndarray, cfg: AutoConfig,
           bg: np.ndarray | None = None) -> np.ndarray:
    h,w = frame.shape[:2]
    m=np.clip(mask,0,1).astype(np.float32)
    if cfg.mode=='mask':
        return cv2.cvtColor(np.uint8(np.round(m*255)),cv2.COLOR_GRAY2BGR)
    if cfg.mode=='blur':
        sigma=float(max(2,min(45,cfg.blur_strength)))
        backdrop=cv2.GaussianBlur(frame,(0,0),sigmaX=sigma,sigmaY=sigma)
    elif cfg.mode=='color':
        backdrop=np.full_like(frame,color_bgr(cfg.bg_color))
    elif cfg.mode=='image':
        if bg is None:
            raise ValueError('Chưa chọn ảnh nền thay thế.')
        backdrop=cv2.resize(bg,(w,h),interpolation=cv2.INTER_LINEAR)
    else:
        raise ValueError(f'Chế độ xuất không hợp lệ: {cfg.mode}')
    return np.uint8(np.clip(frame*m[:,:,None] + backdrop*(1-m[:,:,None]),0,255))


# Tìm chương trình FFmpeg dùng để mã hóa video.
def ffmpeg_bin() -> str | None:
    binary=shutil.which('ffmpeg')
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError,RuntimeError,OSError):
        return None


# Chuyển video đầu ra sang H.264, tùy chọn giữ âm thanh gốc.
def to_h264(temp: Path, output: Path, source: str, audio: bool) -> str:
    ff=ffmpeg_bin()
    if ff:
        cmd=[ff,'-hide_banner','-loglevel','error','-y','-i',str(temp)]
        if audio:
            cmd+=['-i',source,'-map','0:v:0','-map','1:a:0?']
        else:
            cmd+=['-map','0:v:0','-an']
        cmd+=['-c:v','libx264','-crf','22','-preset','veryfast','-pix_fmt','yuv420p',
              '-movflags','+faststart']
        if audio:
            cmd+=['-c:a','aac','-shortest']
        cmd += [str(output)]
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=360)
        if run.returncode==0 and output.exists() and output.stat().st_size>512:
            temp.unlink(missing_ok=True)
            return 'H.264'
        print('WARN FFmpeg:',run.stderr[-600:],flush=True)
    output.unlink(missing_ok=True)
    temp.replace(output)
    return 'mp4v'


# Đổi kích thước frame sang độ phân giải xử lý đã chọn.
def _resize_video_frame(raw: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(raw, (width, height), interpolation=(
        cv2.INTER_AREA if max(raw.shape[:2]) > max(height, width) else cv2.INTER_LINEAR))


# THEO DÕI MASK THEO THỜI GIAN BẰNG OPTICAL FLOW
# Dùng Optical Flow chuyển mask từ frame trước sang frame sau.
def _flow_warp(source_mask: np.ndarray, frame_source: np.ndarray,
               frame_target: np.ndarray, flow_side: int = 400) -> np.ndarray:
    """Move a mask from source to target using *backward* target->source flow.

    Restrict motion estimation to a 400px image for CPU speed, then upscale
    the propagated mask. No object may be added by this operation.
    """
    h, w = frame_source.shape[:2]
    scale = min(1.0, flow_side / max(w, h))
    fw = max(24, round(w * scale)); fh = max(24, round(h * scale))
    size = (fw, fh)
    a = cv2.cvtColor(cv2.resize(frame_target, size), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(frame_source, size), cv2.COLOR_BGR2GRAY)
    # Farneback returns where each pixel in target corresponds in source.
    flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 21, 3, 5, 1.2, 0)
    gx, gy = np.meshgrid(np.arange(fw, dtype=np.float32),
                         np.arange(fh, dtype=np.float32))
    moving = cv2.resize(source_mask.astype(np.float32), size,
                        interpolation=cv2.INTER_LINEAR)
    result = cv2.remap(moving, gx+flow[:,:,0], gy+flow[:,:,1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=0)
    return cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR).clip(0, 1)


# Phục hồi đoạn mất mask bằng dự đoán hai chiều và kiểm tra độ khớp.
def _recover_segment(frames: list[np.ndarray], first_mask: np.ndarray,
                     last_mask: np.ndarray,
                     *, agreement_min: float = .33) -> dict[int, tuple[np.ndarray, float]]:
    """Recover ONLY when forward/backward propagation agrees per frame.

    Frames include two accepted anchor frames, one at each end. The output
    is keyed by the *local* index; anchors are excluded. A failure preserves
    the original frame rather than inventing an unreliable segmentation.
    """
    if len(frames) < 3:
        return {}
    count = len(frames)
    fwd = [None] * count; rev = [None] * count
    fwd[0] = (first_mask.astype(np.float32) / 255.).clip(0, 1)
    rev[-1] = (last_mask.astype(np.float32) / 255.).clip(0, 1)
    for j in range(1, count):
        fwd[j] = _flow_warp(fwd[j-1], frames[j-1], frames[j])
    for j in range(count-2, -1, -1):
        rev[j] = _flow_warp(rev[j+1], frames[j+1], frames[j])

    a0 = float(np.mean(fwd[0] > .5)); a1 = float(np.mean(rev[-1] > .5))
    results = {}
    for j in range(1, count-1):
        am = fwd[j] > .5; bm = rev[j] > .5
        union = int(np.count_nonzero(am | bm))
        agreement = float(np.count_nonzero(am & bm)) / max(union, 1)
        if agreement < agreement_min:
            continue
        t = j / (count-1)
        area = (1-t)*a0 + t*a1
        # Reject masks implausibly collapsed or expanded versus two anchors.
        blended = (1-t)*fwd[j] + t*rev[j]
        # A motion warp may lose opacity at boundaries. Agreement gates it,
        # while blending retains the object's soft boundary.
        candidate = np.clip(blended, 0, 1)
        actual = float(np.mean(candidate > .5))
        if actual < max(.015, area * .53) or actual > min(1.001, max(.12, area * 1.55)):
            continue
        if actual > .97 and max(a0, a1) < .92:
            continue
        results[j] = (np.uint8(np.round(candidate*255)), round(agreement, 3))
    return results


# Tìm các đoạn video bị mất mask và thử phục hồi những đoạn ngắn.
def _temporal_recover(video_path: str, mask_files: dict[int, Path],
                      total: int, width: int, height: int,
                      temp_dir: Path, max_gap: int) -> dict[int, float]:
    """Decode the clip once, keeping at most one short missing stretch in RAM."""
    stretches = []
    i = 0
    while i < total:
        if i in mask_files:
            i += 1; continue
        start = i
        while i < total and i not in mask_files:
            i += 1
        stop = i - 1
        if start > 0 and stop+1 < total and stop-start+1 <= max_gap:
            stretches.append((start-1, stop+1))
    if not stretches:
        return {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    recovered = {}
    active = None; seq = []
    stretches_iter = iter(stretches)
    upcoming = next(stretches_iter, None)
    idx = 0
    try:
        while upcoming is not None:
            ok, raw = cap.read()
            if not ok: break
            frame = _resize_video_frame(raw, width, height)
            if idx == upcoming[0]:
                active = upcoming; seq = []
            if active is not None:
                seq.append(frame)
                if idx == active[1]:
                    left, right = active
                    first = cv2.imread(str(mask_files[left]), cv2.IMREAD_GRAYSCALE)
                    last = cv2.imread(str(mask_files[right]), cv2.IMREAD_GRAYSCALE)
                    if first is not None and last is not None:
                        for offset, (mask, agree) in _recover_segment(seq, first, last).items():
                            frame_index = left + offset
                            path = temp_dir / f'{frame_index:06d}.png'
                            if cv2.imwrite(str(path), mask):
                                mask_files[frame_index] = path
                                recovered[frame_index] = agree
                    active = None; seq = []
                    upcoming = next(stretches_iter, None)
            idx += 1
    finally:
        cap.release()
    return recovered



# Tính độ trùng khớp giữa hai mask nhị phân.
def _mask_agreement(a: np.ndarray, b: np.ndarray, threshold: float = .5) -> float:
    """Binary IoU of two float alpha masks, zero when neither has foreground."""
    aa = a > threshold; bb = b > threshold
    union = int(np.count_nonzero(aa | bb))
    return float(np.count_nonzero(aa & bb)) / union if union else 0.0


# ỔN ĐỊNH MASK GIỮA CÁC KHUNG HÌNH
# Kiểm tra độ ổn định mask một frame với hai frame lân cận.
def _stabilize_one(previous: tuple[np.ndarray,np.ndarray,dict] | None,
                   current: tuple[np.ndarray,np.ndarray,dict],
                   following: tuple[np.ndarray,np.ndarray,dict] | None,
                   *, jump_threshold: float = .18) -> tuple[np.ndarray | None,dict]:
    """Safely refine one frame using independently decoded adjacent masks.

    NEVER shrink solely because an animal grows in the frame. Motion-warped
    before/after masks must agree AND (a) the mask is recovered/low-confidence
    or (b) it is a clear isolated spike. Mask confidence labels in the CSV are
    not ground truth, so unverified candidates are flagged, not auto-trimmed.
    """
    frame, alpha, row = current
    result = {'stability_flag':'ok','stabilized':False,
              'neighbor_agreement':'', 'stability_change_percent':0.0}
    if alpha is None:
        result['stability_flag']='no_mask'
        return None, result
    if previous is None or following is None:
        result['stability_flag']='boundary_frame'
        return None, result
    prev_frame, prev_alpha, prev_row = previous
    next_frame, next_alpha, next_row = following
    if prev_alpha is None or next_alpha is None:
        result['stability_flag']='no_two_neighbors'
        return None, result
    area = float(np.mean(alpha > .5))
    ap = float(np.mean(prev_alpha > .5)); an = float(np.mean(next_alpha > .5))
    jump = max(abs(area-ap),abs(area-an))
    recovered = str(row['status']) == 'recovered_temporal'
    confidence = float(row.get('confidence') or 0.0)
    low_trust = recovered or confidence < .5
    isolated_spike = (abs(ap-an) < .085 and
                      abs(area - .5*(ap+an)) >= jump_threshold)
    suspect = recovered or isolated_spike or (jump >= jump_threshold and low_trust)
    if not suspect:
        if jump >= jump_threshold:
            result['stability_flag']='large_change_kept_possible_zoom'
        return None, result

    # Neighbor warps are estimated in a downscaled space but returned at the
    # original output resolution. Two warps cost CPU, only run for suspects.
    p = _flow_warp(prev_alpha, prev_frame, frame)
    q = _flow_warp(next_alpha, next_frame, frame)
    agree = _mask_agreement(p, q)
    result['neighbor_agreement'] = round(agree, 3)
    if agree < .48:
        result['stability_flag']='review_neighbors_disagree'
        return None, result
    # Strong future+past agreement is evidence, not proof. Work with a
    # conservative motion-registered fusion, not a global erosion/dilation.
    consensus = np.clip(.5*p+.5*q, 0, 1)
    baseline_iou = _mask_agreement(alpha, consensus)
    if recovered:
        if agree >= .68 and area > max(ap,an) + .12:
            # A recovered mask containing much more than *both* reliable
            # neighbors is probably inflated. Suppress the unsupported rim.
            weight = .68
            result['stability_flag']='recovered_inflation_corrected'
        else:
            weight = .38 if agree >= .62 else .22
            result['stability_flag']='recovered_soft_stabilized'
    elif isolated_spike and confidence <= .58 and agree >= .65:
        weight = .66
        result['stability_flag']='isolated_outlier_corrected'
    elif low_trust and jump >= jump_threshold and agree >= .60 and baseline_iou < .72:
        weight = .40
        result['stability_flag']='low_confidence_jump_smoothed'
    else:
        result['stability_flag']='candidate_kept_unverified'
        return None, result

    # Keep hard unanimous interior/background; blend only where at least one
    # source is uncertain. Reduces flicker without softening the whole face.
    fused = np.clip((1-weight)*alpha + weight*consensus, 0, 1)
    sure_fg = (alpha > .92) & (p > .90) & (q > .90)
    sure_bg = (alpha < .08) & (p < .10) & (q < .10)
    fused[sure_fg] = 1.0
    fused[sure_bg] = 0.0
    new_area = float(np.mean(fused > .5))
    # Prevent an algorithmic 'rescue' from creating nearly full-screen cat
    # from a pair of modest masks. Full frame is fine when neighbors are too.
    if new_area > .975 and max(ap,an) < .92:
        result['stability_flag']='review_full_frame_growth'
        return None, result
    change = float(np.mean(np.abs(fused-alpha)))
    if change < .002:
        result['stability_flag']='tiny_change_ignored'
        return None, result
    result['stabilized']=True
    result['stability_change_percent']=round(change*100,3)
    return fused.astype(np.float32),result


# Duyệt video lần hai để ổn định mask và giảm nhấp nháy giữa các frame.
def _stabilize_sequence(video_path: str, mask_files: dict[int,Path],
                        rows: list[dict], total: int, width: int,height: int,
                        temp_dir: Path) -> dict:
    """Second pass, bounded memory: examine a 3-frame window at a time.

    Compute against ORIGINAL masks only. Write refinements to a different
    subdirectory, then update mask file pointers after the scan ends. This
    prevents one correction from recursively distorting future masks.
    """
    from collections import deque as _deque
    dest=temp_dir/'stable'
    dest.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Không đọc được video ở bước ổn định mask.')
    updated={}
    flags={}
    count=0
    n=0
    cache=_deque(maxlen=3)

    # Đọc frame và mask tương ứng để so sánh theo cửa sổ ba frame.
    def load(idx, raw):
        frame=_resize_video_frame(raw,width,height)
        mask_path=mask_files.get(idx)
        arr=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE) if mask_path else None
        if mask_path and arr is None:
            raise RuntimeError(f'Không đọc được mask để ổn định tại frame {idx}.')
        alpha=(arr.astype(np.float32)/255.) if arr is not None else None
        return frame, alpha, rows[idx]

    # Kiểm tra frame giữa và lưu mask đã hiệu chỉnh nếu phù hợp.
    def handle(window, center):
        nonlocal count
        corrected,metadata=_stabilize_one(window[0],window[1],window[2])
        flags[center]=metadata
        if corrected is not None:
            path=dest/f'{center:06d}.png'
            if not cv2.imwrite(str(path),np.uint8(np.round(corrected*255))):
                raise RuntimeError(f'Không ghi được mask đã ổn định ở frame {center}.')
            updated[center]=path
            count+=1
    try:
        while n<total:
            ok,raw=cap.read()
            if not ok:
                raise RuntimeError(f'Không đọc được frame {n} ở bước ổn định mask.')
            cache.append(load(n,raw))
            if n==0:
                flags[0]={'stability_flag':'boundary_frame','stabilized':False,
                          'neighbor_agreement':'','stability_change_percent':0.0}
            if len(cache)==3:
                handle(cache,n-1)
            n+=1
        if total>=2:
            flags[total-1]={'stability_flag':'boundary_frame','stabilized':False,
                            'neighbor_agreement':'','stability_change_percent':0.0}
    finally:
        cap.release()
    mask_files.update(updated)
    for idx,row in enumerate(rows):
        data=flags.get(idx,{'stability_flag':'not_checked','stabilized':False,
                             'neighbor_agreement':'','stability_change_percent':0.0})
        row.update(data)
        if idx in updated:
            corrected=cv2.imread(str(updated[idx]),cv2.IMREAD_GRAYSCALE)
            row['mask_area_percent']=round(100*float(np.mean(corrected>127)),3)
    return {'stabilized_frames':count,
            'review_frames':sum(str(x.get('stability_flag','')).startswith('review')
                                for x in rows),
            'possible_zoom_frames':sum(x.get('stability_flag')=='large_change_kept_possible_zoom'
                                      for x in rows),
            'stabilized_frame_indices':sorted(updated)}


# KIỂM TRA CHẤT LƯỢNG MASK VÀ CẢNH BÁO LỖI
# Tạo các trường thống kê mặc định để đánh giá chất lượng mask.
def _quality_defaults() -> dict:
    return {'quality_flag':'not_checked', 'quality_risk':0,
            'quality_neighbor_iou':'', 'quality_current_iou':'',
            'quality_fragment_percent':0.0,
            'quality_action':'unchanged', 'quality_change_percent':0.0}


# Ước lượng tỷ lệ vùng mask rời rạc, phục vụ cảnh báo nhiễu.
def _component_fragment_percent(alpha: np.ndarray | None) -> float:
    """Foreground fraction outside its largest 8-connected component.

    This is a heuristic: a detached tail or whiskers may be real. We never
    automatically delete components using this signal alone.
    """
    if alpha is None:
        return 0.0
    binary=np.uint8(alpha>.5)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float((areas.sum()-areas.max()) / max(1,areas.sum()) * 100.)


# Phát hiện mask nghi vấn và chỉ sửa khi các frame lân cận đủ nhất quán.
def _quality_one(previous, current, following):
    """Flag questionable segmentation; repair only a verified isolated outlier.

    Uses original neighboring masks and optical-flow registration to account for
    real subject motion. A high risk score is an *inspection priority*, not an
    estimate of segmentation error vs ground truth.
    """
    frame, alpha, row = current
    meta = _quality_defaults()
    if alpha is None:
        meta.update(quality_flag='no_mask',quality_risk=100)
        return None,meta

    area=float(np.mean(alpha>.5))
    fragments=_component_fragment_percent(alpha)
    meta['quality_fragment_percent']=round(fragments,2)
    if previous is None or following is None:
        meta['quality_flag']='boundary_not_verified'
        return None,meta

    prev_frame, prev_alpha, _ = previous
    next_frame, next_alpha, _ = following
    if prev_alpha is None or next_alpha is None:
        meta['quality_flag']='neighbors_missing'
        meta['quality_risk']=65
        return None,meta

    ap=float(np.mean(prev_alpha>.5)); an=float(np.mean(next_alpha>.5))
    jump_prev=abs(area-ap); jump_next=abs(area-an)
    max_jump=max(jump_prev,jump_next)
    isolated=(min(jump_prev,jump_next)>=.12 and abs(ap-an)<=.10)
    suspicious=(max_jump>=.16 or fragments>=7 or
                str(row.get('status'))=='recovered_temporal' or
                float(row.get('confidence') or 0)<.35 or area>=.96)
    if not suspicious:
        meta['quality_flag']='spatial_temporal_ok'
        return None,meta

    # Flow is only used for candidates that need review; CPU-only bounded pass.
    p=_flow_warp(prev_alpha,prev_frame,frame)
    q=_flow_warp(next_alpha,next_frame,frame)
    agreement=_mask_agreement(p,q)
    consensus=np.clip(.5*p+.5*q,0,1)
    current_iou=_mask_agreement(alpha,consensus)
    meta['quality_neighbor_iou']=round(agreement,3)
    meta['quality_current_iou']=round(current_iou,3)
    # Prioritize disagreements, irregular area and disconnected artifacts.
    risk=(.52*(1-current_iou) + .25*min(1,max_jump/.35) +
          .15*min(1,fragments/20) + .08*(1-agreement))
    meta['quality_risk']=int(round(100*np.clip(risk,0,1)))

    recovered=(str(row.get('status'))=='recovered_temporal')
    confidence=float(row.get('confidence') or 0.0)
    # No repair from two disagreeing neighbors. A true zoom or pose change
    # can produce large jumps; require an ISOLATED outlier and high agreement.
    if agreement < .68:
        meta['quality_flag']='review_neighbors_disagree'
        return None,meta
    if area >=.96 and max(ap,an)>=.91:
        meta['quality_flag']='full_frame_possible_closeup'
        return None,meta
    if current_iou >=.72 and fragments < 12:
        meta['quality_flag']='quality_ok_motion_compensated'
        return None,meta
    if not isolated and not (recovered and current_iou<.50 and
                             abs(ap-an)<.085):
        meta['quality_flag']='review_possible_real_motion'
        return None,meta
    if not (recovered or confidence < .65):
        meta['quality_flag']='review_high_confidence_change'
        return None,meta
    if agreement < .78 or current_iou >= .58:
        meta['quality_flag']='review_insufficient_evidence'
        return None,meta

    # Strongly consistent neighbors that both contradict a single weak frame.
    # Retain a small fraction of original alpha for subpixel details.
    candidate=np.clip(.12*alpha + .88*consensus,0,1).astype(np.float32)
    new_area=float(np.mean(candidate>.5))
    expected=(ap+an)/2.
    if new_area < max(.008, .57*expected) or new_area > min(1,.975,max(.09,1.48*expected)):
        meta['quality_flag']='review_candidate_area'
        return None,meta
    if _mask_agreement(candidate,consensus) < .80:
        meta['quality_flag']='review_candidate_shape'
        return None,meta
    change=100*float(np.mean(np.abs(candidate-alpha)))
    if change < .4:
        meta['quality_flag']='quality_ok_tiny_difference'
        return None,meta
    meta.update(quality_flag='isolated_mask_repaired',
                quality_action='repair_with_two_neighbors',
                quality_change_percent=round(change,3))
    return candidate,meta


# Kiểm tra chất lượng toàn video và ghi lại các frame cần xem lại.
def _quality_sequence(video_path: str, mask_files: dict[int,Path],
                      rows: list[dict], total: int, width: int, height: int,
                      temp_dir: Path) -> dict:
    """One streaming 3-frame pass, bounded memory and atomic mask pointer update."""
    dest=temp_dir/'quality_v39'
    dest.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError('Không đọc được video trong bước kiểm tra mask v3.9.')
    buffer=deque(maxlen=3)
    revised={}
    stats={'quality_checked_frames':0, 'quality_suspect_frames':0,
           'quality_repaired_frames':0,'quality_review_frames':0,
           'quality_times_to_review':[]}

    # Đọc một frame cùng mask và kết quả đánh giá ban đầu.
    def get(index,raw):
        frame=_resize_video_frame(raw,width,height)
        path=mask_files.get(index)
        mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE) if path else None
        if path is not None and mask is None:
            raise RuntimeError('Không đọc được mask ở bước kiểm tra v3.9: frame '+str(index))
        alpha=mask.astype(np.float32)/255. if mask is not None else None
        return frame,alpha,rows[index]

    # Đánh giá, thống kê và lưu mask đã sửa của một frame.
    def check(prev,cur,nxt,index):
        proposed,meta=_quality_one(prev,cur,nxt)
        rows[index].update(meta)
        stats['quality_checked_frames']+=1
        flag=str(meta['quality_flag'])
        if flag.startswith('review') or flag in ('isolated_mask_repaired','neighbors_missing','no_mask'):
            stats['quality_suspect_frames']+=1
        if flag.startswith('review') or flag in ('neighbors_missing','no_mask'):
            stats['quality_review_frames']+=1
            stats['quality_times_to_review'].append(round(float(rows[index]['seconds']),3))
        if proposed is not None:
            path=dest/f'{index:06d}.png'
            if not cv2.imwrite(str(path),np.uint8(np.round(proposed*255))):
                raise RuntimeError('Không ghi được mask v3.9 ở frame '+str(index))
            revised[index]=path
            stats['quality_repaired_frames']+=1

    try:
        for idx in range(total):
            ok,raw=cap.read()
            if not ok:
                raise RuntimeError(f'Không đọc được frame {idx} trong lượt v3.9.')
            buffer.append(get(idx,raw))
            if idx==0:
                check(None,buffer[0],None,0)
            if len(buffer)==3:
                check(buffer[0],buffer[1],buffer[2],idx-1)
        if total>=2:
            check(None,buffer[-1],None,total-1)
    finally:
        cap.release()
    mask_files.update(revised)
    for idx,path in revised.items():
        arr=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError('Không thể kiểm tra mask v3.9 đã lưu.')
        rows[idx]['mask_area_percent']=round(100*float(np.mean(arr>127)),3)
    return stats


# Tính mức thay đổi diện tích mask trước và sau khi ổn định.
def _area_change_columns(rows: list[dict]):
    """Export before/after area jumps so results can be audited objectively."""
    prev_original=None;prev_final=None
    for row in rows:
        original=float(row['mask_area_before_stability_percent'])
        final=float(row['mask_area_percent'])
        row['area_jump_before_pp']=(round(abs(original-prev_original),3)
                                    if prev_original is not None else 0.0)
        row['area_jump_after_pp']=(round(abs(final-prev_final),3)
                                   if prev_final is not None else 0.0)
        prev_original,prev_final=original,final


# QUY TRÌNH YOLO TÁCH NỀN TOÀN VIDEO
# Quy trình chính: chạy YOLO, xử lý mask, ghép nền và xuất MP4/CSV.
def process_video(cfg: AutoConfig, model: Any = None,
                  progress: Callable[[int,int], None] | None = None) -> tuple[str,dict]:
    src = Path(cfg.video_path)
    if not src.is_file():
        raise FileNotFoundError('Không tìm thấy video đầu vào.')
    info = video_info(str(src))
    if info['frames'] > 0 and info['duration'] > cfg.max_seconds:
        raise ValueError(f'Video dài {info["duration"]:.1f} giây; tối đa {cfg.max_seconds:.0f} giây.')
    if cfg.max_side not in (480, 640, 720, 960):
        raise ValueError('Độ phân giải phải là 480/640/720/960.')
    out_w, out_h = dimensions(info['width'], info['height'], cfg.max_side)
    model = model if model is not None else load_model()
    segmenter = CatSegmenter(model, cfg.imgsz, cfg.confidence)
    bg = cv2.imread(cfg.background_path) if cfg.background_path else None
    if cfg.mode == 'image' and bg is None:
        raise ValueError('Không mở được ảnh nền JPG/PNG.')
    dst = Path(cfg.output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    csv_path = dst.with_name(dst.stem+'_frames.csv')
    old_box = None; recent = deque(maxlen=9)
    diagnostics = []
    retry_attempts = 0; retry_success = 0
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='aetm_v38_masks_', dir=dst.parent) as cache:
        temp_dir = Path(cache)
        mask_files: dict[int, Path] = {}
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise ValueError('Không mở được video.')
        n = 0
        try:
            while True:
                ok, raw = cap.read()
                if not ok: break
                if n / info['fps'] > cfg.max_seconds:
                    raise ValueError('Video dài hơn thời lượng được phép.')
                frame = _resize_video_frame(raw, out_w, out_h)
                predicted, box, score = segmenter.predict(frame, old_box)
                accepted, first_issue = safe_mask(predicted, score, recent,
                                                  safe=cfg.safe_fallback)
                why = 'detected' if accepted is not None else first_issue
                did_retry = False
                if cfg.retry_hard_frames and (accepted is None or score < .24):
                    did_retry = True; retry_attempts += 1
                    try:
                        second, box2, score2 = segmenter.predict(
                            frame, old_box, imgsz=max(cfg.imgsz,960),
                            confidence=min(cfg.confidence,.08), pad_ratio=.20)
                        second_ok, _ = safe_mask(second, score2, recent,
                                                  safe=cfg.safe_fallback)
                    except Exception as retry_error:
                        print('WARN retry skipped:', str(retry_error)[:200], flush=True)
                        second_ok, box2, score2 = None, None, 0.0
                    if second_ok is not None and (
                        accepted is None or score2 > score + .10 or
                        (np.mean(second_ok>.5) > np.mean(accepted>.5)*1.45 and
                         score2 >= score-.04)):
                        accepted, box, score = second_ok, box2, score2
                        why = 'retried'; retry_success += 1
                if accepted is not None:
                    if box is not None: old_box = box
                    recent.append(float(np.mean(accepted>.5)))
                    if cfg.refine_edges:
                        accepted = refine_mask_edge(frame, accepted)
                    path = temp_dir / f'{n:06d}.png'
                    if not cv2.imwrite(str(path), np.uint8(np.round(accepted*255))):
                        raise RuntimeError('Không ghi được mask tạm trên đĩa.')
                    mask_files[n] = path
                else:
                    why = 'fallback:' + first_issue
                row = diagnostic_row(n, info['fps'], why, score, accepted,
                                     did_retry, first_issue)
                row['recovery_agreement'] = ''
                diagnostics.append(row)
                n += 1
                if progress and n%3 == 0:
                    progress(n, 2*max(info['frames'], n))
        finally:
            cap.release()
        if n == 0:
            raise ValueError('Video không có frame hợp lệ.')
        fallback_before = n - len(mask_files)
        repaired = {}
        if cfg.temporal_rescue and cfg.safe_fallback and fallback_before:
            repaired = _temporal_recover(str(src), mask_files, n, out_w, out_h,
                                         temp_dir, max(1,min(20,int(cfg.max_rescue_gap))))
            for idx, agree in repaired.items():
                row = diagnostics[idx]
                row['status'] = 'recovered_temporal'
                row['recovery_agreement'] = agree
                m = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
                row['mask_area_percent'] = round(100*float(np.mean(m > 127)),3)
        # Inspect temporal anomalies after YOLO + bidirectional rescue.
        for row in diagnostics:
            row['mask_area_before_stability_percent'] = row['mask_area_percent']
        stability_summary={'stabilized_frames':0,'review_frames':0,'possible_zoom_frames':0}
        if cfg.stabilize_masks:
            stability_summary = _stabilize_sequence(str(src),mask_files,diagnostics,n,
                                                    out_w,out_h,temp_dir)
        else:
            for row in diagnostics:
                row.update({'stability_flag':'disabled','stabilized':False,
                            'neighbor_agreement':'','stability_change_percent':0.0})
        quality_summary={'quality_checked_frames':0, 'quality_suspect_frames':0,
                         'quality_repaired_frames':0, 'quality_review_frames':0,
                         'quality_times_to_review':[]}
        if cfg.quality_guard:
            quality_summary=_quality_sequence(str(src),mask_files,diagnostics,n,
                                              out_w,out_h,temp_dir)
        else:
            for row in diagnostics:
                row.update(_quality_defaults())
                row['quality_flag']='disabled'
        _area_change_columns(diagnostics)
        tmp = dst.with_name(dst.stem+'_work_'+uuid.uuid4().hex[:6]+'.mp4')
        cap = cv2.VideoCapture(str(src))
        writer = cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),
                                 info['fps'],(out_w,out_h))
        if not cap.isOpened() or not writer.isOpened():
            cap.release(); writer.release()
            raise RuntimeError('Không tạo được video kết quả.')
        try:
            for idx in range(n):
                ok, raw = cap.read()
                if not ok:
                    raise RuntimeError(f'Không đọc được frame {idx} trong lượt ghép video.')
                frame = _resize_video_frame(raw, out_w, out_h)
                path = mask_files.get(idx)
                if path is None:
                    # Never claim a missing segmentation is a fully white cat.
                    outframe = (np.zeros_like(frame) if cfg.mode == 'mask'
                                else frame.copy()) if cfg.safe_fallback else \
                               render(frame,np.zeros((out_h,out_w),np.float32),cfg,bg)
                else:
                    mask = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise RuntimeError(f'Không đọc được mask đã lưu ở frame {idx}.')
                    outframe = render(frame,mask.astype(np.float32)/255.0,cfg,bg)
                writer.write(outframe)
                if progress and (idx%3 == 0 or idx==n-1):
                    progress(n+idx+1, 2*n)
        finally:
            cap.release(); writer.release()
        elapsed = time.perf_counter()-t0
        codec = to_h264(tmp,dst,str(src),cfg.preserve_audio)
    fallback_indices = [i for i in range(n) if i not in mask_files]
    # mask_files points to temporary paths that have now been deleted: only
    # membership is used beyond this point.
    ai_frames = sum(x['status'] in ('detected','retried') for x in diagnostics)
    with csv_path.open('w',newline='',encoding='utf-8-sig') as handle:
        wr = csv.DictWriter(handle,fieldnames=list(diagnostics[0]))
        wr.writeheader(); wr.writerows(diagnostics)
    summary = {
        'frames':n,'fps_video':round(info['fps'],2),'seconds':round(n/info['fps'],2),
        'fps_processing_excluding_encoding':round(n/max(elapsed,.001),2),
        'resolution':f'{out_w}x{out_h}','codec':codec,
        'cat_detected_frames':ai_frames,
        'first_pass_accepted_frames':sum(x['status']=='detected' for x in diagnostics),
        'retry_attempts':retry_attempts,'retry_saved_frames':retry_success,
        'safe_fallback_frames':len(fallback_indices),
        'fallback_before_temporal':fallback_before,
        'temporal_recovered_frames':len(repaired),
        'temporal_rescue':bool(cfg.temporal_rescue),
        'stabilize_masks':bool(cfg.stabilize_masks),
        'stabilized_frames':stability_summary['stabilized_frames'],
        'quality_guard':bool(cfg.quality_guard),
        **quality_summary,
        'review_frames':stability_summary['review_frames'],
        'possible_zoom_frames':stability_summary['possible_zoom_frames'],
        'large_area_jumps_before':sum(float(r['area_jump_before_pp']) >= 20 for r in diagnostics),
        'large_area_jumps_after':sum(float(r['area_jump_after_pp']) >= 20 for r in diagnostics),
        'temporal_recovery_seconds':[round(i/info['fps'],2) for i in sorted(repaired)],
        'fallback_reasons':dict((k, sum(x['status']==k for x in diagnostics))
                                for k in sorted(set(x['status'] for x in diagnostics))
                                if k.startswith('fallback:')),
        'fallback_seconds':[round(i/info['fps'],2) for i in fallback_indices],
        'safe_fallback':bool(cfg.safe_fallback),
        'retry_hard_frames':bool(cfg.retry_hard_frames),
        'refine_edges':bool(cfg.refine_edges),
        'model':MODEL_FILENAME,'diagnostics_csv':str(csv_path)
    }
    dst.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return str(dst), summary

# CHẠY THỬ VÀ NHẬN THAM SỐ DÒNG LỆNH
# Kiểm tra nhanh khả năng nhận diện mèo ở một số thời điểm của video.
def inspect_video(path: str, model: Any = None, imgsz: int = 640,
                  confidence: float = .15, output_path: str | None = None) -> tuple[str,dict]:
    info=video_info(path)
    if info['frames']<1:
        raise ValueError('Video cần thông tin số khung hình để kiểm tra.')
    model = model if model is not None else load_model()
    seg=CatSegmenter(model,imgsz,confidence)
    target=[0.0,2.7,3.1]
    cap=cv2.VideoCapture(path)
    panels=[]; reports=[]; previous=None
    try:
        for seconds in target:
            idx=int(np.clip(round(seconds*info['fps']),0,info['frames']-1))
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ok,raw=cap.read()
            if not ok:continue
            w,h=dimensions(raw.shape[1],raw.shape[0],480)
            frame=cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA)
            m,box,score=seg.predict(frame,previous)
            if box is not None:previous=box
            accepted=(m is not None and float(np.mean(m>.5))>=.008)
            overlay=frame.copy()
            if accepted:
                red=np.zeros_like(frame);red[:]=(30,55,230)
                overlay=np.uint8(np.clip(frame*(1-.48*m[:,:,None])+.48*m[:,:,None]*red,0,255))
            label=f'{idx/info["fps"]:.2f}s | '+(f'CAT {score:.2f}' if accepted else 'KHONG NHAN DUOC MEO')
            cv2.putText(overlay,label,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.58,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(overlay,label,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
            panels.append(overlay)
            reports.append({'seconds':round(idx/info['fps'],2),'detected':bool(accepted),
                            'confidence':round(score,3),
                            'mask_percent':round(100*float(np.mean(m>.5)),2) if m is not None else 0.0})
    finally:
        cap.release()
    if not panels:raise ValueError('Không đọc được khung hình để kiểm tra.')
    # Width is constant (480 max side), but portrait videos have narrow panels.
    h=max(im.shape[0] for im in panels)
    panels=[cv2.copyMakeBorder(im,0,h-im.shape[0],0,0,cv2.BORDER_CONSTANT,value=(238,238,238)) for im in panels]
    merged=cv2.hconcat(panels)
    out=Path(output_path) if output_path else ROOT/'outputs'/f'auto_inspect_{uuid.uuid4().hex[:8]}.png'
    out.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(out),merged):
        raise RuntimeError('Không lưu được ảnh xem trước.')
    return str(out), {'inspection':reports, 'model':MODEL_FILENAME}


# Nhận các tùy chọn từ Terminal và chạy đánh giá hoặc xử lý video.
def main():
    p=argparse.ArgumentParser(description='AETM AutoCat CPU worker')
    p.add_argument('--video',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--inspect',action='store_true')
    p.add_argument('--mode',choices=['mask','blur','color','image'],default='blur')
    p.add_argument('--blur',type=int,default=19)
    p.add_argument('--color',default='#DDEBFF')
    p.add_argument('--background',default='')
    p.add_argument('--side',type=int,default=640)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--conf',type=float,default=.15)
    p.add_argument('--no-safe',action='store_true')
    p.add_argument('--audio',action='store_true')
    p.add_argument('--no-retry',action='store_true')
    p.add_argument('--no-edge-refine',action='store_true')
    p.add_argument('--no-temporal-rescue',action='store_true')
    p.add_argument('--no-stabilize',action='store_true')
    p.add_argument('--no-quality-guard',action='store_true')
    args=p.parse_args()
    try:
        if args.inspect:
            out,stats=inspect_video(args.video,output_path=args.out,imgsz=args.imgsz,confidence=args.conf)
        else:
            cfg=AutoConfig(video_path=args.video,output_path=args.out,mode=args.mode,
                           blur_strength=args.blur,bg_color=args.color,
                           background_path=args.background or None,max_side=args.side,
                           imgsz=args.imgsz,confidence=args.conf,
                           safe_fallback=not args.no_safe,preserve_audio=args.audio,
                           retry_hard_frames=not args.no_retry,
                           refine_edges=not args.no_edge_refine,
                           temporal_rescue=not args.no_temporal_rescue,
                           stabilize_masks=not args.no_stabilize,
                           quality_guard=not args.no_quality_guard)
            out,stats=process_video(cfg,progress=lambda n,total:print(f'PROGRESS|{n}|{total}',flush=True))
        print('RESULT|'+json.dumps({'path':out,'stats':stats},ensure_ascii=False),flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print('FAILED|'+str(exc),flush=True)
        sys.exit(1)

if __name__=='__main__':
    main()
