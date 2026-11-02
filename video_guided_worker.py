"""3G v4.2 -- guided video object segmentation (Windows CPU worker).

Separate .venv_yolo process. Reuses v3.9 temporal recovery, stability, quality,
rendering and H264 helpers, but selects a specific YOLO instance matching a
user supplied seed mask. Keeps masks on disk for local correction/re-render.
"""
from __future__ import annotations

import argparse
import re
from collections import deque
import csv
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from video_auto_yolo import (
    video_info, dimensions, load_model, render, AutoConfig, to_h264,
    _resize_video_frame, _as_numpy, _iou, refine_mask_edge,
    _temporal_recover, _flow_warp, _stabilize_sequence, _quality_sequence,
    _quality_defaults, _area_change_columns,
)


# Đọc một khung hình tại vị trí chỉ định trong video, có thể đổi kích thước.
def read_frame(path: str, idx: int, size: tuple[int,int] | None = None):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened(): raise ValueError('Không thể mở video.')
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok,frame = cap.read()
        if not ok: raise ValueError(f'Không đọc được frame {idx}.')
        if size is not None: frame = cv2.resize(frame,size)
        return frame
    finally: cap.release()


# Xác định hình chữ nhật bao quanh vùng đối tượng trên mask.
def bbox_of(mask: np.ndarray):
    ys,xs = np.where(mask > .5)
    if len(xs) < 8: return None
    return np.asarray([xs.min(),ys.min(),xs.max()+1,ys.max()+1],dtype=np.float32)


# Tạo biểu đồ màu HSV của đối tượng để hỗ trợ theo dõi đúng mục tiêu.
def hist_for(frame:np.ndarray, mask:np.ndarray):
    # HSV histogram helps disambiguate two animals of the same YOLO class.
    small=cv2.resize(frame,(160,160))
    m=cv2.resize((mask>.5).astype(np.uint8),(160,160),interpolation=cv2.INTER_NEAREST)
    if m.sum()<16: return None
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],m,[16,12],[0,180,0,256])
    return cv2.normalize(hist,None).astype(np.float32)


# So sánh độ giống nhau giữa hai biểu đồ màu HSV.
def hist_similarity(a,b):
    if a is None or b is None:return .5
    return float(np.clip(cv2.compareHist(a,b,cv2.HISTCMP_CORREL),-1,1)*.5+.5)


# Kiểm tra độ sáng để xác định khung hình có thiếu sáng hay không.
def light_metrics(frame: np.ndarray):
    """Estimate illumination on original decoded BGR, not on composited output."""
    gray = cv2.cvtColor(cv2.resize(frame, (160, 160)), cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    p20 = float(np.percentile(gray, 20))
    return (median < 105 or (median < 135 and p20 < 48)), round(median, 1)


# Tăng sáng và tương phản cho ảnh đưa vào AI, không đổi video gốc.
def enhance_low_light_for_ai(frame: np.ndarray):
    """Moderate luminance-only enhancement. NEVER modify source used for export."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(l)
    # Adaptive gamma brightens shadows without aggressively blowing out highlights.
    median = float(np.median(l))
    gamma = 0.78 if median < 105 else 0.90
    lut = np.clip(np.power(np.arange(256, dtype=np.float32)/255., gamma)*255., 0, 255).astype(np.uint8)
    l = cv2.LUT(l, lut)
    return cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)




# Đọc tối đa 5 mốc tham chiếu có mask, chỉ chọn các mốc cùng ID đối tượng.
def load_reference_anchors(manifest_path, job, info, size, selected_id=None):
    """Load only explicit, valid, same-object manual masks; do not mix other IDs.

    The GUI persists PNGs with original width; resize to export resolution.
    """
    path = Path(manifest_path).resolve()
    root = (Path(job).resolve() / 'anchors')
    if path != root / 'manifest.json' or not path.is_file():
        raise ValueError('Không tìm thấy bộ mốc trong đúng thư mục dự án.')
    data = json.loads(path.read_text(encoding='utf-8'))
    if int(data.get('frame_count', -1)) != int(info['frames']):
        raise ValueError('Bộ mốc không khớp số frame của video nguồn.')
    primary_id = str(data.get('primary_anchor_id') or '')
    rows = data.get('anchors', [])
    if not isinstance(rows, list) or len(rows) > 5:
        raise ValueError('Bộ mốc phải có tối đa 5 mục.')
    primary = next((a for a in rows if a.get('anchor_id') == primary_id), None)
    if primary is None:
        raise ValueError('Hãy chọn mốc chính trước.')
    target_id = str(selected_id or primary.get('object_id') or 'doi_tuong_1').strip()
    anchors = []
    for a in rows:
        if str(a.get('object_id') or '') != target_id: continue
        idx = int(a['frame_index'])
        if not 0 <= idx < int(info['frames']): continue
        aid = str(a.get('anchor_id') or '')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,90}', aid):
            raise ValueError('Tên mốc không hợp lệ.')
        f = root / (aid + '.png')
        if not bool(a.get('has_mask')) or not f.is_file(): continue
        m = _read_alpha(f, size)
        if m is None or float(np.mean(m > .5)) < .001: continue
        anchors.append({'id': aid, 'frame': idx, 'mask': m, 'object_id': target_id,
                        'name': str(a.get('name') or 'Mốc')[:100]})
    anchors.sort(key=lambda a:a['frame'])
    if not anchors:raise ValueError('Mốc thuộc ID đã chọn chưa có mask. Vẽ trước khi chạy AI.')
    return anchors, target_id


# Dùng Optical Flow tạo mask gợi ý cho các frame gần mốc tham chiếu.
def _anchor_flow_candidates(source, anchors, size, n, *, radius=12):
    """Short-range OF from every manual keyframe in both directions.

    These are *proposals*, not always accepted; YOLO high-confidence stays intact.
    No dependence on other object IDs or GPU.
    """
    w, h = size
    needed = set()
    for a in anchors:
        lo=max(0,a['frame']-radius);hi=min(n-1,a['frame']+radius)
        needed.update(range(lo,hi+1))
    frames = {}
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():return {}
    try:
        i=0
        while i < n:
            ok,f=cap.read()
            if not ok:break
            if i in needed: frames[i] = _resize_video_frame(f,w,h)
            i += 1
            if i>max(needed,default=-1):break
    finally:cap.release()
    proposals={}
    for a in anchors:
        idx=a['frame'];base=a['mask'].copy()
        if idx not in frames:continue
        for direction in (-1,1):
            current=base.copy();old=frames[idx]
            for d in range(1,radius+1):
                t=idx+d*direction
                if t not in frames:break
                current=_flow_warp(current,old,frames[t],flow_side=360)
                old=frames[t]
                area=float(np.mean(current>.5))
                anchor_area=float(np.mean(base>.5))
                if area < max(.008,anchor_area*.22) or area>(1.001 if anchor_area>.96 else min(.999,max(.92,anchor_area*2.0))):
                    break
                score=(radius+1-d)/(radius+1)
                proposals.setdefault(t,[]).append((current.copy(),score,idx))
    # One candidate per frame, prioritized by nearest anchor.
    chosen={}
    for i,items in proposals.items():
        items.sort(key=lambda x:(-x[1],x[2]))
        chosen[i]=items[0]
    return chosen


# Tính mức độ trùng khớp IoU giữa hai mask nhị phân.
def _binary_iou(a,b):
    x=a>.5;y=b>.5
    union=int(np.count_nonzero(x|y))
    return float(np.count_nonzero(x&y))/max(1,union)


# Lớp quản lý việc nhận diện và theo dõi đúng đối tượng do người dùng chọn.
class GuidedPredictor:
    """Track the user-selected *instance*, optionally across all COCO classes.

    'all': do NOT union every detection. Class label is flexible, but spatial
    continuity, foreground appearance and mask overlap must still match the
    instance chosen with the seed mask. This cannot segment unseen classes.
    """
    # Khởi tạo YOLO, các thông tin đối tượng và mask tham chiếu ban đầu.
    def __init__(self, model, seed_frame, seed_mask, imsize=640, confidence=.15, label='auto', low_light=False):
        self.model = model
        self.low_light = bool(low_light)
        self.last_variant = "original"
        self.last_variant_iou = None
        self.last_is_dark = False
        self.last_brightness = None
        self.imsize = int(imsize)
        self.confidence = float(confidence)
        self.seed_hist = hist_for(seed_frame, seed_mask)
        self.anchor_hists = []   # (frame_index, histogram) of same object
        self.frame_index = 0
        self.target_box = bbox_of(seed_mask)
        self.seed_mask = (seed_mask > .5).astype(np.uint8)
        self.target_class = None
        self.tracked_class = None
        self.target_name = 'Đối tượng đã chọn'
        self.score_seed = 0.0
        self.all_classes = str(label).strip().lower() in ('all', 'all_classes', 'tat_ca')
        self.seed_only = False
        self.last_class = None
        self.last_switched = False
        self.last_match = 0.0
        self._identify(seed_frame, seed_mask, label)

    # Lấy các đối tượng và mask mà YOLO phát hiện trong một khung hình.
    def _candidates(self, frame, *, conf=None, imsize=None, pad=.0):
        h, w = frame.shape[:2]
        py = int(round(h * pad)); px = int(round(w * pad))
        src = (cv2.copyMakeBorder(frame,py,py,px,px,cv2.BORDER_CONSTANT,
                                   value=(114,114,114)) if (px or py) else frame)
        out = self.model.predict(source=src,device='cpu',imgsz=int(imsize or self.imsize),
                                 conf=float(conf if conf is not None else self.confidence),
                                 retina_masks=True,verbose=False,max_det=30)[0]
        if out.boxes is None or out.masks is None: return []
        boxes = _as_numpy(out.boxes.xyxy)
        scores = _as_numpy(out.boxes.conf)
        classes = _as_numpy(out.boxes.cls)
        masks = _as_numpy(out.masks.data)
        if masks.ndim == 2: masks = masks[None]
        candidates = []
        for i in range(min(len(boxes),len(scores),len(classes),len(masks))):
            box = np.asarray(boxes[i],np.float32).copy()
            box[0::2]-=px; box[1::2]-=py
            box[0::2]=np.clip(box[0::2],0,w);box[1::2]=np.clip(box[1::2],0,h)
            if (box[2]-box[0])*(box[3]-box[1]) < w*h*.0008: continue
            mask=np.asarray(masks[i],dtype=np.float32)
            if mask.shape!=src.shape[:2]:mask=cv2.resize(mask,(src.shape[1],src.shape[0]))
            if px or py: mask=mask[py:py+h,px:px+w]
            mask=(mask>.48).astype(np.uint8)
            candidates.append((int(classes[i]),float(scores[i]),box,mask))
        return candidates

    # Lưu thông tin màu của các mốc để đối chiếu khi theo dõi.
    def set_anchors(self, references):
        self.anchor_hists=[]
        for a in references:
            mask=a['mask'];frame=a['image']
            h=hist_for(frame,mask)
            if h is not None:self.anchor_hists.append((int(a['frame']),h))

    # So sánh màu đối tượng hiện tại với các mốc đã chọn.
    def _reference_color(self, frame, mask):
        h=hist_for(frame, mask)
        if h is None:return .5
        if not self.anchor_hists:return hist_similarity(self.seed_hist,h)
        scored=[]
        for t,reference in self.anchor_hists:
            similarity=hist_similarity(reference,h)
            # Prioritize matching an anchor close in time, but still allow
            # another angle/illumination to match when the cat zooms in.
            closeness=max(0.0,1.-abs(self.frame_index-t)/max(45,len(self.anchor_hists)*30))
            scored.append(.86*similarity+.14*closeness)
        return max(scored)

    # Lấy bảng tên lớp đối tượng của mô hình YOLO.
    def _names(self):
        names=self.model.names
        return {i:n for i,n in enumerate(names)} if isinstance(names,list) else names

    # Xác định đối tượng YOLO khớp vùng người dùng khoanh ở mốc chính.
    def _identify(self, frame, seed_mask, label):
        choices=self._candidates(frame,conf=min(.12,self.confidence),imsize=max(640,self.imsize))
        dark, _ = light_metrics(frame)
        if self.low_light and dark:
            choices += self._candidates(enhance_low_light_for_ai(frame),conf=min(.12,self.confidence),
                                        imsize=max(640,self.imsize))
        names=self._names()
        manual=str(label).strip().lower()
        if manual not in ('auto','all','all_classes','tat_ca'):
            ids=[int(i) for i,n in names.items() if str(n).lower()==manual]
            if not ids: raise ValueError(f'Mô hình YOLO không hỗ trợ loại đối tượng "{label}".')
            choices=[c for c in choices if c[0] in ids]
        scored=[]
        target=seed_mask>.5
        seed_area=max(1,int(target.sum()))
        for cls,conf,box,mask in choices:
            inter=int(np.logical_and(mask,target).sum())
            overlap=inter/max(1,int(mask.sum()))
            cover=inter/seed_area
            iou=inter/max(1,int(np.logical_or(mask,target).sum()))
            score=.48*iou+.25*overlap+.20*cover+.07*conf
            scored.append((score,cls,box))
        if not scored:
            # With All classes, manual seed remains valid even when the model
            # cannot identify the class in this frame. Future detections may
            # reconnect to the tracked instance; no invented mask at other frames.
            if self.all_classes:
                self.seed_only=True
                self.target_name='Mục tiêu tự chọn (chưa nhận lớp)'
                return
            raise ValueError('YOLO không nhận diện được đối tượng ở frame đã chọn. Hãy đổi frame hoặc chọn loại phù hợp.')
        score,cls,box=max(scored,key=lambda item:item[0]);self.score_seed=round(float(score),3)
        if score<.055:
            if self.all_classes:
                self.seed_only=True
                self.target_name='Mục tiêu tự chọn (chưa khớp YOLO)'
                return
            raise ValueError('Mask bạn vẽ không khớp YOLO. Hãy chọn một frame khác.')
        self.target_class=int(cls)
        self.tracked_class=int(cls)
        self.target_name=str(names.get(cls,cls))
        self.target_box=box

    # Dự đoán mask và chọn đối tượng phù hợp nhất ở frame tiếp theo.
    def predict(self, frame, previous_box=None, *, previous_mask=None, conf=None, imsize=None, pad=.0):
        self.last_switched=False
        self.last_class=None
        self.last_match=0.0
        dark, brightness = light_metrics(frame)
        self.last_is_dark = bool(dark)
        self.last_brightness = brightness
        self.last_variant = 'original'
        self.last_variant_iou = None
        original = self._candidates(frame,conf=conf,imsize=imsize,pad=pad)
        options=[(*item,'original') for item in original]
        if self.low_light and dark:
            enhanced_frame=enhance_low_light_for_ai(frame)
            enhanced=self._candidates(enhanced_frame,conf=conf,imsize=imsize,pad=pad)
            options.extend((*item,'enhanced') for item in enhanced)
        if not self.all_classes:
            options=[c for c in options if c[0]==self.target_class]
        if not options:return None,None,0.
        prev=previous_box if previous_box is not None else self.target_box
        prev_mask=None
        if previous_mask is not None:
            prev_mask=np.asarray(previous_mask)> .5
            if prev_mask.shape != frame.shape[:2]:
                prev_mask=cv2.resize(prev_mask.astype(np.uint8),(frame.shape[1],frame.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)>0
        scores=[]
        for cls,cf,box,mask,variant in options:
            continuity=_iou(box,prev) if prev is not None else 0.0
            color=self._reference_color(frame,mask)
            area=float(mask.mean())
            mask_iou=0.0
            if prev_mask is not None:
                inter=np.logical_and(mask>0,prev_mask).sum()
                union=np.logical_or(mask>0,prev_mask).sum()
                mask_iou=float(inter/max(1,union))
            # Do not accidentally switch to a background person/hand simply
            # because that detection has a higher YOLO confidence.
            if self.all_classes and self.tracked_class is not None and cls!=self.tracked_class:
                if (continuity < .19 and mask_iou < .19) or color < .60:
                    continue
            if prev is not None and continuity < .018 and mask_iou < .018 and color < .72:
                continue
            same_class=(self.tracked_class is not None and cls==self.tracked_class)
            score=(.32*continuity+.28*color+.20*cf+.16*mask_iou+
                   .04*np.sqrt(area)+(.055 if same_class else 0.))
            # Prefer original if scores differ by less than 0.01 to avoid flicker.
            score -= .01 if variant == 'enhanced' else 0.
            scores.append((score,cls,cf,box,mask,variant))
        if not scores:return None,None,0.
        score,cls,cf,box,mask,variant=max(scores,key=lambda item:item[0])
        self.last_variant=variant
        # Record discrepancies even where YOLO confidence is high; these are
        # potential segmentation mistakes that temporal area metrics miss.
        raw_best=[s for s in scores if s[5]=='original']
        enhanced_best=[s for s in scores if s[5]=='enhanced']
        if raw_best and enhanced_best:
            a=max(raw_best,key=lambda s:s[0])[4]>0
            b=max(enhanced_best,key=lambda s:s[0])[4]>0
            self.last_variant_iou=round(float(np.logical_and(a,b).sum()/
                                               max(1,np.logical_or(a,b).sum())),3)
        self.last_match=round(float(score),4)
        if self.all_classes and self.tracked_class is not None and cls!=self.tracked_class:
            # Class changes are allowed only when geometry AND appearance are plausible.
            # Otherwise let the existing two-way recovery handle the gap.
            color=self._reference_color(frame,mask)
            if color < .70 or (_iou(box,prev) if prev is not None else 0) < .12:
                return None,None,0.
            self.last_switched=True
        if self.all_classes:
            self.tracked_class=int(cls)
            if self.target_class is None:
                self.target_class=int(cls)
                self.seed_only=False
        self.last_class=int(cls)
        return cv2.GaussianBlur(mask.astype(np.float32),(0,0),.8),box,float(cf)


# Tạo đường dẫn lưu mask PNG theo số thứ tự frame.
def _mask_file(directory:Path,n:int):return directory/f'{n:06d}.png'


# Đọc mask PNG và chuẩn hóa thành giá trị alpha từ 0 đến 1.
def _read_alpha(path:Path,size=None):
    if not path.is_file():return None
    m=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if m is None:return None
    if size and m.shape[::-1]!=size:m=cv2.resize(m,size)
    return m.astype(np.float32)/255.


# Lưu mask alpha thành ảnh PNG thang xám.
def _save_alpha(path:Path,mask:np.ndarray):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),np.rint(np.clip(mask,0,1)*255).astype(np.uint8)):
        raise RuntimeError(f'Không lưu được mask {path.name}')


# Thực hiện tách nền video bằng YOLO, hỗ trợ mốc, phục hồi và kiểm tra mask.
def _process(args):
    source=Path(args.video).resolve();job=Path(args.job).resolve()
    job.mkdir(parents=True,exist_ok=True)
    out_masks=job/'masks_ai';out_masks.mkdir(exist_ok=True)
    info=video_info(str(source))
    if info['duration']>30.05:raise ValueError('Video tối đa 30 giây trên máy CPU.')
    w,h=dimensions(info['width'],info['height'],int(args.side))
    seed_frame=int(args.seed_frame)
    if not 0<=seed_frame<info['frames']:raise ValueError('Khung khởi tạo không hợp lệ.')
    seed=_read_alpha(Path(args.seed_mask),(w,h))
    reference_anchors=[]
    target_object_id=''
    if args.anchors_manifest:
        reference_anchors,target_object_id=load_reference_anchors(
            args.anchors_manifest,job,info,(w,h),args.target_object_id)
        main=next((a for a in reference_anchors if a['frame']==seed_frame),None)
        if main is not None:seed=main['mask'].copy()
    if seed is None or float(np.mean(seed>.5))<.001:
        raise ValueError('Mask khởi tạo rỗng. Vẽ vòng Bao đối tượng trước.')
    # Tải mô hình AI và khởi tạo bộ theo dõi từ mốc người dùng đã chọn.
    model=load_model()
    predictor=GuidedPredictor(model,read_frame(str(source),seed_frame,(w,h)),seed,
                               int(args.imgsz),float(args.conf),args.label,
                               low_light=bool(args.low_light))
    fps=float(info['fps']);n=int(info['frames'])
    if not reference_anchors:
        reference_anchors=[{'frame':seed_frame,'mask':seed.copy(),
                            'name':'Mốc chính','object_id':'doi_tuong_1'}]
    references=[]
    for a in reference_anchors:
        b=dict(a)
        b['image']=read_frame(str(source),a['frame'],(w,h))
        references.append(b)
    predictor.set_anchors(references)
    fixed={a['frame']:a for a in reference_anchors}
    status=[None]*n;paths={}
    old_box=predictor.target_box
    previous_mask=None
    class_switches=0
    detected_classes=set()
    start=time.perf_counter()
    # Đọc từng frame, chạy YOLO và ghi mask cùng thống kê.
    cap=cv2.VideoCapture(str(source))
    try:
        for i in range(n):
            ok,raw=cap.read()
            if not ok:raise ValueError('Video kết thúc bất thường.')
            fr=_resize_video_frame(raw,w,h)
            predictor.frame_index=i
            m,box,conf=predictor.predict(fr,old_box,previous_mask=previous_mask)
            stage='yolo'
            if m is None or float(np.mean(m>.5))<.003:
                m2,b2,c2=predictor.predict(fr,old_box,previous_mask=previous_mask,conf=min(.08,args.conf),
                                            imsize=max(960,int(args.imgsz)),pad=.15)
                if m2 is not None and np.mean(m2>.5)>.003:
                    m,box,conf=m2,b2,c2;stage='retry'
            selected_class=predictor.last_class
            light_variant=predictor.last_variant
            variant_iou=predictor.last_variant_iou
            is_dark=predictor.last_is_dark
            brightness=predictor.last_brightness
            changed_class=predictor.last_switched
            match_score=predictor.last_match
            if i in fixed:
                m=fixed[i]['mask'].copy();stage='manual_anchor';conf=1.
                box=bbox_of(m)
                selected_class=predictor.target_class
                changed_class=False
            if m is not None:
                if box is not None:old_box=box
                if args.edges and i not in fixed:m=refine_mask_edge(fr,m)
                path=_mask_file(out_masks,i);_save_alpha(path,m);paths[i]=path
                previous_mask=m.copy()
                if selected_class is not None: detected_classes.add(int(selected_class))
                if changed_class: class_switches+=1
            status[i]={'frame':i,'seconds':round(i/fps,3),'status':stage if m is not None else 'missing',
                       'confidence':round(float(conf),3),
                       'mask_area_percent':round(100*float(np.mean(m>.5)),2) if m is not None else 0.0,
                       'recovery_agreement':'',
                       'anchor_source_frame':i if i in fixed else '',
                       'anchor_used':bool(i in fixed),
                       'anchor_overlap':'',
                       'selected_class':selected_class if selected_class is not None else '',
                       'class_changed':bool(changed_class),
                       'target_match_score':match_score,
                       'low_light_detected':bool(is_dark),
                       'median_brightness':brightness,
                       'ai_variant':light_variant,
                       'ai_variant_iou':variant_iou if variant_iou is not None else ''}
            if i%3==0:print(f'PROGRESS|{i+1}|{n*2}',flush=True)
    finally:cap.release()
    # v4.4 -- guided patches near ALL explicitly annotated keyframes.
    # Important: a short-range flow proposal may fill missing frames or replace
    # a low-confidence, obviously inconsistent prediction, but never overwrite
    # a manually annotated keyframe or a strong YOLO prediction.
    guided_count=0
    # Dùng các mốc phụ để đề xuất bổ sung mask cho frame yếu hoặc bị bỏ sót.
    proposals=_anchor_flow_candidates(source,reference_anchors,(w,h),n,radius=12) if len(reference_anchors)>1 else {}
    for i, (proposal, weight, near) in sorted(proposals.items()):
        if i in fixed:continue
        current=_read_alpha(paths[i],(w,h)) if i in paths else None
        overlap=(_binary_iou(proposal,current) if current is not None else None)
        s=status[i]
        confidence=float(s['confidence'])
        weak=(current is None or (confidence<.30 and (
            overlap is not None and overlap<.27)))
        if not weak:continue
        # Avoid using far, uncertain flow proposals instead of stronger YOLO.
        if weight < .23:continue
        if overlap is not None and confidence>=.20 and overlap<.05:
            # Strong disagreement might reflect genuine movement: report only.
            s['anchor_overlap']=round(overlap,3)
            continue
        _save_alpha(_mask_file(out_masks,i),proposal)
        paths[i]=_mask_file(out_masks,i)
        s.update({'status':'anchor_guided','mask_area_percent':round(100*np.mean(proposal>.5),2),
                  'anchor_source_frame':near,'anchor_used':True,
                  'anchor_overlap':round(overlap,3) if overlap is not None else ''})
        guided_count+=1
    # Phục hồi đoạn bị thiếu và kiểm tra độ ổn định theo thời gian.
    missing=n-len(paths)
    recovered={}
    if missing and args.rescue:
        recovered=_temporal_recover(str(source),paths,n,w,h,out_masks,max(2,int(args.gap)))
        for i,v in recovered.items():
            status[i]['status']='flow_recovered';status[i]['recovery_agreement']=round(float(v),3)
            status[i]['mask_area_percent']=round(100*np.mean(_read_alpha(paths[i])>.5),2)
    for row in status:row['mask_area_before_stability_percent']=row['mask_area_percent']
    stab={'stabilized_frames':0,'review_frames':0,'possible_zoom_frames':0}
    if args.stabilize:
        stab=_stabilize_sequence(str(source),paths,status,n,w,h,out_masks)
    else:
        for row in status:
            row.update({'stability_flag':'disabled','stabilized':False,
                        'neighbor_agreement':'','stability_change_percent':0})
    qual={'quality_checked_frames':0,'quality_suspect_frames':0,
          'quality_repaired_frames':0,'quality_review_frames':0,
          'quality_times_to_review':[]}
    if args.quality:
        qual=_quality_sequence(str(source),paths,status,n,w,h,out_masks)
    else:
        for row in status:row.update(_quality_defaults());row['quality_flag']='disabled'
    # Giữ nguyên mask do người dùng khoanh; kết quả tự động không được ghi đè.
    # Every explicit reference mask is authoritative, even after stabilization/QC.
    for i,a in fixed.items():
        _save_alpha(_mask_file(out_masks,i),a['mask'])
        paths[i]=_mask_file(out_masks,i)
        status[i]['status']='manual_anchor'
        status[i]['mask_area_percent']=round(100*np.mean(a['mask']>.5),2)
        status[i]['anchor_used']=True
        status[i]['anchor_source_frame']=i
    _area_change_columns(status)
    # Make final v3.9 refined masks authoritative; local editor reads this folder.
    import shutil
    for ix,latest in paths.items():
        target=_mask_file(out_masks,ix)
        if Path(latest).resolve()!=target.resolve():
            shutil.copyfile(str(latest),str(target))
    # Xuất thông tin từng frame và lưu cấu hình xử lý để có thể render lại.
    csv_path=job/'diagnostics.csv'
    with csv_path.open('w',newline='',encoding='utf-8-sig') as fp:
        wr=csv.DictWriter(fp,list(status[0]));wr.writeheader();wr.writerows(status)
    cfg={'video':str(source),'fps':fps,'frames':n,'width':w,'height':h,
         'seed_frame':seed_frame,'target_class':predictor.target_class,
         'target_object_id':target_object_id,
         'anchors_used':len(reference_anchors),
         'anchors_frames':[a['frame'] for a in reference_anchors],
         'anchor_guided_frames':guided_count,
         'other_object_ids_not_included':True,
         'selection_mode':'all_classes_targeted' if predictor.all_classes else 'fixed_class',
         'class_switches':class_switches,'observed_classes':sorted(detected_classes),
         'seed_only':predictor.seed_only,
         'low_light_enabled':bool(args.low_light),
         'dark_frames':sum(1 for s in status if s['low_light_detected']),
         'enhanced_selected_frames':sum(1 for s in status if s['ai_variant']=='enhanced' and s['status']!='manual_seed'),
         'low_light_disagreements':sum(1 for s in status if s['ai_variant_iou']!='' and s['ai_variant_iou']<.70),
         'target_name':predictor.target_name,'seed_match':predictor.score_seed,
         'missing_before_rescue':missing,'recovered':len(recovered),
         'still_missing':n-len(paths),**stab,**qual,
         'processing_seconds':round(time.perf_counter()-start,2)}
    (job/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    return cfg


# Ghép mask đã xử lý với nền rồi xuất video; ưu tiên mask người dùng đã sửa.
def _render(args):
    job=Path(args.job);cfg=json.loads((job/'config.json').read_text(encoding='utf-8'))
    path=Path(cfg['video']);out=Path(args.out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    frames=int(cfg['frames']);w=int(cfg['width']);h=int(cfg['height']);fps=float(cfg['fps'])
    blur=int(args.blur)
    params=AutoConfig(video_path=str(path),output_path=str(out),mode=args.mode,
                      blur_strength=blur,bg_color=args.color,background_path=args.background)
    bg=cv2.imread(args.background) if args.background else None
    if args.mode=='image' and bg is None:raise ValueError('Chưa có ảnh nền thay thế.')
    temp=out.with_name(out.stem+f'_tmp_{uuid.uuid4().hex[:6]}.mp4')
    # Ghép từng frame với mask cuối và ghi ra video MP4.
    writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened() or not writer.isOpened():
        writer.release();cap.release();raise RuntimeError('Không mở được video đầu vào/đầu ra.')
    try:
        for i in range(frames):
            ok,raw=cap.read()
            if not ok:raise RuntimeError(f'Lỗi đọc frame {i}')
            frame=_resize_video_frame(raw,w,h)
            # Avoid truth-testing numpy arrays.
            mask=_read_alpha(_mask_file(job/'masks_final',i),(w,h))
            if mask is None:mask=_read_alpha(_mask_file(job/'masks_ai',i),(w,h))
            if mask is None:
                painted=np.zeros_like(frame) if args.mode=='mask' else frame.copy()
            else:painted=render(frame,mask,params,bg)
            writer.write(painted)
            if i%3==0 or i==frames-1:print(f'PROGRESS|{i+1}|{frames}',flush=True)
    finally:cap.release();writer.release()
    # Chuyển mã video sang H.264 khi có công cụ hỗ trợ.
    codec=to_h264(temp,out,str(path),bool(args.audio))
    return {'path':str(out),'stats':{'frames':frames,'codec':codec,'resolution':f'{w}x{h}',
                                    'target_name':cfg['target_name'],
                                    'recovered':cfg['recovered'],'still_missing':cfg['still_missing']}}


# Nhận tham số lệnh và chọn xử lý AI hoặc xuất lại video.
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--action',choices=['segment','render'],required=True)
    p.add_argument('--video',default='');p.add_argument('--job',required=True)
    p.add_argument('--seed-mask',default='');p.add_argument('--seed-frame',type=int,default=0)
    p.add_argument('--anchors-manifest',default='');p.add_argument('--target-object-id',default='')
    p.add_argument('--label',default='auto');p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--conf',type=float,default=.15);p.add_argument('--side',type=int,default=640)
    p.add_argument('--low-light',action='store_true', help='Use optional dual-pass AI for dark frames')
    p.add_argument('--edges',action='store_true');p.add_argument('--rescue',action='store_true')
    p.add_argument('--stabilize',action='store_true');p.add_argument('--quality',action='store_true')
    p.add_argument('--gap',type=int,default=16)
    p.add_argument('--out',default='');p.add_argument('--mode',default='blur')
    p.add_argument('--blur',type=int,default=19);p.add_argument('--color',default='#DDEBFF')
    p.add_argument('--background',default='');p.add_argument('--audio',action='store_true')
    a=p.parse_args()
    try:
        if a.action=='segment':
            cfg=_process(a);result={'path':str(Path(a.job)/'diagnostics.csv'),'stats':cfg}
        else:result=_render(a)
        print('RESULT|'+json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as e:
        print('FAILED|'+str(e),flush=True)
        raise

if __name__=='__main__':main()
