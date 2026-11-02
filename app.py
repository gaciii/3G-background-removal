
from __future__ import annotations

# CÁC THƯ VIỆN: hỗ trợ giao diện, API, xử lý ảnh và quản lý phiên.
import threading
import json
import re
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
from collections import OrderedDict
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse, Response
import uvicorn
import gradio as gr
import numpy as np
from PIL import Image

# CÁC MODULE CỦA 3G: tái sử dụng logic AI, xử lý mask, Video và Webcam.
from image_matting import (
    composite_background,
    make_transparent,
    process_image,
    resize_for_demo,
    save_image,
)
from model_loader import OUTPUT_DIR, U2NetPModel
from processing import ImageRefineConfig
from ui_3g import CSS, GUIDE, intro, step, note, help_box, flow
THREEG_THEME = gr.themes.Soft(primary_hue="red", secondary_hue="slate")
from tab_video import add_guided_v40_tab, register_v40_routes
from tab_webcam import add_webcam_tab

# CẤU HÌNH CHUNG: giới hạn kích thước ảnh, khóa mô hình, thư mục kết quả.
IMAGE_MAX_SIDE = 1280
EDITOR_MAX_SIDE = 880  # Anh xem tren canvas; mask that luu o do phan giai goc
MODEL_LOCK = threading.Lock()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Nạp U2Net-P trên CPU một lần và tái sử dụng cho các lần xử lý.
@lru_cache(maxsize=1)
def get_model() -> U2NetPModel:
    return U2NetPModel(backend="cpu", num_threads=4)


# Kiểm tra ảnh đầu vào rồi chuyển RGB/RGBA sang BGR cho OpenCV.
def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    if rgb is None:
        raise gr.Error("Chua co anh dau vao.")
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise gr.Error("Anh phai la RGB/RGBA.")
    if arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# Chuyển BGR về RGB để hiển thị đúng màu trên giao diện.
def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Chuyển mask 0–1 sang ảnh xám 0–255 để xem trực quan.
def mask_preview(mask: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(mask * 255), 0, 255).astype(np.uint8)


# Tạo ảnh xem trước bằng cách ghép đối tượng lên nền màu.
def composited_preview(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return bgr_to_rgb(composite_background(frame_bgr, mask, mode="color", bg_color="#E6F0FA"))


# Kiểm tra phiên hiện tại đã có ảnh và mask hay chưa.
def require_state(state: dict | None) -> dict:
    if not isinstance(state, dict) or "mask" not in state or "frame_bgr" not in state:
        raise gr.Error("Hay tai anh va nhan 'Tu dong tach nen' truoc.")
    return state


# Trả về ảnh xem trước và mask cho giao diện.
def image_outputs(state: dict) -> tuple[np.ndarray, np.ndarray]:
    return composited_preview(state["frame_bgr"], state["mask"]), mask_preview(state["mask"])


# CHỈNH MASK THỦ CÔNG: quản lý tối đa 5 phiên cọ và lịch sử hoàn tác.
# ---- Chinh sua thu cong bang canvas, khong can Gradio ImageEditor ----
EDITOR_SESSIONS: OrderedDict[str, dict[str, Any]] = OrderedDict()
EDITOR_LOCK = threading.RLock()
MAX_EDITOR_SESSIONS = 5


# Tính kích thước ảnh xem trước, giữ đúng tỉ lệ.
def _preview_size(height: int, width: int) -> tuple[int, int]:
    ratio = min(1.0, EDITOR_MAX_SIDE / max(height, width))
    return max(2, round(width * ratio)), max(2, round(height * ratio))


# Tìm phiên chỉnh mask theo mã token; báo lỗi nếu hết phiên.
def _get_editor(token: str) -> dict[str, Any]:
    with EDITOR_LOCK:
        session = EDITOR_SESSIONS.get(token)
        if session is None:
            raise HTTPException(status_code=404, detail="Phien chinh anh khong con. Hay tach anh lai.")
        EDITOR_SESSIONS.move_to_end(token)
        return session


# Tạo phiên chỉnh mask mới và lưu ảnh, mask, lịch sử.
def _register_editor(frame_bgr: np.ndarray, alpha: np.ndarray) -> str:
    token = uuid.uuid4().hex
    with EDITOR_LOCK:
        EDITOR_SESSIONS[token] = {
            "frame": frame_bgr, "mask": alpha.copy(),
            "original_mask": alpha.copy(), "history": [],
        }
        while len(EDITOR_SESSIONS) > MAX_EDITOR_SESSIONS:
            EDITOR_SESSIONS.popitem(last=False)
    return token


# Tạo khung HTML để người dùng thao tác cọ trên trình duyệt.
def _editor_iframe(token: str) -> str:
    # Token duoc tao boi uuid, khong nhan chuoi tu nguoi dung.
    return (
        f'<iframe title="Chinh mask bang co keo lien tuc" src="/brush/{token}" '
        'style="width:100%;height:735px;border:1px solid #cbd5e1;'
        'border-radius:12px;background:#f8fafc;" loading="eager"></iframe>'
    )


# Tạo JPEG xem trước từ ảnh và mask đang chỉnh.
def _render_editor_jpeg(session: dict[str, Any]) -> bytes:
    frame, mask = session["frame"], session["mask"]
    composed = composite_background(frame, mask, mode="color", bg_color="#E6F0FA")
    h, w = composed.shape[:2]
    dw, dh = _preview_size(h, w)
    if (dw, dh) != (w, h):
        composed = cv2.resize(composed, (dw, dh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", composed, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        raise RuntimeError("Khong tao duoc anh preview de chinh sua")
    return buf.tobytes()


# GIAO DIỆN CỌ (HTML/CSS/JS): bắt chuột, zoom và gửi nét vẽ về API Python.
EDITOR_PAGE = r'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
*{box-sizing:border-box}body{margin:0;padding:10px;font:14px system-ui,Segoe UI,Arial;background:#f8fafc;color:#1f2937}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-bottom:8px}
button{border:1px solid #cbd5e1;border-radius:8px;background:white;color:#111827;padding:9px 13px;font-weight:600;cursor:pointer}
button.active{background:#1d4ed8;color:white;border-color:#1d4ed8}
button.red.active{background:#dc2626;border-color:#dc2626}
button.green.active{background:#16803b;border-color:#16803b}
label{font-size:13px}input[type=range]{width:135px;vertical-align:middle}
#viewport{height:545px;border:1px solid #cbd5e1;background:repeating-conic-gradient(#eee 0% 25%,#fff 0% 50%) 0 0/22px 22px;overflow:auto;position:relative;overscroll-behavior:contain}
#canvas{display:block;margin:auto;touch-action:none;cursor:crosshair;max-width:none}
#circle{border:2px solid #ef4444;position:fixed;z-index:20;border-radius:50%;pointer-events:none;display:none;transform:translate(-50%,-50%);box-shadow:0 0 1px 1px #fff}
#status{min-height:22px;color:#334155;padding:5px 0;font-size:13px}
.small{font-size:12px;color:#64748b;margin:5px 0}
</style></head><body>
<div class="toolbar">
  <button id="erase" class="active red">🔴 Xóa nền</button>
  <button id="restore" class="green">🟢 Giữ lại</button>
  <button id="pan">✋ Di chuyển ảnh</button>
  <label>Độ lớn cọ: <b id="sizeLabel">20</b> px <input id="brushSize" type="range" min="2" max="90" value="20"></label>
</div>
<div class="toolbar">
  <button id="minus">− Thu nhỏ</button><button id="fit">100% — Toàn ảnh</button><button id="plus">+ Phóng to</button>
  <span id="zoomValue">Zoom 100%</span>
  <button id="undo">↶ Hoàn tác</button><button id="reset">↩ Khôi phục mask AI</button>
</div>
<div id="viewport"><canvas id="canvas"></canvas></div>
<div id="status">Đang tải ảnh để chỉnh...</div>
<div class="small">Giữ chuột trái rồi kéo liên tục để xóa/khôi phục. Cọ tròn được hiển thị khi rê chuột. Khi zoom, chọn ✋ Di chuyển ảnh hoặc cuộn thanh cuộn để xem từng góc. Sau khi hoàn thành, bấm «OK — duyệt mask thủ công» bên dưới.</div>
<div id="circle"></div>
<script>
const TOKEN=__TOKEN__;
const endpoint='/brush/'+TOKEN;
const viewport=document.getElementById('viewport');
const canvas=document.getElementById('canvas');
const ctx=canvas.getContext('2d');
const circle=document.getElementById('circle');
const brushSize=document.getElementById('brushSize');
const sizeLabel=document.getElementById('sizeLabel');
const status=document.getElementById('status');
let mode='erase', zoom=1, painting=false, panning=false, pending=false, points=[];
let last=null, lastClient=null, sourceImage=null, initialized=false;
function say(text){status.textContent=text}
function selected(){
  for(const id of ['erase','restore','pan'])document.getElementById(id).classList.toggle('active',id===mode);
  circle.style.borderColor=mode==='restore'?'#22c55e':mode==='erase'?'#ef4444':'#64748b';
  canvas.style.cursor=mode==='pan'?'grab':'crosshair';
  circle.style.display='none';
}
for(const name of ['erase','restore','pan'])document.getElementById(name).onclick=()=>{mode=name;selected();};
brushSize.oninput=()=>{sizeLabel.textContent=brushSize.value;};
function setZoom(z){
  z=Math.min(4,Math.max(1,z));
  const oldW=canvas.getBoundingClientRect().width||canvas.width;
  const cx=(viewport.scrollLeft+viewport.clientWidth/2)/Math.max(oldW,1);
  const cy=(viewport.scrollTop+viewport.clientHeight/2)/Math.max(canvas.getBoundingClientRect().height,1);
  zoom=z;
  canvas.style.width=(canvas.width*zoom)+'px';
  canvas.style.height=(canvas.height*zoom)+'px';
  viewport.scrollLeft=cx*canvas.width*zoom-viewport.clientWidth/2;
  viewport.scrollTop=cy*canvas.height*zoom-viewport.clientHeight/2;
  document.getElementById('zoomValue').textContent='Zoom '+Math.round(zoom*100)+'%';
}
document.getElementById('plus').onclick=()=>setZoom(zoom===1?1.5:zoom===1.5?2:zoom===2?3:4);
document.getElementById('minus').onclick=()=>setZoom(zoom===4?3:zoom===3?2:zoom===2?1.5:1);
document.getElementById('fit').onclick=()=>setZoom(1);
function redraw(){if(sourceImage){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(sourceImage,0,0,canvas.width,canvas.height);}}
function point(e){const box=canvas.getBoundingClientRect(); return [
  Math.max(0,Math.min(canvas.width-1,(e.clientX-box.left)*canvas.width/box.width)),
  Math.max(0,Math.min(canvas.height-1,(e.clientY-box.top)*canvas.height/box.height))];}
function drawPart(a,b){
 ctx.save();ctx.lineJoin='round';ctx.lineCap='round';ctx.strokeStyle=mode==='erase'?'rgba(239,68,68,0.74)':'rgba(34,197,94,0.74)';
 ctx.lineWidth=Number(brushSize.value)*2;
 ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
 ctx.fillStyle=ctx.strokeStyle;ctx.beginPath();ctx.arc(b[0],b[1],Number(brushSize.value),0,Math.PI*2);ctx.fill();ctx.restore();
}
function cursorAt(e){
 if(mode==='pan'){circle.style.display='none';return;}
 circle.style.display='block';circle.style.left=e.clientX+'px';circle.style.top=e.clientY+'px';
 circle.style.width=(Number(brushSize.value)*2*zoom)+'px';circle.style.height=(Number(brushSize.value)*2*zoom)+'px';
}
canvas.addEventListener('pointerenter',cursorAt);
canvas.addEventListener('pointerleave',()=>{if(!painting)circle.style.display='none'});
canvas.addEventListener('pointerdown',e=>{
 if(e.button!==0||pending)return;
 e.preventDefault();canvas.setPointerCapture(e.pointerId);
 if(mode==='pan'){panning=true;lastClient=[e.clientX,e.clientY,viewport.scrollLeft,viewport.scrollTop];return;}
 painting=true;points=[point(e)];last=points[0];drawPart(last,last);cursorAt(e);
});
canvas.addEventListener('pointermove',e=>{
 cursorAt(e);
 if(panning && lastClient){viewport.scrollLeft=lastClient[2]-(e.clientX-lastClient[0]);viewport.scrollTop=lastClient[3]-(e.clientY-lastClient[1]);return;}
 if(!painting)return;e.preventDefault();const p=point(e);points.push(p);drawPart(last,p);last=p;
});
async function loadPreview(){
 const response=await fetch(endpoint+'/view?ts='+Date.now(),{cache:'no-store'});
 if(!response.ok)throw new Error('Khong doc duoc anh xem truoc ('+response.status+')');
 const imageBlob=await response.blob();
 const url=URL.createObjectURL(imageBlob);
 const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=reject;image.src=url});
 URL.revokeObjectURL(url);
 const first=!initialized;
 sourceImage=image;
 if(first){canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;initialized=true;setZoom(1)}
 redraw();
}
async function post(kind,payload={}){
 const response=await fetch(endpoint+'/'+kind,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
 if(!response.ok){let d='Loi server: '+response.status;try{d=JSON.stringify(await response.json())}catch(e){}throw Error(d)}
 return await response.json();
}
async function endStroke(){
 if(!painting)return;painting=false;circle.style.display='none';
 const batch=points;points=[];if(!batch.length)return;
 pending=true;say('Đang cập nhật mask...');
 try{await post('stroke',{mode:mode,radius:Number(brushSize.value),points:batch,view_w:canvas.width,view_h:canvas.height});
  await loadPreview();say('✅ Đã áp dụng một nét kéo. Bạn có thể tô tiếp.');}
 catch(err){await loadPreview().catch(()=>{});say('❌ '+err.message)}finally{pending=false;}
}
canvas.addEventListener('pointerup',e=>{if(panning){panning=false;lastClient=null;return;}void endStroke()});
canvas.addEventListener('pointercancel',e=>{panning=false;lastClient=null;void endStroke()});
async function action(name){if(pending)return;pending=true;say('Đang xử lý...');try{
 const result=await post(name);await loadPreview();say(result.message||'✅ Đã thực hiện.');
 }catch(e){say('❌ '+e.message)}finally{pending=false;}}
document.getElementById('undo').onclick=()=>void action('undo');
document.getElementById('reset').onclick=()=>{if(confirm('Khôi phục mask AI ban đầu và bỏ các nét đã chỉnh?'))void action('reset')};
loadPreview().then(()=>say('Sẵn sàng — giữ chuột trái và kéo cọ tròn để chỉnh.')).catch(e=>say('❌ '+e.message));
</script>
</body></html>'''


# API CỌ: các route /brush để xem ảnh và sửa mask theo token.
# Tạo API FastAPI nhận thao tác vẽ, hoàn tác và khôi phục mask.
def create_editor_api() -> FastAPI:
    api = FastAPI(title="3G manual brush API", docs_url=None, redoc_url=None)

    # Trả trang HTML công cụ cọ cho phiên chỉnh hiện tại.
    @api.get("/brush/{token}", response_class=HTMLResponse)
    def brush_editor(token: str):
        _get_editor(token)
        return HTMLResponse(EDITOR_PAGE.replace("__TOKEN__", json.dumps(token)))

    # Trả ảnh xem trước mới nhất cho trình duyệt.
    @api.get("/brush/{token}/view")
    def brush_view(token: str):
        session = _get_editor(token)
        with EDITOR_LOCK:
            image_bytes = _render_editor_jpeg(session)
        return Response(content=image_bytes, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # Nhận nét cọ, quy đổi tọa độ và cập nhật mask Thêm/Xóa.
    @api.post("/brush/{token}/stroke")
    def brush_stroke(token: str, payload: dict = Body(...)):
        session = _get_editor(token)
        mode = payload.get("mode")
        if mode not in ("erase", "restore"):
            raise HTTPException(422, "Che do co khong hop le")
        raw_points = payload.get("points", [])
        if not isinstance(raw_points, list) or not 1 <= len(raw_points) <= 10000:
            raise HTTPException(422, "So diem trong net ve khong hop le")
        try:
            points = np.asarray(raw_points, dtype=np.float32).reshape(-1, 2)
            radius = float(payload.get("radius", 20))
            view_w = int(payload.get("view_w", 0))
            view_h = int(payload.get("view_h", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(422, "Tham so co sai") from exc
        if not np.isfinite(points).all() or not np.isfinite(radius) or not 1 <= radius <= 90:
            raise HTTPException(422, "Diem hoac radius khong hop le")
        with EDITOR_LOCK:
            mask = session["mask"]
            h, w = mask.shape
            dw, dh = _preview_size(h, w)
            if view_w != dw or view_h != dh:
                raise HTTPException(409, "Kich thuoc anh da thay doi, tai lai trang chinh")
            # Cac diem ve la toa do tren canvas; scale ve kich thuoc mask goc.
            sx, sy = w / dw, h / dh
            raw = np.column_stack((points[:, 0] * sx, points[:, 1] * sy))
            coords = np.rint(raw).astype(np.int32)
            coords[:, 0] = np.clip(coords[:, 0], 0, w - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, h - 1)
            brush_r = max(1, round(radius * (sx + sy) / 2))
            layer = np.zeros((h, w), dtype=np.uint8)
            for a, b in zip(coords[:-1], coords[1:]):
                cv2.line(layer, tuple(map(int, a)), tuple(map(int, b)), 255,
                         thickness=max(1, brush_r * 2), lineType=cv2.LINE_AA)
            for idx in ([0] if len(coords) == 1 else [0, len(coords)-1]):
                cv2.circle(layer, tuple(map(int, coords[idx])), brush_r, 255,
                           thickness=-1, lineType=cv2.LINE_AA)
            alpha = layer.astype(np.float32) / 255.0
            session["history"].append(mask.copy())
            if len(session["history"]) > 10:
                session["history"].pop(0)
            if mode == "erase":
                session["mask"] = np.clip(mask * (1 - alpha), 0, 1).astype(np.float32)
            else:
                session["mask"] = np.clip(mask + (1 - mask) * alpha, 0, 1).astype(np.float32)
        return {"ok": True, "message": "Da ap dung net co"}

    # Khôi phục trạng thái mask trước nét vẽ gần nhất.
    @api.post("/brush/{token}/undo")
    def brush_undo(token: str):
        session = _get_editor(token)
        with EDITOR_LOCK:
            if session["history"]:
                session["mask"] = session["history"].pop()
                return {"ok": True, "message": "Da hoan tac net ve vua roi"}
            return {"ok": True, "message": "Chua co net ve nao de hoan tac"}

    # Khôi phục mask AI ban đầu và xóa lịch sử chỉnh sửa.
    @api.post("/brush/{token}/reset")
    def brush_reset(token: str):
        session = _get_editor(token)
        with EDITOR_LOCK:
            session["mask"] = session["original_mask"].copy()
            session["history"].clear()
        return {"ok": True, "message": "Da khoi phuc mask AI ban dau"}

    return api


# XỬ LÝ ẢNH: tách nền tự động, duyệt mask và xuất kết quả.
# Xóa dữ liệu ảnh cũ khi người dùng thay ảnh đầu vào.
def clear_image_session() -> tuple:
    """Khi nguoi dung doi anh thi khong duoc dung mask cua anh truoc."""
    return (
        None, None, None, gr.update(value=None),
        "Ảnh đã đổi. Nhấn **Tách nền tự động** để bắt đầu.",
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        None, None, None,
    )


# Gọi AI tách nền ảnh, lưu mask và trả kết quả xem trước.
def auto_segment(image_rgb, use_enhanced, use_guided, strong_refine, radius, eps) -> tuple:
    if image_rgb is None:
        raise gr.Error("Ban chua tai anh.")
    if strong_refine and not use_enhanced:
        raise gr.Error("Strong Refine can bat Enhanced.")
    frame = resize_for_demo(rgb_to_bgr(image_rgb), max_side=IMAGE_MAX_SIDE)
    cfg = ImageRefineConfig(
        use_guided_filter=bool(use_guided), radius=int(radius), eps=float(eps),
        color_guide=False, keep_largest=False,
        bg_threshold=0.02, fg_threshold=0.98, alpha_gamma=1.0,
    )
    with MODEL_LOCK:
        result = process_image(
            frame, get_model(), enhanced=bool(use_enhanced),
            strong_refine=bool(strong_refine),
            use_guided_filter=bool(use_guided),
            radius=int(radius), eps=float(eps), refine_config=cfg,
        )
    alpha = np.asarray(result["mask"], dtype=np.float32).copy()
    state = {
        "frame_bgr": frame, "mask": alpha, "auto_mask": alpha.copy(),
        "history": [], "approved": False,
    }
    state["editor_token"] = _register_editor(frame, alpha)
    preview, gray = image_outputs(state)
    return (
        state, preview, gray, gr.update(value=""),
        f"### Da tach tu dong — chua duyet\n"
        f"Thoi gian {result['total_ms']:.0f} ms."
        " Nếu cần sửa, nhấn **Chỉnh bằng cọ Thêm / Xóa**.",
        gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
        None, None, None,
    )


# Hiển thị công cụ cọ để sửa mask đã được AI tạo.
def open_manual(state: dict | None) -> tuple:
    state = require_state(state)
    state["approved"] = False
    token = state.get("editor_token")
    if not token or token not in EDITOR_SESSIONS:
        token = _register_editor(state["frame_bgr"], state["mask"])
        state["editor_token"] = token
    return (
        state, gr.update(visible=True), gr.update(visible=False),
        _editor_iframe(token),
        "Giữ chuột trái và kéo cọ tròn trực tiếp trên ảnh đã tách. Có thể phóng to và hoàn tác.",
    )


# Xác nhận sử dụng mask AI và chuyển sang bước xuất ảnh.
def approve_auto(state: dict | None) -> tuple:
    state = require_state(state)
    state["approved"] = True
    return state, gr.update(visible=True), gr.update(visible=False), "✅ Da duyet mask tu dong. Chuyen sang Buoc 3."


# Lấy mask sau khi chỉnh bằng cọ và xác nhận kết quả.
def approve_manual(state: dict | None) -> tuple:
    state = require_state(state)
    token = state.get("editor_token")
    if token:
        session = _get_editor(token)
        with EDITOR_LOCK:
            state["mask"] = session["mask"].copy()
    state["approved"] = True
    preview, gray = image_outputs(state)
    return (
        state, preview, gray, gr.update(visible=False), gr.update(visible=True),
        "✅ Đã duyệt mask thủ công. Chuyển sang Bước 3 để xuất ảnh.",
    )


# Chuẩn hóa giá trị màu từ giao diện thành mã HEX #RRGGBB.
def normalize_color(color: Any) -> str:
    """Chuyen ColorPicker CSS/RGB/RGBA sang HEX #RRGGBB.

    Gradio/Chrome co the gui mau nhu:
    rgba(72.73249893682384, 121.12186928042381, 169.51123962402343, 1)
    Cac kenh RGB co the la so thuc va duoc lam tron, khong tu choi.

    Ho tro #RGB, #RRGGBB, #RRGGBBAA, rgb(...), rgba(...),
    tuple/list/ndarray RGB(A), dict {r,g,b} hoac {hex/value/color}.
    """
    if color is None:
        return "#F1F5F9"

    # Chuyển ba kênh màu RGB hợp lệ thành mã màu HEX.
    def channels_to_hex(channels: Any) -> str:
        try:
            vals = [float(v) for v in channels]
        except (ValueError, TypeError) as exc:
            raise gr.Error("Mau nen RGB khong hop le.") from exc
        if len(vals) != 3 or not all(np.isfinite(v) and 0 <= v <= 255 for v in vals):
            raise gr.Error("Mau RGB phai co ba kenh, moi kenh trong khoang 0..255.")
        return "#" + "".join(f"{int(np.rint(v)):02X}" for v in vals)

    if isinstance(color, dict):
        if all(key in color for key in ("r", "g", "b")):
            return channels_to_hex([color["r"], color["g"], color["b"]])
        color = color.get("hex") or color.get("value") or color.get("color")
        if color is None:
            raise gr.Error("Mau nen trong giao dien dang rong.")

    if isinstance(color, (tuple, list, np.ndarray)):
        if len(color) < 3:
            raise gr.Error("Mau RGB thieu kenh.")
        return channels_to_hex(color[:3])

    value = str(color).strip()
    if not value:
        return "#F1F5F9"

    if value.startswith("#"):
        digits = value[1:]
        if re.fullmatch(r"[a-fA-F0-9]{3}", digits):
            return "#" + "".join(ch * 2 for ch in digits).upper()
        if re.fullmatch(r"[a-fA-F0-9]{6}", digits):
            return "#" + digits.upper()
        if re.fullmatch(r"[a-fA-F0-9]{8}", digits):
            return "#" + digits[:6].upper()

    # Khong dung mot regex bat buoc tung kenh la so nguyen.
    # Tach RGB/CSS theo dau phay, khoang trang hoac dau '/'.
    lower = value.lower()
    for fn in ("rgb", "rgba"):
        if lower.startswith(fn + "(") and value.endswith(")"):
            body = value[len(fn) + 1:-1]
            parts = [p for p in re.split(r"[,\s/]+", body.strip()) if p]
            if len(parts) not in (3, 4):
                raise gr.Error("RGB/RGBA phai co 3 hoac 4 kenh.")
            try:
                if len(parts) == 4:
                    alpha = float(parts[3])
                    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
                        raise ValueError("alpha khong hop le")
                return channels_to_hex(parts[:3])
            except ValueError as exc:
                raise gr.Error("Gia tri kenh RGB/RGBA khong hop le.") from exc

    raise gr.Error(
        f"Mau nen '{value}' khong hop le. "
        "Hay chon mau trong bang mau hoac dung ma #RRGGBB."
    )


# Hiện đúng tùy chọn theo kiểu nền người dùng chọn.
def output_controls(mode: str):
    """Chi hien tham so can thiet cho loai dau ra da chon."""
    return (
        gr.update(visible=mode == "Nền màu"),
        gr.update(visible=mode == "Ảnh nền khác"),
        gr.update(visible=mode == "Làm mờ nền"),
        {
            "PNG trong suốt": "PNG trong suot: nen se trong suot, khong can chon mau/anh nen.",
            "Nền màu": "Chon mau nen bang bang mau ben duoi, sau do bam Xuat ket qua.",
            "Ảnh nền khác": "BAM vao o Tai anh nen moi ben duoi de chon JPG/PNG tren may.",
            "Làm mờ nền": "Dieu chinh do mo ben duoi va bam Xuat ket qua.",
        }.get(mode, "Chon mot kieu dau ra."),
    )


# Xuất PNG trong suốt, nền màu, nền ảnh hoặc nền mờ.
def export_image(state, mode, background_rgb, color, blur_sigma):
    """Xuat PNG tren mask da duyet; cac che do khong phu thuoc lan nhau."""
    state = require_state(state)
    if not state.get("approved"):
        raise gr.Error("Hay bam OK duyet mask o Buoc 2 truoc khi xuat.")

    frame, mask = state["frame_bgr"], state["mask"]
    if mode == "PNG trong suốt":
        save_array = make_transparent(frame, mask)
        preview_out = cv2.cvtColor(save_array, cv2.COLOR_BGRA2RGBA)
        suffix = "transparent"
    elif mode == "Nền màu":
        selected_color = normalize_color(color)
        save_array = composite_background(frame, mask, mode="color", bg_color=selected_color)
        preview_out = bgr_to_rgb(save_array)
        suffix = "color"
    elif mode == "Ảnh nền khác":
        if background_rgb is None:
            raise gr.Error("Chua tai anh nen! Hay bam 'Tai anh nen moi' de chon file JPG/PNG.")
        save_array = composite_background(frame, mask, mode="image", bg_image_bgr=rgb_to_bgr(background_rgb))
        preview_out = bgr_to_rgb(save_array)
        suffix = "image"
    elif mode == "Làm mờ nền":
        save_array = composite_background(frame, mask, mode="blur", blur_sigma=float(blur_sigma))
        preview_out = bgr_to_rgb(save_array)
        suffix = "blur"
    else:
        raise gr.Error("Hay chon PNG trong suot, Nen mau, Anh nen khac hoac Lam mo nen.")
    path = OUTPUT_DIR / f"3g_{uuid.uuid4().hex[:10]}_{suffix}.png"
    save_image(path, save_array)
    return preview_out, str(path), f"✅ Da xuat {suffix}: `{path.name}`. Ban co the xem va tai anh o ben duoi."


# GIAO DIỆN CHÍNH: bố trí điều hướng và chức năng của 3 tab.
# Dựng giao diện 3 tab và liên kết các sự kiện nút bấm.
def build_app() -> gr.Blocks:
    # 3G: giao diện màu đỏ dựa trên V3 đã duyệt, logic AI và các callbacks giữ nguyên.
    with gr.Blocks(title="3G — Ảnh · Video · Webcam",
                   **({"css": CSS, "theme": THREEG_THEME} if int(gr.__version__.split(".")[0]) < 6 else {})) as demo:
        # Thanh đầu trang gồm logo, chọn tab và nút hướng dẫn.
        with gr.Row(elem_classes=["g3-header"], equal_height=True):
            with gr.Column(scale=3,min_width=105):
                gr.HTML('<div class="g3-logo"><span class="g3-logo-mark">◉</span><span>3G</span></div>')
            with gr.Column(scale=6,min_width=180):
                nav = gr.Radio(["Ảnh","Video","Webcam"],value="Ảnh",show_label=False,
                               container=False,elem_id="g3-nav")
            with gr.Column(scale=2,min_width=88):
                help_btn = gr.Button("Hướng dẫn",variant="secondary",elem_id="g3-help-btn")
        with gr.Column(visible=False,elem_id="g3-help-panel") as guide_panel:
            with gr.Row():
                gr.HTML('<h2 style="color:#18253b;font-size:18px;font-weight:800;margin:8px 0">Hướng dẫn sử dụng 3G</h2>')
                help_close = gr.Button("Đóng ✕",size="sm",scale=0)
            gr.Markdown(GUIDE,elem_classes=["g3-guide-content"])
        help_btn.click(lambda:gr.update(visible=True),outputs=guide_panel,queue=False)
        help_close.click(lambda:gr.update(visible=False),outputs=guide_panel,queue=False)

        # Tab Ảnh: nhập ảnh → xử lý AI → sửa/duyệt mask → xuất PNG.
        with gr.Column(visible=True,elem_classes=["g3-page"],elem_id="g3-photo-page") as image_page:
            gr.HTML(intro("Tách nền ảnh", "Tải ảnh, để AI tách nền, chỉnh bằng cọ nếu cần rồi xuất PNG."))
            gr.HTML(flow(["Tải ảnh", "Duyệt mask", "Xuất ảnh"]))
            st = gr.State(None)
            with gr.Row(equal_height=False,elem_classes=["g3-two-up"]):
                with gr.Column(scale=1,min_width=0,elem_classes=["g3-card"]):
                    gr.HTML('<div class="g3-label">Ảnh gốc</div>')
                    image_input = gr.Image(label="Tải JPG / PNG", type="numpy", image_mode="RGB",
                                           sources=["upload", "clipboard"],height=335,elem_classes=["g3-preview"])
                    with gr.Accordion("Tùy chỉnh AI nâng cao",open=False,elem_classes=["aetm-accordion"]):
                        chk_enhanced = gr.Checkbox(label="AI kép · xử lý ảnh gốc và ảnh tăng tương phản",value=True)
                        chk_guided = gr.Checkbox(label="Guided Filter · làm mịn mép đối tượng",value=True)
                        chk_strong = gr.Checkbox(label="Làm sạch mask mạnh (có thể mất tóc mảnh)",value=False)
                        radius = gr.Slider(1,15,value=5,step=1,label="Bán kính Guided Filter")
                        eps = gr.Slider(0.0001,0.02,value=0.001,step=0.0001,label="Epsilon · độ nhạy theo biên")
                    auto_btn = gr.Button("TÁCH NỀN TỰ ĐỘNG",variant="primary",elem_classes=["g3-primary"])
                    status = gr.Markdown("**Sẵn sàng.** Chọn ảnh để bắt đầu.")
                with gr.Column(scale=1,min_width=0,elem_classes=["g3-card"]):
                    gr.HTML('<div id="aetm-image-review" class="g3-label">Kết quả sau tách nền</div>')
                    preview = gr.Image(label="Ảnh xem trước",type="numpy",interactive=False,height=335,
                                       elem_classes=["g3-preview"])
                    with gr.Accordion("Xem mask trắng / đen",open=False,elem_classes=["aetm-accordion"]):
                        mask_view = gr.Image(label="Mask AI · trắng giữ, đen bỏ",type="numpy",interactive=False,height=250,
                                             elem_classes=["g3-preview"])
                    with gr.Column(visible=False) as choice_group:
                        with gr.Row():
                            auto_ok = gr.Button("✓ Mask đạt",variant="primary",elem_classes=["g3-primary"])
                            manual_open = gr.Button("✎ Chỉnh bằng cọ Thêm / Xóa",variant="secondary")
            with gr.Column(visible=False,elem_classes=["g3-card"]) as manual_group:
                gr.HTML('<div id="aetm-image-manual" class="g3-card-title">Chỉnh mask bằng cọ</div>')
                gr.HTML(help_box("Kéo cọ Giữ lại để phục hồi vùng bị mất; cọ Xóa để loại nền thừa. Có zoom và hoàn tác."))
                edit_canvas = gr.HTML(value="",label="Chỉnh mask trên ảnh")
                manual_status = gr.Markdown("Chọn cọ và kéo trên ảnh.")
                manual_ok = gr.Button("✓ DUYỆT MASK ĐÃ CHỈNH",variant="primary",elem_classes=["g3-primary"])
            with gr.Column(visible=False,elem_classes=["g3-card"]) as export_group:
                gr.HTML('<div id="aetm-image-export" class="g3-card-title">Bước 3 · Xuất kết quả PNG</div>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4,min_width=0):
                        final_mode = gr.Radio(["PNG trong suốt","Nền màu","Ảnh nền khác","Làm mờ nền"],
                                              value="PNG trong suốt",label="Kiểu xuất")
                        mode_hint = gr.Markdown("**PNG trong suốt:** giữ alpha để ghép vào thiết kế khác.")
                        with gr.Column(visible=False) as color_group:
                            color = gr.ColorPicker(label="Màu nền mới",value="#F1F5F9")
                        with gr.Column(visible=False) as bg_image_group:
                            bg_image = gr.Image(label="Tải ảnh hậu cảnh",type="numpy",image_mode="RGB",
                                                sources=["upload"],height=200)
                        with gr.Column(visible=False) as blur_group:
                            blur = gr.Slider(1,40,value=12,step=1,label="Mức làm mờ nền")
                        export_btn = gr.Button("↓ XUẤT ẢNH PNG",variant="primary",elem_classes=["g3-primary"])
                    with gr.Column(scale=6,min_width=0):
                        final_preview = gr.Image(label="Kết quả cuối cùng",type="numpy",interactive=False,height=290)
                        final_file = gr.File(label="Tải PNG về máy")
                final_status = gr.Markdown("")

                # Kết nối thao tác giao diện với các hàm xử lý tương ứng.
                image_input.change(clear_image_session, inputs=[], outputs=[st, preview, mask_view, edit_canvas, status, choice_group, manual_group, export_group, final_preview, final_file, final_status], queue=False, show_progress="hidden")
                event_image_auto = auto_btn.click(auto_segment, inputs=[image_input, chk_enhanced, chk_guided, chk_strong, radius, eps], outputs=[st, preview, mask_view, edit_canvas, status, choice_group, manual_group, export_group, final_preview, final_file, final_status], concurrency_limit=1)
                event_image_auto.then(fn=None,inputs=[],outputs=[],queue=False,
                    js="() => { setTimeout(() => document.getElementById('aetm-image-review')?.scrollIntoView({behavior:'smooth',block:'start'}), 180); }")
                event_image_manual = manual_open.click(open_manual, inputs=[st], outputs=[st, manual_group, export_group, edit_canvas, manual_status], queue=False)
                event_image_manual.then(fn=None,inputs=[],outputs=[],queue=False,
                    js="() => { setTimeout(() => document.getElementById('aetm-image-manual')?.scrollIntoView({behavior:'smooth',block:'start'}), 180); }")
                event_image_approve = auto_ok.click(approve_auto, inputs=[st], outputs=[st, export_group, manual_group, status], queue=False)
                event_image_approve.then(fn=None,inputs=[],outputs=[],queue=False,
                    js="() => { setTimeout(() => document.getElementById('aetm-image-export')?.scrollIntoView({behavior:'smooth',block:'start'}), 180); }")
                event_image_manual_ok = manual_ok.click(approve_manual, inputs=[st], outputs=[st, preview, mask_view, manual_group, export_group, status], queue=False)
                event_image_manual_ok.then(fn=None,inputs=[],outputs=[],queue=False,
                    js="() => { setTimeout(() => document.getElementById('aetm-image-export')?.scrollIntoView({behavior:'smooth',block:'start'}), 180); }")
                final_mode.change(output_controls, inputs=[final_mode], outputs=[color_group, bg_image_group, blur_group, mode_hint], queue=False, show_progress="hidden")
                export_btn.click(export_image, inputs=[st, final_mode, bg_image, color, blur], outputs=[final_preview, final_file, final_status], queue=False)

        # Hai tab Video và Webcam được xây dựng từ module riêng.
        video_page = add_guided_v40_tab()
        webcam_page = add_webcam_tab(get_model=get_model, model_lock=MODEL_LOCK)
        # Chỉ hiển thị tab Ảnh, Video hoặc Webcam đang được chọn.
        def _switch_page(selected):
            return (gr.update(visible=selected=="Ảnh"),gr.update(visible=selected=="Video"),
                    gr.update(visible=selected=="Webcam"))
        nav.change(_switch_page,inputs=[nav],outputs=[image_page,video_page,webcam_page],queue=False)
        gr.HTML('<div class="aetm-footer">3G · Tách nền Ảnh / Video / Webcam bằng AI cục bộ. Kết quả phụ thuộc ánh sáng và chuyển động.</div>')
    return demo


# CHẠY MÁY CHỦ: kết nối FastAPI, Gradio và mở giao diện cục bộ.
# Gắn API chỉnh mask và giao diện Gradio vào máy chủ FastAPI.
def build_server() -> FastAPI:
    """FastAPI cho co Anh/Video + Gradio 3 tab."""
    server = create_editor_api()
    register_v40_routes(server)  # Canvas v4.0: chọn mục tiêu và sửa mask
    demo = build_app()
    demo.queue(default_concurrency_limit=2, max_size=16)
    return gr.mount_gradio_app(
        server, demo, path="/", server_name="127.0.0.1", server_port=7866,
        show_error=True,
        **({"css": CSS, "theme": THREEG_THEME} if int(gr.__version__.split(".")[0]) >= 6 else {}),
    )


# Tự mở trình duyệt khi máy chủ 3G đã sẵn sàng.
def _open_browser_when_ready() -> None:
    """Mo trang giao dien khi server thuc su san sang. Khong can mo tay."""
    import urllib.request
    import webbrowser

    url = "http://127.0.0.1:7866/"
    for _ in range(90):
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    webbrowser.open(url, new=2)
                    return
        except (OSError, ValueError):
            pass
        time.sleep(0.7)


# ĐIỂM BẮT ĐẦU: chạy server ở cổng 7866 và tự mở trình duyệt.
if __name__ == "__main__":
    print("[3G] Giao dien: http://127.0.0.1:7866")
    print("[3G] Giao dien do ban chinh: Anh / Video 5 moc / Webcam ghi MP4.")
    print("[3G] Trinh duyet se tu mo khi server san sang.")
    threading.Thread(target=_open_browser_when_ready, daemon=True).start()
    # Gioi han thoi gian cho cac yeu cau do dang khi nhan Ctrl+C.
    # Khong bam dung khi dang xu ly video can luu.
    uvicorn.run(
        build_server(), host="127.0.0.1", port=7866,
        log_level="info", timeout_graceful_shutdown=5,
    )
