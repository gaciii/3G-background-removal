
from __future__ import annotations

# Các thư viện dùng cho giao diện, xử lý video, quản lý file và gọi tác vụ AI.
from collections import OrderedDict
import csv
import html
import zipfile
import json
import os
from pathlib import Path
import re
import subprocess
import shutil
import threading
import time
import uuid

import cv2
import gradio as gr
from ui_3g import intro, step, note, help_box, flow
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

# Đường dẫn và dữ liệu phiên làm việc của tab Video.
ROOT=Path(__file__).resolve().parent
WORKER=ROOT/'video_guided_worker.py'
OUTPUTS=ROOT/'outputs'/'guided_v40'
SESSIONS:OrderedDict[str,dict]=OrderedDict()
LOCK=threading.RLock()
MAX_ANCHORS=5


# Lấy đường dẫn video từ dữ liệu tải lên trên giao diện Gradio.
def _video_path(value):
    if isinstance(value,(list,tuple)):value=value[0] if value else None
    if isinstance(value,dict):value=value.get('video') or value.get('path')
    return str(value) if value else ''


# Đọc FPS, số frame, kích thước video và kiểm tra giới hạn 30 giây.
def _info(path):
    cap=cv2.VideoCapture(path)
    try:
        if not cap.isOpened():raise ValueError('Video không đọc được.')
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if not .01<fps<=240:fps=25.
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(3));h=int(cap.get(4))
        if n<1 or min(w,h)<2:raise ValueError('Video không hợp lệ.')
        if n/fps>30.1:raise ValueError('Video chỉ hỗ trợ tối đa 30 giây để tránh quá tải CPU.')
        scale=min(1.0,640/max(w,h))
        dw=max(2,int(round(w*scale))//2*2);dh=max(2,int(round(h*scale))//2*2)
        return {'fps':fps,'frames':n,'width':w,'height':h,'dw':dw,'dh':dh,
                'duration':(n-1)/fps}
    finally:cap.release()


# Lấy phiên làm việc video theo token; báo lỗi nếu phiên đã hết.
def _get(token):
    with LOCK:
        state=SESSIONS.get(token)
        if state is None:raise HTTPException(404,'Phiên làm việc không còn. Tải lại video.')
        SESSIONS.move_to_end(token)
        return state


# Xác nhận video hiện tại khớp với phiên đang thao tác.
def _state(state,video):
    path=_video_path(video)
    if not state or path!=state.get('path'):
        raise gr.Error('Hãy tải video gốc trong tab v4.0 để tạo phiên làm việc mới.')
    try:return _get(state['token'])
    except HTTPException as err:raise gr.Error(str(err.detail)) from err


# Đổi thời điểm (giây) thành chỉ số frame hợp lệ.
def _index(session,seconds):
    return max(0,min(session['info']['frames']-1,
                     int(round(float(seconds or 0)*session['info']['fps']))))


# Đọc một frame trong video và thay đổi về kích thước đang xử lý.
def _frame(s,index):
    info=s['info'];job=s.get('job')
    size=((int(job['width']),int(job['height'])) if job else (info['dw'],info['dh']))
    cap=cv2.VideoCapture(s['path'])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,fr=cap.read()
        if not ok:raise HTTPException(400,'Không đọc được khung hình.')
        return cv2.resize(fr,size)
    finally:cap.release()


# Xác định vị trí file mask AI hoặc mask đã sửa thủ công.
def _mask_path(s,index,final=False):
    j=s.get('job_dir')
    if not j:return None
    name=f'{index:06d}.png'
    return Path(j)/('masks_final' if final else 'masks_ai')/name


# Đọc ảnh mask xám và chuẩn hóa kích thước theo frame.
def _readmask(path,shape):
    if not path or not Path(path).is_file():return None
    m=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if m is None:return None
    h,w=shape
    if m.shape!=(h,w):m=cv2.resize(m,(w,h),interpolation=cv2.INTER_LINEAR)
    return m.astype(np.uint8)


# Tìm mốc tham chiếu đã tạo tại một frame cụ thể.
def _anchor_at(s,index):
    return next((a for a in s.get('anchors',{}).values() if a['frame_index']==int(index)),None)


# Lấy mask của mốc; nếu chưa vẽ thì dùng mask rỗng.
def _mask_seed(s,index):
    a=_anchor_at(s,index)
    shape=(s['info']['dh'],s['info']['dw'])
    return (a['mask'].copy() if a and a['mask'] is not None
            else np.zeros(shape,np.uint8))


# Chọn mask đang dùng ở bước tạo mốc hoặc bước sửa sau AI.
def _effective(s,index,stage='seed'):
    if stage=='seed':return _mask_seed(s,index)
    if stage!='local':raise HTTPException(400,'Chế độ chỉnh mask không hợp lệ.')
    if not s.get('job'):return np.zeros((s['info']['dh'],s['info']['dw']),np.uint8)
    shape=(int(s['job']['height']),int(s['job']['width']))
    mask=_readmask(_mask_path(s,index,True),shape)
    if mask is None:mask=_readmask(_mask_path(s,index),shape)
    return mask if mask is not None else np.zeros(shape,np.uint8)


# Đồng bộ frame và mask của mốc chính để gửi sang YOLO.
def _sync_primary(s):
    a=s.get('anchors',{}).get(s.get('primary_id'))
    s['seed_index']=a['frame_index'] if a else 0
    s['seed_mask']=a['mask'].copy() if a and a['mask'] is not None else None


# Lưu tối đa 5 mốc thành mask PNG và tệp manifest.json.
def _persist_anchors(s):
    """Write real, portable PNG masks and metadata; never store only UI state."""
    folder=OUTPUTS/s['token']/'anchors'
    folder.mkdir(parents=True,exist_ok=True)
    anchors=[];names=set()
    for a in s['anchors'].values():
        fn=f"{a['id']}.png"
        mask_file=folder/fn
        if a['mask'] is None:
            mask_file.unlink(missing_ok=True)
        else:
            if not cv2.imwrite(str(mask_file),a['mask']):
                raise RuntimeError('Không lưu được PNG mask cho mốc.')
            names.add(fn)
        anchors.append({'anchor_id':a['id'],'frame_index':a['frame_index'],
                        'time_sec':round(a['frame_index']/s['info']['fps'],5),
                        'name':a['name'],'object_id':a['object_id'],
                        'has_mask':a['mask'] is not None,
                        'mask_file':f'anchors/{fn}' if a['mask'] is not None else None})
    for f in folder.glob('*.png'):
        if f.name not in names:f.unlink()
    manifest={'version':'4.3','source_video':Path(s['path']).name,
              'fps':s['info']['fps'],'frame_count':s['info']['frames'],
              'image_size':[s['info']['dw'],s['info']['dh']],
              'primary_anchor_id':s.get('primary_id'),
              'max_anchors':MAX_ANCHORS,'anchors':anchors}
    target=folder/'manifest.json';temp=folder/'manifest.json.tmp'
    temp.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(target)
    _sync_primary(s)
    return target


# Ghi mask vào mốc hoặc lưu mask đã chỉnh sau khi chạy AI.
def _write_mask(s,index,mask,stage='seed'):
    if stage=='local':
        if not s.get('job'):raise HTTPException(400,'Cần chạy AI trước khi sửa kết quả.')
        target=_mask_path(s,index,True);target.parent.mkdir(exist_ok=True,parents=True)
        if not cv2.imwrite(str(target),mask):raise HTTPException(500,'Không lưu được mask chỉnh sửa.')
        s.setdefault('changed',set()).add(index)
    elif stage=='seed':
        anchor=_anchor_at(s,index)
        if anchor is None:raise HTTPException(400,'Hãy nhấn TẠO MỐC ở thời điểm này trước khi vẽ.')
        anchor['mask']=mask.copy()
        _persist_anchors(s)
    else:raise HTTPException(400,'Chế độ không hợp lệ.')


# Tạo danh sách mốc hiển thị trong ô chọn của Gradio.
def _anchor_choices(s):
    result=[]
    for a in sorted(s.get('anchors',{}).values(),key=lambda x:(x['frame_index'],x['id'])):
        star='★ ' if a['id']==s.get('primary_id') else ''
        ready='✓' if a['mask'] is not None and np.mean(a['mask']>127)>=.001 else '○'
        result.append((f"{star}{ready} {a['frame_index']/s['info']['fps']:.2f}s · {a['name']} · {a['object_id']}",a['id']))
    return result


# Tạo phần mô tả trạng thái, tên, ID và mốc chính.
def _anchor_summary(s):
    anchors=sorted(s.get('anchors',{}).values(),key=lambda a:a['frame_index'])
    lines=[f'**Đã lưu {len(anchors)}/{MAX_ANCHORS} mốc tham chiếu**']
    if not anchors:lines+=['Chọn giây cần đánh dấu rồi nhấn **+ Tạo mốc hiện tại**.']
    for j,a in enumerate(anchors,1):
        indicator='**MỐC CHÍNH**' if a['id']==s.get('primary_id') else 'Mốc hỗ trợ'
        status='Đã vẽ mask' if a['mask'] is not None and (a['mask']>127).sum()>10 else 'Chưa có mask'
        lines.append(f"{j}. **{a['frame_index']/s['info']['fps']:.2f}s** · {html.escape(a['name'])} · {html.escape(a['object_id'])} · {status} · {indicator}")
    lines+=['*v4.4 sử dụng mọi mốc có mask cùng ID với mốc chính. Mốc khác ID được lưu nhưng không tự gộp; nếu muốn tách ID khác, đặt ID đó làm mốc chính rồi chạy lại.*']
    return '\n\n'.join(lines)

# Đổi đường bao người dùng vẽ thành vùng mask nhị phân.
def _polygon(points,shape):
    if not isinstance(points,list) or not 3<=len(points)<=5000:
        raise HTTPException(400,'Hãy vẽ một vòng khép kín ít nhất ba điểm.')
    h,w=shape;coords=[]
    for p in points:
        if not isinstance(p,list) or len(p)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) for x in p):
            raise HTTPException(400,'Tọa độ đường vẽ không hợp lệ.')
        coords.append((max(0,min(w-1,round(p[0]))),max(0,min(h-1,round(p[1])))))
    poly=np.asarray(coords,np.int32)
    if abs(float(cv2.contourArea(poly)))<5:raise HTTPException(400,'Vùng vẽ quá nhỏ.')
    area=np.zeros(shape,np.uint8);cv2.fillPoly(area,[poly],255)
    return area


# Tạo ảnh lớp phủ màu đỏ để hiển thị vùng được giữ.
def _mask_png(mask):
    rgba=np.zeros((*mask.shape,4),np.uint8)
    rgba[:,:,2]=255  # red in BGRA (red index 2)
    rgba[:,:,3]=np.uint8(mask.astype(np.float32)*.48)
    ok,data=cv2.imencode('.png',rgba)
    if not ok:raise HTTPException(500,'Không tạo được ảnh mask.')
    return Response(data.tobytes(),media_type='image/png',headers={'Cache-Control':'no-store'})


# Giao diện HTML/CSS/JavaScript cho thao tác Bao / Thêm / Xóa bằng chuột.
EDITOR_HTML=r'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box}body{font:14px system-ui,Arial;margin:0;padding:10px;color:#14202e;background:#f8fafc}
.toolbar{display:flex;gap:7px;flex-wrap:wrap;margin:0 0 10px}button{padding:9px 12px;border:1px solid #cbd5e1;background:#fff;border-radius:9px;font-weight:600;cursor:pointer}
button.on{background:#0f766e;color:#fff}#erase.on{background:#b91c1c}#add.on{background:#187d3d}
#stage{height:485px;overflow:auto;display:flex;justify-content:center;align-items:center;background:#e2e8f0;border-radius:10px;padding:5px}
canvas{touch-action:none;cursor:crosshair;box-shadow:0 2px 10px #0002}#msg{min-height:38px;margin-top:9px;color:#166534;font-weight:600}
small{color:#475569}
</style></head><body>
<div class="toolbar"><button id="base">1. Bao đối tượng</button><button id="add">2. Thêm vùng thiếu (+)</button><button id="erase">3. Xóa vùng thừa (−)</button></div>
<div id="stage"><canvas id="canvas"></canvas></div>
<div id="msg">Đang nạp khung hình…</div>
<div class="toolbar"><button id="undo">↶ Hoàn tác</button><button id="reset">Xóa mask ở khung này</button><button id="full">Giữ toàn khung</button></div>
<small>Giữ chuột và vẽ một vòng khép kín, thả chuột để lưu ngay. Mask đỏ là phần được giữ. Ở bước 3, các nét chỉ chỉnh mask hiện tại; bạn có thể lan truyền ±N frame bằng nút bên ngoài.</small>
<script>
const T=__TOKEN__,I=__INDEX__,STAGE=__STAGE__,C=document.getElementById('canvas'),X=C.getContext('2d');
const msg=document.getElementById('msg'),root='/guided40/'+T;
let base=null,overlay=null,mode='base',draw=false,stroke=null,busy=false;
function note(t,bad=false){msg.textContent=t;msg.style.color=bad?'#b91c1c':'#166534'}
function choose(m){mode=m;for(const k of ['base','add','erase'])document.getElementById(k).classList.toggle('on',k===m);note('Đã chọn '+({base:'Bao đối tượng',add:'Thêm vùng',erase:'Xóa vùng'}[m])+'. Vẽ đường khép kín trên ảnh.');}
for(const k of ['base','add','erase'])document.getElementById(k).onclick=()=>choose(k);
function xy(e){const r=C.getBoundingClientRect();return [Math.round(Math.max(0,Math.min(C.width-1,(e.clientX-r.left)*C.width/r.width))),Math.round(Math.max(0,Math.min(C.height-1,(e.clientY-r.top)*C.height/r.height)))];}
function show(){if(!base)return;X.clearRect(0,0,C.width,C.height);X.drawImage(base,0,0);if(overlay)X.drawImage(overlay,0,0,C.width,C.height);
if(stroke&&stroke.length>1){X.beginPath();X.moveTo(...stroke[0]);for(let j=1;j<stroke.length;j++)X.lineTo(...stroke[j]);X.closePath();X.strokeStyle=mode==='erase'?'#0284c7':'#eab308';X.lineWidth=3;X.setLineDash([6,3]);X.stroke();X.setLineDash([]);}}
async function json(path,body){const r=await fetch(root+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let j=await r.json();if(!r.ok)throw Error(j.detail||'Không lưu được');return j;}
function loadImg(src){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=reject;im.src=src;});}
async function refresh(){overlay=await loadImg(root+'/mask?i='+I+'&stage='+STAGE+'&t='+Date.now());show();}
async function ready(){try{base=await loadImg(root+'/frame?i='+I);C.width=base.naturalWidth;C.height=base.naturalHeight;
let s=Math.min(1,650/C.width,465/C.height);C.style.width=Math.round(C.width*s)+'px';C.style.height=Math.round(C.height*s)+'px';
choose(STAGE==='local'?'add':'base');await refresh();note('✅ Đã nạp frame. Vẽ để chỉnh vùng đỏ.');}catch(e){note('Không tải được ảnh: '+e.message,true);}}
function append(p){if(!stroke||Math.hypot(p[0]-stroke[stroke.length-1][0],p[1]-stroke[stroke.length-1][1])>1.5)stroke.push(p);}
C.addEventListener('pointerdown',e=>{if(!base||busy||e.button!==0)return;e.preventDefault();C.setPointerCapture(e.pointerId);draw=true;stroke=[xy(e)];});
C.addEventListener('pointermove',e=>{if(!draw)return;append(xy(e));show();});
C.addEventListener('pointerup',async e=>{if(!draw)return;draw=false;append(xy(e));let points=stroke;stroke=null;show();if(points.length<3){note('Vẽ một vòng khép kín.',true);return;}
busy=true;try{const data=await json('/edit',{i:I,stage:STAGE,mode,points});await refresh();note('✅ '+data.message);}catch(err){note('Lỗi: '+err.message,true);}finally{busy=false;}});
C.addEventListener('pointercancel',()=>{draw=false;stroke=null;show();});
for(const [id,action] of [['undo','/undo'],['reset','/reset'],['full','/full']])document.getElementById(id).onclick=async()=>{if(busy)return;busy=true;try{const data=await json(action,{i:I,stage:STAGE});await refresh();note('✅ '+data.message);}catch(err){note(err.message,true);}finally{busy=false;}};
document.addEventListener('keydown',e=>{if(e.key==='Escape'){stroke=null;draw=false;show();}});
ready();
</script></body></html>'''


# Đăng ký các API FastAPI phục vụ canvas vẽ và chỉnh mask video.
def register_v40_routes(server:FastAPI):
    # Trả trang HTML của công cụ vẽ cho frame và bước được chọn.
    @server.get('/guided40/{token}',response_class=HTMLResponse)
    def page(token:str,i:int=0,stage:str='seed'):
        s=_get(token);i=max(0,min(s['info']['frames']-1,i))
        if stage not in ('seed','local'):raise HTTPException(400,'Sai chế độ')
        return (EDITOR_HTML.replace('__TOKEN__',json.dumps(token))
                           .replace('__INDEX__',str(i))
                           .replace('__STAGE__',json.dumps(stage)))

    # Trả hình frame để hiển thị trong canvas video.
    @server.get('/guided40/{token}/frame')
    def frame(token:str,i:int=0):
        s=_get(token);i=max(0,min(s['info']['frames']-1,i))
        data=_frame(s,i);ok,buf=cv2.imencode('.jpg',data,[cv2.IMWRITE_JPEG_QUALITY,94])
        if not ok:raise HTTPException(500,'Không xuất được ảnh')
        return Response(buf.tobytes(),media_type='image/jpeg',headers={'Cache-Control':'no-store'})

    # Trả lớp phủ mask để người dùng xem vùng đã chọn.
    @server.get('/guided40/{token}/mask')
    def mask(token:str,i:int=0,stage:str='seed'):
        s=_get(token);i=max(0,min(s['info']['frames']-1,i))
        return _mask_png(_effective(s,i,stage))

    # Nhận đường bao từ trình duyệt và thực hiện Bao / Thêm / Xóa.
    @server.post('/guided40/{token}/edit')
    def edit(token:str,payload:dict=Body(...)):
        s=_get(token);i=int(payload.get('i',-1));stage=str(payload.get('stage','seed'))
        if not 0<=i<s['info']['frames']:raise HTTPException(400,'Frame sai')
        mode=str(payload.get('mode',''))
        if mode not in ('base','add','erase'):raise HTTPException(400,'Công cụ không hợp lệ.')
        if stage not in ('seed','local'):raise HTTPException(400,'Sai chế độ')
        with LOCK:
            current=_effective(s,i,stage)
            if stage=='seed' and _anchor_at(s,i) is None:
                raise HTTPException(400,'Chưa có mốc tại giây này. Nhấn + Tạo mốc hiện tại trước.')
            if stage=='local' and not s.get('job'):
                raise HTTPException(400,'Chạy YOLO trước khi sửa kết quả.')
            poly=_polygon(payload.get('points'),current.shape)
            anchor=_anchor_at(s,i)
            if stage=='seed' and mode!='base' and anchor['mask'] is None:
                raise HTTPException(400,'Hãy dùng Bao đối tượng trước khi thêm/xóa.')
            key=(i,stage)
            # Remember the mask BEFORE any new manual strokes. Do NOT compare
            # the edited mask with masks_ai: that may contain earlier propagated
            # adjustments and would accidentally send them out a second time.
            if stage=='local':
                s.setdefault('pending_edit_base',{}).setdefault(i,current.copy())
            s.setdefault('history',{}).setdefault(key,[]).append(current.copy())
            if len(s['history'][key])>20:s['history'][key].pop(0)
            if mode=='base':new=poly
            elif mode=='add':new=cv2.bitwise_or(current,poly)
            else:new=cv2.bitwise_and(current,cv2.bitwise_not(poly))
            _write_mask(s,i,new,stage)
        return {'message':f'Đã {dict(base="bao",add="thêm",erase="xóa")[mode]} vùng tại {i/s["info"]["fps"]:.2f}s. Diện tích giữ: {float(np.mean(new>127))*100:.1f}%.'}

    # Hoàn tác lần sửa mask gần nhất của frame đang chọn.
    @server.post('/guided40/{token}/undo')
    def undo(token:str,payload:dict=Body(...)):
        s=_get(token);i=int(payload.get('i',-1));stage=str(payload.get('stage','seed'))
        if not 0<=i<s['info']['frames'] or stage not in ('seed','local'):
            raise HTTPException(400,'Frame hoặc chế độ không hợp lệ.')
        with LOCK:
            stack=s.setdefault('history',{}).get((i,stage),[])
            if not stack:raise HTTPException(400,'Không có bước để hoàn tác.')
            if stage=='local':
                s.setdefault('pending_edit_base',{}).setdefault(i,_effective(s,i,'local').copy())
            _write_mask(s,i,stack.pop(),stage)
        return {'message':'Đã hoàn tác nét cuối.'}

    # Xóa mask của frame hiện tại, không xóa các mốc khác.
    @server.post('/guided40/{token}/reset')
    def reset(token:str,payload:dict=Body(...)):
        s=_get(token);i=int(payload.get('i',-1));stage=str(payload.get('stage','seed'))
        if not 0<=i<s['info']['frames'] or stage not in ('seed','local'):
            raise HTTPException(400,'Frame hoặc chế độ không hợp lệ.')
        with LOCK:
            current=_effective(s,i,stage)
            key=(i,stage)
            s.setdefault('history',{}).setdefault(key,[]).append(current.copy())
            if stage=='local':
                if not s.get('job'):raise HTTPException(400,'Chưa có video AI.')
                path=_mask_path(s,i,True)
                if path:path.unlink(missing_ok=True)
                s.setdefault('changed',set()).discard(i)
                s.setdefault('pending_edit_base',{}).pop(i,None)
            else:
                anchor=_anchor_at(s,i)
                if anchor is None:raise HTTPException(400,'Chưa có mốc tại khung này.')
                anchor['mask']=None
                _persist_anchors(s)
        return {'message':'Đã xóa mask của khung này. Những mốc khác vẫn được giữ.'}

    # Đặt toàn bộ frame thành vùng giữ để sau đó dùng cọ Xóa.
    @server.post('/guided40/{token}/full')
    def full(token:str,payload:dict=Body(...)):
        s=_get(token);i=int(payload.get('i',-1));stage=str(payload.get('stage','seed'))
        if not 0<=i<s['info']['frames'] or stage not in ('seed','local'):
            raise HTTPException(400,'Frame hoặc chế độ không hợp lệ.')
        with LOCK:
            if stage=='seed' and _anchor_at(s,i) is None:
                raise HTTPException(400,'Hãy tạo mốc trước.')
            if stage=='local' and not s.get('job'):
                raise HTTPException(400,'Chưa có video AI.')
            current=_effective(s,i,stage)
            key=(i,stage)
            if stage=='local':
                s.setdefault('pending_edit_base',{}).setdefault(i,current.copy())
            s.setdefault('history',{}).setdefault(key,[]).append(current.copy())
            _write_mask(s,i,np.full_like(current,255),stage)
        return {'message':'Đã giữ toàn bộ khung, có thể dùng Xóa để loại nền thừa.'}


# Nhúng giao diện vẽ mask vào tab Video của Gradio.
def _iframe(s,i,stage='seed'):
    return (f'<iframe title="3G — sửa mask" src="/guided40/{s["token"]}?i={i}&stage={stage}" '
            'style="border:1px solid #cbd5e1;border-radius:12px;width:100%;height:675px;" '
            'loading="eager"></iframe>')


# Tạo phiên mới khi tải video và khởi tạo các thành phần giao diện.
def upload(video):
    path=_video_path(video)
    if not path:
        return (None,'',gr.update(maximum=1,value=0),gr.update(maximum=1,value=0),
                'Chưa tải video.',None,None,None,'',
                gr.update(choices=[],value=None),'Chưa có mốc tham chiếu.')
    try:info=_info(path)
    except ValueError as exc:raise gr.Error(str(exc)) from exc
    token=uuid.uuid4().hex[:16]
    session={'token':token,'path':path,'info':info,'seed_index':0,'seed_mask':None,
             'anchors':OrderedDict(),'primary_id':None,'history':{},'job':None}
    with LOCK:
        SESSIONS[token]=session
        while len(SESSIONS)>6:SESSIONS.popitem(last=False)
    v={'token':token,'path':path}
    return (v,_iframe(session,0,'seed'),
            gr.update(maximum=info['duration'],value=0,step=round(1/info['fps'],5)),
            gr.update(maximum=info['duration'],value=0,step=round(1/info['fps'],5)),
            f'✅ Video có **{info["frames"]} frame**, dài **{info["frames"]/info["fps"]:.2f}s**. '
            'Kéo tới thời điểm mong muốn rồi nhấn **+ Tạo mốc hiện tại** (tối đa 5).',
            None,None,None,'Chưa chạy AI.',
            gr.update(choices=[],value=None),_anchor_summary(session))


# Chuyển thanh thời gian tới frame cần tạo hoặc sửa mốc.
def seek_seed(video,seconds,state):
    s=_state(state,video)
    i=_index(s,seconds)
    a=_anchor_at(s,i)
    if a:
        msg=f'Mốc **{html.escape(a["name"])}** tại {i/s["info"]["fps"]:.2f}s. Chỉnh mask bằng Bao/Thêm/Xóa.'
    else:msg=f'Khung {i} tại {i/s["info"]["fps"]:.2f}s chưa có mốc. Nhấn **+ Tạo mốc hiện tại** trước khi vẽ.'
    return _iframe(s,i,'seed'),msg,gr.update(value=a['id'] if a else None),a['name'] if a else ''


# Kiểm tra ID đối tượng của mốc tham chiếu.
def _anchor_id_input(value):
    value=str(value or 'doi_tuong_1').strip()
    if not 1<=len(value)<=60:raise gr.Error('ID đối tượng phải dài 1–60 ký tự.')
    return value


# Tạo mốc tham chiếu mới tại thời điểm được chọn (tối đa 5).
def add_anchor(video,seconds,state,name,object_id):
    s=_state(state,video);i=_index(s,seconds)
    with LOCK:
        previous=_anchor_at(s,i)
        if previous:
            aid=previous['id']
            msg=f'⚠️ Đã có mốc tại {i/s["info"]["fps"]:.2f}s. Chọn nó rồi chỉnh tiếp; không tạo bản trùng.'
        else:
            if len(s['anchors'])>=MAX_ANCHORS:
                raise gr.Error('Đã đủ 5 mốc. Hãy xóa một mốc cũ nếu muốn thêm mốc mới.')
            aid=uuid.uuid4().hex[:10]
            label=str(name or '').strip()[:80] or f'Mốc {len(s["anchors"])+1}'
            s['anchors'][aid]={'id':aid,'frame_index':i,'name':label,
                               'object_id':_anchor_id_input(object_id),'mask':None}
            if s['primary_id'] is None:s['primary_id']=aid
            _persist_anchors(s)
            msg=f'✅ Đã tạo mốc {len(s["anchors"])}/{MAX_ANCHORS} tại {i/s["info"]["fps"]:.2f}s. Vẽ Bao/Thêm/Xóa ở khung này.'
    return (gr.update(choices=_anchor_choices(s),value=aid),
            _iframe(s,i,'seed'),msg,_anchor_summary(s))


# Mở lại một mốc đã lưu để xem và chỉnh sửa.
def pick_anchor(video,state,aid):
    s=_state(state,video)
    a=s['anchors'].get(str(aid))
    if not a:raise gr.Error('Mốc đã bị xóa. Hãy chọn mốc khác.')
    i=a['frame_index']
    return (gr.update(value=i/s['info']['fps']),_iframe(s,i,'seed'),
            f'✅ Đang chỉnh **{html.escape(a["name"])}**, giây **{i/s["info"]["fps"]:.2f}**. '
            +('Đã có mask.' if a['mask'] is not None else 'Hãy vẽ Bao đối tượng.'),
            a['name'],a['object_id'])


# Chọn mốc chính để xác định đối tượng ưu tiên của YOLO.
def set_primary(video,state,aid):
    s=_state(state,video)
    with LOCK:
        if str(aid) not in s['anchors']:raise gr.Error('Hãy chọn một mốc trước.')
        s['primary_id']=str(aid)
        _persist_anchors(s)
    a=s['anchors'][str(aid)]
    return gr.update(choices=_anchor_choices(s),value=aid),_anchor_summary(s),f'★ Đã chọn **{html.escape(a["name"])}** làm mốc chính; các mốc cùng ID sẽ hỗ trợ YOLO.'


# Lưu tên và ID đối tượng mới cho mốc.
def rename_anchor(video,state,aid,name,object_id):
    s=_state(state,video)
    with LOCK:
        a=s['anchors'].get(str(aid))
        if a is None:raise gr.Error('Hãy chọn mốc để đổi tên.')
        name=str(name or '').strip()
        if not name or len(name)>80:raise gr.Error('Tên mốc phải có 1–80 ký tự.')
        a['name']=name;a['object_id']=_anchor_id_input(object_id)
        _persist_anchors(s)
    return gr.update(choices=_anchor_choices(s),value=aid),_anchor_summary(s),f'✅ Đã cập nhật {html.escape(name)}.'


# Xóa mốc được chọn và cập nhật lại mốc chính khi cần.
def delete_anchor(video,state,aid):
    s=_state(state,video)
    with LOCK:
        if str(aid) not in s['anchors']:raise gr.Error('Hãy chọn mốc cần xóa.')
        s['anchors'].pop(str(aid))
        s['history'].clear()
        if s['primary_id']==str(aid):
            s['primary_id']=next(iter(s['anchors']),None)
        _persist_anchors(s)
        nxt=s['anchors'].get(s['primary_id'])
    i=nxt['frame_index'] if nxt else 0
    msg='✅ Đã xóa mốc. '+('Mốc chính còn lại đang được hiển thị.' if nxt else 'Hãy tạo mốc mới.')
    return gr.update(choices=_anchor_choices(s),value=nxt['id'] if nxt else None),           gr.update(value=i/s['info']['fps']),_iframe(s,i,'seed'),msg,_anchor_summary(s)


# Làm mới danh sách và thông tin các mốc tham chiếu.
def refresh_anchors(video,state,aid):
    s=_state(state,video)
    selected=str(aid) if str(aid) in s['anchors'] else None
    return gr.update(choices=_anchor_choices(s),value=selected),_anchor_summary(s)


# Đóng gói các mốc và manifest vào tệp ZIP để dùng lại.
def download_anchors(video,state):
    s=_state(state,video)
    with LOCK:
        if not s['anchors']:raise gr.Error('Chưa có mốc để xuất.')
        manifest=_persist_anchors(s)
        root=manifest.parent
        dest=OUTPUTS/s['token']/'3g_moc_tham_chieu.zip'
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            z.write(manifest,'manifest.json')
            for a in s['anchors'].values():
                if a['mask'] is not None:
                    file=root/f"{a['id']}.png"
                    z.write(file,f"masks/{a['id']}.png")
            z.writestr('README.txt','3G: tối đa 5 mốc. Tên/ID/thời gian trong manifest.json. '
                        'Các mask PNG có kích thước image_size. Không kèm video gốc.\n')
    return str(dest),f'✅ Đã lưu **{len(s["anchors"])} mốc** vào ZIP. Giữ lại video gốc để sử dụng cùng bộ mốc.'


# Đọc bộ mốc ZIP cũ và kiểm tra tính tương thích với video.
def import_anchors(video,state,zip_path):
    """Restore saved v4.3/v4.4 anchors; never trust ZIP paths or overwrite other video."""
    s=_state(state,video)
    if not zip_path:raise gr.Error('Hãy chọn file ZIP bộ mốc bạn đã tải trước đó.')
    path=Path(str(zip_path))
    if not path.is_file() or path.stat().st_size > 15_000_000:
        raise gr.Error('ZIP không hợp lệ hoặc quá lớn.')
    try:
        with zipfile.ZipFile(path,'r') as z:
            data=json.loads(z.read('manifest.json'))
            items=data.get('anchors',[])
            if not isinstance(items,list) or not 1<=len(items)<=MAX_ANCHORS:
                raise ValueError('Bộ mốc phải có 1-5 mục.')
            if int(data.get('frame_count',-1)) != s['info']['frames']:
                raise ValueError('Bộ mốc thuộc video có số frame khác. Hãy chọn lại video gốc.')
            fps_old=float(data.get('fps',0))
            if abs(fps_old-s['info']['fps'])>.11:
                raise ValueError('FPS video không khớp bộ mốc đã lưu.')
            if list(map(int,data.get('image_size',[])))!=[s['info']['dw'],s['info']['dh']]:
                raise ValueError('Kích thước video không khớp bộ mốc.')
            staged=OrderedDict()
            for a in items:
                aid=str(a.get('anchor_id') or '')
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,90}',aid) or aid in staged:
                    raise ValueError('Mã mốc không hợp lệ hoặc bị trùng.')
                index=int(a['frame_index'])
                if not 0<=index<s['info']['frames'] or any(k['frame_index']==index for k in staged.values()):
                    raise ValueError('Thời điểm mốc không hợp lệ hoặc trùng.')
                mask=None
                if a.get('has_mask'):
                    filename=f'masks/{aid}.png'
                    entry=z.getinfo(filename)
                    if entry.file_size>8_000_000:raise ValueError('Mask PNG quá lớn.')
                    raw=np.frombuffer(z.read(filename),np.uint8)
                    mask=cv2.imdecode(raw,cv2.IMREAD_GRAYSCALE)
                    if mask is None or mask.shape!=(s['info']['dh'],s['info']['dw']):
                        raise ValueError('Mask trong ZIP không đúng kích thước.')
                staged[aid]={'id':aid,'frame_index':index,'name':str(a.get('name','Mốc'))[:80],
                             'object_id':_anchor_id_input(a.get('object_id')),'mask':mask}
            main=str(data.get('primary_anchor_id') or '')
            if main not in staged:main=next(iter(staged))
    except (zipfile.BadZipFile,KeyError,ValueError,TypeError,json.JSONDecodeError) as err:
        raise gr.Error('Không thể nhập bộ mốc: '+str(err)) from err
    with LOCK:
        s['anchors']=staged;s['primary_id']=main
        s['job']=None;s.pop('job_dir',None)
        s['history']={}
        _persist_anchors(s)
    i=staged[main]['frame_index']
    return (gr.update(choices=_anchor_choices(s),value=main),
            gr.update(value=i/s['info']['fps']),_iframe(s,i,'seed'),
            f'✅ Đã khôi phục **{len(staged)} mốc** từ ZIP. Chọn các mốc để xem lại trước khi chạy YOLO.',
            _anchor_summary(s))


# Hiển thị frame và mask để sửa lỗi sau khi YOLO chạy xong.
def seek_local(video,seconds,state):
    s=_state(state,video)
    if not s.get('job'):return '<em>Chưa có video AI để chỉnh.</em>','Hãy chạy Bước 2 trước.'
    i=_index(s,seconds)
    return _iframe(s,i,'local'),f'Khung **{i} ({i/s["info"]["fps"]:.2f}s)**. Dùng Thêm/Xóa, nét sẽ lưu ngay.'


# Chuẩn hóa màu nền người dùng chọn về mã HEX.
def _hex(color):
    val=str(color or '')
    if re.fullmatch('#[a-fA-F0-9]{6}',val):return val.upper()
    nums=re.findall(r'[+-]?(?:\d+\.?\d*|\.\d+)',val)
    if val.startswith('rgb') and len(nums)>=3:
        return '#'+''.join(f'{int(min(255,max(0,round(float(n))))):02X}' for n in nums[:3])
    return '#DDEBFF'


# Xác định Python của môi trường .venv_yolo để chạy AI video.
def _python():
    p=ROOT/'.venv_yolo'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not p.is_file():raise gr.Error('Chưa cài .venv_yolo. Hãy chạy setup_yolo.bat ở D:\\TGMT.')
    return p


# Chạy tác vụ video ở tiến trình riêng, đọc tiến độ và kết quả.
def _run(command,progress=None):
    lines=[];result=None;failure=None
    proc=subprocess.Popen([str(p) for p in command],cwd=ROOT,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,encoding='utf-8',errors='replace',text=True,bufsize=1)
    try:
        for raw in proc.stdout:
            line=raw.strip()
            if line.startswith('PROGRESS|'):
                try:
                    _,i,n=line.split('|')
                    if progress:progress((int(i),int(n)),desc=f'3G: {i}/{n} khung hình')
                except (ValueError,TypeError):pass
            elif line.startswith('RESULT|'):
                result=json.loads(line.split('|',1)[1])
            elif line.startswith('FAILED|'):
                failure=line.split('|',1)[1]
            else:
                if line:print('[3G]',line,flush=True)
                lines.append(line);lines=lines[-25:]
        code=proc.wait()
    finally:
        if proc.poll() is None:proc.kill()
    if code!=0 or result is None:
        raise gr.Error('Lỗi AI: '+(failure or '\n'.join(lines[-10:])[-750:] or str(code)))
    return result


# Gọi tác vụ render để tạo MP4 theo chế độ nền đã chọn.
def _export(s,mode,blur,color,audio,bg,progress=None):
    job=Path(s['job_dir'])
    output=job/f'result_{uuid.uuid4().hex[:7]}.mp4'
    mode=dict(zip(['Làm mờ nền','Video mask','Nền màu','Ảnh nền khác'],['blur','mask','color','image'])).get(mode,'blur')
    cmd=[_python(),'-u',WORKER,'--action','render','--job',job,'--out',output,
         '--mode',mode,'--blur',int(blur),'--color',_hex(color)]
    if audio:cmd+=['--audio']
    if mode=='image':
        if bg is None:raise gr.Error('Hãy tải ảnh nền trước.')
        im=np.asarray(bg,np.uint8)
        if im.ndim!=3 or im.shape[2]<3:raise gr.Error('Ảnh nền phải có 3 kênh RGB.')
        background=job/'background.png'
        cv2.imwrite(str(background),cv2.cvtColor(im[:,:,:3],cv2.COLOR_RGB2BGR))
        cmd+=['--background',background]
    data=_run(cmd,progress)
    out=data['path']
    if not Path(out).is_file():raise gr.Error('Không tìm thấy MP4 đầu ra.')
    return out,data.get('stats',{})


# Kiểm tra mốc, gọi YOLO tách toàn video và xuất kết quả ban đầu.
def run_ai(video,state,label,side,imgsz,conf,edges,rescue,stabilize,quality,low_light,mode,blur,color,bg,audio,
           progress=gr.Progress(track_tqdm=False)):
    s=_state(state,video)
    with LOCK:
        _sync_primary(s)
        seed=s.get('seed_mask')
        if seed is None or np.mean(seed>127)<.001:
            raise gr.Error('Bước 1 chưa có mask đối tượng. Hãy vẽ Bao rồi xác nhận trước.')
        seed=seed.copy();seed_frame=s['seed_index']
    job=OUTPUTS/s['token'];job.mkdir(exist_ok=True,parents=True)
    with LOCK:
        manifest=_persist_anchors(s)
        primary=s['anchors'][s['primary_id']]
        selected_id=primary['object_id']
        valid_anchors=sum(1 for a in s['anchors'].values()
                          if a['object_id']==selected_id and a['mask'] is not None and np.mean(a['mask']>127)>.001)
        if valid_anchors<1:raise gr.Error('Chưa có mask hợp lệ cho ID đối tượng mốc chính.')
    # A new AI run must never inherit manual corrections or stale masks.
    for directory in ('masks_ai','masks_final'):
        old=job/directory
        if old.exists():shutil.rmtree(old)
    s['job']=None;s.pop('job_dir',None)
    seed_path=job/'seed.png';cv2.imwrite(str(seed_path),seed)
    cmd=[_python(),'-u',WORKER,'--action','segment','--video',s['path'],'--job',job,
         '--seed-mask',seed_path,'--seed-frame',seed_frame,
         '--anchors-manifest',manifest,'--target-object-id',selected_id,
         '--label',str(label),
         '--side',int(side),'--imgsz',int(imgsz),'--conf',float(conf)]
    if edges:cmd+=['--edges']
    if rescue:cmd+=['--rescue']
    if stabilize:cmd+=['--stabilize']
    if quality:cmd+=['--quality']
    if low_light:cmd+=['--low-light']
    progress(0,desc='YOLO nhận diện đúng đối tượng bạn đã chọn…')
    stats=_run(cmd,progress)['stats']
    with LOCK:
        s['job']=stats;s['job_dir']=str(job);s['changed']=set()
        s['history']={}
        s['pending_edit_base']={}
        s['last_propagation_backup']=None
    movie,ex=_export(s,mode,blur,color,audio,bg,progress)
    issues=', '.join(f'{t:.2f}s' for t in stats.get('quality_times_to_review',[])[:12]) or 'không có'
    report=(f'✅ **v4.4 sử dụng {stats.get("anchors_used",0)} mốc cùng ID `{selected_id}`**. '
            f'Các frame mốc: {stats.get("anchors_frames",[])}; '
            f'khung hình hỗ trợ bằng Optical Flow gần mốc: **{stats.get("anchor_guided_frames",0)}**.\n\n'
            f'✅ **{stats.get("target_name")}** (chế độ: {stats.get("selection_mode")}; lớp đầu #{stats.get("target_class")}); '
            f'**{stats.get("frames")} frame**, **{stats.get("width")}×{stats.get("height")}**. '
            f'YOLO bỏ sót ban đầu: **{stats.get("missing_before_rescue")}**; '
            f'phục hồi hai chiều: **{stats.get("recovered")}**; '
            f'còn chưa nhận: **{stats.get("still_missing")}**.\n\n'
            f'Lớp YOLO sử dụng: **{stats.get("observed_classes",[])}**, đổi lớp: **{stats.get("class_switches",0)}** lần. '
            f'Mask nghi vấn: **{stats.get("quality_suspect_frames")}** khung; '
            f'Cảnh thiếu sáng: **{stats.get("dark_frames",0)}** khung; '
            f'AI tăng sáng được chọn: **{stats.get("enhanced_selected_frames",0)}** khung; '
            f'Mask gốc/tăng sáng khác nhau nhiều: **{stats.get("low_light_disagreements",0)}** khung. '
            f'có thể xem trước các mốc: **{issues}**. '
            'Đây là cảnh báo kỹ thuật, không phải đánh giá IoU với nhãn chuẩn.\n\n'
            '**Bước 3:** Kéo thanh thời gian đến đoạn bị lỗi, dùng cọ Thêm/Xóa rồi lan truyền trong phạm vi ngắn. '
            'Để xuất lại chỉ chạy ghép video, **không chạy YOLO lại**.')
    return (movie,movie,str(job/'diagnostics.csv'),report,
            _iframe(s,0,'local'),'Bước 3 đã sẵn sàng. Chọn thời điểm cần chỉnh.')


# Ước lượng sự dịch chuyển mask giữa hai frame bằng Optical Flow.
def _flow(source_mask,source_frame,target_frame):
    # Import from v3.9: optical flow target -> source with safe border clipping.
    from video_auto_yolo import _flow_warp
    return _flow_warp(source_mask,source_frame,target_frame,flow_side=400)


# Kiểm tra Optical Flow hai chiều để tránh lan truyền mask sai.
def _safe_warp(mask, frame_before, frame_after, *, min_consistency=.32):
    """Warp a manual correction one frame, with forward/backward consistency.

    Test both directions. When flow loses the drawn region (zoom, occlusion,
    flat texture), do not extrapolate it onto background or erase an object.
    """
    if np.count_nonzero(mask > .30) < 8:
        return None
    moved=np.clip(_flow(mask,frame_before,frame_after),0,1).astype(np.float32)
    if np.count_nonzero(moved > .30) < 8:
        return None
    back=np.clip(_flow(moved,frame_after,frame_before),0,1)
    a=mask > .40;b=back > .30
    i=np.count_nonzero(a & b)
    union=np.count_nonzero(a | b)
    consistency=i/max(1,union)
    size_ratio=np.count_nonzero(moved>.4)/max(1,np.count_nonzero(a))
    if consistency < min_consistency or not .55 <= size_ratio <= 1.85:
        return None
    return moved


# Lan truyền nét Thêm sang frame gần; nét Xóa chỉ lan truyền khi cho phép.
def propagate(video,state,seconds,radius,allow_erase=False):
    """Propagate only NEW manual strokes, never the full AI-vs-final difference.

    Default: positive additions only. Erasing is a destructive operation and
    propagates at most 2 frames when explicitly requested, with stricter flow.
    Previous outputs and explicitly edited frames are preserved. The previous
    propagation can be undone without rerunning YOLO.
    """
    s=_state(state,video)
    if not s.get('job'):raise gr.Error('Cần xử lý AI ở bước 2 trước.')
    i=_index(s,seconds);radius=max(1,min(12,int(radius)))
    with LOCK:
        base=s.setdefault('pending_edit_base',{}).get(i)
        if base is None:
            return ('Không có NÉT MỚI để lan truyền tại frame này. Hãy vẽ Thêm/Xóa '
                    'trước; nét cũ đã lan truyền sẽ không bị truyền lặp lại.')
        now=_effective(s,i,'local')
        if base.shape!=now.shape:
            raise gr.Error('Kích thước mask thay đổi. Hãy tải lại video.')
        # Binary support prevents repeated semi-transparent edge differences
        # from spreading as new additions/deletions.
        pos=np.where((now>127)&(base<=127),1.,0.).astype(np.float32)
        neg=np.where((now<=127)&(base>127),1.,0.).astype(np.float32)
        source_positive=int(np.count_nonzero(pos))
        source_negative=int(np.count_nonzero(neg))
        if source_positive+source_negative<10:
            s['pending_edit_base'].pop(i,None)
            return 'Không còn vùng thay đổi mới đáng kể để lan truyền.'
        if source_positive > .35*pos.size:
            return ('Vùng Thêm chiếm hơn 35% frame. Đã giữ nét tại frame đang sửa, '
                    'nhưng không lan truyền tự động vùng quá lớn để tránh dính nền. '
                    'Hãy tách thành các nét nhỏ hơn.')

        changed=[]; backup={}; halted=[]
        n=int(s['job']['frames'])
        explicit=set(s.get('changed',set()))
        for direction in (-1,1):
            add=pos.copy();remove=neg.copy();old=_frame(s,i)
            for d in range(1,radius+1):
                j=i+direction*d
                if not 0<=j<n:break
                new=_frame(s,j)
                # Both deltas are warped independently, so erase never expands
                # into a broad inverse foreground mask.
                if source_positive:
                    add=_safe_warp(add,old,new)
                    if add is None:
                        halted.append(j);break
                if bool(allow_erase) and source_negative and d<=2:
                    remove=_safe_warp(remove,old,new,min_consistency=.62)
                    if remove is None:
                        # Conservative: stop propagating erasure, but retain add.
                        remove=np.zeros_like(pos)
                else:
                    remove=np.zeros_like(pos)
                old=new
                if j in explicit:
                    break  # Stop at another hand-edited keyframe; never flow across it.
                current=_effective(s,j,'local').astype(np.float32)/255.
                proposal=current.copy()
                if source_positive:
                    # Add only near existing foreground to avoid leaking onto
                    # background. If AI mask is EMPTY, permit cautious nearby
                    # manual recovery rather than fabricating a full object.
                    if np.any(current>.30):
                        hh,ww=current.shape
                        r=max(5,min(35,int(min(hh,ww)*.085)))
                        near=cv2.dilate((current>.25).astype(np.uint8),
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1)))
                        add_new=add*near.astype(np.float32)
                    else:
                        add_new=add
                    dec=max(.35,1.-.65*d/(radius+1))
                    proposal=np.maximum(proposal, np.clip(add_new*dec,0,1))
                if bool(allow_erase) and source_negative and d<=2:
                    # Erase only at pixels already considered foreground here.
                    # Never enlarge the region removed beyond the drawn stroke.
                    erased=np.clip(remove,0,1)*.85
                    proposal=np.minimum(proposal, current*(1-erased))
                # Skip impossible, huge new expansions from optical-flow drift.
                extra=np.count_nonzero((proposal>.5)&(current<=.5))
                if extra > .13*current.size:
                    halted.append(j)
                    continue
                out=np.rint(np.clip(proposal,0,1)*255).astype(np.uint8)
                old_u8=np.rint(current*255).astype(np.uint8)
                if np.array_equal(out,old_u8):continue
                dst=_mask_path(s,j,True)
                dst.parent.mkdir(parents=True,exist_ok=True)
                if j not in backup:
                    backup[j]=_readmask(dst,out.shape) if dst.is_file() else None
                if not cv2.imwrite(str(dst),out):raise gr.Error('Không lưu được mask đã lan truyền.')
                changed.append(j)
        if changed:
            s['last_propagation_backup']=backup
        s['pending_edit_base'].pop(i,None)
        result=(f'✅ Đã lan truyền nét THÊM mới sang **{len(set(changed))} frame** '
                f'gần {i/s["info"]["fps"]:.2f}s. '
                f'Nét XÓA: {"chỉ tối đa ±2 frame, có kiểm tra" if allow_erase else "CHỈ áp dụng tại frame bạn vẽ"}. '
                'Mọi frame đã sửa tay khác đều được giữ nguyên.')
        if halted:
            result+=f' Bộ lọc dừng/bỏ qua {len(set(halted))} frame khi Optical Flow không chắc chắn.'
        if changed:
            result+=' Nếu thấy sai, nhấn **Hoàn tác lần lan truyền gần nhất**.'
        return result


# Lan truyền nét sửa và cập nhật lại khung xem trước.
def propagate_with_refresh(video,state,seconds,radius,allow_erase=False):
    msg=propagate(video,state,seconds,radius,allow_erase)
    s=_state(state,video)
    return msg,_iframe(s,_index(s,seconds),'local')


# Khôi phục mask trước lần lan truyền gần nhất mà không chạy lại YOLO.
def undo_propagation(video,state):
    s=_state(state,video)
    with LOCK:
        backup=s.get('last_propagation_backup')
        if not backup:return 'Không có lượt lan truyền nào để hoàn tác.'
        restored=0
        for i,mask in backup.items():
            if i in s.get('changed',set()):continue
            path=_mask_path(s,i,True)
            if mask is None:
                path.unlink(missing_ok=True)
            else:
                cv2.imwrite(str(path),mask)
            restored+=1
        s['last_propagation_backup']=None
    return f'✅ Đã hoàn tác lần lan truyền gần nhất trên {restored} frame. Nét bạn vẽ tay vẫn được giữ nguyên. Nhấn Xuất lại video để xem kết quả.'

# Xuất lại video sau chỉnh sửa mask mà không suy luận YOLO lần nữa.
def rerender(video,state,mode,blur,color,bg,audio,progress=gr.Progress(track_tqdm=False)):
    s=_state(state,video)
    if not s.get('job'):raise gr.Error('Chạy AI ở Bước 2 trước khi xuất lại.')
    out,stats=_export(s,mode,blur,color,audio,bg,progress)
    return out,out,f'✅ Đã xuất MP4 mới **{stats.get("codec")}**, **{stats.get("frames")} frame**. Không chạy lại YOLO.'


# Giao diện Gradio của ba bước: tạo mốc, chạy YOLO và sửa kết quả.
# Tạo giao diện Video 3 bước và nối các nút với hàm xử lý.
def add_guided_v40_tab():
    # V5.2: chi thay UI, khong thay ham xu ly va mask keyframe.
    with gr.Column(visible=False,elem_classes=['g3-page'],elem_id='g3-video-page') as video_page:
        gr.HTML(intro('Video',
            'Chọn đúng đối tượng ở một vài khung hình, để AI tách toàn clip, rồi sửa khi cần.'))
        state = gr.State(None)

        with gr.Accordion('Bước 1 — Tải video và đánh dấu đối tượng', open=True,
                          elem_classes=['aetm-accordion'],elem_id='aetm-video-step1'):
            with gr.Group(elem_classes=['g3-card']):
                video = gr.File(label='Tải video MP4 · tối đa 30 giây',
                                file_types=['.mp4'],type='filepath',height=116,
                                elem_classes=['aetm-file-compact','g3-video-input'])
                seed_status = gr.Markdown('**Chưa tải video.** Chọn MP4 để bắt đầu.')
            gr.HTML(note('Khung ở dưới chính là hình tại thời điểm bạn chọn; không cần kéo lên để đối chiếu video gốc.'))
            with gr.Row(equal_height=False,elem_id='g3-video-layout'):
                with gr.Column(scale=7,min_width=0,elem_classes=['g3-video-main']):
                    seed_seek = gr.Slider(0,1,value=0,step=.04,
                        label='Thời điểm đang chỉnh · kéo tới giây cần tạo mốc',elem_id='aetm-video-timeline')
                    editor = gr.HTML('<em>Chọn video rồi tạo mốc để hiển thị khung hình.</em>',
                                     elem_classes=['aetm-video-editor'],elem_id='aetm-video-canvas')
                with gr.Column(scale=4,min_width=0,elem_classes=['g3-video-aside']):
                    gr.HTML('<div class="aetm-small-title">Mốc tham chiếu · tối đa 5</div>')
                    anchor_name = gr.Textbox(value='',
                        placeholder='VD: Mèo nhìn thẳng / Mèo phóng gần',label='Tên mốc (dễ nhớ)')
                    object_id = gr.Textbox(value='doi_tuong_1',
                        label='ID đối tượng · 5 mốc của cùng đối tượng dùng cùng ID')
                    add_btn = gr.Button('＋  Tạo mốc tại giây đang chọn',variant='primary',
                                        elem_classes=['aetm-action'])
                    anchor_picker = gr.Dropdown(choices=[],value=None,interactive=True,
                                              label='Mốc đã lưu · chọn để sửa tiếp')
                    with gr.Row():
                        primary_btn = gr.Button('★ Mốc chính',elem_classes=['aetm-secondary'])
                        rename_btn = gr.Button('Lưu tên/ID',elem_classes=['aetm-secondary'])
                        delete_btn = gr.Button('Xóa mốc',elem_classes=['aetm-secondary'])
                    anchor_summary = gr.Markdown('Chưa có mốc tham chiếu.')
                    refresh_btn = gr.Button('↻ Làm mới danh sách',elem_classes=['aetm-secondary'])
                    with gr.Accordion('Lưu hoặc nhập lại bộ 5 mốc',open=False,elem_classes=['aetm-accordion']):
                        download_btn = gr.Button('↓ Xuất bộ mốc ZIP',variant='secondary')
                        anchor_file = gr.File(label='ZIP bộ mốc đã xuất')
                        anchor_file_status = gr.Markdown('')
                        anchor_import = gr.File(label='Chọn ZIP mốc cũ',file_types=['.zip'],type='filepath')
                        import_btn = gr.Button('↥ Khôi phục mốc',variant='secondary')
            gr.HTML(note('Ngay trong khung chỉnh: Bao = vùng giữ ban đầu, Thêm (+) = lấy phần thiếu, Xóa (−) = bỏ phần nền thừa. Mỗi mốc được lưu riêng.'))
            go_step2 = gr.Button('Tiếp tục → Bước 2: AI tách nền',variant='secondary',
                                 elem_classes=['aetm-action'])

        with gr.Accordion('Bước 2 — AI tách nền và xem kết quả',open=False,
                          elem_classes=['aetm-accordion'],elem_id='aetm-video-step2') as stage2:
            gr.HTML(note('Tạo ít nhất một mốc trước khi chạy. Đối tượng có thể phóng to hoặc thay đổi góc nhìn; chọn “Tất cả loại” để AI bám theo vùng bạn đã đánh dấu.'))
            with gr.Row():
                target = gr.Dropdown(
                    choices=[('Tất cả loại · ưu tiên đối tượng đã bao','all'),('Tự nhận lớp rồi khóa lớp','auto'),
                             ('Người','person'),('Mèo','cat'),('Chó','dog'),('Ô tô','car'),
                             ('Xe máy','motorcycle'),('Xe đạp','bicycle'),('Chim','bird'),('Ngựa','horse'),
                             ('Túi xách','handbag'),('Ba lô','backpack'),('Ghế','chair')],
                    value='all',label='Loại đối tượng',scale=3)
                side = gr.Dropdown([480,640,720],value=640,label='Độ phân giải video',scale=1)
                imgsz = gr.Dropdown([416,640,960],value=640,label='Kích thước AI',scale=1)
            mode = gr.Radio(['Làm mờ nền','Video mask','Nền màu','Ảnh nền khác'],
                            value='Làm mờ nền',label='Chọn kiểu kết quả')
            with gr.Row():
                with gr.Column(visible=True) as blur_group:
                    blur = gr.Slider(2,45,19,step=1,label='Mức mờ nền')
                with gr.Column(visible=False) as color_group:
                    color = gr.ColorPicker(value='#DDEBFF',label='Màu nền mới')
                with gr.Column(visible=False) as bg_group:
                    bg = gr.Image(label='Chọn ảnh nền thay thế',type='numpy',height=160)
            with gr.Accordion('Cài đặt AI nâng cao (có thể bỏ qua)',open=False,elem_classes=['aetm-accordion']):
                confidence = gr.Slider(.08,.5,.15,step=.01,label='Ngưỡng tin cậy YOLO')
                with gr.Row():
                    edges = gr.Checkbox(value=True,label='Tinh chỉnh viền')
                    rescue = gr.Checkbox(value=True,label='Phục hồi hai chiều')
                    stabilize = gr.Checkbox(value=True,label='Ổn định mask')
                    quality = gr.Checkbox(value=True,label='Kiểm tra mask nghi vấn')
                    low_light = gr.Checkbox(value=False,label='Hỗ trợ video tối')
                gr.HTML(note('Tăng sáng chỉ áp dụng cho ảnh đầu vào của AI, không đổi màu sắc video xuất.'))
            audio = gr.Checkbox(value=False,label='Giữ âm thanh gốc khi xuất (cần FFmpeg)')
            btn_ai = gr.Button('✦ CHẠY AI TÁCH NỀN',variant='primary',elem_classes=['aetm-action'])
            gr.HTML('<div class="aetm-section-sep" id="aetm-video-result">Kết quả tách nền lần đầu</div>')
            with gr.Row(equal_height=False):
                with gr.Column(scale=7,min_width=320):
                    movie = gr.Video(label='Xem lại video AI đã tách',height=340,format='mp4')
                with gr.Column(scale=3,min_width=240):
                    result_info = gr.Markdown('**Chưa chạy AI.**')
                    download = gr.File(label='Tải MP4 lần đầu')
                    csvfile = gr.File(label='CSV thống kê mask / frame')
            go_step3 = gr.Button('Tiếp tục → Bước 3: Sửa đoạn lỗi',variant='secondary',
                                 elem_classes=['aetm-secondary'])

        with gr.Accordion('Bước 3 — Chỉnh phần còn thiếu và xuất video cuối',open=False,
                          elem_classes=['aetm-accordion'],elem_id='aetm-video-step3') as stage3:
            gr.HTML(note('Chọn đúng giây bị lỗi ở thanh ngay trên khung chỉnh. Vùng Thêm chỉ lan truyền ở phạm vi bạn cho phép; Xóa mặc định chỉ tác động một frame.'))
            with gr.Row(equal_height=False):
                with gr.Column(scale=7,min_width=340):
                    local_seek = gr.Slider(0,1,value=0,step=.04,
                        label='Giây đang sửa · xem trực tiếp hình bên dưới',elem_id='aetm-local-timeline')
                    local_editor = gr.HTML('<em>Chạy AI ở Bước 2 để hiện ảnh cần sửa.</em>',
                            elem_classes=['aetm-local-editor'],elem_id='aetm-local-canvas')
                with gr.Column(scale=4,min_width=250):
                    local_hint = gr.Markdown('Chưa có kết quả AI để chỉnh.')
                    radius = gr.Slider(1,12,4,step=1,label='Lan truyền vùng Thêm ± số frame')
                    propagate_btn = gr.Button('↔ Lan truyền nét Thêm (an toàn)',variant='secondary',
                                            elem_classes=['aetm-action'])
                    undo_prop_btn = gr.Button('↶ Hoàn tác lan truyền gần nhất',
                                              elem_classes=['aetm-secondary'])
                    with gr.Accordion('Nâng cao: lan truyền cả nét Xóa',open=False,elem_classes=['aetm-accordion']):
                        allow_erase = gr.Checkbox(value=False,
                            label='Cho phép lan truyền nét Xóa (tối đa ±2 frame)')
                    propagate_status = gr.Markdown('')
                    rerender_btn = gr.Button('↓ Xuất video đã hoàn thiện',variant='primary',
                                             elem_classes=['aetm-action'])
            gr.HTML('<div class="aetm-section-sep" id="aetm-video-final">Video cuối cùng</div>')
            with gr.Row(equal_height=False):
                with gr.Column(scale=7,min_width=320):
                    final_video = gr.Video(label='Xem lại kết quả hoàn thiện',height=300,format='mp4')
                with gr.Column(scale=3,min_width=240):
                    final_file = gr.File(label='Tải MP4 hoàn thiện')
                    final_status = gr.Markdown('')

        # Hiện đúng ô tùy chọn theo chế độ nền video được chọn.
        def _video_output_controls(selected_mode):
            return (gr.update(visible=selected_mode == 'Làm mờ nền'),
                    gr.update(visible=selected_mode == 'Nền màu'),
                    gr.update(visible=selected_mode == 'Ảnh nền khác'))
        mode.change(_video_output_controls, inputs=[mode], outputs=[blur_group,color_group,bg_group],
                    queue=False, show_progress='hidden')
        video.change(upload,inputs=[video],outputs=[state,editor,seed_seek,local_seek,seed_status,
                                                   movie,download,csvfile,result_info,anchor_picker,anchor_summary])
        seed_seek.release(seek_seed,inputs=[video,seed_seek,state],
                          outputs=[editor,seed_status,anchor_picker,anchor_name],queue=False)
        add_btn.click(add_anchor,inputs=[video,seed_seek,state,anchor_name,object_id],
                      outputs=[anchor_picker,editor,seed_status,anchor_summary],concurrency_limit=1)
        anchor_picker.input(pick_anchor,inputs=[video,state,anchor_picker],
                            outputs=[seed_seek,editor,seed_status,anchor_name,object_id],queue=False)
        refresh_btn.click(refresh_anchors,inputs=[video,state,anchor_picker],
                          outputs=[anchor_picker,anchor_summary],queue=False)
        primary_btn.click(set_primary,inputs=[video,state,anchor_picker],
                          outputs=[anchor_picker,anchor_summary,seed_status],concurrency_limit=1)
        rename_btn.click(rename_anchor,inputs=[video,state,anchor_picker,anchor_name,object_id],
                         outputs=[anchor_picker,anchor_summary,seed_status],concurrency_limit=1)
        delete_btn.click(delete_anchor,inputs=[video,state,anchor_picker],
                         outputs=[anchor_picker,seed_seek,editor,seed_status,anchor_summary],concurrency_limit=1)
        download_btn.click(download_anchors,inputs=[video,state],
                           outputs=[anchor_file,anchor_file_status],concurrency_limit=1)
        import_btn.click(import_anchors,inputs=[video,state,anchor_import],
                         outputs=[anchor_picker,seed_seek,editor,seed_status,anchor_summary],concurrency_limit=1)
        ai_event = btn_ai.click(run_ai,inputs=[video,state,target,side,imgsz,confidence,edges,rescue,stabilize,quality,low_light,
                                    mode,blur,color,bg,audio],
                     outputs=[movie,download,csvfile,result_info,local_editor,local_hint],concurrency_limit=1)
        ai_event.then(lambda: gr.update(open=True),outputs=[stage3],queue=False,
            js="() => { setTimeout(() => document.getElementById('aetm-video-result')?.scrollIntoView({behavior:'smooth',block:'start'}), 240); }")
        local_seek.release(seek_local,inputs=[video,local_seek,state],outputs=[local_editor,local_hint],queue=False)
        propagate_btn.click(propagate_with_refresh,inputs=[video,state,local_seek,radius,allow_erase],outputs=[propagate_status,local_editor],concurrency_limit=1)
        undo_prop_btn.click(undo_propagation,inputs=[video,state],outputs=[propagate_status],concurrency_limit=1)
        render_event = rerender_btn.click(rerender,inputs=[video,state,mode,blur,color,bg,audio],
                           outputs=[final_video,final_file,final_status],concurrency_limit=1)
        render_event.then(fn=None,inputs=[],outputs=[],queue=False,
            js="() => { setTimeout(() => document.getElementById('aetm-video-final')?.scrollIntoView({behavior:'smooth',block:'start'}), 200); }")

        # Dieu huong theo buoc, giu lai trang thai/video/mask trong moi buoc.
        go2_evt = go_step2.click(lambda: gr.update(open=True),outputs=[stage2],queue=False)
        go2_evt.then(fn=None,inputs=[],outputs=[],queue=False,
            js="() => { setTimeout(() => document.getElementById('aetm-video-step2')?.scrollIntoView({behavior:'smooth',block:'start'}), 140); }")
        go3_evt = go_step3.click(lambda: gr.update(open=True),outputs=[stage3],queue=False)
        go3_evt.then(fn=None,inputs=[],outputs=[],queue=False,
            js="() => { setTimeout(() => document.getElementById('aetm-video-step3')?.scrollIntoView({behavior:'smooth',block:'start'}), 140); }")
        # Sau khi hoan thanh AI: mo buoc sua va dua nguoi dung toi ket qua xem truoc.
        return video_page
