"""3G · Giao diện đỏ responsive cho phiên bản AI thật (không phải DEMO).
Các hàm intro/step/help_box giữ API tương thích những tab phiên bản cũ.
"""
from __future__ import annotations
from html import escape

CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root { --g3-red:#b32630; --g3-red-dark:#941b27; --g3-red-soft:#fff1f1;
        --g3-bg:#f3f4f7; --g3-white:#fff; --g3-ink:#18253b;
        --g3-muted:#68778c; --g3-line:#dde3eb; --g3-shadow:0 4px 18px rgba(22,30,48,.045); }
html,body,gradio-app {max-width:none!important;overflow-x:hidden!important;background:var(--g3-bg)!important;}
.gradio-container,.gradio-container .main {width:100%!important;max-width:none!important;min-width:0!important;box-sizing:border-box!important;}
.gradio-container {margin:0!important;padding:16px clamp(10px,2.3vw,38px) 28px!important;background:var(--g3-bg)!important;}
.gradio-container,.gradio-container * {font-family:'Be Vietnam Pro','Segoe UI',Arial,sans-serif!important;}
.gradio-container .gradio-row,.gradio-container .gradio-column,.gradio-container .gradio-group {min-width:0!important;box-sizing:border-box!important;}
/* Header: toàn bộ mục điều khiển nằm trên một hàng */
.g3-header {background:white!important;border:1px solid var(--g3-line)!important;border-radius:17px!important;
  padding:8px 14px!important;gap:12px!important;align-items:center!important;position:sticky!important;top:0!important;
  z-index:40!important;box-shadow:var(--g3-shadow)!important;margin-bottom:10px!important;}
.g3-header > * {min-width:0!important;}
.g3-logo {display:flex;align-items:center;gap:10px;min-height:45px;font-size:22px;font-weight:800;color:var(--g3-ink);}
.g3-logo-mark {width:38px;height:38px;border-radius:11px;background:var(--g3-red);color:white;display:grid;place-items:center;font-size:23px;}
#g3-nav {max-width:370px!important;width:100%!important;margin:auto!important;padding:4px!important;border-radius:999px!important;border:0!important;background:#e9edf3!important;}
#g3-nav .wrap {padding:0!important;border:0!important;}
#g3-nav [role=radiogroup] {display:flex!important;align-items:stretch!important;justify-content:center!important;flex-wrap:nowrap!important;gap:3px!important;}
#g3-nav [role=radiogroup] label {flex:1 1 0!important;display:flex!important;align-items:center!important;justify-content:center!important;
 min-width:0!important;min-height:38px!important;margin:0!important;padding:5px 8px!important;border:0!important;border-radius:999px!important;
 color:#3b495d!important;background:transparent!important;font-size:13px!important;font-weight:750!important;white-space:nowrap!important;}
#g3-nav [role=radiogroup] label:has(input:checked) {background:var(--g3-red)!important;color:white!important;box-shadow:0 3px 8px rgba(179,38,48,.16)!important;}
#g3-nav [role=radiogroup] input {position:absolute!important;opacity:0!important;width:1px!important;height:1px!important;}
#g3-help-btn button,#g3-help-btn {background:transparent!important;border:0!important;box-shadow:none!important;color:var(--g3-red)!important;font-weight:800!important;font-size:13px!important;min-height:38px!important;}
#g3-help-panel {position:fixed!important;right:clamp(10px,2.3vw,38px)!important;top:80px!important;width:min(820px,calc(100vw - 24px))!important;
 max-height:calc(100vh - 91px)!important;overflow:auto!important;z-index:600!important;padding:17px!important;border:1px solid var(--g3-line)!important;
 border-radius:17px!important;background:white!important;box-shadow:0 18px 65px rgba(19,30,48,.3)!important;}
.g3-guide-content {max-height:69vh;overflow:auto!important;padding-right:8px;}
/* Bố cục nội dung: không cố định 1160px và không cho cột chồng nhau. */
.g3-page {width:100%!important;min-width:0!important;max-width:100%!important;}
.g3-intro,.aetm-tab-intro {display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px 20px;margin:14px 0 16px!important;padding:0 2px!important;}
.g3-intro h1,.aetm-tab-intro h2 {font-size:clamp(22px,2.1vw,30px)!important;line-height:1.25!important;margin:0 0 3px!important;color:var(--g3-ink)!important;letter-spacing:-.035em;font-weight:800!important;}
.g3-intro p,.aetm-tab-intro p {font-size:12px!important;line-height:1.65!important;color:var(--g3-muted)!important;margin:0!important;}
.g3-steps,.aetm-mini-steps {display:flex!important;align-items:center;flex-wrap:wrap;gap:8px;}
.g3-step,.aetm-mini-steps span {display:flex;align-items:center;gap:6px;font-size:11px!important;color:#718099;font-weight:750;}
.g3-step strong {width:28px;height:28px;display:grid;place-items:center;border-radius:50%;background:#e2e7ef;color:#546478;}
.g3-step.active {color:var(--g3-ink);} .g3-step.active strong {background:var(--g3-red);color:white;}
.g3-separator {height:2px;width:18px;background:var(--g3-line);}
.g3-card,.aetm-card,.aetm-highlight,.aetm-recording-status {min-width:0!important;max-width:100%!important;box-sizing:border-box!important;
 background:white!important;border:1px solid var(--g3-line)!important;border-radius:16px!important;padding:14px!important;box-shadow:var(--g3-shadow)!important;}
.g3-card-title,.aetm-small-title,.aetm-section-sep {font-size:15px!important;font-weight:800!important;color:var(--g3-ink)!important;margin:2px 0 10px!important;scroll-margin-top:80px!important;}
.g3-label{font-size:11px;font-weight:800;letter-spacing:.05em;text-transform:uppercase;color:#65738a;margin:4px 0 9px;}
.aetm-stephead{display:flex;gap:11px;align-items:center;padding:8px 0;margin:8px 0;}
.aetm-stepno{display:inline-flex;align-items:center;justify-content:center;min-width:68px;height:30px;color:var(--g3-red);
 background:var(--g3-red-soft);font-weight:800;font-size:12px;border-radius:8px;}
.aetm-steptitle{font-size:15px!important;font-weight:790!important;color:var(--g3-ink);margin:0 0 2px;}
.aetm-stepdesc{font-size:11px!important;color:var(--g3-muted)!important;margin:0;line-height:1.5;}
.aetm-help,.aetm-note{border:1px solid #f5d6d7!important;background:#fff7f7!important;color:#83424a!important;border-radius:9px!important;
 font-size:11px!important;line-height:1.6!important;padding:8px 10px!important;margin:5px 0!important;}
.aetm-note{border-left:3px solid var(--g3-red)!important;}
.aetm-accordion{margin:6px 0 12px!important;border-radius:13px!important;max-width:100%!important;min-width:0!important;scroll-margin-top:80px!important;}
.aetm-accordion>.label-wrap{font-weight:750!important;}
.gradio-container button {border-radius:10px!important;font-size:12px!important;font-weight:750!important;}
.gradio-container button.primary,.g3-primary button,.aetm-action button.primary,.aetm-record button.primary{
 background:var(--g3-red)!important;color:white!important;border:1px solid var(--g3-red)!important;min-height:43px!important;}
.gradio-container button.primary:hover,.g3-primary button:hover {background:var(--g3-red-dark)!important;}
.gradio-container input[type=checkbox],.gradio-container input[type=radio]{accent-color:var(--g3-red)!important;}
.g3-video-input,.g3-video-input *{max-width:100%!important;min-width:0!important;box-sizing:border-box!important;}
.g3-video-input{overflow:hidden!important;}
.g3-video-input [data-testid='file-upload'],.g3-video-input .file-drop{min-height:92px!important;max-height:125px!important;}
.aetm-file-compact{max-width:100%!important;min-width:0!important;overflow:hidden!important;}
.aetm-file-compact .file-preview{max-height:85px!important;}
#g3-video-layout{display:grid!important;grid-template-columns:minmax(0,1.85fr) minmax(285px,1fr)!important;gap:18px!important;
 width:100%!important;align-items:start!important;min-width:0!important;}
#g3-video-layout>.column,#g3-video-layout>div {width:100%!important;max-width:100%!important;min-width:0!important;flex:initial!important;}
#g3-video-layout .g3-video-aside{background:#fafbfc!important;border:1px solid var(--g3-line)!important;border-radius:13px!important;padding:12px!important;}
.g3-two-up{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:17px!important;width:100%!important;}
.g3-two-up>.column,.g3-two-up>div{max-width:100%!important;min-width:0!important;width:100%!important;flex:initial!important;}
.gradio-container .wrap,.gradio-container .form,.gradio-container .file-preview{min-width:0!important;max-width:100%!important;}
.g3-preview img,.g3-preview video{max-width:100%!important;object-fit:contain!important;}
.g3-preview{border-radius:11px!important;min-width:0!important;max-width:100%!important;overflow:hidden!important;}
.aetm-video-editor iframe,.aetm-local-editor iframe{height:min(68vh,670px)!important;min-height:465px!important;width:100%!important;max-width:100%!important;border-radius:11px!important;}
.aetm-footer {font-size:11px!important;color:#79869b!important;margin-top:16px;padding:12px 0;border-top:1px solid var(--g3-line);}
footer,#footer,[data-testid='footer'] {display:none!important;}
@media(max-width:960px){#g3-video-layout,.g3-two-up{grid-template-columns:minmax(0,1fr)!important;gap:10px!important;}
 .g3-header{gap:7px!important;padding:7px!important;}.g3-logo{font-size:18px!important;}}
@media(max-width:590px){.gradio-container{padding:7px 9px 25px!important;}
 .g3-header{position:relative!important;flex-wrap:wrap!important;}.g3-logo{min-height:36px!important;}
 #g3-nav{max-width:none!important;}.g3-card{padding:10px!important;}
 .aetm-video-editor iframe,.aetm-local-editor iframe{height:520px!important;}}
"""

def intro(title: str, description: str) -> str:
    return (f'<div class="g3-intro"><div><h1>{escape(title)}</h1>'
            f'<p>{escape(description)}</p></div></div>')

def step(number: int, title: str, description: str) -> str:
    return (f'<div class="aetm-stephead"><span class="aetm-stepno">Bước {int(number)}</span><div>'
            f'<div class="aetm-steptitle">{escape(title)}</div>'
            f'<p class="aetm-stepdesc">{escape(description)}</p></div></div>')

def note(text: str) -> str:
    return f'<div class="aetm-note">{escape(text)}</div>'

def help_box(text: str) -> str:
    return f'<div class="aetm-help">💡 {escape(text)}</div>'

def flow(steps: list[str]) -> str:
    spans=[]
    for idx, item in enumerate(steps, 1):
        spans.append(f'<span class="g3-step {"active" if idx==1 else ""}"><strong>{idx}</strong>{escape(item)}</span>')
        if idx < len(steps):spans.append('<i class="g3-separator"></i>')
    return '<div class="g3-steps">'+''.join(spans)+'</div>'


GUIDE = r"""# HƯỚNG DẪN SỬ DỤNG 3G

**3G** là ứng dụng tách nền ảnh, video và Webcam bằng trí tuệ nhân tạo. Người dùng có thể xử lý tự động, chỉnh sửa vùng đối tượng và xuất kết quả dưới dạng PNG hoặc MP4.

---

## 1. MODULE ẢNH

### Bước 1 — Tải ảnh và tách nền tự động

1. Tải ảnh định dạng **JPG hoặc PNG**.
2. Bật **Tinh chỉnh viền** nếu cần cải thiện đường viền đối tượng.
3. Nhấn **Tách nền tự động**.
4. Xem ảnh đã xử lý và mask kết quả.

Hệ thống sử dụng mô hình **U2Net-P** kết hợp các kỹ thuật xử lý mask như Guided Filter.

### Bước 2 — Kiểm tra và chỉnh sửa mask

Mask là vùng phân tách đối tượng khỏi hậu cảnh:

- **Màu trắng:** Phần đối tượng cần giữ.
- **Màu đen:** Phần nền cần loại bỏ.
- **Vùng xám:** Phần viền bán trong suốt.

Nếu kết quả chính xác, nhấn **Mask đạt**.

Nếu còn thiếu tóc, tay, áo hoặc dính nền:

1. Chọn **Cần chỉnh**.
2. Dùng cọ **Thêm / Giữ lại** để phục hồi vùng bị mất.
3. Dùng cọ **Xóa** để loại bỏ phần nền thừa.
4. Điều chỉnh cỡ cọ, phóng to hoặc thu nhỏ để chỉnh chính xác.
5. Có thể **Hoàn tác** hoặc khôi phục mask AI.
6. Nhấn **Duyệt mask đã chỉnh** để áp dụng thay đổi.

### Bước 3 — Xuất ảnh PNG

Chọn một trong bốn chế độ:

- **PNG trong suốt:** Loại bỏ hậu cảnh.
- **Nền màu:** Thay hậu cảnh bằng màu tùy chọn.
- **Ảnh nền khác:** Ghép đối tượng vào ảnh nền mới.
- **Làm mờ nền:** Giữ đối tượng rõ và làm mờ hậu cảnh.

Nhấn **Xuất ảnh PNG** và tải kết quả về máy.

---

## 2. MODULE VIDEO

### Bước 1 — Chọn đối tượng và tạo mốc tham chiếu

1. Tải video **MP4 tối đa 30 giây**.
2. Kéo thanh thời gian đến khung hình nhìn rõ đối tượng.
3. Nhấn **Tạo mốc tại giây này**.
4. Sử dụng các công cụ:

   - **Bao:** Vẽ đường khép kín quanh đối tượng cần giữ.
   - **Thêm (+):** Bổ sung vùng đối tượng bị thiếu.
   - **Xóa (−):** Loại bỏ vùng nền được chọn nhầm.

5. Đặt tên mốc, ví dụ `Mèo nhìn thẳng`.
6. Nhập ID đối tượng, ví dụ `meo_1`.
7. Kiểm tra mask rồi tiếp tục tạo các mốc khác nếu cần.

Hệ thống hỗ trợ **tối đa 5 mốc tham chiếu**.

**Lưu ý:** Nếu 5 mốc đều thuộc cùng một đối tượng thì phải sử dụng chung ID. Ví dụ, mèo ở xa, mèo phóng gần và mèo quay nghiêng đều dùng `meo_1`.

Nên đánh dấu thêm tại các thời điểm đối tượng thay đổi góc nhìn, phóng to hoặc thiếu sáng.

Có thể chọn **Mốc chính**, sửa lại từng mốc hoặc xuất/nhập bộ mốc ZIP.

### Bước 2 — AI tự động tách nền toàn video

1. Chọn **Loại đối tượng cần giữ**.
2. Chọn độ phân giải **480, 640 hoặc 720px**.
3. Chọn kiểu kết quả:
   - Làm mờ nền.
   - Video mask.
   - Nền màu.
   - Ảnh nền khác.
4. Điều chỉnh thông số AI nếu cần.
5. Nhấn **Chạy AI tách nền toàn video**.

Hệ thống sử dụng **YOLO**, các mốc tham chiếu và các kỹ thuật xử lý video như:

- Optical Flow để hỗ trợ theo dõi.
- Phục hồi mask hai chiều.
- Ổn định mask giữa các khung hình.
- Kiểm tra mask nghi vấn.
- Hỗ trợ xử lý cảnh thiếu sáng.

Sau khi hoàn thành, người dùng có thể xem video kết quả, tải MP4 và CSV thống kê.

### Bước 3 — Chỉnh sửa và hoàn thiện video

Nếu video còn mất một phần đối tượng hoặc dính nền:

1. Kéo thanh thời gian đến khung hình bị lỗi.
2. Dùng cọ **Thêm** để phục hồi vùng thiếu.
3. Dùng cọ **Xóa** để loại bỏ vùng thừa.
4. Có thể lan truyền vùng Thêm sang các khung hình gần đó.
5. Nên thử phạm vi **±4 frame** trước.
6. Nếu kết quả lan truyền không phù hợp, nhấn **Hoàn tác lan truyền**.

**Lưu ý:** Vùng Xóa mặc định chỉ áp dụng cho khung hình đang chỉnh để hạn chế xóa nhầm đối tượng ở các khung khác.

Cuối cùng, nhấn **Xuất video hoàn thiện** để tạo MP4 mới mà không cần chạy lại YOLO.

---

## 3. MODULE WEBCAM

### Bước 1 — Bật Webcam

1. Chọn hiệu ứng nền:
   - Làm mờ nền.
   - Nền màu.
   - Ảnh nền khác.
   - Xem mask.
   - Ảnh gốc.
2. Chọn độ phân giải.
3. Chọn tần suất xử lý AI.
4. Bật Camera và cấp quyền cho trình duyệt.

**Cấu hình khuyến nghị:** 480px, AI mỗi 2 khung hình.

### Bước 2 — Tách nền trực tiếp

- **Khung bên trái:** Camera gốc.
- **Khung bên phải:** Kết quả AI sau tách nền.

Các tùy chọn hỗ trợ:

- **Guided Filter:** Cải thiện đường viền mask.
- **Ổn định mask:** Giảm hiện tượng nhấp nháy.
- **Hỗ trợ cảnh tối:** Cải thiện đầu vào AI khi thiếu sáng.
- **Giảm quầng viền:** Hạn chế viền mờ quanh đối tượng.
- **AI cập nhật khi chuyển động nhanh:** Giảm hiện tượng mask bám chậm.

Nếu đối tượng di chuyển nhanh, thử chọn **AI mỗi khung**.

FPS xử lý AI có thể khác FPS của video xuất ra.

### Bước 3 — Chụp ảnh hoặc ghi video

- **Bắt đầu ghi MP4:** Bắt đầu lưu các khung hình đã xử lý.
- **Dừng ghi & xem lại:** Kết thúc và xem video MP4 H.264.
- **Chụp PNG:** Lưu khung hình Webcam hiện tại.
- **Reset mask:** Khởi tạo lại trạng thái xử lý mask.

**Quan trọng:** Nút Record/Stop nằm trong khung Camera chỉ dùng để bật/tắt nguồn Webcam, không phải nút ghi MP4.

Video Webcam hiện chưa thu âm thanh microphone.

---

## 4. CÁC VẤN ĐỀ THƯỜNG GẶP

| Vấn đề | Cách khắc phục |
|---|---|
| Ảnh mất tóc, tay hoặc viền áo | Dùng cọ Thêm để phục hồi phần thiếu |
| Ảnh còn dính nền | Dùng cọ Xóa để loại bỏ |
| Video bị mất đối tượng | Tạo thêm mốc ở đoạn phóng to, đổi góc hoặc thiếu sáng |
| Video xử lý chậm | Giảm độ phân giải hoặc kích thước AI |
| Video MP4 không phát | Kiểm tra codec H.264 và FFmpeg |
| Webcam không hiển thị | Kiểm tra quyền truy cập camera trong Chrome |
| Webcam bị bóng mờ | Tăng tần suất chạy AI hoặc điều chỉnh ổn định mask |
| Không phân biệt nút Stop và ghi hình | Dùng các nút Bắt đầu ghi MP4 / Dừng ghi & xem lại |

---

## 5. THƯ MỤC DỮ LIỆU

- `models/`: Lưu mô hình U2Net-P.
- `outputs/`: Lưu ảnh PNG, video MP4, mask và CSV.
- `samples/`: Chứa ảnh và video thử nghiệm.
- `.venv/`: Môi trường Python chính.
- `.venv_yolo/`: Môi trường Python xử lý YOLO.

---

**Lưu ý chung:** Chất lượng tách nền phụ thuộc vào ánh sáng, độ phân giải, chuyển động, hậu cảnh và khả năng nhận diện của AI. Những vùng khó có thể cần chỉnh sửa thủ công để đạt kết quả tốt hơn.
"""
