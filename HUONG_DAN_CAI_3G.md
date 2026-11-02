# 3G — bản chính có AI thật, giao diện V3 màu đỏ

**Đây KHÔNG PHẢI DEMO.** Bộ code giữ chức năng tách nền ảnh (U2Net-P), video YOLO 5 mốc và chỉnh mask, webcam trực tiếp + ghi MP4 từ ứng dụng hiện tại. Bố cục V3 responsive và màu đỏ, trên header chỉ ghi **3G**.

## Cài lên dự án đang hoạt động

1. **Dừng ứng dụng ở cổng 7866** và sao lưu toàn bộ thư mục `D:\TGMT` (hoặc ít nhất toàn bộ `.py` và `.bat`) sang chỗ khác.
2. Giải nén gói ZIP này, **chép đè các file Python** vào `D:\TGMT`. Chép thêm `start_3G.bat` và `stop_3G.bat`.
3. **Giữ nguyên** `D:\TGMT\.venv`, `D:\TGMT\.venv_yolo`, `D:\TGMT\models`, các trọng số `.onnx` / `.pt`, `outputs` và `samples`. Không cần chạy lại lệnh pip.
4. Nhấp đúp `start_3G.bat` hoặc gõ ` .\start_3G.bat` trong PowerShell ở `D:\TGMT`.
5. Mở `http://127.0.0.1:7866` và nhấn `Ctrl+F5`. Nếu Chrome còn tên cũ, **kiểm tra đã dừng tiến trình cũ**.
6. Thử đủ ba tab (Ảnh → Video → Webcam), sau đó mới thay shortcut ngoài Desktop nếu muốn.

**Không chép vào D:\Test**: đó là thư mục của bản demo, không có model và môi trường YOLO thật.

## Các file quan trọng

- `app.py`: FastAPI + Gradio + tab Ảnh + Canvas cọ vẽ thật.
- `ui_3g.py`: giao diện 3G màu đỏ + Hướng dẫn trên header.
- `tab_video_v40.py`: giao diện Video 3 bước + 5 mốc + canvas.
- `video_guided_worker.py`, `video_auto_yolo.py`: xử lý YOLO thật trong `.venv_yolo`.
- `tab_webcam_v46.py`: Webcam AI thật + nút ghi MP4, chụp PNG.
- `model_loader.py`, `image_matting.py`, `processing.py`: U2Net-P và xử lý ảnh.
- `evaluate.py`: đánh giá (không bắt buộc để mở giao diện).

**Các tên `v40`, `v46` vẫn được giữ để không phá import.** Không xóa chúng. `aetm_ui.py` phiên bản cũ có thể chuyển vào thư mục backup sau khi bản mới chạy thành công vì các module 3G này đã dùng `ui_3g.py`.

## Cách dùng

### Ảnh
Tải JPG/PNG → **Tách nền tự động** → xem ảnh/mask → **Mask đạt** hoặc dùng cọ **Thêm/Xóa** → chọn loại đầu ra → **Xuất PNG**.

### Video
Tải MP4 tối đa 30 giây → tạo tối đa 5 mốc (một đối tượng nên dùng cùng ID) → vẽ Bao/Thêm/Xóa trên từng mốc → chọn loại nền/độ phân giải → **Chạy AI** → sửa đoạn lỗi ở Bước 3 → **Xuất MP4 hoàn thiện**. Có CSV cảnh báo và chức năng hoàn tác lan truyền.

### Webcam
Bật camera ở ô **Camera gốc** → xem AI trực tiếp bên cạnh → bấm **Bắt đầu ghi MP4** → **Dừng ghi & xem lại** hoặc **Chụp PNG**. Nút Record/Stop nằm *bên trong camera* chỉ bật/tắt nguồn camera, **không phải** ghi MP4. Video hiện không thu microphone.

## Lưu ý kiểm thử

Mã nguồn đã được kiểm tra **cú pháp và khởi tạo giao diện**; kiểm tra HTTP máy chủ bằng mô hình import giả lập. Chất lượng AI và Webcam vật lý trên Windows của bạn vẫn cần thử thực tế. Việc đổi màu/bố cục không nhằm nâng độ chính xác YOLO/U2Net-P.

Nếu gặp lỗi, sao chép lại bản `app.py`, `tab_video_v40.py`, `tab_webcam_v46.py` đã sao lưu và mở lại `start_3G.bat`.
