# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603248 - Đặng Trung Hiếu

> **Chỉ viết phần thay đổi so với v1.0** (không copy cả file). Chốt các câu hỏi mở ở mục 6 của v1.0 (hoặc nói rõ câu nào chưa chốt, vì sao)
> và thêm quy tắc mới rút ra từ lỗi của máy / lỗi của bạn. Mỗi ca khó **có ảnh ví dụ** (`img_0xx`) và khớp một dòng trong `qc/calibration_decisions.csv`.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi | Quyết định | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 4 | Người ngồi trong xe ba bánh có mui che (chỉ thấy thấp thoáng): ngưỡng "còn thấy ≥ 20%" có hợp lý không? | Giữ ngưỡng 20% nhưng đo bằng tiêu chí đếm được thay cho ước lượng % bằng mắt: **gán `person` nếu thấy được đầu VÀ ít nhất vai / thân trên**. Chỉ thấy đầu, hoặc bóng mờ sau kính, thì không gán và tick `can_review` trên box xe. | img_013 (2 khách trong khoang tuk-tuk → gán), img_003 (em bé ngồi trên thùng ba gác → gán) | CAL-02 |
| 5 | Kính chắn gió / cửa sổ xe: thuộc mask xe hay không? | **Kính có tấm kính** (kính chắn gió, cửa kính) **thuộc mask xe**. **Khoang hở không có kính** của tuk-tuk / xích lô (nhìn xuyên ra phố) là **lỗ thủng**: lớn hơn ~5% diện tích xe thì bỏ khỏi polygon (theo 4.2.2). Người ngồi trong là vật riêng; phần xe bị người che không đoán. | img_002 (mask 1 click dính đám người nhìn xuyên qua khoang), img_013 | CAL-03 |
| 7 | Mức tin cậy tối thiểu để chấp nhận box máy không cần sửa? | **Không đặt.** Mọi box máy đều phải được nhìn; `conf` chỉ dùng để **xếp thứ tự duyệt** (box `conf` < 0.5 duyệt trước), không phải tiêu chí chất lượng. Lý do: điểm không tách được đúng / sai. | img_003 (three_wheeler 0.80 chỉ ôm thân xe), img_032 (xích lô đúng chỉ 0.27) | CAL-01 |
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | **Chưa chốt**: chưa gặp ca rõ ràng trong 24 ảnh batch để làm ví dụ. Tạm thời: không chắc thì chọn lớp tin nhất và tick `can_review` (theo 4.3). | — | — |
| 2 | Khối xe máy đỗ dày gần camera: từng xe hay `ignore_region`? | **Chưa chốt**: lô có 7 box xe < 20 px (img_018, img_023, img_027, img_033) nhưng đều là xe ở xa, chưa có ca "dày mà gần camera" để đặt ngưỡng. | — | — |
| 3 | Hàng nhô ra (tre, rổ): box / mask xe có gồm hàng không? | **Chưa chốt**: chưa gặp ca hàng nhô > 20% chiều dài xe. Giữ quy tắc 4.1.5 của v1.0. | — | — |
| 6 | Box máy chồng nhau quá nhiều trong ảnh đêm: gộp hay giữ? | **Chưa chốt** cho ảnh đêm (chưa gặp ca). Với ảnh ngày đã có quy tắc ở mục 2 (CAL-05): một xe đúng một box. | — | — |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| img_011, img_032, img_013 | Xe ba bánh (tuk-tuk, xích lô) trong ảnh | **YOLO26** (closed-set COCO) không có lớp three_wheeler: img_011 gọi `truck` 0.77, img_032 xích lô gọi `bicycle` 0.69, img_013 tuk-tuk gọi `car` 0.41. Tăng `--conf` lên 0.5 không loại được. | Không lấy three_wheeler từ YOLO26; lấy từ YOLOE, merge ưu tiên three_wheeler khi chồng lên car / truck / bicycle. Khi duyệt: box car / truck / bicycle đặt trên xe có mui hoặc thùng mà chỉ có 3 bánh → đổi thành `three_wheeler`. (CAL-04) | Mục 2 (bảng lớp, cột "không gồm" của car / truck / bicycle) và 4.3: thêm 3 ảnh ví dụ này; nhắc **đếm bánh** trước khi chấp nhận box car / truck / bicycle trên xe có mui. |
| img_003, img_009 | Một xe ba bánh bị máy đóng nhiều box; xe khác bị gọi nhầm three_wheeler | **YOLOE** (prompt "tuk-tuk / cyclo / cargo tricycle"): img_003 ngoài box ôm cả ba gác còn box 0.80 chỉ ôm thân và 0.21 chỉ ôm mui; 2 xe tải nhỏ ở xa bị gọi three_wheeler (0.25, 0.61); img_009 box 0.29 thực ra là xe máy. Cả lô: 9/42 box three_wheeler bị xoá (21.4%). | **Mỗi xe ba bánh đúng một box** ôm thân + mui + thùng hàng gắn liền; xoá các box chỉ ôm một phần. Xe ≥ 4 bánh có buồng lái → `truck`; xe 2 bánh → `motorbike`. (CAL-05) | Mục 4.1 (thêm quy tắc 8: "một vật = một box; box chỉ ôm một phần vật là box thừa") và bảng mục 5. |
| img_010, img_031, img_033 | Box đúng chỗ nhưng sai lớp (bẫy wrong_class) | Bẫy đổi lớp: motorbike → bicycle (img_010), person → bicycle (img_031, img_033). Người duyệt để lọt cả 3. | **Không đổi guideline** (định nghĩa lớp đã rõ); lỗi do người duyệt chỉ nhìn khung, không đọc nhãn. (CAL-06) | Không đổi v1.0. Thêm vào checklist duyệt: đọc nhãn mọi box lớp hiếm (bicycle, bus, truck, three_wheeler); box `bicycle` trùng chỗ một người hoặc một xe máy là dấu hiệu sai lớp. |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản giảng viên | — |
| v1.1 | Chốt câu 4 (tiêu chí "đầu + vai / thân trên" cho người trong xe ba bánh), câu 5 (kính thuộc xe, khoang hở là lỗ), câu 7 (không tự chấp nhận theo `conf`); thêm ca khó máy gọi sai lớp xe ba bánh và quy tắc "một xe một box". Câu 1, 2, 3, 6 chưa chốt (chưa có ca làm ví dụ). | CAL-01, CAL-02, CAL-03 (câu hỏi mở); CAL-04, CAL-05 (lỗi máy); CAL-06 (bẫy lọt, chỉ đổi checklist) |
