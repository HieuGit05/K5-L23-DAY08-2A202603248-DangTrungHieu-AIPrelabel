# Báo cáo Day 08 — <<MSHV - Họ và tên>>

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | <<YOLO26m>> | person 0.5 · motorbike 0.35 · bicycle / car / bus / truck: mặc định `--conf` 0.3 | Theo `qc/threshold_sweep.md` (mục tiêu recall ≥ 90%, 4 ảnh gold). person 0.5: recall 13/14 = 93%, precision 13/22 = 59%; lên 0.55 thì recall tụt còn 11/14 = 79%. motorbike 0.35: recall 9/10 = 90%, precision 9/12 = 75%; lên 0.4 thì recall còn 8/10 = 80%. bicycle chỉ có 3 vật gold, bus và truck không có vật gold nào (< 5) nên không chọn ngưỡng riêng. car: không ngưỡng nào ≤ 0.5 đạt 90% (tối đa 6/7 = 86%, 1 xe sót ở mọi ngưỡng), nên giữ 0.3. Ở 0.3, bus có 1 box và truck có 2 box thừa trên gold dù gold không có bus/truck nào; nghi là tuk-tuk bị gọi bus/truck như ở r01, r08a (`model_log.csv`). |
| three_wheeler | <<YOLOE-26s, prompt: ...>> | 0.15 | 0.15: recall 6/6 = 100%, precision 6/9 = 67%; lên 0.2 thì recall còn 5/6 = 83%, lên 0.25 còn 4/6 = 67%. Khớp với r04/r08c: xích lô đúng ở img_032 chỉ đạt 0.27, thấp hơn box sai 0.37–0.40, nên phải để ngưỡng thấp và review tay. |

- Chuỗi ngưỡng theo lớp (từ `qc/threshold_sweep.md`, dùng cho Part 4): `--thr "person=0.5,motorbike=0.35,three_wheeler=0.15"` (lớp không có trong chuỗi dùng `--conf 0.3`)
- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): <<>>
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: <<chất lượng mask, tốc độ, RAM, quyền truy cập>>
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? <<>>
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào): <<>>
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng được chọn và chấm trên cùng 4 ảnh gold. Mỗi lớp chỉ có 6–14 vật, nên chỉ cần thêm hoặc bớt 1 vật là recall đổi 7–17 điểm %. Vì vậy các ngưỡng trên chỉ là ước lượng thô; precision/recall thật trên 24 ảnh batch sẽ thấp hơn.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | <<>> | <<>> | <<box máy>> |
| edited | <<>> | <<>> | <<>> |
| deleted | <<>> | <<>> | <<>> |
| added | <<>> | <<>> | <<đối tượng sau duyệt>> |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? <<>>
- So với gold: precision / recall của **máy** <<>> (mẫu số: <<đối tượng gold>>)

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: <<.../12>>; loại nào lọt: <<>>; vì sao (bạn đã bỏ bước nào của checklist)? <<>>
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt? <<>>

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay <<... phút/ảnh>> vs duyệt <<... phút/ảnh>>. Thiên lệch (bạn gán gold trước nên đã quen cảnh): <<>>
- Kết luận: <<có / không / chỉ cho lớp nào>>, dựa trên <<số liệu>>
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. <<>>
  2. <<>>
  3. <<>>

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| <<img_0xx>> | <<>> | <<>> | <<>> | <<>> | <<>> |

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu? <<>>
- *(Tuỳ chọn)* Interactor SAM trong CVAT: làm được / không (lỗi gì, đã thử gì); khác gì offline: <<>>

## 6. Calibration (Part 6)

- Số quyết định: <<>> (`calibration_decisions.csv`); quan trọng nhất (dẫn `CAL-xx`): <<>>
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`: <<>>
- Nguyên nhân gốc: <<guideline mơ hồ / chưa nắm quy tắc / máy>>

## 7. QC và quyết định (Part 6)

- Mẫu: <<n>> ảnh (<<m>> ảnh `can_review` + <<k>> ngẫu nhiên, seed 42); người kiểm: bạn (**hạn chế**: bạn vừa là người duyệt).
- Lỗi theo **đối tượng**: <<.../... = ...%>> · theo **ảnh**: <<.../... = ...%>> (ngưỡng Release ≤ 2%)
- Top lỗi (trong: bỏ sót, thừa, sai lớp, sai vị trí/kích thước, sai thuộc tính, sai mask) + một ảnh ví dụ mỗi loại: <<>>
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): <<>>
- Script kiểm được / không kiểm được (ví dụ ôm sát, cùng bỏ sót với máy, theo checklist thật): <<>>

**Quyết định: <<Release / Hold / Rework>>**, ngày <<YYYY-MM-DD>>. Lý do (đi theo Hold → Rework → Release): <<>>
Nếu Rework / Hold, việc cần làm tiếp: <<>>

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): <<>>
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? <<không / có, như sau: ...>>
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: <<>>
- Dùng cache `data/cache/` thay vì tự chạy? <<không / có, vì ...>>
- Lỗi / bug script đã gặp (nếu có): <<>>
- Một điều bạn làm khác lần sau: <<>>
