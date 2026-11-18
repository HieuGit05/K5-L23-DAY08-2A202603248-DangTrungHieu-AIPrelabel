# Báo cáo Day 08 — 2A202603248 - Đặng Trung Hiếu


## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m (`yolo26m.pt`, chạy lại `y26_all` bằng `--size m`: r09) | person 0.5 · motorbike 0.35 · bicycle / car / bus / truck: mặc định `--conf` 0.3 | Theo `qc/threshold_sweep.md` (mục tiêu recall ≥ 90%, 4 ảnh gold). person 0.5: recall 13/14 = 93%, precision 13/22 = 59%; lên 0.55 thì recall tụt còn 11/14 = 79%. motorbike 0.35: recall 9/10 = 90%, precision 9/12 = 75%; lên 0.4 thì recall còn 8/10 = 80%. bicycle chỉ có 3 vật gold, bus và truck không có vật gold nào (< 5) nên không chọn ngưỡng riêng. car: không ngưỡng nào ≤ 0.5 đạt 90% (tối đa 6/7 = 86%, 1 xe sót ở mọi ngưỡng), nên giữ 0.3. Ở 0.3, bus có 1 box và truck có 2 box thừa trên gold dù gold không có bus/truck nào; nghi là tuk-tuk bị gọi bus/truck như ở r01, r08a (`model_log.csv`). |
| three_wheeler | YOLOE-26s (`yoloe-26s-seg.pt`), prompt: "tuk-tuk", "cyclo", "cargo tricycle" (r07) | 0.15 | 0.15: recall 6/6 = 100%, precision 6/9 = 67%; lên 0.2 thì recall còn 5/6 = 83%, lên 0.25 còn 4/6 = 67%. Khớp với r04/r08c: xích lô đúng ở img_032 chỉ đạt 0.27, thấp hơn box sai 0.37–0.40, nên phải để ngưỡng thấp và review tay. |

- Chuỗi ngưỡng theo lớp (từ `qc/threshold_sweep.md`, dùng cho Part 4): `--thr "person=0.5,motorbike=0.35,three_wheeler=0.15"` (lớp không có trong chuỗi dùng `--conf 0.3`)
- Lệnh `merge_prelabels.py` đã dùng (dán nguyên): `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --thr "person=0.5,motorbike=0.35,three_wheeler=0.15" --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json` (ra 499 box trên 28 ảnh)
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: SAM 2.1 base (`sam2.1_b.pt`). Có sẵn trong bộ model `core`, không cần xin quyền như SAM 3, chạy trên GPU CUDA của máy (r08d: 2.21 s/ảnh tính cả việc tạo 16 mask). Mask bám tốt mui và thân xe (Part 5, mục 5). Chưa so với `mobile_sam.pt`.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào? YOLO26 không có lớp three_wheeler nên gọi xe ba bánh bằng lớp khác, có khi với điểm cao:
  - img_011: xe ba bánh ở mép trái ra `truck` 0.77 (r08a), tự tin hơn bản cỡ s (0.36).
  - img_032: xích lô ra `bicycle` 0.69.
  - img_013: tuk-tuk ra `car` 0.41 (r08d).
  - img_002: tuk-tuk xanh không có box nào.

  Tăng `--conf` lên 0.5 (r08b) vẫn giữ `truck` 0.77 và `bicycle` 0.69, mà còn mất 43 person và 23 motorbike thật. Cách sửa: lấy three_wheeler từ YOLOE (prompt chữ), để `merge_prelabels.py` ưu tiên box three_wheeler khi chồng lên car/truck/bicycle, rồi review tay đúng các lớp hay nhầm này.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - **Điểm cao mà sai:** img_011 `truck` 0.77 trên xe ba bánh (r08a); img_003 three_wheeler 0.80 bị xoá khi review và three_wheeler 0.61 thực ra là xe tải nhỏ.
  - **Điểm thấp mà đúng:** img_032 xích lô bên phải chỉ được three_wheeler 0.27, trong khi các box sai ở img_002 (xe máy + người lái, pickup, sạp hàng) được 0.37–0.40 (r04/r08c).
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan: ngưỡng được chọn và chấm trên cùng 4 ảnh gold. Mỗi lớp chỉ có 6–14 vật, nên chỉ cần thêm hoặc bớt 1 vật là recall đổi 7–17 điểm %. Vì vậy các ngưỡng trên chỉ là ước lượng thô; precision/recall thật trên 24 ảnh batch sẽ thấp hơn.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 412 | 96.9% | 425 box máy |
| edited | 3 | 0.7% | 425 box máy |
| deleted | 10 | 2.4% | 425 box máy |
| added | 5 | 1.2% | 420 đối tượng sau duyệt |

Nguồn: `qc/review_stats.md`; 24 ảnh batch, đã loại 27 đối tượng quanh bẫy. Trong deleted + added có 3 ca đổi lớp, nên máy chỉ sót thật khoảng 2 vật.

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)? **three_wheeler (YOLOE)**: 9/42 box máy bị xoá (21.4%), trong khi các lớp khác bị xoá ≤ 1 box. Ví dụ:
  - **img_003:** bỏ 7 box three_wheeler (0.16–0.80); trong đó 2 box (0.25 và 0.61) thực ra là xe tải nhỏ nên đổi thành `truck`.
  - **img_009:** xoá 2 box 0.27 và 0.29; một box thực ra là xe máy nên vẽ lại thành `motorbike`.
  - **img_004:** chỉnh lại box three_wheeler 0.62 cho khớp tuk-tuk, và vẽ thêm tuk-tuk xanh dương ở bên trái mà máy bỏ sót.

  Nguyên nhân giống ở lượt pilot (r04, r08c trong `model_log.csv`): YOLOE dùng prompt chữ nên gọi mọi xe có mui/thùng là three_wheeler, và điểm của box sai (0.6–0.8) có thể cao hơn box đúng. Ngưỡng three_wheeler đặt thấp (0.15) để giữ recall nên nhận thêm nhiều box rác.
- So với gold: precision / recall của **máy** 37/60 = 61.7% (KTC 95%: 49–73%) / 37/40 = 92.5% (KTC 95%: 80–97%) (mẫu số: 40 đối tượng gold trên 4 ảnh). Trong 23 FP có 9 person, 4 car, 3 three_wheeler, 3 motorbike, 2 bicycle, 1 truck, 1 bus. 3 FN là 1 motorbike và 1 person ở img_036, 1 car ở img_001. Các số này lạc quan vì ngưỡng được chọn trên chính 4 ảnh gold.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: **2/12 = 16.7%** (KTC 95%: 5–45%; ngưỡng đạt 80%).
  - Bắt được 1 loose (img_009) và 1 extra (img_017).
  - Lọt toàn bộ **wrong_class 3/3** (img_010, img_031, img_033) và **missing 3/3** (img_014, img_018, img_034).
  - Lọt thêm loose 2/3 (img_021, img_026) và extra 2/3 (img_013, img_024).
  - Lưu ý về bẫy missing ở img_014: theo `traps_key.json` (chỉ mở sau khi export), box bị bẫy xoá ở `[291,379,85,168]` là một box three_wheeler **sai** của YOLOE đặt lên gánh hàng của người phụ nữ, không phải xe thật. Vì vậy lọt bẫy này không làm nhãn cuối tệ đi: không vẽ lại box đó mới là đúng. Bẫy loại missing chỉ đo đúng thiên kiến khi vật bị xoá là vật thật. Bẫy này cho thấy một bẫy cài ngẫu nhiên lên pre-label vốn đã sai thì có thể đo sai.
- Vì sao lọt (đã bỏ bước nào của checklist)?
  - **Chỉ sửa ở nửa đầu lô.** Mọi chỉnh sửa nằm trong 10 ảnh đầu (img_003 → img_017; 6/10 ảnh có sửa). 14 ảnh sau (img_018 → img_035) có 0/14 ảnh được sửa. 7/10 bẫy lọt nằm ở 14 ảnh này.
  - **Bỏ bước 1 (ẩn nhãn máy, tự đếm vật trước).** Bẫy missing là vật bị xoá khỏi pre-label: nhìn nhãn máy trước thì không thấy thiếu gì, nên lọt 3/3.
  - **Không kiểm lớp từng box (bước 3: sai nhãn).** Bẫy wrong_class giữ nguyên vị trí box, chỉ đổi lớp: lọt 3/3. Ở img_010 mình có sửa chiều cao box bicycle nhưng bẫy wrong_class trong ảnh này vẫn lọt, tức là chỉ nhìn hình dạng box mà không đọc nhãn.
  - **Lần review đầu (20 phút) export ra không khác pre-label box nào** (439/439 box trùng), có thể do chưa lưu (Ctrl+S) trước khi export. Số liệu trên là của lần review thứ 2 (30 phút).
  - **Lần review thứ 2 không còn "mù":** trước khi review lại, mình đã đọc danh sách 12 ảnh có bẫy và loại bẫy trong `review_stats.md` của lần 1. Vì vậy tỉ lệ 2/12 của lần 2 **không còn là số đo khách quan** về thiên kiến tự động hoá; số "mù" thật là của lần 1 (0/12, export chưa sửa box nào).
  - Ở lần 2, em tập trung sửa box thừa và box sai lớp. Thống kê ghi nhận 10 box bị xoá và 3 ca đổi lớp: img_003 có 2 box three_wheeler đổi thành `truck`; img_009 có 1 box three_wheeler vẽ lại thành `motorbike`. Nhưng các ca này nằm ngoài vùng bẫy: bẫy wrong_class vẫn lọt 3/3, bẫy extra lọt 2/3.
  - Dù đã biết trước ảnh nào có bẫy mà vẫn chỉ bắt 2/12, điều này cho thấy vấn đề không chỉ là không biết chỗ cần tìm. Mình chưa có thói quen kiểm từng box: đọc nhãn lớp, so độ ôm sát, đếm vật bị thiếu.
- Accepted cao mà bắt bẫy thấp nói gì về cách bạn duyệt? Accepted 96.9% mà chỉ bắt 2/12 bẫy cho thấy accepted cao **không** có nghĩa máy tốt. Phần lớn box được giữ vì không ai thực sự kiểm, không phải vì đã kiểm và thấy đúng. Đây là thiên kiến tự động hoá: nhãn máy hiện sẵn thì người duyệt mặc định tin. `added` = 5 (1.2%) cũng không chứng minh máy ít sót, vì 3/3 vật bị xoá có chủ đích vẫn không được vẽ lại.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 5 phút / 4 ảnh = 1.25 phút/ảnh vs duyệt 30 phút / 24 ảnh = 1.25 phút/ảnh. Tính theo vật: gán tay 5 phút / 40 vật ≈ 7.5 giây/vật; duyệt 30 phút / 433 vật ≈ 4.2 giây/vật. Nếu cộng cả lần review đầu (20 phút) thì duyệt là 50/24 ≈ 2.1 phút/ảnh. Thiên lệch:
  - Mình gán gold trước nên đã quen cảnh và quy tắc khi duyệt.
  - Ảnh gold thưa hơn (10 vật/ảnh so với 18 vật/ảnh ở batch).
  - Tốc độ duyệt nhanh một phần vì bỏ qua 14 ảnh cuối (bắt bẫy 2/12), nên con số 4.2 giây/vật **đánh giá quá cao** lợi ích của máy.
- Kết luận: **có, nhưng chỉ cho person / motorbike / car**.
  - Ba lớp này có accepted ≥ 98% trên số box máy (person 185/186, motorbike 126/126, car 53/54), và máy đạt recall 92.5% trên gold. Pre-label giúp vẽ nhanh hơn khoảng 1.8 lần tính theo vật.
  - **three_wheeler** chỉ dùng làm gợi ý vì 9/42 box bị xoá (21%) và điểm số không tách được box đúng với box sai.
  - Điều kiện để dùng: review phải đạt bắt bẫy ≥ 80%. Với 2/12 hiện tại, chất lượng lô này chưa đủ tin cậy để Release.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Ẩn nhãn máy và tự đếm trước.** Với mỗi ảnh, ẩn hết nhãn máy, tự đếm số vật theo lớp rồi ghi ra trước khi hiện nhãn. Nếu số đếm lệch với số box máy thì tìm cho ra vật thiếu hoặc thừa. Kiểm tra: bắt bẫy missing ≥ 2/3, `added` > 0 ở các ảnh có vật bị che.
  2. **Duyệt theo đợt 8 ảnh, nghỉ giữa các đợt, đặt sàn thời gian.** Chia lô 24 ảnh thành 3 đợt 8 ảnh, nghỉ giữa các đợt, tối thiểu 2 phút/ảnh (3 phút với ảnh dày). Kiểm tra: tỉ lệ ảnh có ít nhất 1 chỉnh sửa ở đợt cuối không thấp hơn đợt đầu (lần này 6/10 so với 0/14), và tổng thời gian ≥ 48 phút cho 24 ảnh.
  3. **Đọc nhãn lớp từng box ở các lớp hay nhầm.** Dùng Tab đi qua từng box three_wheeler / truck / bicycle / bus và đọc nhãn lớp, không chỉ nhìn khung. Lọc box có `conf` < 0.5 để duyệt trước. Kiểm tra: bắt bẫy wrong_class ≥ 2/3, ghi số box đã đổi lớp vào REPORT.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| img_013 (tuk-tuk có tài xế + 2 khách) | SAM 2.1 base (`sam2.1_b.pt`) | 9 (3 dương + 6 âm), thử 4 lần | Không: 1 click âm lên mỗi người (tài xế, 2 khách) và 1 lên xe máy phía sau đã tách được hết | Hụt nệm ghế đen và khoang lái đỏ phía trước. Bản thêm 2 click dương vào chỗ hụt thì mui bị lủng lỗ chỗ và mask tách thành 2 polygon, nên quay về bản 9 click | `masks/tuktuk_driver_img013.jpg` |
| img_032 (xích lô phía trước) | SAM 2.1 base | 1 (1 dương lên nệm hoa), thử 3 lần | Không có người trên xe; mask không dính người đội mũ trắng đứng cạnh | Tô kín cả nan bánh xe và khoảng trống dưới gác chân (lỗ lớn nhìn xuyên qua). Đặt click âm vào lòng 2 bánh thì mask co còn mỗi tấm nệm (2.0% ảnh), nên giữ bản 1 click | `masks/cyclo_img032.jpg` |
| img_002 (tuk-tuk xanh bị che) | SAM 2.1 base | 7 (3 dương + 4 âm), thử 3 lần | Không: đã bỏ tài xế và người lái xe máy che phía trước | Bản 1 click còn dính đám người nhìn xuyên qua khoang hở; thêm 2 click âm thì sạch. Bản cuối hụt cột kính chắn gió xanh và phần sàn dưới ghế tài xế | `masks/tuktuk_occluded_img002.jpg` |

Diện tích mask lần lượt 21.1%, 8.6% và 5.1% ảnh, mỗi mask 1 polygon (`sam2_points.py`, dòng `r10` trong `model_log.csv`).

So với r08d (SAM lấy box YOLO26 làm prompt): ở đó tuk-tuk img_013 bị gọi `car` và mask lởm chởm, còn tuk-tuk xanh dương ở img_004 không có box nên không có mask. Prompt bằng click không phụ thuộc lớp của detector nên làm được cả 3 vật, nhưng mỗi vật cần người chọn điểm và xem lại 3–4 lần.

- Đối chiếu guideline mục 4.2: quy tắc nào giúp, quy tắc nào thiếu?
  - **Giúp:** bảng ca khó (mục 5) nói rõ người là vật riêng ("mỗi người một `person`"), và định nghĩa three_wheeler gồm "cả xe lẫn khung mái / thùng hàng". Nhờ vậy biết phải đặt click âm lên tài xế/khách và click dương lên mui.
  - **Giúp:** quy tắc 4.2.4 (vùng chạm thuộc vật ở phía trước). Ở img_002, người lái xe máy đứng trước tuk-tuk nên phần bị che phải bỏ khỏi mask tuk-tuk.
  - **Khó áp dụng:** quy tắc 4.2.2 (lỗ lớn hơn ~5% diện tích thì bỏ ra). SAM không làm được bằng click: click âm vào lỗ làm cả mask co lại (img_032). Lỗ phải xoá tay trên polygon, nhưng guideline không nói khi nào thì chấp nhận mask của SAM còn lỗ.
  - **Thiếu:** guideline không nói khoang hành khách mở của tuk-tuk (nhìn xuyên ra người phía sau, img_002) có tính là "lỗ" không, và người ngồi trong xe che một phần ghế thì mask xe có lấp phần bị che không. Nên thêm vào v1.1.


## 6. Calibration (Part 6)

- Số quyết định: **6** (`calibration_decisions.csv`): 3 `cau_hoi_mo`, 2 `loi_may`, 1 `bay_lot`; 5/6 ca đưa vào v1.1, riêng CAL-06 chỉ đổi checklist duyệt.
- Quan trọng nhất:
  - **CAL-05 (một xe ba bánh, một box):** three_wheeler là lớp máy sai nhiều nhất (9/42 box bị xoá = 21.4%), mà phần lớn là box trùng hoặc box chỉ ôm một phần xe (img_003). Ngay trong nhãn đã duyệt vẫn còn ca vi phạm: img_013 có 2 box three_wheeler chồng nhau trên cùng một tuk-tuk.
  - **CAL-01 (không tự chấp nhận theo `conf`):** điểm số không tách được đúng/sai. Box sai được 0.80 (img_003) trong khi xích lô đúng chỉ được 0.27 (img_032).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:
  - Câu 4: gán `person` trong xe ba bánh khi thấy được đầu và vai/thân trên (CAL-02, img_013, img_003).
  - Câu 5: kính có tấm kính thuộc mask xe; khoang hở là lỗ thủng (CAL-03, img_002).
  - Câu 7: `conf` chỉ dùng để xếp thứ tự duyệt (CAL-01).
  - Khi box car / truck / bicycle nằm trên xe có mui hoặc thùng hàng thì đếm bánh trước khi chấp nhận (CAL-04, img_011, img_032, img_013).
  - Một vật một box; box chỉ ôm một phần vật là box thừa (CAL-05).
  - Câu 1, 2, 3, 6 **chưa chốt** vì trong lô chưa có ca làm ví dụ.
- Nguyên nhân gốc:
  - **Máy (CAL-04, CAL-05):** YOLO26 là closed-set nên không có lớp xe ba bánh. YOLOE dùng prompt chữ nên bắt nhầm và đóng box trùng.
  - **Guideline mơ hồ (CAL-01, CAL-02, CAL-03):** ngưỡng "20%" đo bằng mắt, chưa phân biệt kính với khoang hở, chưa nói gì về `conf`.
  - **Mình chưa áp dụng đúng quy tắc khi duyệt (CAL-06):** định nghĩa lớp đã rõ nhưng mình không đọc nhãn từng box, nên lọt 3/3 bẫy wrong_class. Đây là lỗi lớn nhất, và sửa guideline không giải quyết được.

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ảnh `can_review` + 5 ngẫu nhiên, seed 42), 90 đối tượng: img_003, img_006, img_014, img_031, img_035. Không có ảnh ca khó vì mình không tick `can_review` box nào khi duyệt. Đây cũng là một lỗi quy trình: các ca không chắc (dãy xe đỗ dày ở img_006, xích lô ở img_035) lẽ ra phải được đánh dấu.
- Người kiểm: mình, với bản nháp `qc_review.csv` do AI (Claude) soạn và mình kiểm lại từng dòng. **Hạn chế:**
  - Mình vừa là người duyệt vừa là người kiểm, nên dễ bỏ qua đúng những lỗi mình đã để lọt.
  - Bản nháp của AI chỉ nhìn ảnh ở 1024 px, nên có thể sót box lỏng vài pixel.
- Lỗi theo **đối tượng**: 6/90 = 6.7% · theo **ảnh**: 4/5 = 80% (ngưỡng Release ≤ 2%). Nguồn: `qc/qc_report_auto.md`.
- Top lỗi, mỗi loại kèm ảnh ví dụ:
  - **Thừa** (3/6 lỗi, lặp lại nên là lỗi hệ thống): cả 3 đều là box three_wheeler của YOLOE đặt lên vật không phải xe hoặc trùng xe.
    - img_014 #132: sạp hàng.
    - img_014 #133: gánh hàng (đòn gánh + nồi).
    - img_035 #422: box thứ hai trên cùng một xích lô.
  - **Bỏ sót** (1): img_006, khoảng 3 xe máy đỗ trong dãy sát tường không có box hay `ignore_region`.
  - **Sai lớp** (1): img_031 #368, người bị gán `bicycle` (bẫy wrong_class bị lọt).
  - **Sai vị trí / kích thước** (1): img_031 #367, box bicycle chỉ ôm nửa trước xe đạp.
- **Giới hạn mẫu nhỏ:**
  - Cận trên 95% (Wilson) của tỉ lệ ảnh lỗi là **96%**: 5 ảnh không đủ để đo chính xác. Nhưng ở đây mẫu nhỏ không cứu được kết luận, vì 4/5 ảnh có lỗi. Muốn tỉ lệ ảnh lỗi thật thấp tới 2% thì gần như không thể thấy kết quả như vậy.
  - Ngược lại, kể cả nếu mẫu 5 ảnh không có lỗi nào thì cũng không chứng minh được "≤ 2%". Theo quy tắc 3/n, 0/5 ảnh lỗi vẫn cho phép tỉ lệ thật tới khoảng 60%.
- Script kiểm được / không kiểm được:
  - **Kiểm được:** định dạng file, tên lớp, box nằm trong ảnh, đủ 24 ảnh (`validate_labels.py`: 24 ảnh, 433 đối tượng, OK); accepted / edited / deleted / added và bắt bẫy (`review_stats.py`); đếm lỗi theo phiếu kiểm (`qc_report.py`).
  - **Không kiểm được:** box có ôm sát vật thật không (img_031 #367 qua được validate dù chỉ ôm nửa xe); lớp có đúng thực tế không (img_014: script không biết gánh hàng không phải xe); vật mà cả mình lẫn máy cùng bỏ sót (img_006); mình có làm theo checklist thật không. Những việc này chỉ phát hiện được bằng mắt.

**Quyết định: Rework**, ngày 2026-11-18.

Lý do (đi theo Hold → Rework → Release):
- **Hold?** `qc_report.py` gợi ý Hold vì còn 4 câu hỏi guideline chưa chốt (câu 1, 2, 3, 6). Nhưng trong 6 lỗi tìm thấy, chỉ lỗi bỏ sót ở img_006 dính tới câu hỏi chưa chốt (câu 2: xe đỗ dày). 5 lỗi còn lại sai rõ ràng theo v1.0 / v1.1: sạp hàng và gánh hàng không phải xe, người không phải xe đạp, box phải ôm cả vật, một xe một box. Guideline chưa rõ không phải nguyên nhân chính, nên mình không dừng ở Hold.
- **Rework.** Cả ba điều kiện đều có:
  - Vượt ngưỡng xa: 4/5 ảnh lỗi so với ngưỡng ≤ 2%.
  - Có lỗi lặp: box three_wheeler thừa chiếm 3/6 lỗi, khớp với 9/42 box three_wheeler bị xoá ở bước review.
  - Bắt bẫy thấp: 2/12 = 16.7% (cần ≥ 10/12), tức là review chưa đủ tin cậy để tin các ảnh ngoài mẫu.
- **Release:** không đạt.

Việc cần làm tiếp (vòng Rework):
1. **Duyệt lại cả 24 ảnh** theo 3 quy tắc giảm thiên kiến ở mục 4: ẩn nhãn máy và tự đếm trước; chia đợt 8 ảnh với sàn 2 phút/ảnh; đọc nhãn từng box lớp hiếm.
2. **Áp quy tắc v1.1** với lớp three_wheeler: xoá mọi box three_wheeler không nằm trên xe ba bánh thật, gộp các box chồng nhau về một box mỗi xe (CAL-05). Các ảnh đã biết có lỗi: img_013, img_014, img_035.
3. **Sửa các lỗi đã biết:**
   - Đổi img_031 #368 thành `person` và vẽ lại box bicycle #367 ôm cả xe.
   - Ở img_013: xoá box motorbike thừa (bẫy), xoá nhãn trùng car / truck trên xe van, gộp 2 box three_wheeler thành 1.
   - Chuyển 7 box < 20 px thành `ignore_region` hoặc xoá.
4. **Chốt câu hỏi mở 2** (xe đỗ dày gần camera) bằng ví dụ img_006, rồi gán theo quy tắc đã chốt.
5. Export `labels/qc_fixed.json`, chạy lại `build_release.py` và `qc_sample.py --force` (mẫu mới), kiểm lại và ghi kết quả vòng 2 vào báo cáo.
6. **Recipe:** cân nhắc tăng ngưỡng three_wheeler từ 0.15 lên khoảng 0.3. Recall trên gold sẽ giảm (4/6 ở 0.3), nên phải kèm một bước tìm xe ba bánh bị sót bằng mắt.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài** (công cụ nào, việc gì, kiểm tra thế nào; "không dùng" nếu không): **Có**, dùng Claude Code (extension VS Code). Phân chia như sau:
  - **AI làm, mình kiểm lại số với file nguồn:**
    - Chạy các script: `prelabel.py` cho `y26_all` cỡ m, `threshold_sweep.py`, `merge_prelabels.py`, `inject_traps.py`, `review_stats.py`, `sam2_points.py`, `build_release.py`, `qc_sample.py`, `qc_report.py`.
    - Chép `work/runs.csv` sang `model_log.csv` và viết nhận xét từ ảnh `viz/`.
    - So bản export CVAT với pre-label. Nhờ vậy phát hiện lần export đầu chưa sửa box nào.
    - Nháp REPORT mục 1–7, `calibration_decisions.csv` và `guideline_v1.1_calibrated.md`.
  - **AI chọn click SAM cho 3 mask (Part 5):** AI thử 3–4 bộ click cho mỗi vật rồi chọn bản tốt nhất; mình xem lại ảnh trong `masks/`.
  - **AI nháp `qc_review.csv`:** AI xem 5 ảnh mẫu QC và nháp phiếu kiểm. Cột `reviewer` ghi rõ là AI soạn nháp, mình đối chiếu lại với ảnh `work/qc/review/`. Có một chỗ AI tự đính chính: bẫy "missing" ở img_014 thực ra xoá một box sai, nên không tính là bỏ sót.
  - **Mình tự làm, không dùng AI:** gán gold 4 ảnh (Part 1), review 24 ảnh trong CVAT (Part 4), và chọn quyết định Rework.
  - **Cách kiểm:** mọi con số trong báo cáo đều đối chiếu được với file trong `submission/qc/` hoặc `model_log.csv`.
- Gold có bị nhiễm không (bạn có nhìn pre-label hay hỏi AI trước khi gán xong)? không dùng, gold gán tay
- `can_review`: **không tick box nào**, cả ở gold (0/40 đối tượng) lẫn batch (0/433).
  - Khi gán 4 ảnh gold, mình không gặp ca nào phân vân về quy tắc, nên không tick là đúng với cảm nhận lúc đó.
  - Nhưng nhìn lại thì "không phân vân" chưa chắc là "không có ca mơ hồ". Khi review batch và calibration mới lộ ra các ca guideline chưa rõ: dãy xe máy đỗ dày (img_006, câu hỏi mở 2), xích lô bị máy đóng 2 box (img_035), người ngồi trong xe ba bánh (img_013, CAL-02). Những ca này lẽ ra nên được tick `can_review` khi duyệt batch.
  - Hệ quả: mẫu QC có 0 ca khó (mục 7), và nguồn `can_review` trong calibration để trống.
  - Lần sau mình sẽ tick `can_review` mỗi khi phải tự suy ra quy tắc mà guideline không ghi sẵn, không đợi tới lúc thấy mình phân vân.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: RTX4050, CVAT local
- Dùng cache `data/cache/` thay vì tự chạy? không
- Lỗi / bug script đã gặp (nếu có): **`prelabel.py` ghi đè lịch sử chạy trong `work/runs.csv`**.
  - Khi chạy lại với cùng `--out`, script xoá dòng cũ có cùng `out` rồi ghi dòng mới (`scripts/prelabel.py`, dòng 420).
  - `run_id` được đánh số theo số dòng còn lại (`r{len(rows)+1}`), nên sau khi bị xoá thì bị trùng số.
  - Hậu quả: các lượt yolo26-s ở pilot (r02, r03), YOLOE (r04), SAM (r05) và lượt `--split all` cỡ s (r06), r07 đều không còn trong `runs.csv`, vì sau đó mình chạy lại cùng thư mục `--out` bằng yolo26-m. Có 6 dòng cùng mang `run_id` = `r08`.
  - Vì vậy `model_log.csv` không khớp từng dòng với `runs.csv` hiện tại. Thông số của các lượt bị ghi đè được chép vào `model_log.csv` trước khi chạy lại. Mình đặt lại ID là r08a–r08d và r09 để không trùng.
  - Cách tránh: mỗi lượt chạy dùng một thư mục `--out` riêng (ví dụ `t_y26_s`, `t_y26_m`).
- Một điều bạn làm khác lần sau: dùng model tool để gán nhãn
