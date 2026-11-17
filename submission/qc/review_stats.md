# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 425; đối tượng sau duyệt: 420; ngưỡng accepted: IoU >= 0.9
- Đã **loại 27 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 412 | 412/425 = 96.9% | 425 box máy |
| edited (sửa) | 3 | 3/425 = 0.7% | 425 box máy |
| deleted (xoá) | 10 | 10/425 = 2.4% | 425 box máy |
| added (vẽ thêm) | 5 | 5/420 = 1.2% | 420 đối tượng sau duyệt |

Trong số deleted + added, **3 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 2.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 185 | 1 | 0 | 0 |
| bicycle | 7 | 1 | 0 | 0 |
| motorbike | 126 | 0 | 0 | 1 |
| car | 53 | 0 | 1 | 1 |
| bus | 3 | 0 | 0 | 0 |
| truck | 6 | 0 | 0 | 2 |
| three_wheeler | 32 | 1 | 9 | 1 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 40 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 37 | 23 | 3 | 61.7% (KTC 95%: 49–73%) | 92.5% (KTC 95%: 80–97%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 2/12 = 16.7% (KTC 95%: 5–45%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - wrong_class: 0/3
  - loose: 1/3
  - extra: 1/3
  - missing: 0/3
  - lọt: img_010.jpg (wrong_class), img_026.jpg (loose), img_033.jpg (wrong_class), img_024.jpg (extra), img_013.jpg (extra), img_014.jpg (missing), img_034.jpg (missing), img_018.jpg (missing), img_021.jpg (loose), img_031.jpg (wrong_class)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
