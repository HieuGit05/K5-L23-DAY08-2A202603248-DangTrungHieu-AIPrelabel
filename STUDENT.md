# Thông tin học viên

- Họ và tên: Đặng Trung Hiếu
- MSHV: 2A202603248
- Lớp (L2 / L3 / khác): L2
- Tài khoản GitHub: HieuGit05
- Repo bài nộp (URL): https://github.com/HieuGit05/K5-L23-DAY08-2A202603248-DangTrungHieu-AIPrelabel
- Commit chốt bài (hash): a47b96f
- Máy: Windows 11, GPU NVIDIA GeForce RTX 4050 Laptop (CUDA, torch 2.6.0+cu124)
- CVAT (Docker): v2.81.0

## Kết quả tóm tắt (điền cuối bài; số lấy từ `submission/qc/*`, kèm mẫu số)

- Máy: accepted / edited / deleted: 412/425 = 96.9% / 3/425 = 0.7% / 10/425 = 2.4% (mẫu số: 425 box máy); added: 5/420 = 1.2% (mẫu số: 420 đối tượng sau duyệt)
- So với gold: precision / recall của **máy** 37/60 = 61.7% / 37/40 = 92.5% (40 đối tượng gold, 4 ảnh; lạc quan vì ngưỡng chọn trên chính gold)
- Bắt bẫy: 2/12 = 16.7%
- Thời gian: gán tay 1.25 phút/ảnh (5 phút / 4 ảnh); duyệt có pre-label 1.25 phút/ảnh (30 phút / 24 ảnh, chưa tính lần đầu 20 phút)
- Lỗi QC: 80% theo ảnh (4/5), 6.7% theo đối tượng (6/90)
- **Quyết định:** Rework — lý do một dòng: 4/5 ảnh mẫu có lỗi (ngưỡng ≤ 2%), lỗi box three_wheeler thừa lặp lại (3/6 lỗi) và chỉ bắt được 2/12 bẫy.
