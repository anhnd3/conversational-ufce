# UPV-2025: độ nhạy của xếp hạng MI theo số dòng train

Ngày 2026-09-27. Dùng đúng train split theo student, `EncodedSpace`, 29 biến mutable (28 numeric `MODERATE` + `dedicacion`), `mutual_info_regression` của scikit-learn 0.24.2, `random_state=0`, 406 cặp. Mẫu 10k và 100k chọn theo student groups với seed 42; do cùng thứ tự shuffle, mẫu 10k nằm trong mẫu 100k. Full train gồm 325.483 dòng. Tám tiến trình chỉ phân phối các cặp độc lập, không thay estimator hoặc điểm MI.

| MI reference | Dòng thực tế | Wall time cho 406 cặp | Top 100 numeric trùng với 10k | Top 5 numeric trùng | Top 5 `dedicacion` trùng | Tổng 105 cặp trùng | Spearman trên 406 thứ hạng |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 10.007 | 1,81 s | — | — | — | — | — |
| 100k | 100.002 | 23,41 s | 97/100 | 4/5 | 1/5 | 98/105 = 93,3% | 0,9863 |
| Full train | 325.483 | 131,08 s | 97/100 | 4/5 | 1/5 | 98/105 = 93,3% | 0,9858 |

Giữa 100k và full train, top 100 numeric và top 5 numeric có đúng cùng thành phần; top 5 `dedicacion` chỉ trùng 2/5. Top 100 numeric là lựa chọn khá ổn định trên các kích thước này. Nhánh categorical top 5 thay đổi đáng kể và điểm MI rất nhỏ (full top 5: 0,00419–0,00488), vì vậy không nên gộp 105 cặp thành một tỷ lệ rồi kết luận mọi nhánh đều ổn định. Top 5 numeric có 4 cặp chung giữa 10k và full; thứ tự cũng thay đổi.

Tổng thời gian của 406 lần gọi estimator trên full train là 1.040 giây trên 8 worker; trung vị một lần gọi là 2,53 giây. Pilot cũ chỉ chạy tuần tự 378 cặp numeric trên full train và đã được dừng thủ công sau hơn 720 giây khi chưa xong. Đây là chi phí setup của cài đặt gọi từng cặp một, không phải timeout mỗi truy vấn hay giới hạn cứng của MI. Số 131 giây là wall time của phép chạy song song mới, không phải thời gian của pilot tuần tự cũ.

Kiểm tra tái lập: bảng encoded của 10k khớp SHA-256 trong artifact gốc; toàn bộ 406 cặp có đúng thứ tự và điểm MI gốc (sai khác số học lớn nhất 8,33e-17); không có estimator call lỗi. Chỉ tính MI, không sinh lại CF hoặc đo lại tỷ lệ CF khả thi. `dedicacion` đang được mã hóa ordinal rồi được estimator coi như continuous, nên điểm của cặp categorical–numeric có giới hạn phương pháp luận. Phép này không kiểm tra global top 5 của toàn bộ 81 feature; với bộ lọc giữ hai feature đều mutable, thứ hạng các cặp mutable được tính độc lập và có thể đối chiếu trực tiếp với 406 cặp ở đây.

Đầu ra chi tiết: `summary.json`, `rankings.csv`. Tái lập: `.venv/bin/python -m scripts.external_binary_eval.mi_sample_sensitivity --workers 8`.
