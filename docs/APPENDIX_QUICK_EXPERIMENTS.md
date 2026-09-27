# Phụ lục phản biện 2026-09-27: hai thí nghiệm nhanh

Tài liệu này đối chiếu các số ở slide Phụ lục 24–27 của bản trình bày bảo vệ với ledger đã khử thông tin nhận diện và các báo cáo tổng hợp tại [`evidence/appendix_20260927/`](../evidence/appendix_20260927/). Đây là các phân tích bổ sung, không thay các bảng benchmark chính của luận văn. Chạy `python3 scripts/appendix_evidence/summarize.py` từ bất kỳ thư mục nào để tính lại bảng tóm tắt từ CSV/JSON đã commit; kết quả cố định ở [`derived_summary.json`](../evidence/appendix_20260927/derived_summary.json). [`SHA256SUMS.txt`](../evidence/appendix_20260927/SHA256SUMS.txt) khóa byte của từng artifact.

## 1. UPV-2025: chi phí và độ ổn định của MI trên dữ liệu lớn (slide 24–25)

| Câu hỏi | Kết quả và nguồn |
| --- | --- |
| Quy mô | 464.739 dòng, 81 biến thô; 325.483 dòng train theo nhóm sinh viên; 207 query trong cohort kiểm tra. `upv_2025_base/dataset_summary.json`, `upv_2025_author_alignment_v1/run_manifest.json`. |
| Phạm vi MI | 29 biến được phép thay đổi (28 numeric + `dedicacion`) tạo 406 cặp. `upv_2025_mi_sensitivity_20260927/{summary.json,rankings.csv}`. |
| Thời gian cho 406 cặp | 10.007 dòng: 1,81 giây; 100.002 dòng: 23,41 giây; full train 325.483 dòng: 131,08 giây, cùng 8 worker. `summary.json` trong thư mục MI sensitivity. |
| Độ ổn định so với 10k | Top 100 cặp numeric trùng 97/100 ở cả 100k và full train; Spearman toàn bộ 406 thứ hạng lần lượt 0,9863 và 0,9858. Riêng top 5 cặp với `dedicacion` chỉ trùng 1/5. `report.md`, `summary.json`, `rankings.csv` trong thư mục MI sensitivity. |
| Dùng train reference lớn khi sinh CF | Lượt author-alignment dùng cùng cohort 207 query, so sánh reference 10k và full train; *cả hai arm đều dùng MI tính trên 10k*. Trong full-train `SHARED_FF1`, median query 87,207 ms; các arm FF2/FF3 có median 64–66 ms. `upv_2025_author_alignment_v1/{run_manifest.json,report.md,scaling_summary.csv}`. |

MI sensitivity chỉ tính lại thứ hạng MI, không sinh CF. Số 131 giây là wall time khi phân 406 phép tính cho 8 worker, không phải chi phí mỗi query. Mẫu 10k và 100k chọn theo nhóm sinh viên với seed 42. Điểm MI của `dedicacion` có giới hạn: biến được mã hóa ordinal và estimator xử lý như continuous. Báo cáo chi tiết còn ghi top 5 categorical kém ổn định; không dùng tỷ lệ top 100 numeric để tuyên bố mọi cặp đều ổn định.

**Cổng mô hình UPV:** `upv_2025_base/model_quality_gate.json` và bản sao trong lượt author-alignment đều ghi `passed=false`, `status=STOP`, nguyên nhân `CONVERGENCE_WARNING` của LogisticRegression. `model_metrics.json` ghi ROC-AUC 0,885 và PR-AUC 0,437 nhưng các số này chỉ mô tả snapshot mô hình, không xóa trạng thái STOP. Lượt author-alignment là chẩn đoán bổ sung trên snapshot này; không xem kết quả CF của nó như benchmark đã vượt cổng chất lượng. `dedicacion` đổi TC/TP là kiểm tra khả năng thuật toán, không phải can thiệp giáo dục đã được xác nhận.

Mã nguồn: `scripts/external_binary_eval/` và `ufce/ufce_ff/`. `run_manifest.json` của lượt author-alignment ghi checksum của `adapters.py` khác file hiện có, vì vậy cũng không khẳng định một lượt chạy lại bằng source hiện tại sẽ tái tạo byte-for-byte kết quả lịch sử. Để tính lại phần MI khi có ba archive UPV theo Zenodo: `./.venv/bin/python -m scripts.external_binary_eval.mi_sample_sensitivity --workers 8`. Loader trong `dataset.py` kiểm tra MD5 công bố; archive dữ liệu gốc không được đưa vào Git. Chi tiết đường dẫn và tham số của lượt author-alignment nằm trong `scripts/external_binary_eval/README.md` và các manifest đã commit.

## 2. Student Outcomes: một mô hình đa lớp nguyên bản (slide 26–27)

Bộ UCI 697 có 4.424 học viên và 36 biến đầu vào. Một LogisticRegression đa thức dự đoán trực tiếp ba lớp Dropout, Enrolled, Graduate. Chỉ tám biến kết quả học tập học kỳ 1/2 được phép thay đổi; các biến còn lại bất biến. Split 70/15/15 theo lớp, mô hình và policy được khóa trước final test. Baseline hoàn tất 1.328 cặp `(query, target)` trên 664 dòng test; `student_outcomes_20260924_resume1/{freeze_manifest.json,final_report.json,run_status.json}` là bằng chứng.

Bảng slide chỉ xét 546 cặp theo ba hướng **Dropout→Enrolled (190), Dropout→Graduate (190), Enrolled→Graduate (166)**. `F` = có ít nhất một candidate đúng target và khả thi theo ràng buộc; `X` = có ít nhất một CF qua cổng để xuất ra. Thời gian là tổng `runtime_ms` của 546 cặp chia 1.000; `giây/F` là tổng đó chia số cặp có F. Setup adapter và precompute MI không nằm trong đồng hồ query.

| Phương pháp | F / 546 | X / 546 | Tổng runtime | Giây/F |
| --- | ---: | ---: | ---: | ---: |
| UFCE-FF1 | 81 (14,8%) | 81 (14,8%) | 258,3 s | 3,2 |
| UFCE-FF2 | 77 (14,1%) | 77 (14,1%) | 860,2 s | 11,2 |
| UFCE-FF3 | 68 (12,5%) | 68 (12,5%) | 865,2 s | 12,7 |
| DiCE random | 90 (16,5%) | 25 (4,6%) | 2.404,0 s | 26,7 |
| DiCE genetic | 106 (19,4%) | 100 (18,3%) | 24.136,6 s | 227,7 |
| DiCE kdtree | 0 (0%) | 0 (0%) | 219,8 s | — |

Trong 546 cặp có **112 cặp** mà ít nhất một phương pháp tìm được candidate khả thi; UFCE-FF1 tìm được 81/112 cặp đó. Nguồn công khai: `multiclass_improvement_ledger.csv` gồm 546 cặp × 6 phương pháp, chỉ có chỉ số cặp ngẫu nhiên, F, X, thời gian và trạng thái; được tạo từ baseline `final_test_query_results.csv` và supplement `final_test_query_results.partial.csv` lưu tại local. Script `sanitize.py` mô tả phép biến đổi, còn `LOCAL_RAW_SHA256SUMS.txt` ghi checksum của log gốc. DiCE genetic có **380/546** query báo `TIMEOUT`; số F có thể đến từ candidate tìm được trước timeout. DiCE random có 90 F nhưng chỉ 25 X vì cổng xuất bản kiểm tra lại điều kiện. Không đồng nhất F với X.

**Phạm vi hoàn tất:** baseline bốn phương pháp hoàn tất cả 1.328 cặp. Supplement DiCE genetic/kdtree là hậu kiểm sau baseline, có log local `.partial.csv` vì lượt chạy toàn bộ 1.328 cặp/method chưa hoàn tất; riêng cả ba hướng cải thiện đã hoàn tất đủ 546/546 cho *mỗi* backend. Không trình bày kết quả supplement như kết quả hoàn tất sáu hướng. `supplement_manifest.json`, `supplement_freeze.json` và ledger khử nhận diện xác định phạm vi công bố.

**Tính tái lập mã nguồn:** `freeze_manifest.json` của baseline ghi SHA-256 của source tại thời điểm chạy. Các file runtime hiện có khớp checksum đã khóa trừ `scripts/native_multiclass_eval/config.py`, `scripts/native_multiclass_eval/runner.py`, và `ufce/ufce_ff/ufce.py`; chúng được sửa sau baseline. File kiểm thử được liệt kê trong freeze manifest không nằm trong gói phát hành này (trong đó có MI cache và cache reference index). Không có bản source byte-for-byte của ba file cũ trong gói; một lượt chạy mới bằng source hiện tại không thể được khẳng định có cùng thời gian lịch sử. Ngược lại, source của supplement DiCE khớp toàn bộ `source_sha256` trong `supplement_manifest.json`. Báo cáo [`order_effect_20260927.md`](../evidence/appendix_20260927/order_effect_20260927.md) và hai ledger order-ablation kiểm tra FF2/FF3 trên cùng 546 cặp dưới source hiện tại; so sánh thứ tự không thay kết quả F/X nhưng không dùng thời gian mới thay số lịch sử.

Mã nguồn: `scripts/native_multiclass_eval/` và `ufce/ufce_ff/`. UCI archive không được commit; loader có URL và checksum dữ liệu trong freeze manifest. Xem `scripts/native_multiclass_eval/README.md` để chạy lại baseline, MI cache và supplement. Các file joblib trong evidence là artifact mô hình của lượt chạy, chỉ load trong môi trường tin cậy.

## Cấu trúc và giới hạn công bố

`evidence/appendix_20260927/` giữ ledger ghép cặp đã khử nhận diện, báo cáo tổng hợp, manifest có checksum mô hình và kết quả MI đủ để tính lại các số trên slide. Bộ kiểm duyệt tự động đã chặn công bố log candidate có thuộc tính cá nhân/học tập theo từng sinh viên và bundle mô hình học từ dữ liệu giáo dục. Vì vậy log query/candidate gốc, bundle mô hình, bảng ID, raw dataset archive, checkpoint dư và bản `.orig`/`.rej` chỉ nằm local; manifest vẫn giữ SHA-256 mô hình. `LOCAL_RAW_SHA256SUMS.txt` giữ checksum để đối chiếu với bản local mà không công bố hồ sơ cá nhân; `SHA256SUMS.txt` khóa các file thực sự được phát hành. Đường dẫn tuyệt đối trong manifest ghi máy chạy ban đầu; sau khi clone, dùng đường dẫn tương đối dưới `evidence/appendix_20260927/` theo đúng tên thư mục run. Lượt chạy lại phụ thuộc môi trường Python/OS và thư viện được ghi trong manifest nên thời gian wall clock có thể đổi.
