# Few-Shot Training & Testing on MVTec AD

## Overview
Em thực hiện **train** và **test** lại trên bộ dữ liệu **MVTec AD** với kĩ thuật few-shot.  
Em lưu checkpoints trên drive sau https://drive.google.com/drive/u/0/folders/1ch6Z-FwQntD23VKfRxwIuMXTqUwm-fUv.
Em sử dụng hai file chính:

- `train_few_mvtec.py` sửa từ `train_few.py`
- `test_few_mvtec.py` sửa từ `test_few.py`

## Các thay đổi chính
- **Loader** được chỉnh lại để chỉ load **normal data** (MVTec chỉ training trên dữ liệu bình thường).
- Bỏ hoàn toàn phần xử lý anomaly data trong training.
- Viết thêm file `dataset/mvtec_few.py` để config cho dataset MVTec.
- Trong `dataset/`, k-shot seed được lấy từ **WinCLIP** (k-shot = 5, chọn ngẫu nhiên từ tập train).
- MVFA-AD dùng **data augmentation** từ `utils.py` để sinh thêm dữ liệu training.
- Folder **CLIP/** để load pretrain **CLIP ViT** phục vụ huấn luyện.
- Dữ liệu MVTec được lưu trên Google Drive và unzip trực tiếp khi train/test trên Colab.

## Dataset
- Bộ dữ liệu: **MVTec AD**
- Cấu hình dữ liệu nằm trong: `dataset/mvtec_few.py`
- K-shot: **5-shot**
- Lấy mẫu ngẫu nhiên từ tập train theo cùng seed với WinCLIP.
- Tất cả dữ liệu train chỉ bao gồm ảnh **normal**, kết hợp với data augmentation.

## Notebook Training
- Notebook dùng để train:  
  [Google Colab Link](https://colab.research.google.com/drive/1VPN03REi1EVkOz_TWtlB7SXvoQwgUaGU#scrollTo=E-d8trXsMnMb)
- Train trên **GPU T4**.
- Dữ liệu MVTec được unzip từ Google Drive trước khi train.

## Performance (Results)

### 5 shot

| Class        | pAUROC   | AUROC   |
|-------------|---------|--------|
| carpet      | 0.989618 | 0.994382 |
| grid        | 0.931836 | 0.939850 |
| leather     | 0.992606 | 0.998981 |
| tile        | 0.949158 | 0.971140 |
| wood        | 0.963595 | 0.997368 |
| bottle      | 0.983490 | 0.998413 |
| cable       | 0.871252 | 0.622751 |
| capsule     | 0.908385 | 0.702832 |
| hazelnut    | 0.977062 | 0.989643 |
| metal_nut   | 0.906313 | 0.605572 |
| pill        | 0.988642 | 0.861702 |
| screw       | 0.975779 | 0.671244 |
| toothbrush  | 0.973655 | 0.925000 |
| transistor  | 0.791644 | 0.609167 |
| zipper      | 0.934509 | 0.914653 |

### 4 shot

| **Class**      | **pAUROC** | **AUROC** |
|----------------|-------------|------------|
| carpet         | 0.9908      | 0.9839     |
| grid           | 0.8199      | 0.7494     |
| leather        | 0.9924      | 0.9969     |
| tile           | 0.9284      | 0.9484     |
| wood           | 0.9598      | 0.9904     |
| bottle         | 0.9776      | 0.9976     |
| cable          | 0.8476      | 0.6529     |
| capsule        | 0.9242      | 0.6506     |
| hazelnut       | 0.9732      | 0.9886     |
| metal_nut      | 0.9193      | 0.5103     |
| pill           | 0.9839      | 0.8350     |
| screw          | —           | —          |
| toothbrush     | 0.9607      | 0.8167     |
| transistor     | 0.7693      | 0.6475     |
| zipper         | 0.9392      | 0.9112     |

### 10 shot

| **Class**      | **pAUROC** | **AUROC** |
|----------------|-------------|------------|
| carpet         | 0.9850      | 0.9555     |
| grid           | 0.9084      | 0.8329     |
| leather        | 0.9930      | 0.9990     |
| tile           | 0.9393      | 0.9394     |
| wood           | 0.9636      | 0.9930     |
| bottle         | 0.9832      | 0.9905     |
| cable          | 0.8196      | 0.6239     |
| capsule        | 0.9133      | 0.6406     |
| hazelnut       | 0.9826      | 0.9868     |
| metal_nut      | 0.9256      | 0.5401     |
| pill           | 0.9885      | 0.8541     |
| screw          | —           | —          |
| toothbrush     | 0.9776      | 0.9000     |
| transistor     | 0.8148      | 0.6717     |
| zipper         | 0.9337      | 0.8889     |


> **Note:**  
> - Trong quá trình train, em thấy performance chưa đạt như kết quả report trong paper gốc trên MVTec.  
> - Đánh giá của CRANE https://arxiv.org/pdf/2504.11055 trên AC tính ra probability bằng **cosine similarity giữa text feature và global image features**, sau đó qua softmax (`test.py` & `metrics.py`).  
> - Cách này khác với MVFA-AD em tiếp cận, sử dụng **max pooling trên mask** và chọn pixel có anomaly score cao nhất (`test.py`, hàm `test`).
> - Em chỉ train khác với MVFA-AD gốc là dùng k-shot bằng 5 thay vì là 4.
> - Dạ trung bình AUC của pixel-level là 94.25, trong khi image-level (AC) chỉ có 85.35
> - Dạ em có lưu lại checkpoint cho việc lưu trữ kết quả về sau, mỗi class trong MVTEC là một file `.pth`


## Chạy files
```bash
# Train
python train_few_mvtec.py --dataset mvtec --k_shot 5

# Test
python test_few_mvtec.py --dataset mvtec --k_shot 5
