# Setup

## Nghiên cứu sử dụng CIFAR-100 dataset. Để tải và sử dụng:

```bash
  pip install requests
  python ./data/download_cifar100.py
```

## Folder `attention`

   chứa 2 folder con là `T_pred` và `F_pred` trực quan hóa attention maps ở từng layers, heads và tổng thể.

## Folder `./model` chứa các file tạo dựng biến thể mô hình vit.

   Tham khảo từ thư viện timm `https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py`

## Folder `./notebooks`

   chứa folder `EDA` và các file `.ipynb` dùng để xử lý ảnh và trực quan hóa một số thử nghiệm ban đầu.

## Folder `./results`

   chứa các biến thể kịch bản mô hình vit, ở mỗi folder con sẽ chứa kết quả đánh giá dưới dạng ảnh plot gồm confusion matrix, loss distribution, train/test accuracy và train/test loss.

## Sử dụng script trong `./script `

   đọc file `README.md` trong thư mục để chạy lại các task nếu cần.

## Folder `./weights`

   chứa các mô hình đã huấn luyện qua 100 epochs. Lưu ở link `./link_to_model_list.txt` nếu folder rỗng
