
----- Train train-vit-custom.py -----

#### Huấn luyện 
chạy gpu, e.g.,--device '0', '0,1', or 'cpu'

# TH01( Kiến trúc gốc ViT-B/16 ): giữ nguyên kiến trúc của vitb-16 và huấn luyện từ đầu với CIFAR100, batch size 32 giống với pretrained
## version 01: pz16-img_sz224-h12-d12-hid768-mlp768*4 (mặc định ratio gấp 4) (vitb-16)

'''
python train-vit-custom-2.py --train 1 --epochs 100 --batch_size 32 --lr 0.0001 --patch_size 16 --embed_dim 768 --num_heads 12 --depth 12 --image_sz 224 --save checkpoints --save_freq 5 --device 1 --name v1_pz16-img224-h12-d12-hd768-2
''' 

Ưu điểm:
- Kiến trúc này đã được chứng minh hiệu quả trên các bộ dữ liệu lớn như ImageNet.
- Dùng kích thước ảnh 224x224 và patch size 16 đảm bảo độ chi tiết phù hợp.

Hạn chế:
- Kiến trúc phức tạp, đòi hỏi tài nguyên lớn (bộ nhớ và thời gian huấn luyện).
- CIFAR-100 là tập dữ liệu nhỏ hơn (32x32), nên việc upscale lên 224x224 có thể không mang lại cải thiện đáng kể.

Dự đoán: 
- Mô hình có thể đạt được kết quả tốt nhất trong 3 trường hợp nhờ kiến trúc phức tạp và khả năng biểu diễn mạnh.
- Tuy nhiên, yêu cầu tài nguyên cao, thời gian huấn luyện lâu.
- Có nguy cơ overfitting nếu không áp dụng đủ augmentation hoặc regularization.

Thực tế:
- Training time: 5867.10s
- Accuracy: 46.61%
- Tài nguyên sử dụng: 1404.97 MB 5172MiB / 11264MiB

# TH02 (Giảm thông số): giảm patch size, resize chỉ gấp đôi kích thước, giảm heads, giảm layers, giảm hidden layer thành 512, có nghĩa mlp sẽ có 2072 neuron 
## version 02: pz8-img_sz64-h8-d8-hid512-mlp512*4 (mặc định ratio gấp 4)    

"""
python train-vit-custom-2.py --train 1 --epochs 100 --batch_size 32 --lr 0.0001 --patch_size 8 --embed_dim 512 --num_heads 8 --depth 8 --image_sz 64 --save checkpoints --save_freq 5 --device 2 --name v2_pz8-img64-h8-d8-hd512-2
"""

Ưu điểm:
- Patch nhỏ giúp mô hình bắt được các chi tiết cục bộ hơn so với patch 16x16.
- Giảm num_heads và embed_dim giúp mô hình nhẹ hơn, giảm yêu cầu tài nguyên.
- Phù hợp hơn với tập dữ liệu nhỏ như CIFAR-100.

Hạn chế:
- Số lượng thông số ít có thể làm giảm khả năng học các đặc trưng phức tạp.

Dự đoán:
- Kết quả có thể thấp hơn TH01 nhưng mô hình sẽ nhanh hơn nhiều, sử dụng ít tài nguyên hơn.
- Patch nhỏ (8x8) và kích thước ảnh 64x64 phù hợp hơn với CIFAR-100.
- Đây là thiết lập cân bằng giữa hiệu quả và hiệu suất.

Thực tế:
- Training time: 740.86 s
- Accuracy: 47.06%
- Tài nguyên sử dụng: 431 MB 1258MiB / 11264MiB


# TH03 (Tăng depth và heads): giống với th02 nhưng ta tăng heads và layers
## version 03: pz8-img64-h16-d16-hid512-mlp512*4 (mặc định ratio gấp 4)

"""
python train-vit-custom.py --train 1 --epochs 100 --batch_size 32 --lr 0.0001 --patch_size 8 --embed_dim 512 --num_heads 16 --depth 16 --image_sz 64 --save checkpoints --save_freq 5 --device 3 --name v3_pz8-img64-h16-d16-hd512
"""

Ưu điểm:
- Số lớp và số heads tăng giúp mô hình học được các đặc trưng phức tạp hơn.
- Kích thước ảnh và patch size giữ nguyên như TH02, vẫn tối ưu cho CIFAR-100.

Hạn chế:
- Tăng số layers và heads làm tăng số lượng thông số, có thể dẫn đến overfitting trên dữ liệu nhỏ.

Dự đoán:
- Hiệu suất có thể cải thiện so với TH02 nhờ số layers và heads tăng.
- Tuy nhiên, nguy cơ overfitting vẫn cao, và tài nguyên cần thiết sẽ gần với TH01.

Thực tế:
- Training time: 4:01:56
- Accuracy: 98.88%
- Tài nguyên sử dụng: 2182MiB / 11264MiB   839.14 MB

#### Tiếp tục huấn luyện nếu gặp gián đoạn
python train-vit-custom.py --train 1 --resume checkpoints/checkpoint_epoch_10.pth

### Test từ checkpoint tốt nhất
python train-vit-custom.py --train 0 --save checkpoints --name experiment_name

python train-vit-custom.py --train 0 --save checkpoints --name v3_pz8-img64-h16-d16-hd512-2



----------------------------------------------------------------------------------
----- Train train-vit-pure.py ---------


Xem danh sách các mô hình có sẵn: "https://github.com/huggingface/pytorch-image-models/blob/5f9aff395c224492e9e44248b15f44b5cc095d9c/timm/models/vision_transformer.py#L503" 
### Huấn luyện 

python train-vit-pure.py --name vit-b16 --epochs 10 --batch_size 32

python train-vit-pure-2.py --train 1 --epochs 100 --batch_size 32 --lr 0.0001 \
--model_name vit_base_patch16_224 --image_sz 224 \
--save checkpoints --name vit-b16-2 --save_freq 2 --device 0    


### Tiếp tục huấn luyện nếu gặp gián đoạn
python train-vit-pure.py --train 1 --resume checkpoints/checkpoint_epoch_10.pth

### Test từ checkpoint tốt nhất
python train-vit-pure.py --train 0 --save checkpoints


Thực tế:
- Training time: 5944.89s
- Accuracy:
- Tài nguyên sử dụng: 5172MiB / 11264MiB 



--------
Khi so sánh trên bộ gốc có hidden layer là 768 nếu mô hình checkpoint có kiến trúc 512 thì ko thể test, cần load mô hình test với kiến trúc 512