# 🐾 Animal Recognition System V3 (Hybrid MobileNetV2 + XGBoost)

Dự án xây dựng hệ thống nhận dạng và phân loại động vật sử dụng bộ dữ liệu **Animal-10 Dataset** (bao gồm khoảng 28,000 hình ảnh tự nhiên của 10 loài động vật khác nhau). 

Để xử lý bài toán phân loại hình ảnh quy mô lớn này một cách hiệu quả nhất, dự án áp dụng kiến trúc kết hợp (Hybrid Architecture): **MobileNetV2 làm Feature Extractor** kết hợp với **XGBoost Classifier** để tối ưu hóa độ chính xác và tốc độ xử lý, tích hợp giao diện người dùng trực quan bằng **Tkinter**.

---

## 🛠️ Kiến trúc hệ thống & Phương pháp kỹ thuật

Với một bộ dữ liệu lớn và nhiều nhiễu như **Animal-10** (ảnh chụp trong điều kiện tự nhiên, ánh sáng và góc nhìn thay đổi phức tạp), việc sử dụng các mạng CNN thuần thông thường dễ dẫn đến hiện tượng Overfitting hoặc tốn rất nhiều tài nguyên huấn luyện. Do đó, dự án áp dụng giải pháp lai (Hybrid Model):

1. **Trích xuất đặc trưng (Feature Extraction):** Sử dụng mô hình `MobileNetV2` đã được pre-trained trên tập ImageNet khổng lồ. Bằng cách đóng băng các tầng dưới và loại bỏ tầng phân loại gốc (`include_top=False`), mô hình có khả năng trích xuất các đặc trưng cao cấp (đường nét, hình khối của động vật) cực tốt. Ảnh đầu vào kích thước (128, 128, 3) truyền qua mạng sẽ được nén thành vector đặc trưng 256 chiều tại tầng `feature_layer`.
2. **Chuẩn hóa dữ liệu:** Vector đặc trưng được đưa qua `StandardScaler` nhằm đồng bộ phân phối dữ liệu đầu vào cho bộ phân loại phía sau.
3. **Bộ phân loại nâng cao (Classification):** Thay vì dùng tầng Dense + Softmax truyền thống, vector đặc trưng được phân loại bằng mạng thực thể cây quyết định tăng cường `XGBoost Classifier`. Sự kết hợp này tận dụng tối đa sức mạnh biểu diễn không gian của CNN và khả năng phân tách ranh giới dữ liệu cực mạnh của Gradient Boosting, giúp mô hình đạt độ chính xác cao và tối ưu thời gian suy luận (inference time).

---

## 📁 Cấu trúc thư mục dự án

```text
├── models/                  # Thư mục chứa các file trọng số (hướng dẫn tải bên dưới)
├── src/                     # Mã nguồn chính của ứng dụng
│   ├── __init__.py          # File đánh dấu package
│   ├── model_pipeline.py    # Pipeline xử lý AI (Load model, Preprocessing, Predict)
│   └── app_gui.py           # Giao diện người dùng Tkinter (Chạy file này)
├── README.md                # Tài liệu hướng dẫn dự án
└── requirements.txt         # Danh sách các thư viện cần thiết
git clone https://github.com/phuocle123123-sys/animal-recognition.git
cd animal-recognition
pip install -r requirements.txt