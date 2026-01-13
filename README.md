# Introduction to Data Science - Final Project
**Giáo viên hướng dẫn:** Thầy Lê Nhựt Nam  
**Tên đề tài:** Phân tích các yếu tố ảnh hưởng đến hành vi mua hàng và dự đoán số lượng bán ra trên Tiki

**Thông tin nhóm:**  
23120025 - Phan Thị Phương Chi  
23120038 - Lê Hoàng Mỹ Hạ  
23120248 - Nguyễn Mạnh Hà  
23120257 - Bùi Trung Hiếu  

***

# TỔNG QUAN DỰ ÁN

## 1. Mục tiêu dự án
Bằng cách thu thập và phân tích dữ liệu thực tế, nhóm mong muốn tìm ra những insight quan trọng về mối quan hệ giữa giá cả, hình ảnh, nguồn gốc, uy tín cửa hàng với số lượng sản phẩm bán ra (`quantity_sold`). Cuối cùng, áp dụng các thuật toán máy học để xây dựng mô hình dự đoán doanh số, giúp người bán tối ưu hóa chiến lược kinh doanh.

## 2. Mô tả dự án

### 2.1. Thu thập dữ liệu
* **File thực hiện**:
    * `notebooks/data_collection.ipynb` & `src/data_collection.py`: Thực thi quy trình thu thập bằng các hàm gọi API Tiki.
* **Mô tả**: Quy trình thu thập dữ liệu được tự động hóa hoàn toàn thông qua việc gọi chuỗi các API của Tiki (Listing, Product Detail, Shop API, Performance API) (không sử dụng dataset có sẵn). Nhóm tập trung thu thập đa dạng các trường dữ liệu từ thông tin cơ bản, chỉ số bán hàng, đến các chỉ số uy tín của shop.

### 2.2. Khám phá và Tiền xử lý dữ liệu
* **File thực hiện**: 
    * `notebooks/data_preprocessing.ipynb` & `src/data_preprocessing.py`: Làm sạch dữ liệu, xử lý giá trị thiếu và chuẩn hóa dữ liệu.
    * `notebooks/data_exploration.ipynb` & `src/data_exploration.py`: Phân tích thống kê mô tả, kiểm tra phân phối và tương quan giữa các biến.
* **Mô tả**: Giai đoạn này tập trung vào việc làm sạch và chuẩn hóa dữ liệu thô: xử lý các giá trị bị khuyết, tách gộp thông tin, và loại bỏ các outlier nhiễu. Song song đó, nhóm sử dụng các biểu đồ thống kê (Histogram, Boxplot, Scatter Plot) để khám phá phân phối của các biến quan trọng như giá và lượt bán. Điều này giúp nhận diện sớm các vấn đề của dữ liệu và định hình các đặc trưng tiềm năng cho bài toán mô hình hóa.

### 2.3. Đặt câu hỏi và Trả lời
* **File thực hiện**: `notebooks/questions.ipynb`: Phân tích số liệu, vẽ biểu đồ và đưa ra nhận định cho 7 câu hỏi nghiên cứu.
* **Mô tả**: Nhóm đã xác định 7 câu hỏi nghiên cứu trọng tâm nhằm khai thác insight về hành vi khách hàng. Mỗi câu hỏi được trả lời bằng quy trình phân tích chặt chẽ: từ tính toán số liệu thống kê đến trực quan hóa bằng biểu đồ và rút ra kết luận. Kết quả giúp người bán hiểu rõ hơn về thị hiếu người dùng và các yếu tố quyết định đến số lượng bán hàng trên sàn Tiki.

### 2.4. Mô hình hóa dữ liệu
* **File thực hiện**:
    * `notebooks/modeling.ipynb` & `src/models.py`: Xây dựng các mô hình hồi quy cơ sở (Linear, Random Forest, XGBoost).
    * `notebooks/modeling_v2.ipynb` & `src/models_v2.py`: Cải thiện hiệu suất mô hình bằng chiến lược phân khúc dữ liệu (High/Low) và thử nghiệm thêm các thuật toán khác.
* **Mô tả**: Xây dựng pipeline học máy để dự đoán `quantity_sold`, bắt đầu từ việc lựa chọn feature và thử nghiệm các mô hình hồi quy như Linear Regression, Random Forest, XGBoost. 
* **<font color="orange">Xem xét từ ý kiến của thầy</font>**: Để giải quyết vấn đề phân phối lệch của dữ liệu bán hàng, nhóm đã phân tách dữ liệu thành High Segment và Low Segment. Việc huấn luyện riêng biệt trên từng phân khúc giúp tối ưu hóa độ chính xác (RMSE, R²) và đưa ra các dự báo sát thực tế hơn cho từng nhóm sản phẩm.

## 3. Tổ chức dự án
Cấu trúc thư mục chi tiết của dự án:
```
├── data/                         # Thư mục dữ liệu
│   ├── raw/                        # Dữ liệu thô thu thập từ API
│   ├── processed/                  # Dữ liệu trong quá trình Data Exploration và Data Preprocessing
│   ├── predicted/                  # Kết quả dự đoán từ mô hình
│   └── images/                     # Hình ảnh biểu đồ trực quan hóa
├── notebooks/                    # Các Jupyter Notebooks thực thi từng giai đoạn
│   ├── data_collection.ipynb       # Thu thập dữ liệu (Data Collection)
│   ├── data_preprocessing.ipynb    # Tiền xử lý dữ liệu (Data Preprocessing)
│   ├── data_exploration.ipynb      # Khám phá dữ liệu (Data Exploration)
│   ├── questions.ipynb             # Trả lời 7 câu hỏi phân tích chi tiết
│   ├── modeling.ipynb              # Huấn luyện và so sánh 3 mô hình cơ bản
│   └── modeling_v2.ipynb           # Mô hình cải tiến phân khúc High/Low
├── src/                          # Mã nguồn Python (các hàm xử lý chính)
│   ├── data_collection.py          # Hàm gọi API Tiki để lấy sản phẩm, shop
│   ├── data_preprocessing.py       # Các hàm làm sạch, xử lý null, chuẩn hóa
│   ├── data_exploration.py         # Các hàm vẽ biểu đồ, tính thống kê
│   ├── models.py                   # Hàm huấn luyện cho modeling.ipynb
│   └── models_v2.py                # Hàm huấn luyện cải tiến cho modeling_v2.ipynb
└── README.md                     # Tài liệu tổng quan dự án
```
***
