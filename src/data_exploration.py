import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

# Bỏ qua các cảnh báo không mong muốn trong quá trình thực thi
warnings.filterwarnings('ignore')

# Cấu hình Matplotlib để hiển thị tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans' 

# Thiết lập để xử lý dấu trừ Unicode (đảm bảo hiển thị số âm đúng)
plt.rcParams['axes.unicode_minus'] = False

# ==================== ĐỌC FILE DATA ====================

def load_data(file_path):
    """
    Chức năng:
        - Đọc dữ liệu từ file csv đã được cung cấp đường dẫn
        - In ra các thông tin cơ bản của dataframe đọc được từ csv

    Tham số:
        - file_path: Đường dẫn tới file csv
    
    Giá trị trả về:
        - Dataframe chứa dữ liệu
    """
    try:
        # Sử dụng pandas.read_csv để đọc dữ liệu từ file
        df = pd.read_csv(file_path)
        print(f"Đã đọc file: {file_path}")
        print(f"Số dòng: {len(df)}")
        print(f"Số cột: {len(df.columns)}")
        return df
    except FileNotFoundError: # Lỗi file không tồn tại
        print(f"Lỗi: Không tìm thấy file {file_path}")
        raise
    except Exception as e: # Các lỗi ngoại lệ khác
        print(f"Lỗi khi đọc file {file_path}: {str(e)}")
        raise

# ==================== XỬ LÝ KIỂU DỮ LIỆU ====================

def process_data_types(input_path='../data/raw/data.csv', 
                       output_path='../data/processed/data_typed.csv'):
    """
    Chức năng: 
        - Đọc dữ liệu từ file csv gốc
        - Thực hiện việc xử lý kiểu dữ liệu của các cột
            + Chuyển đổi 2 cột chứa ký tự phần trăm "%" (xóa "%" và chuyển sang float64)
            + Chuyển các cột số đang ở định dạng object về float64
            + Chuyển đổi các giá trị không hợp lệ thành NaN
    
    Tham số:
        - input_path: Đường dẫn tới file csv gốc
        - output_path: Đường dẫn lưu file sau khi đã xử lý
    
    Giá trị trả về:
        - Dataframe đã xử lý kiểu dữ liệu
    """
    print("=" * 60)
    print("XỬ LÝ KIỂU DỮ LIỆU")
    print("=" * 60)
    
    # Gọi hàm load_data để đọc dữ liệu
    df = load_data(input_path)

    print("\n--- Dữ liệu trước khi xử lý kiểu dữ liệu ---")
    print(df.dtypes)
    print(df.describe())
    df.info()
    
    # Xử lý cột chứa ký tự "%"
    print("\n--- Xử lý các cột chứa ký tự '%' ---")
    # Xử lý cột 'cancel_by_seller_rate' và 'return_rate'
    percent_columns = ['cancel_by_seller_rate', 'return_rate']
    
    for col in percent_columns:
        if col in df.columns: # Kiểm tra cột có tồn tại trong dataframe không
            # Đếm số lượng giá trị không phải NaN
            non_null_count = df[col].notna().sum()
            
            if non_null_count > 0:
                # Kiểm tra xem có chứa ký tự '%' không
                has_percent = df[col].astype(str).str.contains('%').any()
                
                # Xử lý chuyển đổi
                if has_percent:
                    print(f"Đang xử lý cột '{col}'...")

                    # Loại bỏ ký tự '%'
                    df[col] = df[col].astype(str).str.replace('%', '', regex=False)

                    # Loại bỏ khoảng trắng
                    df[col] = df[col].str.strip()

                    # Chuyển đổi sang số, không hợp lệ sẽ thành NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Chia cho 100 để chuyển từ phần trăm sang số thập phân
                    df[col] = df[col] / 100
                else:
                    # Nếu không có '%', thử chuyển trực tiếp sang số
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Đảm bảo kiểu dữ liệu cuối cùng là float64
            df[col] = df[col].astype('float64')
    
    # Đưa các cột số về đúng kiểu dữ liệu
    print("\n--- Chuyển đổi các cột số về kiểu float64 ---")
    # Danh sách các cột số cần kiểm tra
    numeric_columns_to_check = [
        'price', 'original_price', 'discount_rate', 'quantity_sold',
        'rating_average', 'review_count', 'is_return_policy',
        'is_freeship_xtra', 'is_authentic', 'image_count', 'video_count',
        'is_brand', 'store_review_count', 'total_follower', 'is_official',
    ]
    
    for col in numeric_columns_to_check:
        if col in df.columns:
            if df[col].dtype == 'object':
                print(f"Cột '{col}' có kiểu object, đang chuyển sang numeric...")
                
                # Đếm số giá trị không phải NaN trước khi chuyển
                before_count = df[col].notna().sum()

                # Chuyển đổi sang numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')

                # Đếm số giá trị không phải NaN sau khi chuyển
                after_count = df[col].notna().sum()
                
                # Số lượng giá trị bị mất hoặc chuyển thành NaN
                lost_values = before_count - after_count
                if lost_values > 0:
                    print(f"Mất {lost_values} giá trị không thể chuyển đổi")
                
                print(f"Kiểu dữ liệu sau: {df[col].dtype}")
            
            # Đảm bảo kiểu dữ liệu cuối cùng là float64
            df[col] = df[col].astype('float64')
    
    # Lưu file
    df.to_csv(output_path, index=False)
    print(f"\nĐã lưu file đã xử lý tại: {output_path}")
    
    # Hiển thị kiểu dữ liệu sau khi xử lý
    print("\n--- Dữ liệu sau khi xử lý kiểu dữ liệu ---")
    print(df.dtypes)
    print(df.describe())
    df.info()
    
    return df

# ==================== BASIC STATISTICS ====================

def calculate_descriptive_stats(df, numerical_cols):
    """
    Chức năng:
        - Tính toán thống kê mô tả cơ bản cho các cột số
            + Trung bình (mean)
            + Trung vị (median)
            + Độ lệch chuẩn (std)
            + Min
            + Max
    
    Tham số:
        - df: Dataframe đầu vào
        - numerical_cols: danh sách các cột số cần tính toán thống kê
    
    Giá trị trả về:
        - Dataframe chứa thống kê mô tả
    """
    print("--- Bắt đầu tính toán Thống kê Mô tả ---")
    
    # Kiểm tra các cột có tồn tại
    existing_cols = [col for col in numerical_cols if col in df.columns]
    if len(existing_cols) == 0:
        print("Cảnh báo: Không có cột số nào tồn tại để tính thống kê")
        return pd.DataFrame()
    # Tính toán các thống kê (mean, median, std, min, max) cho các cột tồn tại
    # Sử dụng list các tên hàm string thay vì đối tượng numpy để đảm bảo tên cột
    stats = df[existing_cols].agg(['mean', 'median', 'std', 'min', 'max']).T.reset_index()
    stats.columns = ['Thuộc tính', 'mean', 'median', 'std', 'min', 'max']
    
    # Dictionary ánh xạ tên cột sang tên hiển thị
    display_name_map = {
        'price': 'Giá bán (VND)', 
        'original_price': 'Giá gốc (VND)', 
        'discount_rate': 'Tỷ lệ giảm giá (%)',
        'quantity_sold': 'SL Đã bán', 
        'rating_average': 'Rating TB', 
        'review_count': 'Số lượng Review',
        'image_count': 'Số lượng Ảnh', 
        'video_count': 'Số lượng Video', 
        'store_review_count': 'Review Shop', 
        'total_follower': 'Follower Shop'
    }

    # Định dạng dữ liệu cho dễ đọc
    for index, row in stats.iterrows():
        # Cập nhật tên hiển thị bằng tiếng Việt
        stats.at[index, 'Thuộc tính'] = display_name_map.get(row['Thuộc tính'], row['Thuộc tính'])
        
        if 'VND' in stats.at[index, 'Thuộc tính']:
            # Định dạng tiền tệ (số nguyên, có dấu phân cách hàng nghìn)
            for col in ['mean', 'median', 'std', 'min', 'max']:
                stats.at[index, col] = f"{row[col]:,.0f} VND"
        elif 'Tỷ lệ' in stats.at[index, 'Thuộc tính'] or 'Rating' in stats.at[index, 'Thuộc tính']:
            # Định dạng tỉ lệ/rating (2 chữ số thập phân)
            for col in ['mean', 'median', 'std', 'min', 'max']:
                stats.at[index, col] = f"{row[col]:,.2f}"
        else: 
            # Định dạng số nguyên, có dấu phân cách hàng nghìn
            for col in ['mean', 'median', 'std', 'min', 'max']:
                stats.at[index, col] = f"{row[col]:,.0f}"

    # Đổi tên cột sang tiếng Việt
    stats.columns = ['Thuộc tính', 'Trung bình (Mean)', 'Trung vị (Median)', 
                     'Độ lệch chuẩn (Std Dev)', 'Giá trị Tối thiểu (Min)', 
                     'Giá trị Tối đa (Max)']
            
    print("--- Thống kê Mô tả hoàn tất ---")
    return stats

# ==================== VISUALIZATION - HISTOGRAM ====================

def plot_all_numerical_distributions(df, numerical_cols, output_dir='../data/images'):
    """
    Chức năng:
        - Vẽ histogram cho tất cả các cột số
    
    Tham số:
        - df: Dataframe đầu vào
        - numerical_cols: danh sách các cột số cần vẽ histogram
        - output_dir: Thư mục lưu hình biểu đồ sau khi vẽ
    """
    # Kiểm tra các cột có tồn tại
    existing_cols = [col for col in numerical_cols if col in df.columns]
    if len(existing_cols) == 0:
        print("Cảnh báo: Không có cột số nào tồn tại để vẽ histogram")
        return
    
    # Cấu hình biểu đồ (5 hàng, 2 cột) cho 10 biến
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    # Biến ma trận 5x2 thành 1 mảng 1D để dễ lặp
    axes = axes.flatten()
    
    # Cập nhật numerical_cols để chỉ dùng các cột tồn tại
    numerical_cols = existing_cols
    
    # Các cột cần dùng Log Transformation
    log_scale_cols = ['price', 'original_price', 'quantity_sold', 
                      'review_count', 'store_review_count', 'total_follower']

    # Dictionary ánh xạ tên cột gốc sang tên hiển thị
    display_name_map = {
        'price': 'Giá bán (VND)', 
        'original_price': 'Giá gốc (VND)', 
        'discount_rate': 'Tỷ lệ giảm giá (%)',
        'quantity_sold': 'SL Đã bán', 
        'rating_average': 'Rating TB', 
        'review_count': 'Số lượng Review',
        'image_count': 'Số lượng Ảnh', 
        'video_count': 'Số lượng Video', 
        'store_review_count': 'Review Shop', 
        'total_follower': 'Follower Shop'
    }

    for i, col in enumerate(numerical_cols):
        is_log = col in log_scale_cols

        # Lấy dữ liệu và loại bỏ NaN
        plot_data = df[col].copy().dropna()
        
        # Kiểm tra có dữ liệu hợp lệ không
        if len(plot_data) == 0:
            axes[i].text(0.5, 0.5, 'Không có dữ liệu', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'Phân phối {display_name_map.get(col, col)} (Không có dữ liệu)', fontsize=14)
            continue
        
        if is_log:
            # np.log1p(x) tính log(1+x), an toàn khi có giá trị 0
            # Dữ liệu cần được clip để đảm bảo không có giá trị âm nếu chưa xử lý
            plot_data = plot_data.clip(lower=0) 
            plot_data = np.log1p(plot_data)

        # Vẽ Histogram trên dữ liệu đã được transform (nếu có)
        sns.histplot(plot_data, kde=True, ax=axes[i], bins=20, color='teal' if is_log else 'salmon')
        
        # Đặt tiêu đề và nhãn
        title = display_name_map.get(col, col)

        # Cập nhật nhãn cho cột đã được Log Transform
        axes[i].set_title(f'Phân phối {title}', fontsize=14)
        axes[i].set_xlabel(title + (' (Log Scale)' if is_log else ''), fontsize=10)
        axes[i].set_ylabel('Tần suất', fontsize=10)

        # Thêm đường Median
        original_data = df[col].dropna()
        if len(original_data) > 0:
            if is_log:
                # Tính median trên dữ liệu đã log transform
                original_median = original_data.median()
                if not np.isnan(original_median):
                    median_val = np.log1p(max(0, original_median))
                    median_label = f'Trung vị: {original_median:,.0f}'
                    axes[i].axvline(median_val, color='red', linestyle='--', label=median_label)
            else:
                # Tính median trên dữ liệu gốc và vẽ
                median_val = original_data.median()
                if not np.isnan(median_val):
                    median_label = f'Trung vị: {median_val:,.0f}'
                    axes[i].axvline(median_val, color='red', linestyle='--', label=median_label)
            axes[i].legend()

    # Tùy chỉnh Layout
    plt.suptitle('Phân phối của 10 Thuộc tính Số Quan trọng', fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 1.0])
    
    # Lưu ảnh
    output_path = f'{output_dir}/all_numerical_distributions_histogram.png'
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ Histogram vào: {output_path}")
    plt.close(fig)

# ==================== VISUALIZATION - HEATMAP ====================

def plot_correlation_heatmap(df, numerical_cols, output_dir='../data/images'):
    """
    Chức năng: 
        - Vẽ Heatmap tương quan giữa các biến số
    
    Tham số:
        - df: Dataframe đầu vào
        - numerical_cols: danh sách các cột số cần vẽ tương quan
        - output_dir: Thư mục lưu hình
    """
    # Kiểm tra các cột có tồn tại
    existing_cols = [col for col in numerical_cols if col in df.columns]
    if len(existing_cols) == 0:
        print("Cảnh báo: Không có cột số nào tồn tại để vẽ heatmap")
        return
    if len(existing_cols) < 2:
        print("Cảnh báo: Cần ít nhất 2 cột để tính correlation")
        return
    
    plt.figure(figsize=(15, 20))
    # Tính ma trận tương quan (Correlation Matrix)
    corr_matrix = df[existing_cols].corr()
    
    # Vẽ Heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                cbar=True, linewidths=.5, linecolor='black')
    
    plt.title('Heatmap Ma trận Tương quan giữa các Thuộc tính Số', fontsize=18)
    # Xoay nhãn x để tránh chồng chéo
    plt.xticks(rotation=40, ha='right')
    plt.yticks(rotation=0)
    
    # Lưu ảnh
    output_path = f'{output_dir}/correlation_heatmap.png'
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ Heatmap Tương quan vào: {output_path}")
    plt.close()


# ==================== VISUALIZATION - BAR CHART ====================

def plot_category_composition(df, output_dir='../data/images'):
    """
    Chức năng:
        - Vẽ Bar Chart thể hiện phân bố số lượng sản phẩm của từng danh mục lớn
    
    Tham số:
        - df: Dataframe đầu vào
        - output_dir: Thư mục lưu hình
    """
    if 'category_root_name' not in df.columns:
        print("Cảnh báo: Cột 'category_root_name' không tồn tại trong DataFrame")
        return
    
    # Đếm số lượng sản phẩm theo từng danh mục gốc và sắp xếp giảm dần
    category_counts = df['category_root_name'].value_counts().sort_values(ascending=False)
    
    if len(category_counts) == 0:
        print("Cảnh báo: Không có dữ liệu category để vẽ biểu đồ")
        return
    
    plt.figure(figsize=(12, 12))
    # Vẽ Bar Plot
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
    
    plt.title('Thành phần Sản phẩm theo Danh mục Cấp cao', fontsize=16)
    plt.xlabel('Danh mục Gốc', fontsize=12)
    plt.ylabel('Số lượng Sản phẩm', fontsize=12)
    # Xoay nhãn trục x
    plt.xticks(rotation=45, ha='right')
    # Thêm lưới trục y
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Lưu ảnh
    output_path = f'{output_dir}/category_composition_bar.png'
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ Thành phần Danh mục vào: {output_path}")
    plt.close()


# ==================== MISSING VALUE ANALYSIS (TEXT) ====================

def analyze_text_missing(df):
    """
    Chức năng:
        - Phân tích và thống kê missing value cho các cột text
    
    Tham số:
        - df: Dataframe đầu vào
    
    Giá trị trả về:
        - Dictionary chứa số lượng missing value
    """
    # Các cột text
    text_cols = [
        'product_name',
        'product_url',
        'category_name',
        'category_root_name',
        'brand_name',
        'origin',
        'store_name',
        'cancel_by_seller_rate_status',
        'return_rate_status'
    ]
    
    # Dictionary để lưu cột và số dòng bị thiếu ở cột đó
    missing_dict = {}
    count = 0
    
    for col in text_cols:
        if col in df.columns:
            # Đếm số lượng NaN trong cột
            number_missing_value = df[col].isnull().sum()
            if number_missing_value > 0: 
                count += 1
            missing_dict[col] = number_missing_value
    
    # In kết quả ra console dưới dạng bảng
    print(f"{'Cột':30} {'Số lượng':10} {'Phần trăm':10}")
    print("-" * 55)
    
    for col, val in missing_dict.items():
        percent = round(val / len(df) * 100, 3) if len(df) > 0 else 0
        print(f"{col:30} {val:<10} {percent:<10}%")
    
    print(f"\nSố cột chứa missing value: {count}")

    return missing_dict


# ==================== MISSING VALUE ANALYSIS (NUMERIC) ====================

def analyze_numeric_missing(df):
    """
    Chức năng:
        - Phân tích và thống kê missing value cho các cột numeric
    
    Tham số:
        - df: DataFrame đầu vào
    
    Giá trị trả về:
        - DataFrame chứa thống kê missing value
    """
    # Danh sách các cột cần kiểm tra
    columns_to_check = [
        'product_id', 'category_id', 'price', 'original_price', 
        'discount_rate', 'quantity_sold', 'rating_average', 
        'review_count', 'is_return_policy', 'is_freeship_xtra', 
        'is_authentic', 'image_count', 'video_count', 'is_brand', 
        'store_id', 'store_review_count', 'total_follower', 
        'is_official', 'cancel_by_seller_rate', 'return_rate'
    ]
    
    # Kiểm tra xem các cột có tồn tại trong dataframe không
    existing_columns = [col for col in columns_to_check if col in df.columns]
    # Tính missing values
    total_rows = len(df)
    
    if total_rows == 0:
        print("Cảnh báo: DataFrame rỗng, không có dữ liệu để phân tích")
        # Trả về DataFrame rỗng với cấu trúc cột mong muốn
        return pd.DataFrame(columns=['Thuộc tính', 'Số lượng missing', 'Tỉ lệ missing (%)'])
    
    missing_data = []
    for col in existing_columns:
        # Đếm số lượng NaN
        missing_count = df[col].isnull().sum()
        # Tính tỷ lệ phần trăm
        missing_percent = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        
        missing_data.append({
            'Thuộc tính': col,
            'Số lượng missing': missing_count,
            'Tỉ lệ missing (%)': round(missing_percent, 2)
        })
    
    # Tạo dataframe missing_df
    missing_df = pd.DataFrame(missing_data)
    
    print("\nBẢNG TỈ LẾ MISSING VALUES:")
    # In toàn bộ DataFrame mà không bị cắt
    print(missing_df.to_string(index=False))

    return missing_df


# ==================== VISUALIZE MISSING VALUE ====================

def visualize_missing_values(missing_df, output_dir='../data/images'):
    """
    Chức năng:
        - Trực quan hóa tỉ lệ missing values
    
    Tham số:
        - missing_df: DataFrame từ analyze_numeric_missing()
        - output_dir: Thư mục lưu hình
    """
    # Kiểm tra DataFrame có rỗng không
    if missing_df.empty or 'Thuộc tính' not in missing_df.columns or 'Tỉ lệ missing (%)' not in missing_df.columns:
        print("Cảnh báo: Không có dữ liệu để vẽ biểu đồ missing values")
        return
    
    # Tạo figure và axes
    plt.figure(figsize=(12, 6))
    # Vẽ Bar Chart
    bars = plt.bar(missing_df['Thuộc tính'], missing_df['Tỉ lệ missing (%)'], color='skyblue')
    # Thêm đường ngang đánh dấu ngưỡng 25%
    plt.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='Ngưỡng 25%')
    
    plt.xlabel('Thuộc tính')
    plt.ylabel('Tỉ lệ Missing Values (%)')
    plt.title('TỈ LẾ MISSING VALUES THEO TỪNG THUỘC TÍNH')
    # Xoay nhãn trục x
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    # Điều chỉnh layout để tránh nhãn bị chồng lên nhau
    plt.tight_layout()
    
    # Tô màu đỏ cho các cột có missing > 25%
    for i, (bar, rate) in enumerate(zip(bars, missing_df['Tỉ lệ missing (%)'])):
        if rate > 25:
            bar.set_color('red')
    
    # Lưu ảnh 
    output_path = f'{output_dir}/missing_values_visualization.png'
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ Missing Values vào: {output_path}")
    plt.close()


# ==================== THỐNG KÊ PHÂN BỐ CỦA CÁC CỘT DƯỚI DẠNG BẢNG ====================

def get_detailed_statistics(df):
    """
    Chức năng:
        - Bảng tổng hợp chi tiết về phân bổ của các cột số
            + Kiểu dữ liệu
            + Số giá trị unique
            + Thống kê missing values
            + Các chỉ số thống kê cơ bản min, max, median, std, var, q1, q3
    
    Tham số:
        - df: DataFrame đầu vào
    
    Giá trị trả về:
        - DataFrame chứa thống kê chi tiết
    """ 
    # Danh sách các cột cần kiểm tra
    columns_to_check = [
        'product_id', 'category_id', 'price', 'original_price', 
        'discount_rate', 'quantity_sold', 'rating_average', 
        'review_count', 'is_return_policy', 'is_freeship_xtra', 
        'is_authentic', 'image_count', 'video_count', 'is_brand', 
        'store_id', 'store_review_count', 'total_follower', 
        'is_official', 'cancel_by_seller_rate', 'return_rate'
    ]
    
    # Lọc ra các cột tồn tại trong DataFrame
    existing_columns = [col for col in columns_to_check if col in df.columns]
    total_rows = len(df)
    
    if total_rows == 0:
        print("Cảnh báo: DataFrame rỗng, không có dữ liệu để phân tích")
        return pd.DataFrame()
    
    summary_stats = []
    
    for col in existing_columns:
        col_data = df[col]
        # Kiểu dữ liệu
        dtype = col_data.dtype
        # Số lượng giá trị unique
        n_unique = col_data.nunique()
        # Số lượng missing
        missing_count = col_data.isnull().sum()
        # Tỉ lệ missing
        missing_percentage = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        
        # Khởi tạo các giá trị thống kê bằng NaN
        min_val = max_val = mean_val = median_val = std_val = var_val = q25 = q75 = np.nan
        
        # Chỉ tính toán các thống kê cho cột số
        if pd.api.types.is_numeric_dtype(col_data):
            numeric_data = col_data.dropna()
            if len(numeric_data) > 0:
                min_val = numeric_data.min()
                max_val = numeric_data.max()
                mean_val = numeric_data.mean()
                median_val = numeric_data.median()
                std_val = numeric_data.std()
                var_val = numeric_data.var()
                q25 = numeric_data.quantile(0.25)
                q75 = numeric_data.quantile(0.75)
        
        # Thêm kết quả vào danh sách
        summary_stats.append({
            'Thuộc tính': col,
            'Kiểu dữ liệu': dtype,
            'Số giá trị unique': n_unique,
            'Số lượng missing': missing_count,
            'Tỉ lệ missing (%)': round(missing_percentage, 2),
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val,
            'Median': median_val,
            'Std': std_val,
            'Var': var_val,
            'Q1 (25%)': q25,
            'Q3 (75%)': q75
        })
    
    # Tạo DataFrame từ kết quả
    summary_df = pd.DataFrame(summary_stats)
    
    # Sắp xếp cột cho đẹp
    cols_order = [
        'Thuộc tính', 'Kiểu dữ liệu', 'Số giá trị unique',
        'Số lượng missing', 'Tỉ lệ missing (%)',
        'Min', 'Max', 'Mean', 'Median', 'Std', 'Var',
        'Q1 (25%)', 'Q3 (75%)'
    ]
    summary_df = summary_df[cols_order]

    print("\nTHÔNG TIN CHI TIẾT VỀ PHÂN BỐ CỦA TẤT CẢ CÁC CỘT:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    
    return summary_df


# ==================== DUPLICATE ANALYSIS ====================

def check_duplicates(df):
    """
    Chức năng:
        - Kiểm tra trùng lặp theo product_id và product_url
    
    Tham số:
        - df: DataFrame đầu vào
    """
    print("\n--- KIỂM TRA TRÙNG LẶP ---")
    
    # Các cột nhận dạng chính
    product_cols = ['product_id', 'product_url']
    
    for col in product_cols:
        if col in df.columns:
            # Lọc ra các giá trị bị trùng lặp (lần xuất hiện thứ 2 trở đi)
            duplicates = df[col][df[col].duplicated()]
            if not duplicates.empty:
                print(f"Có giá trị trùng trong cột '{col}':")
                print(f"Số lượng: {len(duplicates)}")
                # Chỉ in các giá trị unique bị trùng
                print(duplicates.unique())
            else:
                print(f"Không có giá trị trùng trong cột '{col}'")


# ==================== OUTLIER ANALYSIS ====================

def analyze_outliers(df):
    """
    Chức năng:
        - Phân tích và thống kê Outliers cho các cột số bằng phương pháp IQR
        - Cung cấp các thông tin về IQR, giới hạn trên/dưới
    
    Tham số:
        - df: DataFrame đầu vào
    
    Giá trị trả về:
        - DataFrame chứa thống kê outliers
    """
    # Danh sách các cột cần phân tích
    columns_for_boxplot = [
        'price', 'original_price', 'discount_rate', 'quantity_sold',
        'rating_average', 'review_count', 'image_count', 'video_count',
        'store_review_count', 'total_follower', 
        'cancel_by_seller_rate', 'return_rate'
    ]
    
    print("=" * 80)
    print("PHÂN TÍCH OUTLIERS THEO PHƯƠNG PHÁP IQR")
    print("=" * 80)
    
    results = []
    for col in columns_for_boxplot:
        if col in df.columns:
            # Lọc dữ liệu không phải NaN
            data = df[col].dropna()
            
            # Kiểm tra có dữ liệu hợp lệ không
            if len(data) == 0:
                results.append({
                    'Cột': col,
                    'IQR': "N/A",
                    'Lower Bound': "N/A",
                    'Upper Bound': "N/A",
                    'Số outliers': 0,
                    'Tỉ lệ outliers': "0%",
                    'Min outlier': "Không có",
                    'Max outlier': "Không có",
                    'Phân phối': "Không có dữ liệu"
                })
                continue
            
            # Tính Q1 (25%), Q3 (75%)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            # Tính Khoảng Tứ phân vị
            IQR = Q3 - Q1
            
            # Kiểm tra IQR hợp lệ
            if np.isnan(IQR) or IQR == 0:
                results.append({
                    'Cột': col,
                    'IQR': round(IQR, 2) if not np.isnan(IQR) else "N/A",
                    'Lower Bound': "N/A",
                    'Upper Bound': "N/A",
                    'Số outliers': 0,
                    'Tỉ lệ outliers': "0%",
                    'Min outlier': "Không có",
                    'Max outlier': "Không có",
                    'Phân phối': "Không đủ dữ liệu"
                })
                continue
            
            # Tính giới hạn dưới và trên theo công thức IQR
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Lọc ra các outliers (nằm ngoài [lower_bound, upper_bound])
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            median = data.median()
            mean = data.mean()
            
            # Nhận xét sơ bộ về độ lệch (Skewness) bằng cách so sánh Mean và Median
            if median < mean:
                skewness_note = "Lệch phải (right-skewed)"
            elif median > mean:
                skewness_note = "Lệch trái (left-skewed)"
            else:
                skewness_note = "Cân bằng"
            
            data_length = len(data)
            outlier_percent = (len(outliers) / data_length * 100) if data_length > 0 else 0
            
            # Thêm kết quả vào danh sách
            results.append({
                'Cột': col,
                'IQR': round(IQR, 2),
                'Lower Bound': round(lower_bound, 2),
                'Upper Bound': round(upper_bound, 2),
                'Số outliers': len(outliers),
                'Tỉ lệ outliers': f"{outlier_percent:.1f}%",
                'Min outlier': round(outliers.min(), 2) if len(outliers) > 0 else "Không có",
                'Max outlier': round(outliers.max(), 2) if len(outliers) > 0 else "Không có",
                'Phân phối': skewness_note
            })
    
    # In bảng kết quả
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    return results_df


# ==================== VẼ BOXPLOT PHÂN TÍCH OUTLIERS ====================

def plot_boxplots(df, output_dir='../data/images'):
    """
    Chức năng:
        - Vẽ Box Plot cho phân tích và nhận diện outliers
    
    Tham số:
        - df: DataFrame đầu vào
        - output_dir: Thư mục lưu hình vẽ
    """
    # Danh sách các cột cần vẽ Box Plot
    columns_for_boxplot = [
        'price', 'original_price', 'discount_rate', 'quantity_sold',
        'rating_average', 'review_count', 'image_count', 'video_count',
        'store_review_count', 'total_follower', 
        'cancel_by_seller_rate', 'return_rate'
    ]
    
    print("\n" + "=" * 80)
    print("Vẽ BOX PLOT")
    print("=" * 80)

    # Lấy thông tin outliers từ hàm analyze_outliers
    results = analyze_outliers(df)
    
    # Tạo mapping từ tên cột sang index trong results để tránh lỗi index mismatch
    results_dict = {row['Cột']: row for _, row in results.iterrows()}
    
    # Tạo figure lớn với nhiều subplot
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    # Chuyển mảng 2D thành 1D
    axes = axes.flatten()
    
    plot_idx = 0
    for col in columns_for_boxplot:
        if plot_idx < len(axes) and col in df.columns and col in results_dict:
            ax = axes[plot_idx]
            
            # Lọc dữ liệu và kiểm tra có dữ liệu không
            plot_data = df[col].dropna()
            if len(plot_data) == 0:
                ax.text(0.5, 0.5, 'Không có dữ liệu', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f"{col}\n(Không có dữ liệu)", fontsize=12)
                plot_idx += 1
                continue
            
            # Vẽ box plot
            boxplot = ax.boxplot(plot_data, patch_artist=True)

            # Tùy chỉnh màu sắc 
            boxplot['boxes'][0].set_facecolor('lightblue')
            boxplot['medians'][0].set_color('red')
            
            # Thêm thông tin từ results_dict
            result_row = results_dict[col]
            outliers_count = result_row['Số outliers']
            
            # Đặt tiêu đề với thông tin phân phối
            dist_text = result_row['Phân phối']
            # Đặt màu chữ tiêu đề dựa trên độ lệch (red=lệch phải, blue=lệch trái, green=cân bằng)
            color = 'red' if 'Lệch phải' in dist_text else 'blue' if 'Lệch trái' in dist_text else 'green'
            ax.set_title(f"{col}\n({dist_text})", 
                        fontsize=12, fontweight='bold', color=color)
            ax.set_ylabel('Giá trị')
            # Thêm lưới
            ax.grid(True, alpha=0.3)
            
            # Hiển thị số outliers
            ax.text(0.95, 0.95, f'Outliers: {outliers_count}', 
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            
            plot_idx += 1
    
    # Ẩn các subplot không sử dụng
    for j in range(plot_idx, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('BOX PLOT - PHÂN TÍCH OUTLIERS VÀ PHÂN PHỐI', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = f'{output_dir}/boxplots_outliers.png'
    plt.savefig(output_path)
    print(f"Đã lưu biểu đồ Box Plot vào: {output_path}")
    plt.close()


# ==================== DATA REDUCTION PREVIEW ====================

def preview_data_reduction(df):
    """
    Chức năng: 
        - Thống kê sơ bộ trước khi reduction
            + Số lượng unique
            + Tỉ lệ missing values
    
    Tham số:
        - df: DataFrame đầu vào
    """
    # --- THỐNG KÊ CỘT SỐ (NUMERIC) ---
    # Cột numeric
    numeric_cols = [
        'product_id', 'category_id', 'price',
        'original_price', 'discount_rate', 'quantity_sold',
        'rating_average', 'review_count', 'is_return_policy',
        'is_freeship_xtra', 'is_authentic','image_count', 
        'video_count', 'is_brand', 'store_id', 'store_review_count', 
        'total_follower', 'is_official', 'cancel_by_seller_rate', 'return_rate',
    ]
    
    total_rows = len(df)
    
    if total_rows == 0:
        print("Cảnh báo: DataFrame rỗng, không có dữ liệu để phân tích")
        return
    
    print("=" * 50)
    print("--- TÓM TẮT CỘT SỐ (NUMERIC) ---")
    print("=" * 50)
    
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n[Cột: {col}]")
            
            # Tính toán thống kê
            # Số lượng giá trị unique (không kể NaN)
            unique_count = df[col].nunique(dropna=True)
            # .describe() để lấy các thống kê cơ bản
            stats = df[col].describe(percentiles=[]).to_dict()
            # Số lượng missing
            missing_count = df[col].isna().sum()
            
            print(f"  > Số lượng giá trị khác nhau (không kể NaN): {int(unique_count)}")
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
            print(f"  > Giá trị thiếu (NaN): {missing_count} ({missing_pct:.2f}%)")
            
            # Trích xuất các chỉ số thống kê
            mean_val = stats.get('mean', None)
            std_val = stats.get('std', None)
            min_val = stats.get('min', None)
            max_val = stats.get('max', None)
            
            # In các chỉ số thống kê, định dạng 4 chữ số thập phân
            print(f"> Trung bình (Mean): {mean_val:.4f}" if mean_val is not None else "  > Trung bình (Mean): N/A")
            print(f"> Độ lệch chuẩn (Std): {std_val:.4f}" if std_val is not None else "  > Độ lệch chuẩn (Std): N/A")
            print(f"> Min/Max: {min_val} / {max_val}" if (min_val is not None and max_val is not None) else "  > Min/Max: N/A / N/A")
    
    print("\n" + "=" * 50)
    

    # --- THỐNG KÊ CỘT CHUỖI (TEXT/OBJECT) ---
    # Cột text
    text_cols = [
        'product_name',
        'product_url',
        'category_name',
        'category_root_name',
        'brand_name',
        'origin',
        'store_name',
        'cancel_by_seller_rate_status',
        'return_rate_status'
    ]
    
    print("--- TÓM TẮT CỘT CHUỖI (TEXT/OBJECT) ---")
    print("=" * 50)
    
    for col in text_cols:
        if col in df.columns:
            print(f"\n[Cột: {col}]")
            
            # Số lượng giá trị unique
            unique_count = df[col].nunique(dropna=True)
            # Số lượng missing
            missing_count = df[col].isna().sum()
            
            print(f"> Số lượng giá trị khác nhau (không kể NaN): {unique_count}")
            missing_pct = (missing_count / total_rows * 100) if total_rows > 0 else 0
            print(f"> Giá trị thiếu (NaN): {missing_count} ({missing_pct:.2f}%)")