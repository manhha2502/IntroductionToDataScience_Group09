import numpy as np
import pandas as pd
import re
import unicodedata
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.model_selection import train_test_split


# ==================== DATA CLEANING ====================

def handle_numeric_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chức năng:
        - Xử lý Missing Value dạng numeric bằng hai cách:
            + Điền 0: cho các cột mà giá trị thiếu được hiểu là không có
            + KNN Imputer: cho các cột còn lại mà giá trị thiếu cần được ước tính dựa trên các điểm dữ liệu gần nhất

    Tham số:
        - df: dataframe đầu vào
    
    Giá trị trả về:
        - dataframe đã xử lý missing value numeric
    """
    df_imputed = df.copy()
    
    # Danh sách các cột cần xử lý
    columns_to_check = [
        'quantity_sold', 'is_freeship_xtra', 'is_authentic', 
        'store_id', 'store_review_count', 'total_follower', 
        'is_official', 'cancel_by_seller_rate', 'return_rate'
    ]
    
    # Chỉ lấy các cột có trong dataframe
    existing_columns = [col for col in columns_to_check if col in df_imputed.columns]
    
    # Xác định các cột cần điền 0.0
    zero_impute_cols = []
    
    # 1. quantity_sold
    if 'quantity_sold' in existing_columns:
        zero_impute_cols.append('quantity_sold')
        missing_count = df_imputed['quantity_sold'].isnull().sum()
        if missing_count > 0:
            df_imputed['quantity_sold'] = df_imputed['quantity_sold'].fillna(0.0)
            print(f"Đã điền 0.0 cho quantity_sold: {missing_count} giá trị")
    
    # 2. Các cột bắt đầu bằng 'is_'
    is_cols = [col for col in existing_columns if col.startswith('is_')]
    for col in is_cols:
        zero_impute_cols.append(col)
        missing_count = df_imputed[col].isnull().sum()
        if missing_count > 0:
            df_imputed[col] = df_imputed[col].fillna(0.0)
            print(f"Đã điền 0.0 cho {col}: {missing_count} giá trị")
    
    # 3. store_id
    if 'store_id' in existing_columns:
        zero_impute_cols.append('store_id')
        missing_count = df_imputed['store_id'].isnull().sum()
        if missing_count > 0:
            df_imputed['store_id'] = df_imputed['store_id'].fillna(0.0)
            print(f"Đã điền 0.0 cho store_id: {missing_count} giá trị")
    
    # Xác định các cột còn lại cần dùng KNN (các cột số không phải là cột đặc biệt)
    remaining_numeric_cols = []
    for col in existing_columns:
        # Lọc các cột số, không thuộc nhóm đã điền 0.0
        if col not in zero_impute_cols and pd.api.types.is_numeric_dtype(df_imputed[col]):
            remaining_numeric_cols.append(col)

    print(f"Các cột sẽ dùng KNN Imputer ({len(remaining_numeric_cols)} cột):")
    for i, col in enumerate(remaining_numeric_cols, 1):
        missing_count = df_imputed[col].isnull().sum()
        print(f"{i}. {col}: {missing_count} missing")
    
    # Kiểm tra xem còn missing values không
    missing_after_zero = {}

    for col in remaining_numeric_cols:
        missing_count = df_imputed[col].isnull().sum()
        if missing_count > 0:
            missing_after_zero[col] = missing_count
    
    # ÁP DỤNG KNN IMPUTER
    if missing_after_zero:
        # Tạo subset dữ liệu chỉ chứa các cột cần impute
        knn_data = df_imputed[remaining_numeric_cols].copy()
        
        # Kiểm tra xem có đủ dữ liệu để thực hiện KNN không
        min_non_missing = knn_data.notna().sum().min()
        if min_non_missing < 2:
            print("Cảnh báo: Một số cột có quá ít dữ liệu không bị missing (<2 giá trị)")
        
        # Áp dụng KNN Imputer
        # Tùy chọn 1: KNN với tham số distance ưu tiên các điểm gần hơn
        # n_neighbors=5: lấy 5 điểm gần nhất để ước tính
        # weights='distance': ưu tiên các điểm gần hơn (trọng số nghịch đảo với khoảng cách)
        knn_imputer = KNNImputer(n_neighbors=5, weights='distance')

        # Thực hiện imputation trên tập dữ liệu đã chọn
        knn_imputed_array = knn_imputer.fit_transform(knn_data)
        
        # Cập nhật dữ liệu đã impute vào dataframe
        knn_imputed_df = pd.DataFrame(
            knn_imputed_array, 
            columns=remaining_numeric_cols, 
            index=df_imputed.index
        )
        
        # Cập nhật các cột đã impute
        for col in remaining_numeric_cols:
            df_imputed[col] = knn_imputed_df[col]
    
    return df_imputed


def handle_text_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chức năng:
        - Xử lý Missing Value cho các cột text bằng các cách:
            + Điền "Unknown": cho brand_name, origin
            + Mapping từ ID: Mapping store_name bằng cách dùng store_id để điền tên store từ các bản ghi có sẵn, nếu không có thì "Unknown"
            + Điền Mode: điền mode nếu có rate, "Unknown" nếu không có rate
    
    Tham số: 
        - df: dataframe đầu vào
    
    Giá trị trả về:
        - Dataframe đã xử lý missing value text    
    """
    df_cleaned = df.copy()
    
    # Đối với brand_name và origin điền "Unknown"
    for col in ['brand_name', 'origin']:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')
    
    # Đối với store_name, lấy store_id và đối chiếu với các hàng khác để tìm store_name, nếu không có thì điền "Unknown"
    if 'store_name' in df_cleaned.columns and 'store_id' in df_cleaned.columns:
        # Tạo dictionary mapping: store_id -> store_name từ các hàng không NaN
        store_map = df_cleaned.dropna(subset=['store_name']).drop_duplicates('store_id')
        if not store_map.empty:
            store_map = store_map.set_index('store_id')['store_name'].to_dict()
            
            def fill_store_name(row):
                # Nếu store_name là NaN
                if pd.isna(row['store_name']):
                    # Tra cứu tên cửa hàng bằng store_id trong store_map, nếu không tìm thấy thì gán "Unknown"
                    return store_map.get(row['store_id'], "Unknown")
                return row['store_name']
            
            # Áp dụng hàm fill_store_name theo từng dòng
            df_cleaned['store_name'] = df_cleaned.apply(fill_store_name, axis=1)
        else:
            # Nếu store_map rỗng, chỉ điền "Unknown"
            df_cleaned['store_name'] = df_cleaned['store_name'].fillna("Unknown")
    
    # Đối với cancel_by_seller_rate_status nếu cancel_by_seller_rate tồn tại thì điền mode, nếu không tồn tại thì điền Unknown
    if 'cancel_by_seller_rate_status' in df_cleaned.columns:
        # Lấy mode (giá trị xuất hiện nhiều nhất)
        mode_status = df_cleaned['cancel_by_seller_rate_status'].mode()
        mode_status = mode_status.iloc[0] if len(mode_status) > 0 else "Unknown"
        
        # Điền mode nếu cột rate tồn tại (notnull) nhưng status là NaN, ngược lại điền "Unknown"
        df_cleaned['cancel_by_seller_rate_status'] = df_cleaned.apply(
            lambda row: mode_status if pd.notnull(row.get('cancel_by_seller_rate', None)) and pd.isnull(row['cancel_by_seller_rate_status'])
                        else ("Unknown" if pd.isnull(row.get('cancel_by_seller_rate', None)) and pd.isnull(row['cancel_by_seller_rate_status'])
                              else row['cancel_by_seller_rate_status']),
            axis=1
        )
    
    # Đối với return_rate_status nếu return_rate tồn tại thì điền mode, nếu không tồn tại thì điền Unknown
    if 'return_rate_status' in df_cleaned.columns:
        # Lấy mode (giá trị xuất hiện nhiều nhất)
        mode_status = df_cleaned['return_rate_status'].mode()
        mode_status = mode_status.iloc[0] if len(mode_status) > 0 else "Unknown"
        
        # Điền mode nếu cột rate tồn tại (notnull) nhưng status là NaN, ngược lại điền "Unknown"
        df_cleaned['return_rate_status'] = df_cleaned.apply(
            lambda row: mode_status if pd.notnull(row.get('return_rate', None)) and pd.isnull(row['return_rate_status'])
                        else ("Unknown" if pd.isnull(row.get('return_rate', None)) and pd.isnull(row['return_rate_status'])
                              else row['return_rate_status']),
            axis=1
        )
    
    return df_cleaned


def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    """
    Chức năng:
        - Hàm Winsorizing để xử lý outliers bằng cách giữ lại các giá trị nằm ngoài phạm vi nhưng đưa chúng về ngưỡng tối đa/tối thiểu
    
    Tham số:
        - s: chuỗi dữ liệu đầu vào cần được Winsorizing
        - lower_q: ngưỡng tứ phân vị dưới
        - upper_q: ngưỡng tứ phân vị trên

    Giá trị trả về:
        - Chuỗi dữ liệu mới sau khi đã được Winsorizing
    """
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    # Giới hạn giá trị dưới lower về lower, giá trị trên upper về upper
    return s.clip(lower, upper)


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chức năng: 
        - Xử lý Outliers cho các cột có phân phối lệch mạnh bằng các phương pháp:
            + Winsorizing + Log Transformation
            + Chỉ Winsorizing
    
    Tham số:
        - df: dataframe đầu vào

    Giá trị trả về:
        - Dataframe đã xử lý Outliers
    """
    df_pre = df.copy()
    
    # Các cột cần Winsor + Log
    winsor_log_cols = [
        "price", "original_price", "quantity_sold",
        "review_count", "video_count", "store_review_count",
        "total_follower"
    ]
    
    # Các cột chỉ cần Winsor
    winsor_only_cols = ["discount_rate", "image_count"]
    
    # Xử lý Winsor + Log
    for col in winsor_log_cols:
        if col in df_pre.columns:
            numeric_col = pd.to_numeric(df_pre[col], errors="coerce")

            # Winsorize
            numeric_col = winsorize_series(numeric_col)

            # Clip lower = 0 để đảm bảo không âm khi lấy log
            numeric_col = numeric_col.clip(lower=0)

            # Áp dụng log1p
            df_pre[col] = np.log1p(numeric_col)
    
    # Chỉ Winsor
    for col in winsor_only_cols:
        if col in df_pre.columns:
            numeric_col = pd.to_numeric(df_pre[col], errors="coerce")
            # Chỉ áp dụng Winsorize
            df_pre[col] = winsorize_series(numeric_col)
    
    return df_pre


def normalize_text(s: str) -> str:
    """
    Chức năng:
        - Chuẩn hóa text cơ bản: 
            + Loại bỏ các ký tự đặc biệt
            + Chuyển về chữ thường
            + Chuẩn hóa unicode
    
    Tham số: 
        - s: chuỗi text đầu vào

    Giá trị trả về:
        - Chuỗi text đã được chuẩn hóa
    """
    if not isinstance(s, str):
        return ""
    
    # Chuẩn hóa Unicode
    s = unicodedata.normalize("NFC", s)
    s = s.replace("...", " ")

    # Loại bỏ bất kỳ nội dung nào trong ngoặc đơn
    s = re.sub(r"\(.*?\)", "", s)

    # Thay thế nhiều khoảng trắng bằng một khoảng trắng
    s = re.sub(r"\s+", " ", s)

    # Chuyển về chữ thường và cắt bỏ khoảng trắng
    return s.strip().lower()


def clean_origin(origin: str) -> str:
    """
    Chức năng:
        - Chuẩn hóa tên quốc gia trong cột origin về một định dạng chuẩn
    
    Tham số:
        - origin: chuỗi các tên quốc gia/xuất xứ cần chuẩn hóa

    Giá trị trả về:
        - Tên quốc gia/xuất xử đã được chuẩn hóa
    """
    if not isinstance(origin, str) or origin.strip() == "":
        # Điền "Other" nếu giá trị rỗng/không hợp lệ
        return "Other"
    
    # Chuẩn hóa text cơ bản
    text = normalize_text(origin)
    
    # Tách theo nhiều dạng dấu phân cách
    parts = re.split(r"[/,;|]", text)
    # Loại bỏ khoảng trắng thừa và chuỗi rỗng
    parts = [p.strip() for p in parts if p.strip()]
    
    # Từ điển chuẩn hóa về tên quốc gia tiếng Anh
    mapping = {
        # Việt Nam
        "việt nam": "Vietnam", "viet nam": "Vietnam", "vietnam": "Vietnam",
        
        # Trung Quốc
        "trung quốc": "China", "trung quoc": "China", "china": "China",
        "tq": "China", "hn/tq": "China", "hk/tq": "China",
        
        # Nhật Bản
        "nhật bản": "Japan", "nhat ban": "Japan", "japan": "Japan",
        
        # Hàn Quốc
        "hàn quốc": "South Korea", "han quoc": "South Korea",
        "korea": "South Korea", "south korea": "South Korea",
        
        # Mỹ
        "mỹ": "USA", "my": "USA", "usa": "USA",
        "united states": "USA", "us": "USA",
        
        # Hong Kong
        "hong kong": "Hong Kong", "hồng kông": "Hong Kong", "hk": "Hong Kong",
        
        # Đài Loan
        "đài loan": "Taiwan", "dai loan": "Taiwan", "taiwan": "Taiwan",
        
        # Ấn Độ
        "ấn độ": "India", "an do": "India", "india": "India",
        
        # Thái Lan
        "thái lan": "Thailand", "thai lan": "Thailand", "thailand": "Thailand",
        
        # Đức
        "đức": "Germany", "duc": "Germany", "germany": "Germany",
        
        # Anh
        "anh": "UK", "england": "UK", "uk": "UK",
        
        # Pháp
        "pháp": "France", "phap": "France", "france": "France",
        
        # Úc
        "úc": "Australia", "uc": "Australia", "australia": "Australia",
        
        # Italy
        "ý": "Italy", "y": "Italy", "italia": "Italy", "italy": "Italy",
        
        # Malaysia
        "malaysia": "Malaysia", "mã lai": "Malaysia",
        
        # Philippines
        "philippines": "Philippines", "philipin": "Philippines",
        "philipines": "Philippines",
        
        # Singapore
        "singapore": "Singapore",
        
        # Hà Lan
        "hà lan": "Netherlands", "ha lan": "Netherlands",
        
        # Thổ Nhĩ Kỳ
        "thổ nhĩ kì": "Turkey", "thổ nhĩ kỳ": "Turkey",
        "tho nhi ky": "Turkey", "turkey": "Turkey",
        
        # Pakistan
        "pakistan": "Pakistan",
        
        # Iran
        "iran": "Iran",
        
        # Myanmar
        "myanmar": "Myanmar",
        
        # Cambodia
        "campuchia": "Cambodia", "cam pu chia": "Cambodia", "cambodia": "Cambodia",
        
        # Ai Cập
        "ai cập": "Egypt", "ai cap": "Egypt", "egypt": "Egypt",
        
        # Nga
        "nga": "Russia", "russia": "Russia",
        
        # Canada
        "canada": "Canada",
        
        # Ba Lan
        "ba lan": "Poland", "poland": "Poland",
        
        # Hungary
        "hungary": "Hungary",
        
        # Cộng Hòa Séc
        "cộng hòa séc": "Czech Republic", "sec": "Czech Republic",
        "czech": "Czech Republic",
        
        # Bỉ
        "bỉ": "Belgium", "bi": "Belgium", "belgium": "Belgium",
        
        # Israel
        "israel": "Israel",
        
        # Chile
        "chile": "Chile",
        
        # Mexico
        "mexico": "Mexico",
        
        # Ireland
        "ireland": "Ireland",
        
        # Nam Phi
        "nam phi": "South Africa", "south africa": "South Africa",
        
        # Ukraine
        "ukraine": "Ukraine",
        
        # Tunisia
        "tunisia": "Tunisia",
        
        # Cuba
        "cuba": "Cuba",
        
        # Sri Lanka
        "sri lanka": "Sri Lanka",
        
        # Brazil
        "brazil": "Brazil",
        
        # Madagascar
        "madagascar": "Madagascar",
        
        # Đan Mạch
        "đan mạch": "Denmark", "dan mach": "Denmark",
        
        # Hy Lạp
        "hy lạp": "Greece", "hy lap": "Greece", "greece": "Greece",
        
        # Thụy Điển
        "thụy điển": "Sweden", "thuy dien": "Sweden", "sweden": "Sweden",
        
        # Thụy Sĩ
        "thụy sỹ": "Switzerland", "thuy sy": "Switzerland",
        "switzerland": "Switzerland",
        
        # United Kingdom, Scotland
        "scotland": "UK",
        
        # New Zealand
        "new zealand": "New Zealand",
        
        # Slovenia
        "slovenia": "Slovenia",
        
        # Áo
        "áo": "Austria", "ao": "Austria", "austria": "Austria",
        
        # Finland
        "phần lan": "Finland", "phan lan": "Finland", "finland": "Finland",
        
        # Nhiều giá trị đặc biệt
        "nhiều quốc gia": "Other",
        "khác": "Other",
        "unknown": "Other",
        "không xác định": "Other",
    }
    
    result = set()
    
    for p in parts:
        if p in mapping:
            # Thêm tên chuẩn hóa vào set
            result.add(mapping[p])
            continue
        # Không nhận diện được -> Other
        result.add("Other")
    
    if len(result) == 0:
        return "Other"
    
    # Sắp xếp cho đẹp
    return ", ".join(sorted(result))


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chức năng:
        - Áp dụng normalize_text và clean_origin cho các cột text cụ thể
    
    Tham số:
        - df: dataframe đầu vào

    Giá trị trả về:
        - Dataframe đã làm sạch text
    """
    df_cleaned = df.copy()
    
    # Chuẩn hóa origin
    if 'origin' in df_cleaned.columns:
        df_cleaned['origin'] = df_cleaned['origin'].apply(clean_origin)
    
    return df_cleaned


# ==================== DATA REDUCTION ====================

def reduce_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chức năng:
        - Drop các cột không cần thiết
            + Numeric: drop return_rate, cancel_by_seller_rate
            + Text: drop cancel_by_seller_rate_status, return_rate_status, product_url, category_name, store_name, brand_name (giữ category_root_name)
            + ID: drop product_id, category_id, store_id

    Tham số:
        - df: dataframe đầu vào

    Giá trị trả về:
        - Dataframe đã được loại bỏ các cột không cần thiết
    """
    df_reduced = df.copy()
    
    # Danh sách các cột cần loại bỏ:
    drop_cols = [
        # Cột số đã được kiểm tra (gần như 0 hoặc 1 giá trị unique)
        'cancel_by_seller_rate', 'return_rate',
        # Cột text/status không cần thiết
        'cancel_by_seller_rate_status', 'return_rate_status', 'product_url',
        # Cột định danh/unique cao
        'category_name', 'store_name', 'brand_name',
        # Cột trùng lặp/chi tiết không cần thiết
        'product_id', 'store_id', 'category_id'
    ]
    
    # Chỉ drop các cột có trong dataframe
    drop_cols = [col for col in drop_cols if col in df_reduced.columns]
    df_reduced.drop(columns=drop_cols, inplace=True)
    
    return df_reduced


# ==================== DATA TRANSFORMATION ====================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chức năng:
        - Feature Engineering: Tạo các feature mới, có ý nghĩa cho mô hình từ các thuộc tính đã có
            + reputation_score
            + trust_level
            + review_to_sold_ratio
            + shop_potential
            + total_visuals
            + has_video
            + discount_amount
            + price_vs_category
            + hot_keyword_count
            + name_length
            + name_word_count

    Tham số:
        - df: dataframe đầu vào

    Giá trị trả về:
        - Dataframe đã được thêm các feature mới
    """
    df_featured = df.copy()
    
    # Reputation Score
    if 'rating_average' in df_featured.columns and 'review_count' in df_featured.columns:
        df_featured['reputation_score'] = df_featured['rating_average'] * np.log1p(df_featured['review_count'])
    
    # Trust Level
    trust_cols = ['is_official', 'is_authentic', 'is_brand']
    if all(col in df_featured.columns for col in trust_cols):
        df_featured['trust_level'] = (df_featured['is_official'] + 
                                      df_featured['is_authentic'] + 
                                      df_featured['is_brand'])
    
    # Review to Sold Ratio
    if 'review_count' in df_featured.columns and 'quantity_sold' in df_featured.columns:
        df_featured['review_to_sold_ratio'] = df_featured['review_count'] / (df_featured['quantity_sold'].fillna(0) + 1)
    
    # Shop Potential
    if 'store_review_count' in df_featured.columns and 'total_follower' in df_featured.columns:
        df_featured['shop_potential'] = df_featured['store_review_count'].fillna(0) + df_featured['total_follower'].fillna(0)
    
    # Total Visuals
    if 'image_count' in df_featured.columns and 'video_count' in df_featured.columns:
        df_featured['total_visuals'] = df_featured['image_count'].fillna(0) + df_featured['video_count'].fillna(0)
    
    # Has Video
    if 'video_count' in df_featured.columns:
        df_featured['has_video'] = df_featured['video_count'].apply(lambda x: 1 if x > 0 else 0)
    
    # Discount Amount
    if 'original_price' in df_featured.columns and 'price' in df_featured.columns:
        df_featured['discount_amount'] = df_featured['original_price'] - df_featured['price']
        df_featured['discount_amount'] = df_featured['discount_amount'].apply(lambda x: x if x > 0 else 0)
    
    # Price vs Category
    if 'category_root_name' in df_featured.columns and 'price' in df_featured.columns:
        df_featured['category_avg_price'] = df_featured.groupby('category_root_name')['price'].transform('mean')
        df_featured['price_vs_category'] = df_featured['price'] / (df_featured['category_avg_price'] + 1)
        df_featured.drop(columns='category_avg_price', inplace=True, errors='ignore')
    elif 'price' in df_featured.columns:
        df_featured['price_vs_category'] = 1.0

    keywords = [
        'chính hãng', 'cao cấp', 'freeship', 'xịn', 'hot',
        'giảm', 'sale', 'tặng', 'combo', 'bảo hành',
        'nhập khẩu', 'siêu rẻ'
    ]

    for kw in keywords:
        col_name = f'kw_{kw}'
        df_featured[col_name] = (
            df_featured['product_name']
            .astype(str)
            .str.lower()
            .str.contains(kw)
            .astype(int)
    )
    
    # Hot Keyword Count
    if 'product_name' in df_featured.columns:
        keywords = ['chính hãng', 'cao cấp', 'freeship', 'xịn', 'hot', 'giảm', 'sale', 'tặng', 'combo', 'bảo hành', 'nhập khẩu', 'siêu rẻ']
        pattern = '|'.join(keywords)
        df_featured['hot_keyword_count'] = df_featured['product_name'].astype(str).str.lower().str.count(pattern).fillna(0).astype(int)
        
        # Name Length và Name Word Count
        df_featured['name_length'] = df_featured['product_name'].astype(str).apply(len)
        df_featured['name_word_count'] = df_featured['product_name'].astype(str).apply(lambda x: len(x.split()))
        
        # Drop product_name sau khi dùng xong
        df_featured.drop(columns='product_name', inplace=True, errors='ignore')
    
    return df_featured


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chức năng:
        - Chuẩn hóa dữ liệu: One-hot encoding cho category_root_name và origin

    Tham số:
        - df: dataframe đầu vào
    
    Giá trị trả về:
        - Dataframe đã được mã hóa  
    """
    df_encoded = df.copy()
    
    # One-hot encoding cho category_root_name
    if 'category_root_name' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['category_root_name'], drop_first=True, dtype=int)
        print("One-hot encoding cho category_root_name hoàn tất")

    # Drop cột category_avg_price sau khi dùng xong
    df_encoded.drop(columns='category_avg_price', inplace=True, errors='ignore')

    # One-hot encoding cho origin
    if 'origin' in df_encoded.columns:
        def split_origin(s):
            """Tách tất cả các giá trị trong cột origin thành list"""
            if not isinstance(s, str):
                return []
            
            # Tách theo dấu phẩy, '/', '-', giữ các giá trị quốc gia
            parts = re.split(r'[,/\-]', s)
            # Xóa khoảng trắng thừa
            parts = [p.strip() for p in parts if p.strip()]
            return parts
        
        # Tạo cột mới dạng list
        df_encoded['origin_list'] = df_encoded['origin'].apply(split_origin)
        
        # One-hot encoding
        mlb = MultiLabelBinarizer()
        origin_encoded = pd.DataFrame(
            mlb.fit_transform(df_encoded['origin_list']),
            columns=mlb.classes_,
            index=df_encoded.index
        )
        
        # Lưu lại các cột
        origin_cols = origin_encoded.columns.tolist()

        # Nối vào DataFrame gốc
        df_encoded = pd.concat([df_encoded, origin_encoded], axis=1)

        # Xóa cột tạm
        df_encoded.drop(columns=['origin_list', 'origin'], inplace=True, errors='ignore')
        
        # Gộp các cột origin có tỉ lệ <= 2% vào Others
        if origin_cols:
            # Tính tổng từng cột one-hot
            origin_per = (df_encoded[origin_cols].sum() / len(df_encoded) * 100)

            # Cột giữ lại (> 2%)
            keep_origin = origin_per[origin_per > 2].index.tolist()

            # Cột gộp vào Others (<= 2%)
            other_origin = origin_per[origin_per <= 2].index.tolist()
            
            if other_origin:
                # Tạo cột Others
                if 'Other' in df_encoded.columns:
                    df_encoded['Others'] = df_encoded[other_origin].sum(axis=1) + df_encoded['Other']
                    df_encoded.drop(columns='Other', inplace=True, errors='ignore')
                else:
                    df_encoded['Others'] = df_encoded[other_origin].sum(axis=1)
                
                # Xóa các cột nhỏ
                df_encoded.drop(columns=other_origin, inplace=True, errors='ignore')
    
    return df_encoded


# ==================== SCALING ====================

def scale_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, 
                   save_files: bool = False, columns_to_exclude: list = None) -> tuple:
    """
    Chức năng:
        - Chia tập train, text
        - Scaling: áp dụng MinMaxScaler
    
    Tham số:
        - df : DataFrame đầu vào (đã được preprocessing)
        - test_size : tỉ lệ tập test
        - random_state : random seed
        - save_files : có lưu file CSV không 
        - columns_to_exclude : danh sách cột không cần scale
    
    Giá trị trả về:
        - tuple: (train_df, test_df, scaler)
    """
    # Chia tập train-test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Nếu không có columns_to_exclude, sử dụng binary columns mặc định
    if columns_to_exclude is None:
        columns_to_exclude = ['is_official', 'is_authentic', 'is_brand', 
                              'is_freeship_xtra', 'is_return_policy', 'has_video', 'quantity_sold']
    
    # Xác định các cột cần scale (loại bỏ columns_to_exclude)
    cols_to_scale = [col for col in train_df.columns 
                     if train_df[col].dtype in ['float64', 'int64'] 
                     and col not in columns_to_exclude]
    
    # Xử lý NaN và inf
    train_df[cols_to_scale] = train_df[cols_to_scale].replace([np.inf, -np.inf], 0).fillna(0)
    test_df[cols_to_scale] = test_df[cols_to_scale].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Áp dụng MinMaxScaler (0-1) cho tất cả các cột liên tục (đã log hoặc chưa)
    scaler = MinMaxScaler()
    train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    
    # Lưu file nếu cần
    if save_files:
        train_df.to_csv(f'../data/processed/train_data_final.csv', index=False)
        test_df.to_csv(f'../data/processed/test_data_final.csv', index=False)
        df.to_csv(f'../data/processed/full_data_final.csv', index=False)
    
    return train_df, test_df, scaler
