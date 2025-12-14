import requests
import time
import pandas as pd
import csv
from typing import List, Dict, Optional

# ==================== CẤU HÌNH VÀ THAM SỐ TOÀN CỤC ====================

# Giả lập trình duyệt (quan trọng để tránh bị chặn bởi API)
HEADERS = HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
    "x-guest-token": "V087vTNnajoLWykYUSH2EzsrxtemRhig"
}

# Tên các file output trung gian và cuối cùng
# Kết quả bước 1 (Danh sách sản phẩm cơ bản)
OUTPUT_FILE_1 = "../data/raw/tiki_initial_data.csv"
# Kết quả bước 2 (Thêm chi tiết sản phẩm)
OUTPUT_FILE_2 = "../data/raw/tiki_products_data.csv"
# Kết quả bước 3 (Thêm chi tiết shop và performance)
OUTPUT_FILE_3 = "../data/raw/tiki_products.csv"

# Thời gian chờ giữa các lần gọi API
TIME_SLEEP = 1

# Các URL API cần thiết cho việc crawl
# Dùng để lấy danh sách danh mục nhỏ
CATEGORY_API_URL = "https://tiki.vn/api/v2/categories" 
# Dùng để lấy danh sách các sản phẩm
LISTINGS_API_URL = "https://tiki.vn/api/personalish/v1/blocks/listings"
# Dùng để lấy chi tiết 1 sản phẩm
PRODUCT_API_URL = "https://tiki.vn/api/v2/products/{}"
# Dùng để lấy thông tin shop
SHOP_API_URL = "https://api.tiki.vn/product-detail/v2/widgets/seller"
# Dùng để lấy hiệu suất shop
PERFORMANCES_API_URL = "https://seller-store-api.tiki.vn/ovl-performances/{}"

# Danh sách các danh mục gốc cần crawl
CATEGORIES = [
    {"id": 8322, "name": "Nhà Sách Tiki"},
    {"id": 1883, "name": "Nhà Cửa - Đời Sống"},
    {"id": 1789, "name": "Điện Thoại - Máy Tính Bảng"},
    {"id": 2549, "name": "Đồ chơi - Mẹ & Bé"},
    {"id": 1815, "name": "Thiết bị số - Phụ kiện số"},
    {"id": 1882, "name": "Điện Gia Dụng"},
    {"id": 1520, "name": "Làm Đẹp - Sức Khỏe"},
    {"id": 8594, "name": "Ô Tô – Xe Máy – Xe Đạp"},
    {"id": 4384, "name": "Bách Hóa Online"},
    {"id": 1975, "name": "Thể Thao – Dã Ngoại"},
    {"id": 915, "name": "Thời trang nam"},
    {"id": 1846, "name": "Laptop – Máy Vi Tính – Linh kiện"} ,
    {"id": 931, "name": "Thời trang nữ"},
    {"id": 1686, "name": "Giày - Dép nam"},
    {"id": 4221, "name": "Điện tử - Điện lạnh"},
    {"id": 1801, "name": "Máy ảnh - Máy quay phim"},
    {"id": 15078, "name": "Chăm sóc nhà cửa"},
    {"id": 1703, "name": "Giày - Dép nữ"},
    {"id": 27498, "name": "Phụ kiện thời trang"},
    {"id": 44792, "name": "NGON"},
    {"id": 8371, "name": "Đồng hồ và Trang sức"},
    {"id": 6000, "name": "Balo và Vali"},
    {"id": 976, "name": "Túi thời trang nữ"},
    {"id": 27616, "name": "Túi thời trang nam"},   
]

# ==================== CÁC HÀM CƠ BẢN ĐỂ LẤY DỮ LIỆU ====================

def get_subcategories(category_id: int) -> List[Dict]:
    """
    Chức năng:
        - Lấy danh sách các danh mục con (subcategories) từ một danh mục gốc (category root) bằng API.
    
    Cách thực hiện:
        1. Thiết lập tham số `parent_id` bằng `category_id` và `include: children`.
        2. Gửi GET request đến `CATEGORY_API_URL`.
        3. Trả về danh sách các danh mục con.
    
    Tham số:
        category_id: ID của danh mục cha.
    
    Giá trị trả về:
        Danh sách các danh mục con.
    """
    params = {
        "include": "children", 
        "parent_id": category_id
    }

    try:
        # Gửi request GET tới API danh mục
        response = requests.get(CATEGORY_API_URL, params=params, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            # Lấy dữ liệu từ trường data trong response JSON
            return data.get("data", [])
        else:
            print(f"Lỗi khi lấy danh mục từ danh mục lớn {category_id}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Lỗi: {e}")
        return []

def get_products_from_subcategory(category_id: int, category_name: str, url_key: str, category_root_name: str):
    """
    Chức năng:
        - Lấy danh sách sản phẩm từ một danh mục con cụ thể.
        - Hàm chỉ crawl tối đa 3 trang và sắp xếp theo sản phẩm bán chạy nhất để đảm bảo lấy được các sản phẩm có chất lượng cao và phổ biến.
    
    Cách thực hiện
        1. Lặp qua các trang (từ 1 đến 3).
        2. Thiết lập tham số: `limit`, `sort`, `page`, `urlKey`, `category`.
        3. Gửi GET request đến `LISTINGS_API_URL`.
        4. Trích xuất `product_id`, `product_name`, `product_url` và thông tin danh mục, sau đó lưu vào danh sách.
        5. Dừng giữa các trang bằng `time.sleep(TIME_SLEEP)`.
    
    Tham số:
        - category_id: ID của danh mục con.
        - category_name: Tên của danh mục con.
        - url_key: URL key của danh mục.
        - category_root_name: Tên của danh mục gốc.
    
    Giá trị trả về:
        Danh sách các sản phẩm.
    """
    products = []

    # Giới hạn crawl 3 trang (page 1, 2, 3)
    for page in range(1, 4):
        params = {
            "limit": 40, 
            "sort": "top_seller",
            "page": page,
            "urlKey": url_key,
            "category": category_id
        }

        try:
            print(f"Đang crawl trang {page} của {category_name}")
            # Gửi request GET
            response = requests.get(LISTINGS_API_URL, params=params, headers=HEADERS, timeout=10)

            if response.status_code == 200:
                data = response.json()
                page_products = data.get("data", [])

                # Lặp qua từng sản phẩm trong trang
                for product in page_products:
                    products.append({
                        "product_id": product.get("id"),
                        "product_name": product.get("name"),
                        # Tạo URL đầy đủ từ 'url_path'
                        "product_url": f"https://tiki.vn/{product.get('url_path', '')}",
                        "category_id": category_id,
                        "category_name": category_name,
                        "category_root_name": category_root_name
                    })

                print(f"Trang {page}: Đã lấy {len(page_products)} sản phẩm")

                # Kiểm tra nếu đã đạt trang cuối cùng thì dừng
                paging = data.get("paging", {})
                if page >= paging.get("last_page", page):
                    break

                # Tạm dừng giữa các trang
                time.sleep(TIME_SLEEP)

            else:
                print(f"Lỗi khi lấy trang {page}: {response.status_code}")
                break

        except Exception as e:
            print(f"Lỗi: {e}")
            break

    return products

def process_1():
    """
    Chức năng:
        - QUY TRÌNH 1: Thu thập thông tin cơ bản của sản phẩm.
        - Thực hiện crawl các danh mục con của các danh mục gốc đã định nghĩa và thu thập danh sách sản phẩm ban đầu.
    
    Cách thực hiện:
        1. Lặp qua `CATEGORIES`.
        2. Đối với mỗi danh mục gốc, gọi `get_subcategories()` để lấy danh mục con.
        3. Đối với mỗi danh mục con, gọi `get_products_from_subcategory()` để lấy sản phẩm.
        4. Tổng hợp tất cả sản phẩm vào DataFrame và lưu vào `OUTPUT_FILE_1`.
    """
    products_final = []

    for c in CATEGORIES:
        print(f"\nĐang crawl danh mục: {c['name']}")

        # Bước 1: Lấy danh mục con
        subcategories = get_subcategories(c["id"])
        i = 0

        for sub_cat in subcategories:
            i += 1
            print(f"\nĐang xử lý danh mục con: {sub_cat['name']}")

            # Bước 2: Lấy sản phẩm từ danh mục con (tối đa 3 trang)
            products = get_products_from_subcategory(sub_cat["id"],sub_cat["name"],sub_cat["url_key"],c["name"])

            products_final.extend(products)
            print(f"Đã lấy {len(products)} sản phẩm từ danh mục: {sub_cat['name']}")

            time.sleep(TIME_SLEEP)

        time.sleep(TIME_SLEEP)

    # Chuyển danh sách kết quả thành DataFrame
    df = pd.DataFrame(products_final)

    # Sắp xếp lại thứ tự cột cho DataFrame
    columns_order = [
        'product_id', 'product_name', 'product_url',
        'category_id', 'category_name', 'category_root_name'
    ]
    df = df[columns_order]

    # Lưu DataFrame vào file CSV
    df.to_csv(OUTPUT_FILE_1, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    print(f"Tổng số sản phẩm: {len(products_final)}")
    print(f"Đã lưu vào file {OUTPUT_FILE_1}")


# ==================== CÁC HÀM XỬ LÝ DỮ LIỆU SẢN PHẨM ====================

def extract_spid_from_url(product_url: str) -> Optional[str]:
    """
    Chức năng: 
    - Trích xuất SPID (Seller Product ID) từ URL sản phẩm.
    - SPID là tham số cần thiết để lấy đúng biến thể sản phẩm khi gọi API chi tiết.
    
    Cách thực hiện:
        - Tìm chuỗi 'spid=' trong URL và lấy giá trị cho đến ký tự '&' tiếp theo.
    
    Tham số:
        - product_url: URL sản phẩm.
    
    Giá trị trả về:
        - SPID hoặc None.
    """
    try:
        if 'spid=' in product_url:
            # Tách chuỗi sau 'spid=' và trước ký tự '&'
            return product_url.split('spid=')[1].split('&')[0]
        return None
    except:
        return None

def has_freeship_xtra(product_data: Dict) -> Optional[int]:
    """
    Chức năng:
        - Kiểm tra xem sản phẩm có dịch vụ Freeship Xtra hay không.
    
    Cách thực hiện:
        - Trích xuất giá trị boolean từ trường `tracking_info -> amplitude -> is_freeship_xtra`.
        - Chuyển đổi boolean thành 1 hoặc 0.
    
    Tham số:
        - product_data: Dữ liệu JSON chi tiết sản phẩm.
    
    Giá trị trả về:
        1 (có) hoặc 0 (không) hoặc None.
    """
    try:
        is_freeship = product_data.get('tracking_info', {}).get('amplitude', {}).get('is_freeship_xtra')
        if is_freeship is None:
            return None
        return 1 if is_freeship else 0
    except:
        return None

def is_authentic(product_data: Dict) -> Optional[int]:
    """
    Chức năng:
        - Kiểm tra xem sản phẩm có cờ Authentic (hàng chính hãng) hay không.
    
    Cách thực hiện:
        - Trích xuất giá trị boolean từ trường `tracking_info -> amplitude -> is_authentic`.
        - Chuyển đổi boolean thành 1 hoặc 0.
    
    Tham số:
        - product_data: Dữ liệu JSON chi tiết sản phẩm.
    
    Giá trị trả về:
        1 (có) hoặc 0 (không) hoặc None.
    """
    try:
        is_authentic = product_data.get('tracking_info', {}).get('amplitude', {}).get('is_authentic')
        if is_authentic is None:
            return None
        return 1 if is_authentic else 0
    except:
        return None

def get_origin(product_data: Dict) -> Optional[str]:
    """
    Chức năng:
        - Lấy thông tin xuất xứ/nguồn gốc của sản phẩm từ phần thông số kỹ thuật (specifications).
    
    Cách thực hiện:
        1. Lặp qua danh sách `specifications`.
        2. Tìm thông số có `code` là 'origin'.
        3. Trả về giá trị của thông số đó.
    
    Tham số:
        - product_data: Dữ liệu JSON chi tiết sản phẩm.
    
    Giá trị trả về:
        - Tên xuất xứ hoặc None.
    """
    specifications = product_data.get('specifications')
    if not specifications or not isinstance(specifications, list):
        return None

    # Lặp qua từng nhóm thông số
    for s in specifications:
        if not s or not isinstance(s, dict):
            continue

        attributes = s.get('attributes')
        if not attributes or not isinstance(attributes, list):
            continue

        # Lặp qua từng thông số chi tiết
        for attr in attributes:
            if not attr or not isinstance(attr, dict):
                continue

            if attr.get('code') == 'origin':
                value = attr.get('value')
                # Trả về giá trị (tên quốc gia)
                return value if value else None 
    return None

def get_product_details(product_id: int, product_url: str) -> Dict:
    """
    Chức năng:
        - Gọi API để lấy tất cả thông tin chi tiết của một sản phẩm (giá, rating, review, images, video...).
    
    Cách thực hiện:
        1. Trích xuất SPID từ URL.
        2. Thiết lập URL và tham số (bao gồm SPID nếu có).
        3. Gửi GET request đến `PRODUCT_API_URL`.
        4. Trả về toàn bộ response JSON.
    
    Tham số:
        - product_id: ID sản phẩm.
        - product_url: URL sản phẩm.
    
    Giá trị trả về:
        - Dữ liệu JSON chi tiết sản phẩm.
    """
    # Trích xuất SPID
    spid = extract_spid_from_url(product_url)

    params = {
        "platform": "web",
        "version": "3"
    }
    if spid:
        # Thêm SPID vào tham số để lấy đúng biến thể
        params["spid"] = spid

    try:
        # Format URL với product_id
        url = PRODUCT_API_URL.format(product_id)
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Lỗi khi lấy chi tiết sản phẩm {product_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Lỗi khi lấy sản phẩm {product_id}: {e}")
        return None

def parse_product_data(product_data: Dict, product_url: str) -> Dict:
    """
    Chức năng:
        - Trích xuất các thuộc tính cần thiết (giá, rating, số lượng bán, brand, origin,...) từ dữ liệu JSON chi tiết sản phẩm.
    
    Cách thực hiện:
        1. Trích xuất trực tiếp các trường cơ bản.
        2. Dùng các hàm phụ trợ (`has_freeship_xtra`, `is_authentic`, `get_origin`) để lấy các giá trị phức tạp hơn.
        3. Trả về một dictionary chứa các thuộc tính đã được làm sạch.
    
    Tham số:
        - product_data: Dữ liệu JSON chi tiết sản phẩm.
        - product_url: URL sản phẩm.
    
    Giá trị trả về:
        - Dictionary chứa các thuộc tính đã trích xuất.
    """
    if not product_data:
        return {}

    # Đếm video 
    video_count = 0
    # Lặp qua tất cả các key-value trong dữ liệu sản phẩm để tìm video
    for key, value in product_data.items():
        if 'video_url' in key.lower() and value is not None:
            if isinstance(value, list):
                video_count += len(value)
            elif isinstance(value, str) and value.strip():
                video_count += 1
            elif value:
                video_count += 1

    # Trích xuất các trường cơ bản
    result = {
        'price': product_data.get('price'),
        'original_price': product_data.get('original_price'),
        'discount_rate': product_data.get('discount_rate'),
        'quantity_sold': product_data.get('all_time_quantity_sold'),
        'rating_average': product_data.get('rating_average'),
        'review_count': product_data.get('review_count'),
        'is_return_policy': 1 if product_data.get('return_policy') is not None else 0,
        'is_freeship_xtra': has_freeship_xtra(product_data),
        'is_authentic': is_authentic(product_data),
        'image_count': len(product_data.get('images', [])),
        'video_count': video_count,
        'is_brand': 1 if product_data.get('brand') else 0,
        'brand_name': product_data.get('brand', {}).get('name') if product_data.get('brand') else None,
        'origin': get_origin(product_data),
        'spid': extract_spid_from_url(product_url),
    }

    # Thông tin người bán 
    seller_info = product_data.get('current_seller', {})
    result.update({
        'store_id': seller_info.get('id') if seller_info else None,
    })

    return result

def process_2(input_file: str = OUTPUT_FILE_1):
    """
    Chức năng:
        - QUY TRÌNH 2: Thu thập thông tin chi tiết của từng sản phẩm.
        - Đọc danh sách sản phẩm từ bước 1 và gọi API chi tiết cho từng sản phẩm để lấy các thuộc tính số.
    
    Cách thực hiện:
        1. Đọc file `OUTPUT_FILE_1`.
        2. Lặp qua từng dòng (sản phẩm).
        3. Gọi `get_product_details()` để lấy dữ liệu JSON chi tiết.
        4. Gọi `parse_product_data()` để trích xuất các thuộc tính cần thiết.
        5. Cập nhật các thuộc tính mới vào DataFrame.
        6. Lưu DataFrame mới vào `OUTPUT_FILE_2`.
    
    Tham số:
        - input_file: Tên file output từ process_1.
    """
    try:
        # Đọc file kết quả từ bước 1
        df_1 = pd.read_csv(input_file)
        print(f"Đã đọc {len(df_1)} sản phẩm từ {input_file}")
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file}")
        return

    df_2 = df_1.copy()

    # Khai báo các cột mới sẽ được thêm vào DataFrame
    columns = [
        'price', 'original_price', 'discount_rate', 'quantity_sold',
        'rating_average', 'review_count', 'is_return_policy', 'is_freeship_xtra',
        'is_authentic', 'image_count', 'video_count', 'is_brand', 'brand_name',
        'origin', 'spid', 'store_id'
    ]

    # Khởi tạo các cột mới với giá trị None
    for c in columns:
        df_2[c] = None

    total_products = len(df_2)

    # Lặp qua từng sản phẩm để crawl chi tiết
    for index, row in df_2.iterrows():
        product_id = row['product_id']
        product_url = row['product_url']

        print(f"Đang crawl sản phẩm {index + 1}/{total_products}")

        # Lấy dữ liệu chi tiết sản phẩm
        product_detail = get_product_details(product_id, product_url)

        if product_detail:
            # Trích xuất dữ liệu
            parsed_data = parse_product_data(product_detail, product_url)

            # Cập nhật vào DataFrame
            for key, value in parsed_data.items():
                if key in df_2.columns:
                    df_2.at[index, key] = value

            print(f"Đã cập nhật sản phẩm")
        else:
            print(f"Không lấy được sản phẩm sản phẩm")

        # Tạm dừng giữa các request
        time.sleep(TIME_SLEEP)

    # Lưu kết quả bước 2
    df_2.to_csv(OUTPUT_FILE_2, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    print(f"Đã lưu vào file {OUTPUT_FILE_2}")


# ==================== CÁC HÀM XỬ LÝ DỮ LIỆU SHOP VÀ PERFORMANCE ====================

def get_shop_details(seller_id: int, product_id: int, spid: str) -> Dict:
    """
    Chức năng:
    - Gọi API để lấy thông tin cơ bản của Shop (tên shop, số review shop, số follower, is_official).
    
    Cách thực hiện:
        1. Thiết lập các tham số cần thiết: seller_id, mpid (product_id), spid.
        2. Gửi GET request đến `SHOP_API_URL`.
    
    Tham số:
        - seller_id: ID của Shop/Seller.
        - product_id: ID sản phẩm.
        - spid: SPID của sản phẩm.
    
    Giá trị trả về:
        - Dữ liệu JSON chi tiết Shop.
    """
    params = {
        "seller_id": seller_id,
        "mpid": product_id,
        "spid": spid,
        "trackity_id": "c2c11d48-c1a6-0ceb-52d5-9ae068d40d8f",
        "platform": "desktop",
        "version": "3"
    }

    try:
        # Gửi request GET tới API shop
        response = requests.get(SHOP_API_URL, params=params, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Lỗi khi lấy chi tiết shop {seller_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Lỗi khi lấy shop {seller_id}: {e}")
        return None

def get_shop_performance(store_id: int) -> Dict:
    """
    Chức năng:
    - Gọi API để lấy các chỉ số hiệu suất bán hàng của store.
    
    Cách thực hiện:
        1. Format URL với `store_id`.
        2. Gửi GET request đến `PERFORMANCES_API_URL`.
    
    Tham số:
        - store_id: ID của store.
    
    Giá trị trả về:
        - Dữ liệu JSON chỉ số hiệu suất.
    """
    try:
        # Format URL với store_id
        url = PERFORMANCES_API_URL.format(store_id)
        response = requests.get(url, headers=HEADERS, timeout=10)

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Lỗi khi lấy performance shop {store_id}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Lỗi khi lấy performance shop {store_id}: {e}")
        return None

def parse_shop_data(shop_data: Dict) -> Dict:
    """
    Chức năng:
    - Trích xuất các thuộc tính cơ bản của Shop (tên, số review, follower, is_official) từ dữ liệu JSON.
    
    Cách thực hiện:
        1. Lấy dữ liệu từ trường 'seller' bên trong trường 'data'.
        2. Trích xuất trực tiếp các trường: 'name', 'review_count', 'total_follower'.
        3. Chuyển đổi 'is_official' (boolean) thành 1 hoặc 0.
    
    Tham số:
        - shop_data: Dữ liệu JSON chi tiết Shop.
    
    Giá trị trả về:
        - Dictionary chứa các thuộc tính Shop đã trích xuất.
    """
    if not shop_data:
        return {}
    
    # Khởi tạo mặc định nếu không tìm thấy dữ liệu
    default_result = {
        'store_name': None,
        'store_review_count': None,
        'total_follower': None,
        'is_official': None
    }

    result = {}

    data = shop_data.get('data')
    if not data or not isinstance(data, dict):
        return default_result

    seller = data.get('seller')
    if not seller or not isinstance(seller, dict):
        return default_result

    # Chuyển đổi cờ is_official sang 1/0
    is_official = seller.get('is_official')
    if is_official is None:
        is_official_value = None
    else:
        is_official_value = 1 if is_official else 0

    result = {
        'store_name': seller.get('name'),
        'store_review_count': seller.get('review_count'),
        'total_follower': seller.get('total_follower'),
        'is_official': is_official_value
    }

    return result

def parse_performance_data(performance_data: Dict) -> Dict:
    """
    Chức năng:
        - Trích xuất các chỉ số hiệu suất bán hàng của Shop từ dữ liệu JSON.
    
    Cách thực hiện:
        - Trích xuất trực tiếp các trường: `cancel_by_seller_rate_l4w`, `return_rate_l4w` (tỷ lệ), và các trường `_status` tương ứng.
    
    Tham số:
        - performance_data: Dữ liệu JSON chỉ số hiệu suất.
    
    Giá trị trả về:
        - Dictionary chứa các chỉ số hiệu suất đã trích xuất.
    """
    if not performance_data:
        return {}

    # Lấy các chỉ số trong 4 tuần gần nhất (l4w: last 4 weeks)
    result = {
        'cancel_by_seller_rate': performance_data.get('cancel_by_seller_rate_l4w'),
        'cancel_by_seller_rate_status': performance_data.get('cancel_by_seller_rate_l4w_status'),
        'return_rate': performance_data.get('return_rate_l4w'),
        'return_rate_status': performance_data.get('return_rate_l4w_status')
    }

    return result

def process_3(input_file: str = OUTPUT_FILE_2):
    """
    Chức năng:
        - QUY TRÌNH 3: Thu thập thông tin Shop và Hiệu suất.
        - Đọc dữ liệu từ bước 2, và thu thập thông tin của người bán (Shop) và chỉ số hiệu suất bán hàng của họ.
    
    Cách thực hiện:
        1. Đọc file `OUTPUT_FILE_2`.
        2. Lặp qua từng sản phẩm, lấy `store_id`, `product_id`, `spid`.
        3. Gọi `get_shop_details()` và `parse_shop_data()` để lấy thông tin Shop.
        4. **Sử dụng Cache:** Dùng `performance_cache` để chỉ gọi `get_shop_performance()` một lần duy nhất cho mỗi `store_id` (tránh request trùng lặp, tối ưu tốc độ).
        5. Cập nhật các thuộc tính Shop và Performance vào DataFrame.
        6. Xóa cột `spid` không cần thiết.
        7. Lưu DataFrame cuối cùng vào `OUTPUT_FILE_3`.
    
    Tham số:
        - input_file: Tên file output từ process_2.
    """
    try:
        # Đọc file kết quả từ bước 2
        df_2 = pd.read_csv(input_file)
        print(f"Đã đọc {len(df_2)} sản phẩm từ {input_file}")
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file}")
        return

    df_3 = df_2.copy()

    # Khai báo các cột mới sẽ được thêm vào DataFrame
    columns = [
        'store_name', 'store_review_count', 'total_follower', 'is_official',
        'cancel_by_seller_rate', 'cancel_by_seller_rate_status',
        'return_rate', 'return_rate_status'
    ]

    # Khởi tạo các cột mới với giá trị None
    for c in columns:
        df_3[c] = None

    total_products = len(df_3)
    # Dictionary dùng để cache kết quả performance của shop (key: store_id)
    performance_cache = {}

    for index, row in df_3.iterrows():
        store_id = row['store_id']
        product_id = row['product_id']
        spid = row['spid']

        print(f"Đang crawl shop của sản phẩm {index + 1}/{total_products}")

        # Kiểm tra xem có đủ thông tin cần thiết để crawl shop không
        if pd.notna(store_id) and pd.notna(product_id) and pd.notna(spid):
            # Lấy thông tin Shop cơ bản (tên, follower, is_official)
            shop_detail = get_shop_details(int(store_id), int(product_id), str(spid))

            if shop_detail:
                parsed_shop_data = parse_shop_data(shop_detail)

                # Cập nhật thông tin shop
                for key, value in parsed_shop_data.items():
                    if key in df_3.columns:
                        df_3.at[index, key] = value

                # Lấy chỉ số Performance (Sử dụng Cache)
                if store_id not in performance_cache:
                    # Nếu chưa có trong cache, gọi API performance
                    performance_detail = get_shop_performance(int(store_id))
                    if performance_detail:
                        parsed_performance_data = parse_performance_data(performance_detail)
                        # Lưu vào cache
                        performance_cache[store_id] = parsed_performance_data
                    else:
                        # Lưu rỗng nếu lỗi
                        performance_cache[store_id] = {}

                # Lấy dữ liệu performance từ cache
                performance_data = performance_cache[store_id]
                # Cập nhật thông tin performance
                for key, value in performance_data.items():
                    if key in df_3.columns:
                        df_3.at[index, key] = value

                print(f"Đã cập nhật thông tin shop và performance")
            else:
                print(f"Không lấy được thông tin shop")
        else:
            print(f"Thiếu thông tin để crawl shop")

        # Tạm dừng giữa các request
        time.sleep(TIME_SLEEP)

    # Xóa cột 'spid' vì nó không cần thiết cho phân tích mô hình
    if 'spid' in df_3.columns:
        df_3 = df_3.drop(columns=['spid'])

    # Lưu file
    df_3.to_csv(OUTPUT_FILE_3, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)

    print(f"Đã lưu vào file {OUTPUT_FILE_3}")