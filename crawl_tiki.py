import requests
import time
import pandas as pd
import csv
from typing import List, Dict, Optional

HEADERS = HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
    "x-guest-token": "V087vTNnajoLWykYUSH2EzsrxtemRhig"
}

OUTPUT_FILE_1 = "tiki_initial_data.csv"
OUTPUT_FILE_2 = "tiki_products_data.csv"
OUTPUT_FILE_3 = "tiki_products.csv"

TIME_SLEEP = 1

CATEGORY_API_URL = "https://tiki.vn/api/v2/categories"
LISTINGS_API_URL = "https://tiki.vn/api/personalish/v1/blocks/listings"
PRODUCT_API_URL = "https://tiki.vn/api/v2/products/{}"
SHOP_API_URL = "https://api.tiki.vn/product-detail/v2/widgets/seller"
PERFORMANCES_API_URL = "https://seller-store-api.tiki.vn/ovl-performances/{}"

CATEGORIES = [
    {"id": 8322, "name": "Nhà Sách Tiki"},
    {"id": 1883, "name": "Nhà Cửa - Đời Sống"},
    {"id": 1789, "name": "Điện Thoại - Máy Tính Bảng"},
    {"id": 2549, "name": "Đồ chơi - Mẹ & Bé"},
    {"id": 1815, "name": "Thiết bị số - Phụ kiện số"},
    {"id": 1882, "name": "Điện Gia Dụng"},
    {"id": 1520, "name": "Làm Đẹp - Sức Khỏe"}    
]

def get_subcategories(category_id: int) -> List[Dict]:
    params = {
        "include": "children", 
        "parent_id": category_id
    }

    try:
        response = requests.get(CATEGORY_API_URL, params=params, headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            print(f"Lỗi khi lấy danh mục từ danh mục lớn {category_id}: {response.status_code}")
            return []
    except Exception as e:
        print(f"Lỗi: {e}")
        return []

def get_products_from_subcategory(category_id: int, category_name: str, url_key: str, category_root_name: str):
    products = []

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
            response = requests.get(LISTINGS_API_URL, params=params, headers=HEADERS, timeout=10)

            if response.status_code == 200:
                data = response.json()
                page_products = data.get("data", [])

                for product in page_products:
                    products.append({
                        "product_id": product.get("id"),
                        "product_name": product.get("name"),
                        "product_url": f"https://tiki.vn/{product.get('url_path', '')}",
                        "category_id": category_id,
                        "category_name": category_name,
                        "category_root_name": category_root_name
                    })

                print(f"Trang {page}: Đã lấy {len(page_products)} sản phẩm")

                paging = data.get("paging", {})
                if page >= paging.get("last_page", page):
                    break

                time.sleep(TIME_SLEEP)

            else:
                print(f"Lỗi khi lấy trang {page}: {response.status_code}")
                break

        except Exception as e:
            print(f"Lỗi: {e}")
            break

    return products

def process_1():
    products_final = []

    for c in CATEGORIES:
        print(f"\nĐang crawl danh mục: {c['name']}")

        subcategories = get_subcategories(c["id"])
        i = 0

        for sub_cat in subcategories:
            i += 1
            print(f"\nĐang xử lý danh mục con: {sub_cat['name']}")

            products = get_products_from_subcategory(sub_cat["id"],sub_cat["name"],sub_cat["url_key"],c["name"])

            products_final.extend(products)
            print(f"Đã lấy {len(products)} sản phẩm từ danh mục: {sub_cat['name']}")

            time.sleep(TIME_SLEEP)

        time.sleep(TIME_SLEEP)

    df = pd.DataFrame(products_final)

    columns_order = [
        'product_id', 'product_name', 'product_url',
        'category_id', 'category_name', 'category_root_name'
    ]
    df = df[columns_order]

    df.to_csv(OUTPUT_FILE_1, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

    print(f"Tổng số sản phẩm: {len(products_final)}")
    print(f"Đã lưu vào file {OUTPUT_FILE_1}")

def extract_spid_from_url(product_url: str) -> Optional[str]:
    try:
        if 'spid=' in product_url:
            return product_url.split('spid=')[1].split('&')[0]
        return None
    except:
        return None

def has_freeship_xtra(product_data: Dict) -> Optional[int]:
    try:
        is_freeship = product_data.get('tracking_info', {}).get('amplitude', {}).get('is_freeship_xtra')
        if is_freeship is None:
            return None
        return 1 if is_freeship else 0
    except:
        return None

def is_authentic(product_data: Dict) -> Optional[int]:
    try:
        is_authentic = product_data.get('tracking_info', {}).get('amplitude', {}).get('is_authentic')
        if is_authentic is None:
            return None
        return 1 if is_authentic else 0
    except:
        return None

def get_origin(product_data: Dict) -> Optional[str]:
    specifications = product_data.get('specifications')
    if not specifications or not isinstance(specifications, list):
        return None

    for s in specifications:
        if not s or not isinstance(s, dict):
            continue

        attributes = s.get('attributes')
        if not attributes or not isinstance(attributes, list):
            continue

        for attr in attributes:
            if not attr or not isinstance(attr, dict):
                continue

            if attr.get('code') == 'origin':
                value = attr.get('value')
                return value if value else None
    return None

def get_product_details(product_id: int, product_url: str) -> Dict:
    spid = extract_spid_from_url(product_url)

    params = {
        "platform": "web",
        "version": "3"
    }
    if spid:
        params["spid"] = spid

    try:
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
    if not product_data:
        return {}

    video_count = 0
    for key, value in product_data.items():
        if 'video_url' in key.lower() and value is not None:
            if isinstance(value, list):
                video_count += len(value)
            elif isinstance(value, str) and value.strip():
                video_count += 1
            elif value:
                video_count += 1

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

    seller_info = product_data.get('current_seller', {})
    result.update({
        'store_id': seller_info.get('id') if seller_info else None,
    })

    return result

def process_2(input_file: str = OUTPUT_FILE_1):
    try:
        df_1 = pd.read_csv(input_file)
        print(f"Đã đọc {len(df_1)} sản phẩm từ {input_file}")
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file}")
        return

    df_2 = df_1.copy()

    columns = [
        'price', 'original_price', 'discount_rate', 'quantity_sold',
        'rating_average', 'review_count', 'is_return_policy', 'is_freeship_xtra',
        'is_authentic', 'image_count', 'video_count', 'is_brand', 'brand_name',
        'origin', 'spid', 'store_id'
    ]

    for c in columns:
        df_2[c] = None

    total_products = len(df_2)

    for index, row in df_2.iterrows():
        product_id = row['product_id']
        product_url = row['product_url']

        print(f"Đang crawl sản phẩm {index + 1}/{total_products}")

        product_detail = get_product_details(product_id, product_url)

        if product_detail:
            parsed_data = parse_product_data(product_detail, product_url)

            for key, value in parsed_data.items():
                if key in df_2.columns:
                    df_2.at[index, key] = value

            print(f"Đã cập nhật sản phẩm")
        else:
            print(f"Không lấy được sản phẩm sản phẩm")

        time.sleep(TIME_SLEEP)

    df_2.to_csv(OUTPUT_FILE_2, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

    print(f"Đã lưu vào file {OUTPUT_FILE_2}")

def get_shop_details(seller_id: int, product_id: int, spid: str) -> Dict:
    params = {
        "seller_id": seller_id,
        "mpid": product_id,
        "spid": spid,
        "trackity_id": "c2c11d48-c1a6-0ceb-52d5-9ae068d40d8f",
        "platform": "desktop",
        "version": "3"
    }

    try:
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
    try:
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
    if not shop_data:
        return {}

    result = {}

    data = shop_data.get('data')
    if not data or not isinstance(data, dict):
        return {
            'store_name': None,
            'store_review_count': None,
            'total_follower': None,
            'is_official': None
        }

    seller = data.get('seller')
    if not seller or not isinstance(seller, dict):
        return {
            'store_name': None,
            'store_review_count': None,
            'total_follower': None,
            'is_official': None
        }

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
    if not performance_data:
        return {}

    result = {
        'cancel_by_seller_rate': performance_data.get('cancel_by_seller_rate_l4w'),
        'cancel_by_seller_rate_status': performance_data.get('cancel_by_seller_rate_l4w_status'),
        'return_rate': performance_data.get('return_rate_l4w'),
        'return_rate_status': performance_data.get('return_rate_l4w_status')
    }

    return result

def process_3(input_file: str = OUTPUT_FILE_2):
    try:
        df_2 = pd.read_csv(input_file)
        print(f"Đã đọc {len(df_2)} sản phẩm từ {input_file}")
    except FileNotFoundError:
        print(f"Không tìm thấy file {input_file}")
        return

    df_3 = df_2.copy()

    columns = [
        'store_name', 'store_review_count', 'total_follower', 'is_official',
        'cancel_by_seller_rate', 'cancel_by_seller_rate_status',
        'return_rate', 'return_rate_status'
    ]

    for c in columns:
        df_3[c] = None

    total_products = len(df_3)
    performance_cache = {}

    for index, row in df_3.iterrows():
        store_id = row['store_id']
        product_id = row['product_id']
        spid = row['spid']

        print(f"Đang crawl shop của sản phẩm {index + 1}/{total_products}")

        if pd.notna(store_id) and pd.notna(product_id) and pd.notna(spid):
            shop_detail = get_shop_details(int(store_id), int(product_id), str(spid))

            if shop_detail:
                parsed_shop_data = parse_shop_data(shop_detail)

                for key, value in parsed_shop_data.items():
                    if key in df_3.columns:
                        df_3.at[index, key] = value

                if store_id not in performance_cache:
                    performance_detail = get_shop_performance(int(store_id))
                    if performance_detail:
                        parsed_performance_data = parse_performance_data(performance_detail)
                        performance_cache[store_id] = parsed_performance_data
                    else:
                        performance_cache[store_id] = {}

                performance_data = performance_cache[store_id]
                for key, value in performance_data.items():
                    if key in df_3.columns:
                        df_3.at[index, key] = value

                print(f"Đã cập nhật thông tin shop và performance")
            else:
                print(f"Không lấy được thông tin shop")
        else:
            print(f"Thiếu thông tin để crawl shop")

        time.sleep(TIME_SLEEP)

    if 'spid' in df_3.columns:
        df_3 = df_3.drop(columns=['spid'])

    df_3.to_csv(OUTPUT_FILE_3, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_NONNUMERIC)

    print(f"Đã lưu vào file {OUTPUT_FILE_3}")

def process():
    process_1()
    process_2()
    process_3()

if __name__ == "__main__":
    process()