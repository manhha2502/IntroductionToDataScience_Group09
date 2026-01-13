import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CÁC HÀM DÙNG CHUNG
# ============================================================

def load_data(
    train_csv="../data/processed/train_data_final.csv",
    test_csv="../data/processed/test_data_final.csv",
    leak_col=("review_to_sold_ratio",)
):
    """
    Chức năng:
        - Đọc dữ liệu Train và Test từ file CSV.
        - Tự động loại bỏ các cột gây rò rỉ dữ liệu nếu tồn tại.

    Cách thực hiện:
        1. Đọc file CSV train và test bằng pandas.
        2. Kiểm tra danh sách các cột leak (mặc định là review_to_sold_ratio).
        3. Drop các cột này nếu chúng tồn tại trong DataFrame.

    Tham số:
        - train_csv (str): Đường dẫn đến file CSV chứa dữ liệu huấn luyện.
        - test_csv (str): Đường dẫn đến file CSV chứa dữ liệu kiểm tra.
        - leak_col (tuple/list): Danh sách tên các cột cần loại bỏ (mặc định: review_to_sold_ratio).

    Giá trị trả về:
        - train_df (DataFrame): DataFrame dữ liệu train đã được làm sạch cột leak.
        - test_df (DataFrame): DataFrame dữ liệu test đã được làm sạch cột leak.
    """
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    # Ép kiểu leak_col thành list nếu là string
    if isinstance(leak_col, str):
        leak_col = [leak_col]
        
    for col in leak_col:
        if col in train_df.columns:
            train_df = train_df.drop(columns=[col])
        if col in test_df.columns:
            test_df = test_df.drop(columns=[col])

    # --- ALIGN COLUMNS: Ensure Test has all columns from Train ---
    missing_cols = set(train_df.columns) - set(test_df.columns)
    if missing_cols:
        print(f"Warning: Test data is missing {len(missing_cols)} columns found in Train. Filling with 0.")
        for col in missing_cols:
            test_df[col] = 0
            
    # Reorder test columns to match train (optional but safer)
    # Keep extra columns in test just in case, but ensure Train columns exist
    # test_df = test_df[train_df.columns] # Warning: This might drop target if not in train features?
    # Train usually has target. Test might or might not. 
    # For now, just adding missing cols is enough.

    return train_df, test_df

def split_low_high_hard(df, target="quantity_sold", threshold=2.0):
    """
    Chức năng:
        - Chia tập dữ liệu thành 2 phân khúc riêng biệt: Low và High.

    Cách thực hiện:
        1. Tạo mask cho High: quantity_sold >= threshold.
        2. Tạo mask cho Low: quantity_sold < threshold.
        3. Tách DataFrame ban đầu thành 2 DataFrame con tương ứng.

    Tham số:
        - df (DataFrame): DataFrame đầu vào chứa cột target.
        - target (str): Tên cột mục tiêu dùng để phân loại (mặc định 'quantity_sold').
        - threshold (float): Ngưỡng để chia cắt (mặc định 2.0).
          + Low: < threshold
          + High: >= threshold

    Giá trị trả về:
        - df_low (DataFrame): DataFrame chứa các mẫu thuộc phân khúc Low.
        - df_high (DataFrame): DataFrame chứa các mẫu thuộc phân khúc High.
    """
    df_low = df[df[target] < threshold].copy()
    df_high = df[df[target] >= threshold].copy()
    print(f"Split Data: Low (<{threshold}): {len(df_low)} | High (>={threshold}): {len(df_high)}")
    return df_low, df_high

def train_gate_rf(X_train, y_train, threshold=2.0, seed=42):
    """
    Chức năng:
        - Huấn luyện một mô hình phân loại để dự đoán xem một mẫu dữ liệu thuộc nhóm High hay Low.

    Cách thực hiện:
        1. Chuyển đổi nhãn y_train thành nhãn nhị phân: 1 nếu >= threshold, 0 nếu < threshold.
        2. Khởi tạo RandomForestClassifier với 100 cây.
        3. Huấn luyện mô hình trên toàn bộ X_train.

    Tham số:
        - X_train (DataFrame): Features huấn luyện.
        - y_train (Series): Biến mục tiêu thực (số lượng bán).
        - threshold (float): Ngưỡng để tạo nhãn phân loại (mặc định 2.0).
        - seed (int): Random seed để tái lập kết quả (mặc định 42).

    Giá trị trả về:
        - clf (RandomForestClassifier): Mô hình phân loại đã được huấn luyện.
    """
    y_train_class = (y_train >= threshold).astype(int) 
    
    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train_class)
    
    print("Đã train xong Gate Model RandomForestClassifier.")
    return clf

def plot_feature_importance(model, feature_names=None, title="Feature Importance", top_k=10):
    """
    Chức năng:
        - Trích xuất và vẽ biểu đồ mức độ quan trọng (Feature Importance) hoặc hệ số (Coefficients) của mô hình.

    Cách thực hiện:
        1. Tự động phát hiện loại mô hình (Tree-based, Linear, Pipeline) để lấy thuộc tính `feature_importances_` hoặc `coef_`.
        2. Nếu là Linear model, lấy giá trị tuyệt đối của hệ số (abs).
        3. Lấy tên features từ danh sách feature_names truyền vào.
        4. Sắp xếp features theo độ quan trọng giảm dần.
        5. Vẽ biểu đồ cột ngang (Bar plot) cho Top K features quan trọng nhất.

    Tham số:
        - model (object): Mô hình đã được huấn luyện (sklearn, xgboost, lightgbm...).
        - feature_names (list, optional): Danh sách tên features (nếu model không tự lưu).
        - title (str): Tiêu đề biểu đồ.
        - top_k (int): Số lượng features hàng đầu muốn hiển thị (mặc định 10).
    """
    importances = None
    
    # Check feature_importances_ (RF, XGB, LGBM)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    
    # Check Pipeline -> steps -> model -> feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "feature_importances_"):
        importances = model.named_steps["model"].feature_importances_
        
    # Check coef_ (Linear/ElasticNet) - lay abs
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        
    # Check Pipeline -> steps -> model -> coef_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps.get("model"), "coef_"):
        importances = np.abs(model.named_steps["model"].coef_)

    if importances is None:
        print(f"Model {type(model).__name__} không hỗ trợ trích xuất Feature Importance trực tiếp.")
        return

    # Lấy tên features
    if feature_names is not None:
        feats = feature_names
    else:
        feats = [f"Feature {i}" for i in range(len(importances))]
        
    # Tạo DataFrame để sort
    df_imp = pd.DataFrame({"feature": feats, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_k)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="viridis")
    plt.title(f"{title} (Top {top_k})")
    plt.xlabel("Importance/Coef (Abs)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# ============================================================
# LOW MODELS
# ============================================================

def train_lgbm(X_train, y_train, seed=42):
    """
    Chức năng:
        - Huấn luyện mô hình LightGBM Regressor cho tập dữ liệu Low.

    Cách thực hiện:
        1. Thiết lập không gian tham số (param_dist) gồm n_estimators, learning_rate, num_leaves, max_depth...
        2. Sử dụng RandomizedSearchCV để tìm tham số tối ưu (n_iter=10, cv=3).
        3. Huấn luyện mô hình tốt nhất trên toàn bộ tập train.

    Tham số:
        - X_train (DataFrame): Features đầu vào (phân khúc Low).
        - y_train (Series): Biến mục tiêu (Low ).
        - seed (int): Random seed (mặc định 42).

    Giá trị trả về:
        - best_estimator_ (LGBMRegressor): Mô hình LightGBM đã được tối ưu tham số và huấn luyện.
    """
    print("\n" + "-"*60)
    print("HUẤN LUYỆN MODEL LOW: LightGBM")
    print("-"*60)
    
    param_dist = {
        'n_estimators': [100, 500, 1000],
        'learning_rate': [0.05, 0.1, 0.2],
        'num_leaves': [31, 50, 100],
        'max_depth': [5, 7, 9, -1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    search = RandomizedSearchCV(
        estimator=LGBMRegressor(random_state=seed, verbose=-1),
        param_distributions=param_dist,
        n_iter=10,
        scoring='neg_mean_squared_error',
        cv=3,
        random_state=seed,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    
    best_model = search.best_estimator_
    feats = X_train.columns.tolist()

    return best_model, feats

def train_svr(X_train, y_train, top_k=20, seed=42):
    """
    Chức năng:
        - Huấn luyện mô hình Support Vector Regressor (SVR) cho tập dữ liệu Low.

    Cách thực hiện:
        1. Feature Selection: Dùng RandomForestRegressor để chọn ra `top_k` features quan trọng nhất.
        2. Hyperparameter Tuning: Chạy GridSearchCV để tìm C, epsilon, kernel tối ưu cho SVR trên tập features sơ bộ.
        3. Feature Refinement:
           - Tính Permutation Importance trên mô hình SVR tốt nhất.
           - Loại bỏ các features có tầm quan trọng thấp hơn ngưỡng (threshold = avg * 0.1).
        4. Re-training:
           - Train lại SVR với bộ tham số tối ưu trên tập features đã tinh chỉnh.
           - So sánh kết quả, nếu tốt hơn mô hình gốc bước 2 thì chọn, không thì giữ lại bước 2.

    Tham số:
        - X_train (DataFrame): Features đầu vào.
        - y_train (Series): Biến mục tiêu.
        - top_k (int): Số lượng features tối đa chọn ban đầu bằng RF (mặc định 20).
        - seed (int): Random seed (mặc định 42).

    Giá trị trả về:
        - final_model (SVR): Mô hình SVR tốt nhất sau quy trình tinh chỉnh.
          (Trả về kèm danh sách feature cuối cùng).
    """
    print("\n" + "-"*60)
    print("HUẤN LUYỆN MODEL LOW: SVR")
    print("-"*60)
    
    # 1. Feature selection bằng Random Forest
    print("Bước 1: Feature selection bằng Random Forest...")
    rf = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    low_feats = fi.head(min(top_k, len(fi))).index.tolist()
    print(f"  Đã chọn {len(low_feats)} features quan trọng nhất.")
    
    # 2. GridSearchCV SVR
    print("Bước 2: GridSearchCV cho SVR...")
    pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
    grid = {
        'svr__C': [0.1, 1, 10, 100],
        'svr__epsilon': [0.01, 0.1, 0.2],
        'svr__kernel': ['linear', 'rbf', 'poly']
    }
    gs = GridSearchCV(pipe, grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    gs.fit(X_train[low_feats], y_train)
    
    print(f"Best params: {gs.best_params_}")
    print(f"Best score (MSE): {-gs.best_score_:.6f}")
    
    # 3. Permutation importance để tinh chỉnh (Refine)
    print("Bước 3: Permutation importance để tinh chỉnh features...")
    perm = permutation_importance(gs.best_estimator_, X_train[low_feats], y_train, n_repeats=10, random_state=seed, scoring='neg_mean_squared_error')
    perm_imp = pd.Series(perm.importances_mean, index=low_feats).sort_values(ascending=False)
    
    threshold = perm_imp.mean() * 0.1
    refined_feats = perm_imp[perm_imp > threshold].index.tolist()
    
    if len(refined_feats) < 10:
        refined_feats = perm_imp.head(min(20, len(perm_imp))).index.tolist()
        
    print(f"Refined còn: {len(refined_feats)} features (từ {len(low_feats)})")
    
    # 4. Re-train nếu cần
    final_model = gs.best_estimator_
    final_feats = low_feats
    
    if len(refined_feats) < len(low_feats):
        print("Bước 4: Re-training với features đã tinh chỉnh...")
        gs_refined = GridSearchCV(pipe, grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        gs_refined.fit(X_train[refined_feats], y_train)
        
        print(f"  Refined score (MSE): {-gs_refined.best_score_:.6f}")
        
        if gs_refined.best_score_ > gs.best_score_:
            print("-> Sử dụng mô hình đã tinh chỉnh (Hiệu suất tốt hơn).")
            final_model = gs_refined.best_estimator_
            final_feats = refined_feats
        else:
            print("-> Giữ nguyên mô hình gốc.")
            
    # Lưu danh sách features vào model để dùng khi predict (tránh lỗi thiếu cột)
 
    return final_model, final_feats

def train_knn(X_train, y_train, top_k=40, seed=42):
    """
    Chức năng:
        - Huấn luyện mô hình K-Nearest Neighbors (KNN) Regressor cho tập dữ liệu Low.

    Cách thực hiện (theo TrungHieu/knn_regression.ipynb):
        1. Feature Selection: Sử dụng RandomForestRegressor để xếp hạng features, chọn `top_k` features quan trọng nhất.
        2. Hyperparameter Tuning: Chạy GridSearchCV tìm số lượng n_neighbors (3, 5, 7...), weights (uniform/distance), p (Minkowski).
        3. Huấn luyện mô hình tốt nhất trên tập features đã chọn.

    Tham số:
        - X_train (DataFrame): Features đầu vào.
        - y_train (Series): Biến mục tiêu.
        - top_k (int): Số lượng features tối đa được chọn (mặc định 40).
        - seed (int): Random seed (dùng cho bước Feature Selection).

    Giá trị trả về:
        - best_model (Pipeline): Pipeline chứa scaler và mô hình KNN tốt nhất.
          (Trả về kèm danh sách feature cuối cùng).
    """
    print("\n" + "-"*60)
    print("HUẤN LUYỆN MODEL LOW: KNN")
    print("-"*60)
    
    # 1. Feature selection bằng RF
    print("Bước 1: Feature selection (RF Top-K)...")
    rf = RandomForestRegressor(n_estimators=400, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    fi = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feats = fi.head(min(top_k, len(fi))).index.tolist()
    print(f"  Đã chọn {len(feats)} features.")
    
    # 2. GridSearchCV KNN
    print("Bước 2: GridSearchCV cho KNN...")
    X_sel = X_train[feats]
    
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ])
    grid = {
        "knn__n_neighbors": [3, 5, 7, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2]
    }
    
    gs = GridSearchCV(pipe, grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
    gs.fit(X_sel, y_train)
    
    print(f"  Best params: {gs.best_params_}")
    
    best_model = gs.best_estimator_

    
    return best_model, feats

# ============================================================
# 3. HIGH MODELS
# ============================================================

def train_elasticnet(X_train, y_train, seed=42, coef_threshold=1e-8):
    """
    Chức năng:
        - Huấn luyện mô hình ElasticNet (Linear Regression + L1/L2 Regularization) cho tập High.

    Cách thực hiện (logic Refinement):
        1. Tìm tham số: Chạy ElasticNetCV để tự động tìm alpha và l1_ratio tốt nhất.
        2. Sơ loại Feature: Loại bỏ các features có hệ số (coefficient) xấp xỉ 0 (< 1e-8).
        3. Feature Refinement:
           - Xem xét các features còn lại.
           - Lọc bỏ 25% features có giá trị tuyệt đối của hệ số nhỏ nhất (quantile 0.25).
        4. Re-training: Huấn luyện ElasticNet cuối cùng trên tập features tinh chỉnh.

    Tham số:
        - X_train, y_train: Dữ liệu huấn luyện.
        - seed: Random seed.
        - coef_threshold: Ngưỡng hệ số tối thiểu để được giữ lại ở bước sơ loại.

    Giá trị trả về:
        - final_model (Pipeline): Pipeline chứa Scaler và ElasticNet model đã tinh chỉnh.
    """
    print("\n" + "-"*60)
    print("HUẤN LUYỆN MODEL HIGH: ElasticNet (Refined)")
    print("-"*60)
    
    # 1. ElasticNetCV
    print("Bước 1: ElasticNetCV tìm tham số tối ưu...")
    enet_cv = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10],
            cv=5, max_iter=10000, random_state=seed
        ))
    ])
    enet_cv.fit(X_train, y_train)
    best_alpha = enet_cv.named_steps["model"].alpha_
    best_l1 = enet_cv.named_steps["model"].l1_ratio_
    print(f"  Best: alpha={best_alpha}, l1_ratio={best_l1}")
    
    # 2. Feature Selection theo Coef
    print("Bước 2: Chọn features dựa trên hệ số (Coefficient)...")
    final_pipeline_tmp = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=10000, random_state=seed))
    ])
    final_pipeline_tmp.fit(X_train, y_train)
    coef = final_pipeline_tmp.named_steps["model"].coef_
    
    high_feats = X_train.columns[np.abs(coef) > coef_threshold].tolist()
    if len(high_feats) == 0: high_feats = X_train.columns.tolist()
    print(f"  Selected (Sơ bộ): {len(high_feats)} features")
    
    # 3. Refine features (loại bỏ 25% yếu nhất)
    print("Bước 3: Tinh chỉnh features (Refinement)...")
    coef_df = pd.DataFrame({"feature": high_feats, "coef": final_pipeline_tmp.named_steps["model"].coef_[np.abs(coef) > coef_threshold]})
    coef_df["abs"] = coef_df["coef"].abs()
    threshold_refined = coef_df["abs"].quantile(0.25)
    refined_feats = coef_df[coef_df["abs"] >= threshold_refined]["feature"].tolist()
    
    if len(refined_feats) < 10:
        refined_feats = coef_df.nlargest(15, "abs")["feature"].tolist()
        
    print(f"  Refined còn: {len(refined_feats)} features")
    
    # 4. Final Train
    print("Bước 4: Train model cuối cùng...")
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=10000, random_state=seed))
    ])
    final_model.fit(X_train[refined_feats], y_train)
    

    return final_model, refined_feats

def train_rf(X_train, y_train, seed=42):
    """
    Chức năng:
        - Huấn luyện Random Forest Regressor cho tập High.

    Cách thực hiện (logic Refinement):
        1. Hyperparameter Tuning: Chạy GridSearchCV cho Random Forest (n_estimators, max_depth, max_features...).
        2. Feature Refinement:
           - Lấy Feature Importance từ model tốt nhất.
           - Loại bỏ 20% features yếu nhất (quantile 0.20).
        3. Re-training: Huấn luyện lại Random Forest với các features đã chọn và tham số tối ưu.
        4. Kiểm tra: Chỉ sử dụng model tinh chỉnh nếu hiệu suất (R2) không bị giảm quá nhiều so với model gốc (chấp nhận giảm 2% để model nhẹ hơn).

    Tham số:
        - X_train, y_train: Dữ liệu huấn luyện.
        - seed: Random seed.

    Giá trị trả về:
        - final_model (RandomForestRegressor): Mô hình RF đã tinh chỉnh.
    """
    print("\n" + "-"*60)
    print("HUẤN LUYỆN MODEL HIGH: Random Forest (Refined)")
    print("-"*60)
    
    # 1. GridSearchCV
    print("Bước 1: GridSearchCV...")
    rf = RandomForestRegressor(random_state=seed, n_jobs=-1)
    grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"]
    }
    gs = GridSearchCV(rf, grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best score (R2): {gs.best_score_:.6f}")
    
    # 2. Refine Features
    print("Bước 2: Tinh chỉnh features dựa trên importance...")
    best_estimator = gs.best_estimator_
    fi = pd.Series(best_estimator.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    
    threshold = fi.quantile(0.20) # Loại 20% yếu
    refined_feats = fi[fi >= threshold].index.tolist()
    if len(refined_feats) < 15:
        refined_feats = fi.head(30).index.tolist()
        
    print(f"  Refined còn: {len(refined_feats)} features")
    
    # 3. Retrain
    print("Bước 3: Re-training với refined features...")
    gs_refined = GridSearchCV(rf, grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
    gs_refined.fit(X_train[refined_feats], y_train)
    
    final_model = gs_refined.best_estimator_
    
    # Kiểm tra xem có tốt hơn không
    if gs_refined.best_score_ >= gs.best_score_ * 0.98: # Cho phép giảm nhẹ để model gọn hơn
        print("  -> Sử dụng model tinh chỉnh.")
        final_model = gs_refined.best_estimator_
    else:
        print("  -> Giữ nguyên model gốc (do performance giảm nhiều).")
        final_model = best_estimator
        refined_feats = X_train.columns.tolist()
        

    return final_model, refined_feats

def train_xgboost(X_train, y_train, seed=42, n_iter=10):
    """
    Chức năng:
        - Huấn luyện XGBoost Regressor cho tập High.

    Cách thực hiện (logic Refinement):
        1. Feature Selection (Initial): Train một model XGBoost sơ khởi để lấy Feature Importance và sắp xếp các features.
        2. Hyperparameter Tuning: Chạy RandomizedSearchCV để tìm tham số tối ưu (learning_rate, max_depth, subsample...).
        3. Feature Refinement:
           - Train thử model với tham số tốt nhất.
           - Lọc 20% features yếu nhất.
        4. Re-training: Huấn luyện model cuối cùng trên tập features tinh chỉnh.

    Tham số:
        - X_train, y_train: Dữ liệu huấn luyện.
        - seed: Random seed.
        - n_iter: Số lần lặp cho RandomizedSearchCV (mặc định 10).

    Giá trị trả về:
        - final_model (XGBRegressor): Mô hình XGBoost đã tinh chỉnh.
    """
    print("\n" + "-"*60)
    print("HUẤN LUYỆN MODEL HIGH: XGBoost (Refined)")
    print("-"*60)
    
    # 1. Initial Train để lấy Feature Importance
    print("Bước 1: Initial training để lọc feature...")
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
    
    first = XGBRegressor(n_estimators=10000, learning_rate=0.1, max_depth=6, random_state=seed, 
                         early_stopping_rounds=50, n_jobs=-1)
    first.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    
    fi = pd.Series(first.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feats = fi.index.tolist() # Dùng tất cả, nhưng sắp xếp
    
    # 2. RandomizedSearchCV
    print("Bước 2: RandomizedSearchCV...")
    param_dist = {
        "n_estimators": [100, 500, 1000, 5000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }
    
    rs = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=seed, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=3, random_state=seed, n_jobs=-1, verbose=0
    )
    rs.fit(X_tr, y_tr)
    print(f"  Best params: {rs.best_params_}")
    
    # 3. Refine & Retrain Final
    print("Bước 3: Tinh chỉnh & Train final model...")
    # Train tạm model đầy đủ để lấy importance chính xác
    full_model = XGBRegressor(**rs.best_params_, random_state=seed, n_jobs=-1)
    full_model.fit(X_train, y_train)
    
    fi_final = pd.Series(full_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    threshold = fi_final.quantile(0.20)
    refined_feats = fi_final[fi_final >= threshold].index.tolist()
    
    if len(refined_feats) < 15:
        refined_feats = fi_final.head(25).index.tolist()
        
    print(f"  Refined còn: {len(refined_feats)} features")
    
    final_model = XGBRegressor(**rs.best_params_, random_state=seed, n_jobs=-1)
    final_model.fit(X_train[refined_feats], y_train)
    

    return final_model, refined_feats

# ============================================================
# COMPARISON
# ============================================================

def run_model_comparison(results_dict, dataset_name="Data"):
    """
    Chức năng:
        - So sánh hiệu suất của nhiều mô hình khác nhau trên cùng một tập dữ liệu.
        - Tính toán các chỉ số: MAE, RMSE, R2.
        - Hiển thị bảng kết quả và vẽ biểu đồ so sánh trực quan (Bar & Scatter).

    Cách thực hiện:
        1. Duyệt qua dictionary chứa kết quả dự đoán của từng model.
        2. Với mỗi model, tính toán:
           - MAE (Mean Absolute Error)
           - RMSE (Root Mean Squared Error)
           - R2 Score
        3. Tổng hợp kết quả vào DataFrame.
        4. Vẽ biểu đồ so sánh Metrics (RMSE, MAE, R2).
        5. Vẽ biểu đồ tương quan Thực tế vs Dự đoán (Scatter plot) cho từng model.
        6. Lưu các biểu đồ vào file PNG.

    Tham số:
        - results_dict (dict): Dictionary chứa kết quả. 
          Format: {'ModelName': {'y_true': actual_values, 'y_pred': predicted_values}}
        - dataset_name (str): Tên tập dữ liệu để hiển thị trên tiêu đề (ví dụ "Low", "High").

    Giá trị trả về:
        - metrics_df (DataFrame): Bảng tổng hợp các chỉ số đánh giá của các mô hình.
    """
    metrics_list = []
    
    for name, res in results_dict.items():
        y_true = res['y_true']
        y_pred = res['y_pred']
        
        y_pred = np.maximum(y_pred, 0)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics_list.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    print(f"\n--- Bảng So Sánh Hiệu Suất ({dataset_name}) ---")
    print(metrics_df.to_string(index=False))
    
    # 1. Vẽ biểu đồ Metrics
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(f'So sánh Metrics trên tập {dataset_name}', fontsize=16)
    
    sns.barplot(x="Model", y="RMSE", data=metrics_df, ax=axes1[0], palette="Blues_d")
    axes1[0].set_title("RMSE (Thấp là tốt)")
    for i, v in enumerate(metrics_df["RMSE"]):
        axes1[0].text(i, v, f"{v:.3f}", ha='center', va='bottom')
        
    sns.barplot(x="Model", y="MAE", data=metrics_df, ax=axes1[1], palette="Greens_d")
    axes1[1].set_title("MAE (Thấp là tốt)")
    for i, v in enumerate(metrics_df["MAE"]):
        axes1[1].text(i, v, f"{v:.3f}", ha='center', va='bottom')
        
    sns.barplot(x="Model", y="R2", data=metrics_df, ax=axes1[2], palette="Reds_d")
    axes1[2].set_title("R2 (Cao là tốt)")
    for i, v in enumerate(metrics_df["R2"]):
        axes1[2].text(i, v, f"{v:.3f}", ha='center', va='bottom')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_name = dataset_name.replace(" ", "_").lower()
    plt.savefig(f"../data/images/comparison_metrics_{safe_name}.png")
    print(f"Đã lưu biểu đồ metrics: ../data/images/comparison_metrics_{safe_name}.png")
    plt.show()
    
    # 2. Vẽ biểu đồ Scatter (Thực tế vs Dự đoán)
    n_models = len(results_dict)
    fig2, axes2 = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: axes2 = [axes2]
    
    fig2.suptitle(f'Thực tế vs Dự đoán - Tập {dataset_name}', fontsize=16)
    
    for idx, (name, res) in enumerate(results_dict.items()):
        ax = axes2[idx]
        y_true = res['y_true']
        y_pred = res['y_pred']
        y_pred = np.maximum(y_pred, 0)
        
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, alpha=0.5, color='blue')
        
        # Đường chéo reference y=x
        m_min = min(y_true.min(), y_pred.min())
        m_max = max(y_true.max(), y_pred.max())
        ax.plot([m_min, m_max], [m_min, m_max], 'r--', lw=2)
        
        ax.set_title(f"Model: {name}")
        ax.set_xlabel("Thực tế (Ground Truth)")
        ax.set_ylabel("Dự đoán (Predicted)")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"../data/images/comparison_scatter_{safe_name}.png")
    print(f"Đã lưu biểu đồ scatter: ../data/images/comparison_scatter_{safe_name}.png")
    plt.show()

    return metrics_df
