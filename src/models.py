import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# LINEAR REGRESSION
# ============================================================

def train_linear_regression_elasticnet(
    train_csv="../data/processed/train_data_final.csv",
    test_csv="../data/processed/test_data_final.csv",
    target="quantity_sold",
    leak_col="review_to_sold_ratio",
    coef_threshold=1e-8,
    out_fi_png="../data/images/feature_importance_lr.png",
    out_pred_csv="../data/predicted/lr_predictions.csv",
    top_n=10,
    seed=42,
):
    """
    Chức năng:
        - Train mô hình ElasticNet (Linear Regression) theo pipeline sau:
            + Loại leak column
            + ElasticNetCV tìm best hyperparameters
            + Chọn feature theo |coef| > threshold
            + Train final model trên tập feature đã chọn
            + Vẽ feature importance (Top 10) và lưu ảnh
            + Predict + tính metric trên test
            + Lưu file lr_predictions.csv

    Cách thực hiện:
        1. Đọc train/test từ:
            - ../data/processed/train_data_final.csv
            - ../data/processed/test_data_final.csv
        2. Tách X/y theo TARGET="quantity_sold".
        3. Drop leak column (mặc định: review_to_sold_ratio) ở cả train và test.
        4. Dùng ElasticNetCV để tìm best alpha và best l1_ratio:
            - l1_ratio: [0.1 .. 0.99]
            - alphas: [0.0001 .. 10]
            - cv=5
        5. Train ElasticNet lần 1 với best params -> lấy coef -> chọn feature theo:
            |coef| > coef_threshold (mặc định 1e-8)
        6. Train final model trên tập feature đã chọn.
        7. Tạo bảng hệ số (coef_final_df), sắp xếp theo |coef| giảm dần.
        8. Vẽ Top 10 feature theo abs_coefficient, lưu ảnh:
            - ../data/images/feature_importance_lr.png
        9. Predict trên test, tính MAE/MSE/RMSE/R2.
        10. Lưu kết quả dự đoán:
            - lr_predictions.csv (index=True, index=test_df.index)

    Tham số:
        - train_csv: str
            Đường dẫn file train_data_final.csv
        - test_csv: str
            Đường dẫn file test_data_final.csv
        - target: str
            Tên biến mục tiêu (mặc định "quantity_sold")
        - leak_col: str
            Cột leak cần drop (mặc định "review_to_sold_ratio")
        - coef_threshold: float
            Ngưỡng chọn feature dựa trên |coef| (mặc định 1e-8)
        - out_fi_png: str
            Đường dẫn lưu ảnh feature importance
        - out_pred_csv: str
            Tên/đường dẫn lưu file dự đoán (mặc định "lr_predictions.csv")
        - top_n: int
            Số feature top để vẽ
        - seed: int
            Random seed (mặc định 42)

    Giá trị trả về:
        dict gồm:
            - model: Pipeline final
            - best_alpha, best_l1_ratio
            - selected_features: list
            - coef_df: DataFrame hệ số final (có abs_coefficient)
            - y_pred: mảng dự đoán
            - metrics: dict MAE/MSE/RMSE/R2
            - pred_df: DataFrame prediction đã lưu
    """
    np.random.seed(seed)

    # Load train & test
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Split train / target
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Drop leak column
    if leak_col in X_train.columns:
        X_train = X_train.drop(columns=[leak_col])
    if leak_col in X_test.columns:
        X_test = X_test.drop(columns=[leak_col])

    # ElasticNetCV
    enet_cv = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNetCV(
            l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            alphas=[0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
            cv=5,
            max_iter=10000,
            random_state=seed
        ))
    ])
    enet_cv.fit(X_train, y_train)

    # Lấy best params từ ElasticNetCV
    best_alpha = enet_cv.named_steps["model"].alpha_
    best_l1_ratio = enet_cv.named_steps["model"].l1_ratio_
    print("Best alpha:", best_alpha)
    print("Best l1_ratio:", best_l1_ratio)

    # Fit lần 1 và chọn features theo coef threshold
    best_enet = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1_ratio,
            max_iter=10000,
            random_state=seed
        ))
    ])
    best_enet.fit(X_train, y_train)

    # Feature selection dựa vào hệ số coef
    coef = best_enet.named_steps["model"].coef_
    selected_features = X_train.columns[np.abs(coef) > coef_threshold].tolist()

    print(f"Total selected features: {len(selected_features)}")

    # Train với selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    # Train final model với selected features
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(
            alpha=best_alpha,
            l1_ratio=best_l1_ratio,
            max_iter=10000,
            random_state=seed
        ))
    ])
    final_model.fit(X_train_selected, y_train)

    # Feature importance dataframe
    final_coef = final_model.named_steps["model"].coef_
    coef_final_df = pd.DataFrame({
        "feature": X_train_selected.columns,
        "coefficient": final_coef
    })
    coef_final_df["abs_coefficient"] = coef_final_df["coefficient"].abs()
    coef_final_df = coef_final_df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

    print(f"[FINAL MODEL] Number of selected features: {len(final_coef)}")

    # Plot top 10 feature importance (dựa trên abs_coefficient)
    data_to_plot = coef_final_df.head(top_n)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=data_to_plot, x="abs_coefficient", y="feature", palette='viridis')

    for i, (abs_val, orig_val) in enumerate(zip(data_to_plot["abs_coefficient"], data_to_plot["coefficient"])):
        ax.text(abs_val + 0.005, i, f'{abs_val:.4f}', color='black', va='center', fontweight='bold')

    plt.title(
        f'Top {top_n} Yếu tố ảnh hưởng mạnh nhất (Linear Regression - Absolute Values)',
        fontsize=16, fontweight='bold', pad=20
    )
    plt.xlabel('Trọng số tuyệt đối (Absolute Coefficient)', fontsize=12, fontweight='bold')
    plt.ylabel('Các đặc trưng (Features)', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_fi_png, dpi=300)
    plt.show()

    # Predict + evaluation
    y_pred = final_model.predict(X_test_selected)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    print("\nTest Set Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Lưu predictions ra CSV
    results_df = pd.DataFrame({
        "quantity_sold_ground_truth": y_test.values,
        "quantity_sold_predicted": y_pred
    }, index=test_df.index)

    results_df.to_csv(out_pred_csv, index=True)

    return {
        "model": final_model,
        "best_alpha": best_alpha,
        "best_l1_ratio": best_l1_ratio,
        "selected_features": selected_features,
        "coef_df": coef_final_df,
        "y_pred": y_pred,
        "metrics": metrics,
        "pred_df": results_df,
    }


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest_regressor(
    train_csv="../data/processed/train_data_final.csv",
    test_csv="../data/processed/test_data_final.csv",
    target="quantity_sold",
    out_fi_png="../data/images/feature_importance_rfr.png",
    out_pred_csv="../data/predicted/rfr_predictions.csv",
    top_n=10,
    seed=42,
    verbose=1,
):
    """
    Chức năng:
        - Train RandomForestRegressor theo đúng pipeline sau:
            + GridSearchCV để tìm best params
            + Đánh giá R2 và RMSE
            + Vẽ feature importance Top 10 và lưu ảnh
            + Lưu dự đoán ra ../data/predicted/rfr_predictions.csv

    Cách thực hiện:
        1. Đọc dữ liệu train/test từ ../data/processed/
        2. Tách X/y theo target="quantity_sold"
        3. Khởi tạo RandomForestRegressor(random_state=42, n_jobs=-1)
        4. Thiết lập param_grid:
            - n_estimators: [200, 500]
            - max_depth: [None, 10, 20]
            - min_samples_split: [2, 5]
            - min_samples_leaf: [1, 2]
            - max_features: ["sqrt", "log2"]
        5. GridSearchCV:
            - cv=3, scoring="r2", n_jobs=-1, verbose=1
        6. Lấy best_estimator_ -> predict test
        7. Tính R2 và RMSE, in ra.
        8. Lấy feature_importances_ -> vẽ Top N, lưu ảnh:
            - ../data/images/feature_importance_rfr.png
        9. Lưu file prediction:
            - ../data/predicted/rfr_predictions.csv

    Tham số:
        - train_csv, test_csv: str
            Đường dẫn file train/test
        - target: str
            Tên biến mục tiêu
        - out_fi_png: str
            Đường dẫn lưu ảnh FI
        - out_pred_csv: str
            Đường dẫn lưu prediction csv
        - op_n: int
            Số feature top để vẽ
        - seed: int
            Random seed
        - verbose: int
            Verbose của GridSearchCV

    Giá trị trả về:
        dict gồm:
            - grid: GridSearchCV object
            - best_params: dict
            - model: rf_best
            - feature_importances: Series đã sort giảm dần
            - y_pred: prediction
            - metrics: {"R2": ..., "RMSE": ...}
            - pred_df: DataFrame prediction đã lưu
    """
    # Load dữ liệu train/test
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # Tách X/y
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]

    X_test = df_test.drop(columns=[target])
    y_test = df_test[target]

    # Khởi tạo Random Forest base model
    rf = RandomForestRegressor(
        random_state=seed,
        n_jobs=-1
    )

    # Param grid để GridSearchCV
    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    # GridSearchCV để tìm best params theo R2 (cv=3)
    grid_rf = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=verbose
    )

    grid_rf.fit(X_train, y_train)

    print("Best RF params:")
    print(grid_rf.best_params_)

    # Lấy mô hình tốt nhất và predict test
    rf_best = grid_rf.best_estimator_

    y_test_pred = rf_best.predict(X_test)

    # Tính metrics
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"RF R2 (test): {r2:.4f}")
    print(f"RF RMSE (test): {rmse:.4f}")

    # Feature importance 
    rf_feature_importances = pd.Series(
        rf_best.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("Random Forest Feature Importances:")
    print(rf_feature_importances.to_string())

    # Plot top 10 feature importance
    data_to_plot = rf_feature_importances.head(top_n)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette='viridis')

    for i, v in enumerate(data_to_plot.values):
        ax.text(v + 0.001, i, f'{v:.4f}', color='black', va='center', fontweight='bold')

    plt.title(f'Top {top_n} yếu tố ảnh hưởng nhất đến Doanh số (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Mức độ quan trọng (Importance Score)', fontsize=12, fontweight='bold')
    plt.ylabel('Các đặc trưng (Features)', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_fi_png, dpi=300)
    plt.show()

    # Lưu prediction CSV
    result_df = pd.DataFrame({
        'quantity_sold_ground_truth': y_test.values,
        'quantity_sold_predicted': y_test_pred
    })
    result_df.to_csv(out_pred_csv, index=False)

    return {
        "grid": grid_rf,
        "best_params": grid_rf.best_params_,
        "model": rf_best,
        "feature_importances": rf_feature_importances,
        "y_pred": y_test_pred,
        "metrics": {"R2": r2, "RMSE": rmse},
        "pred_df": result_df,
    }


# ============================================================
# XGBOOST
# ============================================================

def train_xgboost_regressor(
    train_csv="../data/processed/train_data_final.csv",
    test_csv="../data/processed/test_data_final.csv",
    target="quantity_sold",
    cols_to_drop=('review_count', 'review_to_sold_ratio', 'has_video'),
    out_fi_png="../data/images/feature_importance_xgbr.png",
    out_pred_csv="../data/predicted/xgbr_predictions.csv",
    top_n=10,
    seed=42,
    n_iter=10,
):
    """
    Chức năng:
        - Train XGBoost Regressor theo đúng pipeline sau:
            + Chia train/val 80/20 để train lần 1 và lấy feature importance
            + Loại các cột gây nhiễu khỏi danh sách feature dùng để tune/train
            + RandomizedSearchCV để tìm hyperparameters tốt
            + Train final model trên toàn bộ dữ liệu (train + val)
            + Đánh giá trên thang gốc bằng expm1 (RMSE/MAE/R2)
            + Vẽ feature importance Top N và lưu ảnh
            + Lưu dự đoán ra ../data/predicted/xgbr_predictions.csv

    Cách thực hiện:
        1. Đọc train_data_final.csv và test_data_final.csv
        2. Tách X_test/y_test, và chia train thành:
            - X_train, X_val, y_train, y_val (test_size=0.2, random_state=42)
        3. Train XGBRegressor lần 1 để lấy feature_importances_:
            - n_estimators=10000
            - learning_rate=0.1
            - max_depth=6
            - early_stopping_rounds=50
            - eval_set=[(X_val, y_val)]
        4. Tạo danh sách valid_features bằng cách loại cols_to_drop khỏi ranking importance.
        5. RandomizedSearchCV trên X_train_top:
            - n_iter=10, cv=3, scoring='neg_mean_squared_error'
            - param_dist gồm n_estimators, learning_rate, max_depth, subsample, colsample_bytree
        6. Gộp train + val:
            - X_full_train = concat(X_train_top, X_val_top)
            - y_full_train = concat(y_train, y_val)
        7. Train final_model với best_params trên X_full_train.
        8. Predict trên X_test_top.
        9. Đánh giá trên thang gốc:
            - y_test_real = expm1(y_test)
            - y_pred_real = expm1(y_pred)
            - tính RMSE/MAE/R2
        10. Lấy feature_importances_ từ final_model -> vẽ Top 10 và lưu ảnh:
            - ../data/images/feature_importance_xgbr.png
        11. Lưu prediction (log-scale) ra:
            - ../data/predicted/xgbr_predictions.csv

    Tham số:
        - train_csv, test_csv: str
            Đường dẫn file train/test
        - target: str
            Tên biến mục tiêu
        - cols_to_drop: tuple/list
            Danh sách cột loại bỏ trước khi tune/train
        - out_fi_png: str
            Đường dẫn lưu ảnh FI
        - out_pred_csv: str
            Đường dẫn lưu prediction csv
        - top_n: int
            Số feature top để vẽ
        - seed: int
            Random seed
        - n_iter: int
            Số lần random search

    Giá trị trả về:
        dict gồm:
            - model_first: model train lần 1 (để xem importance ban đầu)
            - feature_importances_first: Series importance lần 1
            - valid_features: list feature dùng để tune/train
            - search: RandomizedSearchCV object
            - best_params: dict
            - model: final_model
            - feature_importances_final: Series importance final
            - y_pred: prediction (log-scale)
            - metrics_real: {"RMSE": ..., "MAE": ..., "R2": ...} (real-scale)
            - pred_df: DataFrame prediction đã lưu
    """
    # Load dữ liệu
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)

    # Tách X_test/y_test
    y_test = test_data[target]
    X_test = test_data.drop(columns=[target])

    # Split train thành train/val (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        train_data.drop(columns=[target]),
        train_data[target],
        test_size=0.2,
        random_state=seed
    )

    # train lần 1 để lấy feature importance
    xgb_model = XGBRegressor(
        n_estimators=10000,
        learning_rate=0.1,
        max_depth=6,
        random_state=seed,
        early_stopping_rounds=50
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Feature importance lần 1
    feature_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    print("Feature Importances:")
    print(feature_importances.to_string())

    # Bỏ cột nhiễu
    valid_features = [col for col in feature_importances.index if col not in set(cols_to_drop)]

    X_train_top = X_train[valid_features]
    X_val_top = X_val[valid_features]
    X_test_top = X_test[valid_features]

    # RandomizedSearchCV
    param_dist = {
        'n_estimators': [100, 500, 1000, 5000, 10000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    random_search = RandomizedSearchCV(
        estimator=XGBRegressor(random_state=seed),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        random_state=seed,
        n_jobs=-1
    )

    random_search.fit(X_train_top, y_train)
    best_xgb_model = random_search.best_estimator_

    print("Đang huấn luyện model cuối cùng trên toàn bộ dữ liệu (Train + Val)...")

    # Gộp train + val
    X_full_train = pd.concat([X_train_top, X_val_top])
    y_full_train = pd.concat([y_train, y_val])

    best_params = random_search.best_params_

    final_model = XGBRegressor(
        **best_params,
        random_state=seed,
        n_jobs=-1
    )

    final_model.fit(X_full_train, y_full_train)

    print("Đã train xong Final Model!")

    # predict + metric (expm1)
    y_test_pred = final_model.predict(X_test_top)
    y_test_pred_real = np.expm1(y_test_pred)
    y_test_real = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_real, y_test_pred_real))
    mae = mean_absolute_error(y_test_real, y_test_pred_real)
    r2 = r2_score(y_test_real, y_test_pred_real)

    print(f"Test RMSE: {rmse}")
    print(f"Test MAE: {mae}")
    print(f"Test R²: {r2}")

    # feature importance final
    final_feature_importances = pd.Series(
        final_model.feature_importances_,
        index=X_full_train.columns
    ).sort_values(ascending=False)

    print("="*50)
    print("TOP 20 FEATURE IMPORTANCES FROM FINAL MODEL")
    print("="*50)
    print(final_feature_importances.head(20).to_string())
    print("\n" + "="*50)
    print(f"Total features used: {len(final_feature_importances)}")
    print("="*50)

    # plot top 10
    data_to_plot = final_feature_importances.head(top_n)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette='viridis')

    for i, v in enumerate(data_to_plot.values):
        ax.text(v + 0.001, i, f'{v:.4f}', color='black', va='center', fontweight='bold')

    plt.title(f'Top {top_n} Yếu tố ảnh hưởng nhất đến Doanh số (XGBoost)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Mức độ quan trọng (Importance Score)', fontsize=12, fontweight='bold')
    plt.ylabel('Các đặc trưng (Features)', fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_fi_png, dpi=300)
    plt.show()

    # save predictions 
    pred_df = pd.DataFrame({
        'quantity_sold_ground_truth': y_test.values,
        'quantity_sold_predicted': y_test_pred
    })
    pred_df.to_csv(out_pred_csv, index=False)

    return {
        "model_first": xgb_model,
        "feature_importances_first": feature_importances,
        "valid_features": valid_features,
        "search": random_search,
        "best_params": best_params,
        "model": final_model,
        "feature_importances_final": final_feature_importances,
        "y_pred": y_test_pred,
        "metrics_real": {"RMSE": rmse, "MAE": mae, "R2": r2},
        "pred_df": pred_df,
    }


# ============================================================
# 4) COMPARISON
# ============================================================

def run_model_comparison(
    files=(
        '../data/predicted/xgbr_predictions.csv',
        '../data/predicted/rfr_predictions.csv',
        '../data/predicted/lr_predictions.csv'
    ),
    model_names=(
        'XGBoost Regressor',
        'Random Forest Regressor',
        'Linear Regression'
    ),
    out_dir="../data/images/"
):
    """
    Chức năng:
        - So sánh 3 mô hình dựa trên các file prediction CSV:
            + Tính metric trên log-scale (RMSE/MAE/R²)
            + Tính metric trên real-scale sau khi expm1 (RMSE Real/MAE Real/R² Real)
            + Vẽ và lưu các biểu đồ:
                - model_comparison_1.png (MAE vs MAE Real)
                - model_comparison_2.png (R² vs R² Real)
                - model_comparison_3.png (RMSE vs RMSE Real)
                - predicted_vs_actual_1.png (scatter log-scale)
                - predicted_vs_actual_2.png (scatter real-scale)

    Cách thực hiện:
        1. Với mỗi file prediction:
            - Đọc y_true, y_pred từ 2 cột:
              quantity_sold_ground_truth, quantity_sold_predicted
            - Tạo y_true_real, y_pred_real = expm1(...)
            - Tính metric log-scale và real-scale
            - Lưu lại (y_true, y_pred) để vẽ scatter
        2. Gom tất cả kết quả thành DataFrame comparison_df và in ra.
        3. Vẽ 3 nhóm bar chart (MAE, R², RMSE) cho log & real, lưu ảnh.
        4. Vẽ scatter Predicted vs Actual cho từng model (log-scale), lưu ảnh.
        5. Vẽ scatter Predicted vs Actual (real-scale), lưu ảnh.
        6. Trả về comparison_df.

    Tham số:
        - files: tuple/list[str]
            Danh sách đường dẫn prediction csv
        - model_names: tuple/list[str]
            Tên model tương ứng với files
        - out_dir: str
            Thư mục lưu ảnh output

    Giá trị trả về:
        pandas.DataFrame:
            Bảng tổng hợp metric của các mô hình (log-scale và real-scale).
    """
    summary = []
    results = {}

    for file, name in zip(files, model_names):
        df = pd.read_csv(file)

        # y_true/y_pred hiện tại đang ở log-scale
        y_true = df['quantity_sold_ground_truth']
        y_pred = df['quantity_sold_predicted']

        # chuyển sang thang gốc
        y_true_real = np.expm1(y_true)
        y_pred_real = np.expm1(y_pred)

        # lưu lại để vẽ scatter
        results[name] = (y_true, y_pred)

        # Metrics trên log-scale
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Metrics trên real-scale (sau expm1)
        rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        mae_real = mean_absolute_error(y_true_real, y_pred_real)
        r2_real = r2_score(y_true_real, y_pred_real)

        # gom vào summary table
        summary.append({
            'Model': name,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'RMSE Real': rmse_real,
            'MAE Real': mae_real,
            'R² Real': r2_real
        })

    # Tạo bảng tổng hợp
    comparison_df = pd.DataFrame(summary)
    print(comparison_df)

    # ===== Bar chart 1: MAE =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='MAE', data=comparison_df, palette='viridis')
    plt.title('So sánh MAE\n(Càng thấp mô hình càng tốt)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Giá trị MAE', fontweight='bold')

    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='MAE Real', data=comparison_df, palette='viridis')
    plt.title('So sánh MAE Real\n(Càng thấp mô hình càng tốt)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Giá trị MAE Real', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir + 'model_comparison_1.png')

    # ===== Bar chart 2: R2 =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='R²', data=comparison_df, palette='magma')
    plt.title('So sánh $R^2$ Score\n(Càng cao mô hình càng tốt)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Giá trị $R^2$', fontweight='bold')

    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='R² Real', data=comparison_df, palette='magma')
    plt.title('So sánh $R^2$ Score Real\n(Càng cao mô hình càng tốt)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Giá trị $R^2$ Real', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir + 'model_comparison_2.png')

    # ===== Bar chart 3: RMSE =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='RMSE', data=comparison_df, palette='plasma')
    plt.title('So sánh RMSE', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Giá trị RMSE', fontweight='bold')

    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='RMSE Real', data=comparison_df, palette='plasma')
    plt.title('So sánh RMSE Real', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Giá trị RMSE Real', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_dir + 'model_comparison_3.png')

    # ===== Scatter: Predicted vs Actual (log) =====
    plt.figure(figsize=(18, 5))

    for i, name in enumerate(model_names):
        y_true, y_pred = results[name]

        plt.subplot(1, 3, i + 1)
        plt.scatter(y_true, y_pred, alpha=0.5, s=10)

        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', lw=2)

        plt.title(f'{name}: Predicted vs Actual', fontweight='bold')
        plt.xlabel('Giá trị thực tế', fontweight='bold')
        plt.ylabel('Giá trị dự đoán', fontweight='bold')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir + 'predicted_vs_actual_1.png')

    # ===== Scatter: Predicted vs Actual (real) =====
    plt.figure(figsize=(18, 5))

    for i, name in enumerate(model_names):
        y_true, y_pred = results[name]
        y_true_real = np.expm1(y_true)
        y_pred_real = np.expm1(y_pred)

        plt.subplot(1, 3, i + 1)
        plt.scatter(y_true_real, y_pred_real, alpha=0.5, s=10)

        max_val = max(y_true_real.max(), y_pred_real.max())
        plt.plot([0, max_val], [0, max_val], 'r--', lw=2)

        plt.title(f'{name}: Predicted vs Actual (Real)', fontweight='bold')
        plt.xlabel('Giá trị thực tế', fontweight='bold')
        plt.ylabel('Giá trị dự đoán', fontweight='bold')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir + 'predicted_vs_actual_2.png')

    print("Đã tạo xong file comparison & predicted_vs_actual!")

    return comparison_df
