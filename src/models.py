import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# LINEAR REGRESSION
# ============================================================

def train_lgbm_linear_regression_elasticnet(
    train_csv="../data/processed/train_data_final.csv",
    test_csv="../data/processed/test_data_final.csv",
    target="quantity_sold",
    leak_col="review_to_sold_ratio",
    coef_threshold=1e-8,
    out_fi_png="../data/images/feature_importance_lgbm_enet.png",
    out_pred_csv="../data/predicted/lgbm_enet_predictions.csv",
    top_n=10,
    seed=42,
    y_threshold=2.0,
):
    """
    Chức năng:
        - Train mô hình kết hợp ElasticNet (y >= threshold) + LightGBM (y < threshold):
            + Train ElasticNet trên dữ liệu y >= y_threshold theo pipeline:
                * Loại leak column
                * ElasticNetCV tìm best hyperparameters
                * Chọn feature theo |coef| > threshold
                * Train final model trên tập feature đã chọn
            + Train LightGBM trên dữ liệu y < y_threshold (cũng drop leak column)
            + Ghép predictions và lưu file CSV hoàn chỉnh
            + Vẽ feature importance (Top 10) và lưu ảnh

    Cách thực hiện:
        1. Chia dữ liệu theo y_threshold thành 2 phần: y < threshold và y >= threshold
        2. Train ElasticNet trên phần y >= threshold:
            - Drop leak column (mặc định: review_to_sold_ratio)
            - ElasticNetCV tìm best alpha và best l1_ratio
            - Feature selection theo |coef| > coef_threshold
            - Train final model trên tập feature đã chọn
        3. Train LightGBM trên phần y < threshold (RandomizedSearchCV, cũng drop leak column)
        4. Ghép predictions từ ElasticNet và LightGBM
        5. Vẽ Top 10 feature importance của ElasticNet và lưu ảnh
        6. Lưu kết quả dự đoán hoàn chỉnh ra file CSV

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
            Tên/đường dẫn lưu file dự đoán
        - top_n: int
            Số feature top để vẽ
        - seed: int
            Random seed (mặc định 42)
        - y_threshold: float
            Ngưỡng chia dữ liệu (mặc định 2.0)

    Giá trị trả về:
        dict gồm:
            - lgbm_model: LightGBM model
            - elasticnet_model: ElasticNet Pipeline final
            - best_alpha, best_l1_ratio
            - selected_features: list
            - coef_df: DataFrame hệ số final (có abs_coefficient)
            - combined_predictions: mảng dự đoán kết hợp
            - metrics: dict MAE/MSE/RMSE/R2
            - pred_df: DataFrame prediction đã lưu
    """
    np.random.seed(seed)

    # Load train & test
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"Original train size: {len(train_df)}")
    print(f"Original test size: {len(test_df)}")

    # Chia data theo threshold
    train_low = train_df[train_df[target] < y_threshold].copy()  # y < threshold
    train_high = train_df[train_df[target] >= y_threshold].copy()  # y >= threshold
    
    test_low = test_df[test_df[target] < y_threshold].copy()
    test_high = test_df[test_df[target] >= y_threshold].copy()
    
    print(f"Train data with y < {y_threshold}: {len(train_low)}")
    print(f"Train data with y >= {y_threshold}: {len(train_high)}")
    print(f"Test data with y < {y_threshold}: {len(test_low)}")
    print(f"Test data with y >= {y_threshold}: {len(test_high)}")
    
    all_predictions = []
    
    # =================================================================
    # 1. ElasticNet cho dữ liệu y >= threshold
    # =================================================================
    elasticnet_model = None
    best_alpha = None
    best_l1_ratio = None
    selected_features = []
    coef_final_df = None
    
    if len(train_high) > 0 and len(test_high) > 0:
        print("\n" + "="*60)
        print("TRAINING ELASTICNET ON DATA WITH y >= threshold")
        print("="*60)

        # Split train / target cho phần y >= threshold
        X_train = train_high.drop(columns=[target])
        y_train = train_high[target]

        X_test = test_high.drop(columns=[target])
        y_test = test_high[target]

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
        elasticnet_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=best_alpha,
                l1_ratio=best_l1_ratio,
                max_iter=10000,
                random_state=seed
            ))
        ])
        elasticnet_model.fit(X_train_selected, y_train)

        # Feature importance dataframe
        final_coef = elasticnet_model.named_steps["model"].coef_
        coef_final_df = pd.DataFrame({
            "feature": X_train_selected.columns,
            "coefficient": final_coef
        })
        coef_final_df["abs_coefficient"] = coef_final_df["coefficient"].abs()
        coef_final_df = coef_final_df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

        print(f"[FINAL MODEL] Number of selected features: {len(final_coef)}")

        # Predict cho phần y >= threshold
        y_pred_enet = elasticnet_model.predict(X_test_selected)
        
        # Tạo DataFrame cho ElasticNet predictions
        enet_pred_df = pd.DataFrame({
            'quantity_sold_ground_truth': y_test.values,
            'quantity_sold_predicted': y_pred_enet
        }, index=test_high.index)
        
        all_predictions.append(enet_pred_df)
        
    # =================================================================
    # 2. LightGBM cho dữ liệu y < threshold
    # =================================================================
    lgbm_model = None
    if len(train_low) > 0 and len(test_low) > 0:
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM ON DATA WITH y < threshold")
        print("="*60)
        
        X_train_low = train_low.drop(columns=[target])
        y_train_low = train_low[target]
        X_test_low = test_low.drop(columns=[target])
        y_test_low = test_low[target]
        
        # Drop leak column (giống ElasticNet)
        if leak_col in X_train_low.columns:
            X_train_low = X_train_low.drop(columns=[leak_col])
        if leak_col in X_test_low.columns:
            X_test_low = X_test_low.drop(columns=[leak_col])
        
        # RandomizedSearchCV cho LightGBM
        param_dist_lgb = {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'max_depth': [5, 7, 9, -1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        search_lgb = RandomizedSearchCV(
            estimator=LGBMRegressor(random_state=seed, verbose=-1),
            param_distributions=param_dist_lgb,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            random_state=seed,
            n_jobs=-1
        )
        
        search_lgb.fit(X_train_low, y_train_low)
        
        # Final LightGBM model
        lgbm_model = LGBMRegressor(**search_lgb.best_params_, random_state=seed, verbose=-1)
        lgbm_model.fit(X_train_low, y_train_low)
        
        y_pred_lgb = lgbm_model.predict(X_test_low)
        
        # Tạo DataFrame cho LightGBM predictions
        lgb_pred_df = pd.DataFrame({
            'quantity_sold_ground_truth': y_test_low.values,
            'quantity_sold_predicted': y_pred_lgb
        }, index=test_low.index)
        
        all_predictions.append(lgb_pred_df)
        
        print(f"LightGBM (y < {y_threshold}) - Best params: {search_lgb.best_params_}")

    # =================================================================
    # Tính Feature Importance Tổng Kết hợp
    # =================================================================
    combined_feature_importances = None
    if elasticnet_model is not None and lgbm_model is not None and coef_final_df is not None:
        # Lấy feature importance từ ElasticNet (y >= threshold)
        elasticnet_features = coef_final_df.set_index('feature')['abs_coefficient']
        
        # Lấy feature importance từ LightGBM (y < threshold) 
        # Lưu ý: X_train_low.columns đã loại bỏ leak column
        if 'X_train_low' in locals() and len(X_train_low.columns) > 0:
            lgbm_features = pd.Series(
                lgbm_model.feature_importances_,
                index=X_train_low.columns  # Đã được filter leak column
            )
        else:
            lgbm_features = pd.Series([])
        
        # Tính weight dựa trên số lượng samples
        n_high = len(train_high) if len(train_high) > 0 else 0
        n_low = len(train_low) if len(train_low) > 0 else 0
        total_samples = n_high + n_low
        
        if total_samples > 0:
            weight_high = n_high / total_samples
            weight_low = n_low / total_samples
            
            # Tạo combined feature importance
            all_features = set(elasticnet_features.index).union(set(lgbm_features.index))
            # Đảm bảo không có leak column (phòng trường hợp)
            all_features = all_features - {leak_col}
            combined_importance = {}
            
            for feature in all_features:
                enet_importance = elasticnet_features.get(feature, 0) * weight_high
                lgbm_importance = lgbm_features.get(feature, 0) * weight_low
                combined_importance[feature] = enet_importance + lgbm_importance
            
            combined_feature_importances = pd.Series(combined_importance).sort_values(ascending=False)
            
            
            print("\n" + "="*60)
            print("COMBINED FEATURE IMPORTANCE (ElasticNet + LightGBM)")
            print("="*60)
            print(f"ElasticNet weight: {weight_high:.3f} (n={n_high})")
            print(f"LightGBM weight: {weight_low:.3f} (n={n_low})")
            print("\nTop 15 Combined Features:")
            print(combined_feature_importances.head(15).to_string())
            print("="*60)

    # =================================================================
    # 3. Ghép kết quả và lưu file
    # =================================================================
    if all_predictions:
        combined_predictions_df = pd.concat(all_predictions).sort_index()
        combined_predictions_df.to_csv(out_pred_csv, index=False)
        
        # Tính metrics trên toàn bộ combined predictions
        y_true_all = combined_predictions_df['quantity_sold_ground_truth'].values
        y_pred_all = combined_predictions_df['quantity_sold_predicted'].values
        
        metrics = {
            "MAE": mean_absolute_error(y_true_all, y_pred_all),
            "MSE": mean_squared_error(y_true_all, y_pred_all),
            "RMSE": np.sqrt(mean_squared_error(y_true_all, y_pred_all)),
            "R2": r2_score(y_true_all, y_pred_all)
        }
    else:
        combined_predictions_df = pd.DataFrame()
        metrics = {}
        y_pred_all = []

    print(f"\nCombined Test Set Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot top 10 feature importance nếu có combined_feature_importances
    if combined_feature_importances is not None:
        data_to_plot = combined_feature_importances.head(top_n)
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette='viridis')

        for i, v in enumerate(data_to_plot.values):
            ax.text(v + 0.001, i, f'{v:.4f}', color='black', va='center', fontweight='bold')

        plt.title(
            f'Top {top_n} Yếu tố ảnh hưởng mạnh nhất (Combined: ElasticNet + LightGBM)',
            fontsize=16, fontweight='bold', pad=20
        )
        plt.xlabel('Mức độ quan trọng kết hợp (Combined Importance)', fontsize=12, fontweight='bold')
        plt.ylabel('Các đặc trưng (Features)', fontsize=12, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_fi_png, dpi=300)
        plt.show()

    return {
        "lgbm_model": lgbm_model,
        "elasticnet_model": elasticnet_model,
        "best_alpha": best_alpha,
        "best_l1_ratio": best_l1_ratio,
        "selected_features": selected_features,
        "coef_df": coef_final_df,
        "combined_feature_importances": combined_feature_importances,
        "combined_predictions": y_pred_all,
        "metrics": metrics,
        "pred_df": combined_predictions_df,
    }


# ============================================================
# RANDOM FOREST
# ============================================================

def train_lgbm_random_forest_regressor(
    train_csv="../data/processed/train_data_final.csv",
    test_csv="../data/processed/test_data_final.csv",
    target="quantity_sold",
    leak_col="review_to_sold_ratio",
    out_fi_png="../data/images/feature_importance_lgbm_rf.png",
    out_pred_csv="../data/predicted/lgbm_rf_predictions.csv",
    top_n=10,
    seed=42,
    verbose=1,
    y_threshold=2.0,
):
    """
    Chức năng:
        - Train mô hình kết hợp RandomForest (y >= threshold) + LightGBM (y < threshold):
            + Train RandomForest trên dữ liệu y >= y_threshold theo pipeline:
                * Drop leak column
                * GridSearchCV để tìm best params
                * Đánh giá R2 và RMSE
            + Train LightGBM trên dữ liệu y < y_threshold (cũng drop leak column)
            + Ghép predictions và lưu file CSV hoàn chỉnh
            + Vẽ feature importance (Top 10) và lưu ảnh

    Tham số:
        - leak_col: str
            Cột leak cần drop (mặc định "review_to_sold_ratio")
        - y_threshold: float
            Ngưỡng chia dữ liệu (mặc định 2.0)
        [Các tham số khác giống như trước]

    Giá trị trả về:
        dict gồm:
            - lgbm_model: LightGBM model
            - rf_model: RandomForest model
            - combined_predictions: predictions kết hợp
            - metrics: dict R2/RMSE/MAE
            - pred_df: DataFrame prediction đã lưu
    """
    # Load dữ liệu train/test
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    print(f"Original train size: {len(df_train)}")
    print(f"Original test size: {len(df_test)}")

    # Chia data theo threshold
    train_low = df_train[df_train[target] < y_threshold].copy()  # y < threshold
    train_high = df_train[df_train[target] >= y_threshold].copy()  # y >= threshold
    
    test_low = df_test[df_test[target] < y_threshold].copy()
    test_high = df_test[df_test[target] >= y_threshold].copy()
    
    print(f"Train data with y < {y_threshold}: {len(train_low)}")
    print(f"Train data with y >= {y_threshold}: {len(train_high)}")
    print(f"Test data with y < {y_threshold}: {len(test_low)}")
    print(f"Test data with y >= {y_threshold}: {len(test_high)}")
    
    all_predictions = []
    
    # =================================================================
    # 1. RandomForest cho dữ liệu y >= threshold
    # =================================================================
    rf_model = None
    rf_feature_importances = None
    
    if len(train_high) > 0 and len(test_high) > 0:
        print("\n" + "="*60)
        print("TRAINING RANDOMFOREST ON DATA WITH y >= threshold")  
        print("="*60)

        # Tách X/y cho phần y >= threshold
        X_train = train_high.drop(columns=[target])
        y_train = train_high[target]

        X_test = test_high.drop(columns=[target])
        y_test = test_high[target]

        # Drop leak column
        if leak_col in X_train.columns:
            X_train = X_train.drop(columns=[leak_col])
        if leak_col in X_test.columns:
            X_test = X_test.drop(columns=[leak_col])

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
        rf_model = grid_rf.best_estimator_

        y_test_pred = rf_model.predict(X_test)

        # Feature importance 
        rf_feature_importances = pd.Series(
            rf_model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        print("Random Forest Feature Importances:")
        print(rf_feature_importances.to_string())
        
        # Tạo DataFrame cho RandomForest predictions
        rf_pred_df = pd.DataFrame({
            'quantity_sold_ground_truth': y_test.values,
            'quantity_sold_predicted': y_test_pred
        }, index=test_high.index)
        
        all_predictions.append(rf_pred_df)
        
    # =================================================================
    # 2. LightGBM cho dữ liệu y < threshold
    # =================================================================
    lgbm_model = None
    if len(train_low) > 0 and len(test_low) > 0:
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM ON DATA WITH y < threshold")
        print("="*60)
        
        X_train_low = train_low.drop(columns=[target])
        y_train_low = train_low[target]
        X_test_low = test_low.drop(columns=[target])
        y_test_low = test_low[target]
        
        # Drop leak column (giống RandomForest)
        if leak_col in X_train_low.columns:
            X_train_low = X_train_low.drop(columns=[leak_col])
        if leak_col in X_test_low.columns:
            X_test_low = X_test_low.drop(columns=[leak_col])
        
        # RandomizedSearchCV cho LightGBM
        param_dist_lgb = {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'max_depth': [5, 7, 9, -1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        search_lgb = RandomizedSearchCV(
            estimator=LGBMRegressor(random_state=seed, verbose=-1),
            param_distributions=param_dist_lgb,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            random_state=seed,
            n_jobs=-1
        )
        
        search_lgb.fit(X_train_low, y_train_low)
        
        # Final LightGBM model
        lgbm_model = LGBMRegressor(**search_lgb.best_params_, random_state=seed, verbose=-1)
        lgbm_model.fit(X_train_low, y_train_low)
        
        y_pred_lgb = lgbm_model.predict(X_test_low)
        
        # Tạo DataFrame cho LightGBM predictions
        lgb_pred_df = pd.DataFrame({
            'quantity_sold_ground_truth': y_test_low.values,
            'quantity_sold_predicted': y_pred_lgb
        }, index=test_low.index)
        
        all_predictions.append(lgb_pred_df)
        
        print(f"LightGBM (y < {y_threshold}) - Best params: {search_lgb.best_params_}")

    # =================================================================
    # Tính Feature Importance Tổng Kết hợp
    # =================================================================
    combined_feature_importances = None
    if rf_feature_importances is not None and lgbm_model is not None:
        # Lấy feature importance từ RandomForest (y >= threshold)
        rf_features = rf_feature_importances
        
        # Lấy feature importance từ LightGBM (y < threshold)
        # Lưu ý: X_train_low.columns đã loại bỏ leak column
        if 'X_train_low' in locals() and len(X_train_low.columns) > 0:
            lgbm_features = pd.Series(
                lgbm_model.feature_importances_,
                index=X_train_low.columns  # Đã được filter leak column
            )
        else:
            lgbm_features = pd.Series([])
            
        # Tính weight dựa trên số lượng samples
        n_high = len(train_high) if len(train_high) > 0 else 0
        n_low = len(train_low) if len(train_low) > 0 else 0
        total_samples = n_high + n_low
        
        if total_samples > 0:
            weight_high = n_high / total_samples
            weight_low = n_low / total_samples
            
            # Tạo combined feature importance
            all_features = set(rf_features.index).union(set(lgbm_features.index))
            # Đảm bảo không có leak column (phòng trường hợp)
            all_features = all_features - {leak_col}
            combined_importance = {}
            
            for feature in all_features:
                rf_importance = rf_features.get(feature, 0) * weight_high
                lgbm_importance = lgbm_features.get(feature, 0) * weight_low
                combined_importance[feature] = rf_importance + lgbm_importance
            
            combined_feature_importances = pd.Series(combined_importance).sort_values(ascending=False)
            
            print("\n" + "="*60)
            print("COMBINED FEATURE IMPORTANCE (RandomForest + LightGBM)")
            print("="*60)
            print(f"RandomForest weight: {weight_high:.3f} (n={n_high})")
            print(f"LightGBM weight: {weight_low:.3f} (n={n_low})")
            print("\nTop 15 Combined Features:")
            print(combined_feature_importances.head(15).to_string())
            print("="*60)

    # =================================================================
    # 3. Ghép kết quả và tính metrics
    # =================================================================
    if all_predictions:
        combined_predictions_df = pd.concat(all_predictions).sort_index()
        combined_predictions_df.to_csv(out_pred_csv, index=False)
        
        # Tính metrics trên toàn bộ combined predictions
        y_true_all = combined_predictions_df['quantity_sold_ground_truth'].values
        y_pred_all = combined_predictions_df['quantity_sold_predicted'].values
        
        # Tính metrics
        r2 = r2_score(y_true_all, y_pred_all)
        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae = mean_absolute_error(y_true_all, y_pred_all)

        metrics = {"R2": r2, "RMSE": rmse, "MAE": mae}
    else:
        combined_predictions_df = pd.DataFrame()
        metrics = {}
        y_pred_all = []

    print(f"\nCombined RF Test Set Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Plot top 10 feature importance nếu có RandomForest model
    if combined_feature_importances is not None:
        data_to_plot = combined_feature_importances.head(top_n)

        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette='viridis')

        for i, v in enumerate(data_to_plot.values):
            ax.text(v + 0.001, i, f'{v:.4f}', color='black', va='center', fontweight='bold')

        plt.title(f'Top {top_n} yếu tố ảnh hưởng nhất đến Doanh số (Combined: RandomForest + LightGBM)', fontsize=14, fontweight='bold')
        plt.xlabel('Mức độ quan trọng kết hợp (Combined Importance)', fontsize=12, fontweight='bold')
        plt.ylabel('Các đặc trưng (Features)', fontsize=12, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(out_fi_png, dpi=300)
        plt.show()

    return {
        "lgbm_model": lgbm_model,
        "rf_model": rf_model,
        "grid": grid_rf if 'grid_rf' in locals() else None,
        "best_params": grid_rf.best_params_ if 'grid_rf' in locals() else None,
        "feature_importances": rf_feature_importances,
        "combined_feature_importances": combined_feature_importances,
        "combined_predictions": y_pred_all,
        "metrics": metrics,
        "pred_df": combined_predictions_df,
    }


# ============================================================
# XGBOOST
# ============================================================

def train_lgbm_xgboost_regressor(
    train_csv="../data/processed/train_data_final.csv",
    test_csv="../data/processed/test_data_final.csv",
    target="quantity_sold",
    cols_to_drop=("review_to_sold_ratio",),
    out_fi_png="../data/images/feature_importance_lgbm_xgb.png",
    out_pred_csv="../data/predicted/lgbm_xgb_predictions.csv",
    top_n=10,
    seed=42,
    n_iter=10,
    y_threshold=2.0,
):
    """
    Chức năng:
        - Train mô hình kết hợp XGBoost (y >= threshold) + LightGBM (y < threshold):
            + Train XGBoost trên dữ liệu y >= y_threshold theo pipeline:
                * Chia train/val 80/20 để train lần 1 và lấy feature importance
                * Loại các cột gây nhiễu khỏi danh sách feature dùng để tune/train
                * RandomizedSearchCV để tìm hyperparameters tốt
                * Train final model trên toàn bộ dữ liệu (train + val)
                * Đánh giá trên thang gốc bằng expm1 (RMSE/MAE/R2)
            + Train LightGBM trên dữ liệu y < y_threshold (cũng drop cols_to_drop)
            + Ghép predictions và lưu file CSV hoàn chỉnh
            + Vẽ feature importance Top N và lưu ảnh

    Tham số:
        - y_threshold: float
            Ngưỡng chia dữ liệu (mặc định 2.0)
        [Các tham số khác giống như trước]

    Giá trị trả về:
        dict gồm:
            - lgbm_model: LightGBM model
            - xgb_model: XGBoost final_model
            - combined_predictions: predictions kết hợp
            - metrics: {"RMSE": ..., "MAE": ..., "R2": ...} (log-scale)
            - pred_df: DataFrame prediction đã lưu
    """
    # Load dữ liệu
    train_data = pd.read_csv(train_csv)
    test_data = pd.read_csv(test_csv)
    
    print(f"Original train size: {len(train_data)}")
    print(f"Original test size: {len(test_data)}")

    # Chia data theo threshold
    train_low = train_data[train_data[target] < y_threshold].copy()  # y < threshold
    train_high = train_data[train_data[target] >= y_threshold].copy()  # y >= threshold
    
    test_low = test_data[test_data[target] < y_threshold].copy()
    test_high = test_data[test_data[target] >= y_threshold].copy()
    
    print(f"Train data with y < {y_threshold}: {len(train_low)}")
    print(f"Train data with y >= {y_threshold}: {len(train_high)}")
    print(f"Test data with y < {y_threshold}: {len(test_low)}")
    print(f"Test data with y >= {y_threshold}: {len(test_high)}")
    
    all_predictions = []
    
    # =================================================================
    # 1. XGBoost cho dữ liệu y >= threshold
    # =================================================================
    xgb_model = None
    final_feature_importances = None
    
    if len(train_high) > 0 and len(test_high) > 0:
        print("\n" + "="*60)
        print("TRAINING XGBOOST ON DATA WITH y >= threshold")
        print("="*60)

        # Tách X_test/y_test cho phần y >= threshold
        y_test = test_high[target]
        X_test = test_high.drop(columns=[target])

        # Split train thành train/val (80/20) cho phần y >= threshold
        X_train, X_val, y_train, y_val = train_test_split(
            train_high.drop(columns=[target]),
            train_high[target],
            test_size=0.2,
            random_state=seed
        )

        # train lần 1 để lấy feature importance
        xgb_model_first = XGBRegressor(
            n_estimators=10000,
            learning_rate=0.1,
            max_depth=6,
            random_state=seed,
            early_stopping_rounds=50
        )

        xgb_model_first.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Feature importance lần 1
        feature_importances = pd.Series(xgb_model_first.feature_importances_, index=X_train.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        print("Feature Importances:")
        print(feature_importances.to_string())

        # Drop leak columns SAU KHI lấy feature importance
        print(f"Columns to drop: {cols_to_drop}")
        print(f"Available columns in X_train: {list(X_train.columns)}")
        print(f"Available columns in X_test: {list(X_test.columns)}")
        
        for col in cols_to_drop:
            if col in X_test.columns:
                X_test = X_test.drop(columns=[col])
            if col in X_train.columns:
                X_train = X_train.drop(columns=[col])
            if col in X_val.columns:
                X_val = X_val.drop(columns=[col])
                

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

        random_search.fit(X_train, y_train)
        best_xgb_model = random_search.best_estimator_

        print("Đang huấn luyện model cuối cùng trên toàn bộ dữ liệu (Train + Val)...")

        # Gộp train + val (đã drop leak columns)
        X_full_train = pd.concat([X_train, X_val])
        y_full_train = pd.concat([y_train, y_val])

        best_params = random_search.best_params_

        xgb_model = XGBRegressor(
            **best_params,
            random_state=seed,
            n_jobs=-1
        )

        xgb_model.fit(X_full_train, y_full_train)

        # feature importance của XGBoost (sau khi train toàn bộ data high)
        # Chỉ lấy từ valid_features đã filter leak columns
        final_feature_importances = pd.Series(
            xgb_model.feature_importances_,
            index=X_full_train.columns
        ).sort_values(ascending=False)

        # predict + metric
        y_test_pred = xgb_model.predict(X_test)
        
        # Tạo DataFrame cho XGBoost predictions
        xgb_pred_df = pd.DataFrame({
            'quantity_sold_ground_truth': y_test.values,
            'quantity_sold_predicted': y_test_pred
        }, index=test_high.index)
        
        all_predictions.append(xgb_pred_df)
        
    # =================================================================
    # 2. LightGBM cho dữ liệu y < threshold
    # =================================================================
    lgbm_model = None
    if len(train_low) > 0 and len(test_low) > 0:
        print("\n" + "="*60)
        print("TRAINING LIGHTGBM ON DATA WITH y < threshold")
        print("="*60)
        
        X_train_low = train_low.drop(columns=[target])
        y_train_low = train_low[target]
        X_test_low = test_low.drop(columns=[target])
        y_test_low = test_low[target]
        
        # Drop leak columns (giống XGBoost)
        for col in cols_to_drop:
            if col in X_train_low.columns:
                X_train_low = X_train_low.drop(columns=[col])
            if col in X_test_low.columns:
                X_test_low = X_test_low.drop(columns=[col])
        
        # RandomizedSearchCV cho LightGBM
        param_dist_lgb = {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.05, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'max_depth': [5, 7, 9, -1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        search_lgb = RandomizedSearchCV(
            estimator=LGBMRegressor(random_state=seed, verbose=-1),
            param_distributions=param_dist_lgb,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=3,
            random_state=seed,
            n_jobs=-1
        )
        
        search_lgb.fit(X_train_low, y_train_low)
        
        # Final LightGBM model
        lgbm_model = LGBMRegressor(**search_lgb.best_params_, random_state=seed, verbose=-1)
        lgbm_model.fit(X_train_low, y_train_low)
        
        y_pred_lgb = lgbm_model.predict(X_test_low)
        
        # Tạo DataFrame cho LightGBM predictions
        lgb_pred_df = pd.DataFrame({
            'quantity_sold_ground_truth': y_test_low.values,
            'quantity_sold_predicted': y_pred_lgb
        }, index=test_low.index)
        
        all_predictions.append(lgb_pred_df)
        
        print(f"LightGBM (y < {y_threshold}) - Best params: {search_lgb.best_params_}")

    # =================================================================
    # Tính Feature Importance Sau khi kết hợp
    # =================================================================
    combined_feature_importances = None
    if final_feature_importances is not None and lgbm_model is not None:
        # Lấy feature importance từ XGBoost (y >= threshold)
        xgb_features = final_feature_importances
        
        # Lấy feature importance từ LightGBM (y < threshold) 
        # Lưu ý: X_train_low.columns đã loại bỏ cols_to_drop
        if 'X_train_low' in locals() and len(X_train_low.columns) > 0:
            lgbm_features = pd.Series(
                lgbm_model.feature_importances_,
                index=X_train_low.columns  # Đã được filter cols_to_drop
            )
        else:
            lgbm_features = pd.Series([])
            
        # Tính weight dựa trên số lượng samples
        n_high = len(train_high) if len(train_high) > 0 else 0
        n_low = len(train_low) if len(train_low) > 0 else 0
        total_samples = n_high + n_low
        
        if total_samples > 0:
            weight_high = n_high / total_samples
            weight_low = n_low / total_samples
            
            # Tạo combined feature importance
            all_features = set(xgb_features.index).union(set(lgbm_features.index))
            # Đảm bảo không có leak columns (phòng trường hợp)
            all_features = all_features - set(cols_to_drop)
            combined_importance = {}
            
            for feature in all_features:
                xgb_importance = xgb_features.get(feature, 0) * weight_high
                lgbm_importance = lgbm_features.get(feature, 0) * weight_low
                combined_importance[feature] = xgb_importance + lgbm_importance
            
            combined_feature_importances = pd.Series(combined_importance).sort_values(ascending=False)
            
            print("\n" + "="*60)
            print("COMBINED FEATURE IMPORTANCE (XGBoost + LightGBM)")
            print("="*60)
            print(f"XGBoost weight: {weight_high:.3f} (n={n_high})")
            print(f"LightGBM weight: {weight_low:.3f} (n={n_low})")
            print("\nTop 15 Combined Features:")
            print(combined_feature_importances.head(15).to_string())
            print("="*60)

    # =================================================================
    # 3. Ghép kết quả và tính metrics
    # =================================================================
    if all_predictions:
        combined_predictions_df = pd.concat(all_predictions).sort_index()
        combined_predictions_df.to_csv(out_pred_csv, index=False)
        
        # Tính metrics trên toàn bộ combined predictions
        y_true_all = combined_predictions_df['quantity_sold_ground_truth'].values
        y_pred_all = combined_predictions_df['quantity_sold_predicted'].values

        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae = mean_absolute_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    else:
        combined_predictions_df = pd.DataFrame()
        metrics = {}
        y_pred_all = []

    print(f"\nCombined XGB Test Set Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # plot top 10 nếu có XGBoost model và final_feature_importances
    if combined_feature_importances is not None and len(combined_feature_importances) > 0:
        data_to_plot = combined_feature_importances.head(top_n)
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette='viridis')

        for i, v in enumerate(data_to_plot.values):
            ax.text(v + 0.001, i, f'{v:.4f}', color='black', va='center', fontweight='bold')

        plt.title(f'Top {top_n} Yếu tố ảnh hưởng nhất đến Doanh số (Combined: XGBoost + LightGBM)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Mức độ quan trọng kết hợp (Combined Importance)', fontsize=12, fontweight='bold')
        plt.ylabel('Các đặc trưng (Features)', fontsize=12, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_fi_png, dpi=300)
        plt.show()

    return {
        "lgbm_model": lgbm_model,
        "xgb_model": xgb_model,
        "model_first": xgb_model_first if 'xgb_model_first' in locals() else None,
        "feature_importances_first": feature_importances if 'feature_importances' in locals() else None,
        "search": random_search if 'random_search' in locals() else None,
        "best_params": best_params if 'best_params' in locals() else None,
        "combined_feature_importances": combined_feature_importances,
        "combined_predictions": y_pred_all,
        "metrics": metrics,
        "pred_df": combined_predictions_df,
    }


# ============================================================
# 4) COMPARISON
# ============================================================

def run_model_comparison(
    files=(
        '../data/predicted/lgbm_enet_predictions.csv',
        '../data/predicted/lgbm_rf_predictions.csv',
        '../data/predicted/lgbm_xgb_predictions.csv'
    ),
    model_names=(
        'LightGBM + ElasticNet',
        'LightGBM + RandomForest',
        'LightGBM + XGBoost'
    ),
    out_dir="../data/images/"
):
    """
    Chức năng:
        - So sánh 3 mô hình conditional training dựa trên các file prediction CSV:
            + Tính metric trên log-scale (RMSE/MAE/R²)
            + Tính metric trên real-scale sau khi expm1 (RMSE Real/MAE Real/R² Real)
            + Vẽ và lưu các biểu đồ:
                - model_comparison_1.png (MAE vs MAE Real)
                - model_comparison_2.png (R² vs R² Real)
                - model_comparison_3.png (RMSE vs RMSE Real)
                - predicted_vs_actual_1.png (scatter log-scale)
                - predicted_vs_actual_2.png (scatter real-scale)

    Cách thực hiện:
        1. Với mỗi file prediction (LightGBM+ElasticNet, LightGBM+RandomForest, LightGBM+XGBoost):
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
            Danh sách đường dẫn prediction csv cho conditional training
        - model_names: tuple/list[str]
            Tên model combination tương ứng với files
        - out_dir: str
            Thư mục lưu ảnh output

    Giá trị trả về:
        pandas.DataFrame:
            Bảng tổng hợp metric của các mô hình conditional (log-scale và real-scale).
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
    plt.figure(figsize=(15, 5))

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
    plt.figure(figsize=(15, 5))

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
