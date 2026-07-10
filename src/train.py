import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

def load_and_split_features(train_path, test_path, target_col='success_score'):
    """
    Loads train and test CSVs and separates them into feature matrices (X) and target vectors (y).
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # List of metadata/unrelated columns to exclude from training
    drop_cols = [
        'AppID', 'Name', 'Release date', 'Genres', 'Tags', 'About the game', 
        'Header image', 'Support email', 'Developers', 'Publishers', 
        'Categories', 'Screenshots', 'Supported languages', 'Full audio languages',
        'Positive', 'Negative', 'Review_Count', 'wilson_lb', 'success_score',
        'Min owners', 'Max owners', 'Avg owners', 'price_tier', 'primary_genre',
        'primary_genre_mapped', 'Recommendations', 'Peak CCU'
    ]
    
    cols_to_drop = [c for c in drop_cols if c in train_df.columns]
    
    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df[target_col]
    
    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df[target_col]
    
    # Pre-fit NaN verification
    assert X_train.isnull().sum().sum() == 0, "NaNs found in X_train"
    assert X_test.isnull().sum().sum() == 0, "NaNs found in X_test"
    
    return X_train, y_train, X_test, y_test

def train_ridge_baseline(X_train, y_train, X_test, y_test, alpha=1.0):
    """
    Trains a Ridge regression model and logs metrics/parameters to MLflow.
    """
    with mlflow.start_run(run_name="ridge_baseline", nested=True):
        mlflow.log_param("alpha", alpha)
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Predict & Evaluate
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        log_mlflow_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test, tune=False):
    """
    Trains a Random Forest Regressor, optionally runs tuning, and logs to MLflow.
    """
    with mlflow.start_run(run_name="random_forest", nested=True):
        if tune:
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10]
            }
            rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
            search = GridSearchCV(
                estimator=rf_base,
                param_grid=rf_params,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            for param, val in search.best_params_.items():
                mlflow.log_param(param, val)
        else:
            model = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 20)
            mlflow.log_param("min_samples_split", 5)
            
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        log_mlflow_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics

def train_xgboost(X_train, y_train, X_test, y_test, tune=False):
    """
    Trains an XGBoost Regressor, optionally runs tuning, and logs to MLflow.
    """
    with mlflow.start_run(run_name="xgboost", nested=True):
        if tune:
            xgb_params = {
                'n_estimators': [100, 200],
                'learning_rate': [0.03, 0.1],
                'max_depth': [5, 7],
                'subsample': [0.8, 1.0]
            }
            xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            search = GridSearchCV(
                estimator=xgb_base,
                param_grid=xgb_params,
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            for param, val in search.best_params_.items():
                mlflow.log_param(param, val)
        else:
            model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=7, subsample=0.8, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 7)
            mlflow.log_param("subsample", 0.8)
            
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred)
        log_mlflow_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")
        
        return model, metrics

def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred):
    """
    Calculates evaluation metrics (RMSE, MAE, R2) for train and test splits.
    """
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred))
    }

def log_mlflow_metrics(metrics):
    """
    Logs dict of metrics to active MLflow run.
    """
    for metric_name, val in metrics.items():
        mlflow.log_metric(metric_name, val)

def run_experiment(project_root_path, tune=False):
    """
    Main orchestrator that runs the entire baseline and model comparisons.
    """
    project_root = Path(project_root_path)
    
    # Configure MLflow SQLite tracking backend
    db_path = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("steam_success_prediction")
    
    train_path = project_root / "data" / "processed" / "games_train.csv"
    test_path = project_root / "data" / "processed" / "games_test.csv"
    
    X_train, y_train, X_test, y_test = load_and_split_features(train_path, test_path)
    
    print("Starting MLflow parent run...")
    with mlflow.start_run(run_name="model_comparison_pipeline"):
        print("Training Ridge Baseline...")
        ridge_model, ridge_metrics = train_ridge_baseline(X_train, y_train, X_test, y_test)
        
        print("Training Random Forest...")
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test, tune=tune)
        
        print("Training XGBoost...")
        xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, tune=tune)
        
        # Build comparison summary
        summary = pd.DataFrame([
            {"Model": "Ridge Baseline", "Train R2": ridge_metrics["train_r2"], "Test RMSE": ridge_metrics["test_rmse"], "Test MAE": ridge_metrics["test_mae"], "Test R2": ridge_metrics["test_r2"]},
            {"Model": "Random Forest", "Train R2": rf_metrics["train_r2"], "Test RMSE": rf_metrics["test_rmse"], "Test MAE": rf_metrics["test_mae"], "Test R2": rf_metrics["test_r2"]},
            {"Model": "XGBoost", "Train R2": xgb_metrics["train_r2"], "Test RMSE": xgb_metrics["test_rmse"], "Test MAE": xgb_metrics["test_mae"], "Test R2": xgb_metrics["test_r2"]}
        ])
        
        print("\n=== Model Leaderboard ===")
        print(summary)
        
        # Export best model
        best_row = summary.sort_values(by="Test R2", ascending=False).iloc[0]
        best_name = best_row["Model"]
        print(f"\nBest Model: {best_name} (Test R2: {best_row['Test R2']:.4f})")
        
        best_model_map = {
            "Ridge Baseline": ridge_model,
            "Random Forest": rf_model,
            "XGBoost": xgb_model
        }
        
        export_dir = project_root / "models"
        export_dir.mkdir(exist_ok=True)
        export_path = export_dir / "best_model.joblib"
        joblib.dump(best_model_map[best_name], export_path)
        print(f"Serialized best model successfully saved to {export_path}")
        
        # Export feature columns order
        feature_names_path = export_dir / "feature_names.joblib"
        joblib.dump(X_train.columns.tolist(), feature_names_path)
        print(f"Feature column order successfully saved to {feature_names_path}")

if __name__ == "__main__":
    # Fallback to current working directory
    proj_root = Path.cwd()
    if not (proj_root / "data" / "processed").exists():
        proj_root = proj_root.parent
        
    run_experiment(proj_root, tune=True)
