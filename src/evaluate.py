import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_predictions(y_true, y_pred):
    """
    Compute regression metrics (R2, RMSE, MAE).
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    
    Returns:
    dict: Dictionary containing R2, RMSE, and MAE scores
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae
    }

def calculate_residuals(y_true, y_pred):
    """
    Calculate residuals (Actual - Predicted).
    
    Parameters:
    y_true (array-like): Actual values
    y_pred (array-like): Predicted values
    
    Returns:
    array-like: Residual values
    """
    return y_true - y_pred

def plot_residuals_vs_predicted(y_pred, residuals, title="Residuals vs. Predicted Success Score", save_path=None):
    """
    Plot residuals vs predicted scatter plot.
    
    Parameters:
    y_pred (array-like): Predicted values
    residuals (array-like): Residual values
    title (str): Plot title
    save_path (str/Path): Optional file path to save the generated plot
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.1, color='purple')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Success Score")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_residuals_by_group(df, group_col, residual_col, title="Residuals Distribution by Group", rotation=45, save_path=None):
    """
    Plot residuals distribution by a group column using boxplots.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing the data
    group_col (str): Column name to group by
    residual_col (str): Column containing residuals
    title (str): Plot title
    rotation (int): X-axis tick rotation angle
    save_path (str/Path): Optional file path to save the generated plot
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=group_col, y=residual_col, hue=group_col, legend=False, palette='muted')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(group_col.replace('_', ' ').title())
    plt.ylabel("Residual (Actual - Predicted)")
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
