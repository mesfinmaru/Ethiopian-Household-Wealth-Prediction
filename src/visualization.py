"""
Visualization Module - Standardized plotting functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class WealthVisualizer:

    @staticmethod
    def distribution(data, title="Distribution", bins=50, figsize=(8, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(data.dropna(), bins=bins, edgecolor='black', alpha=0.7)
        ax.axvline(data.median(), color='red', linestyle='--', label=f'Median: {data.median():.2f}')
        ax.axvline(data.mean(), color='green', linestyle='--', label=f'Mean: {data.mean():.2f}')
        ax.set_title(title); ax.legend()
        plt.tight_layout(); return fig

    @staticmethod
    def actual_vs_predicted(y_true, y_pred, figsize=(8, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(y_true, y_pred, alpha=0.3, s=5)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--')
        ax.set_xlabel('Actual'); ax.set_ylabel('Predicted'); ax.set_title('Actual vs Predicted')
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
        ax.text(0.05, 0.95, f'R²={r2:.4f}', transform=ax.transAxes, fontsize=12, va='top')
        plt.tight_layout(); return fig

    @staticmethod
    def feature_importance(imp_df, top_n=20, figsize=(10, 7)):
        top = imp_df.head(top_n)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(len(top)), top.values, color='steelblue')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.set_title('Feature Importance'); ax.invert_yaxis()
        plt.tight_layout(); return fig

    @staticmethod
    def model_comparison(results_df, figsize=(16, 10)):
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        results_df = results_df.sort_values('R2')
        for ax, metric, color, title in [
            (axes[0,0], 'R2', 'steelblue', 'R² Score'),
            (axes[0,1], 'RMSE', 'coral', 'RMSE'),
            (axes[1,0], 'MAE', 'mediumpurple', 'MAE'),
            (axes[1,1], 'CV_R2', 'lightgreen', 'CV R²')
        ]:
            ax.barh(results_df['Model'], results_df[metric], color=color)
            ax.set_xlabel(metric); ax.set_title(title)
            for i, v in enumerate(results_df[metric]):
                ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=8)
        plt.suptitle('Model Performance Comparison', fontweight='bold')
        plt.tight_layout(); return fig

    @staticmethod
    def residuals(y_true, y_pred, figsize=(14, 5)):
        res = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].scatter(y_pred, res, alpha=0.3, s=5)
        axes[0].axhline(0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Residual'); axes[0].set_title('Residuals vs Predicted')
        axes[1].hist(res, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='r', linestyle='--')
        axes[1].set_xlabel('Residual'); axes[1].set_title('Residual Distribution')
        plt.tight_layout(); return fig

    @staticmethod
    def time_trend(x, y, figsize=(10, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, 'o-', linewidth=2, markersize=10, color='steelblue')
        ax.set_xlabel('Wave'); ax.set_ylabel('Value')
        ax.set_title('Trend Over Time'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); return fig