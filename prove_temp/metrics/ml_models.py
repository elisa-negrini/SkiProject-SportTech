"""
Machine Learning Models for Ski Jumping Score Prediction
========================================================

This module implements various ML models for predicting ski jumping scores
and extracting feature importance to understand what matters most.

MODELS IMPLEMENTED:
-------------------
1. Random Forest Regressor: Robust ensemble method with feature importance
2. Gradient Boosting (XGBoost): State-of-the-art for tabular data
3. Ridge Regression: Linear baseline with regularization
4. Neural Network: Simple MLP for non-linear relationships

FEATURE ENGINEERING:
--------------------
- Time-series statistics (mean, std, min, max, skew)
- Phase-specific features
- Interaction terms
- Polynomial features

EVALUATION:
-----------
- Leave-One-Out Cross-Validation (for small datasets)
- K-Fold Cross-Validation
- Permutation Importance
- SHAP values (if available)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# FEATURE EXCLUSION LIST
# =============================================================================
# Features excluded from ML models for various reasons
EXCLUDED_FEATURES = [
    # -------------------------------------------------------------------------
    # DATA LEAKAGE - These derive directly from target scores
    # -------------------------------------------------------------------------
    'normalized_distance',     # Physical_Score is computed from distance
    'AthleteDistance',         # Same as above
    'DistancePoints',          # Same as above
    
    # -------------------------------------------------------------------------
    # DEMOGRAPHIC BIAS - Not biomechanical, can introduce bias
    # -------------------------------------------------------------------------
    'AthleteGender',           # Gender should not predict style score
    'AthleteNat',              # Nationality bias
    'HillLocation',            # Event-specific bias
    'CompetitionDate',         # Temporal bias
    
    # -------------------------------------------------------------------------
    # REDUNDANT FEATURES - Correlated with better alternatives
    # -------------------------------------------------------------------------
    'flight_range',            # Redundant with flight_std (r~0.85)
    'flight_trend',            # Low predictive power
    'knee_mean_velocity',      # Redundant with knee_peak_velocity
    'knee_extension_range',    # Redundant with knee_peak_velocity
    'takeoff_acceleration_peak',  # Redundant with takeoff_peak_velocity (removed from advanced_metrics)
    'takeoff_smoothness',      # Not interpretable - removed from advanced_metrics
    'flight_stability_std',    # Redundant with flight_std (removed from core_metrics)
    'avg_telemark_offset_x',   # Redundant with telemark_proj_ski (removed from core_metrics)
    'landing_absorption_rate', # Often NaN, less reliable
    'landing_hip_drop',        # Depends on jump height, not technique
    'landing_smoothness_score', # Engineered feature, model can learn combination
    
    # -------------------------------------------------------------------------
    # IDENTIFIERS - Not features
    # -------------------------------------------------------------------------
    'jump_id',
    'AthleteName',
    'AthleteSurname',
    'AthleteScore',            # This is Style + Physical (leakage)
    'ID',
]

# Core ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score, cross_val_predict
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.inspection import permutation_importance
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("❌ scikit-learn not installed - ML features unavailable")

# Optional: XGBoost
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Optional: SHAP for explainability
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


class SkiJumpingMLModels:
    """
    Machine Learning pipeline for ski jumping score prediction.
    
    This class provides:
    - Data loading and feature engineering
    - Multiple model training and evaluation
    - Feature importance extraction
    - Visualization of results
    
    The primary goal is NOT accurate prediction (small dataset), but
    understanding WHICH FEATURES MATTER through importance analysis.
    """
    
    def __init__(self):
        """Initialize the ML pipeline with paths and configurations."""
        
        # =====================================================================
        # PATH CONFIGURATION
        # =====================================================================
        self.base_path = Path(__file__).parent.parent
        
        # Input files (from subfolders)
        self.jp_data_file = self.base_path / 'dataset' / 'JP_data.csv'
        self.timeseries_metrics_file = self.base_path / 'metrics' / 'timeseries_metrics' / 'timeseries_summary.csv'
        self.advanced_metrics_file = self.base_path / 'metrics' / 'advanced_metrics' / 'advanced_metrics_summary.csv'
        self.old_metrics_file = self.base_path / 'metrics' / 'core_metrics' / 'metrics_summary_per_jump.csv'
        
        # Output directory (to models subfolder)
        self.output_dir = self.base_path / 'metrics' / 'models'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # MODEL CONFIGURATION
        # =====================================================================
        self.random_state = 42
        
        # Feature columns to use (will be populated during loading)
        self.feature_columns = []
        
        # Target columns
        self.target_columns = ['Physical_Score', 'Style_Score', 'AthleteDistance']
    
    # =========================================================================
    # DATA LOADING AND FEATURE ENGINEERING
    # =========================================================================
    
    def load_and_prepare_data(self) -> Optional[pd.DataFrame]:
        """
        Load all data sources and merge into a single dataset.
        
        This method:
        1. Loads JP_data (scores and metadata)
        2. Loads computed metrics (time-series, advanced, old)
        3. Merges all data on jump_id
        4. Computes derived features
        5. Handles missing values
        
        Returns:
            pd.DataFrame: Merged dataset ready for ML, or None if loading fails
        """
        if not HAS_SKLEARN:
            print("❌ scikit-learn required for ML")
            return None
        
        # -----------------------------------------------------------------
        # LOAD JP DATA (SCORES)
        # -----------------------------------------------------------------
        if not self.jp_data_file.exists():
            print(f"❌ JP_data not found: {self.jp_data_file}")
            return None
        
        df_jp = pd.read_csv(self.jp_data_file)
        print(f"✅ Loaded JP data: {len(df_jp)} jumps")
        
        # Compute target scores
        scores = []
        for _, row in df_jp.iterrows():
            # Get judge scores
            judge_scores = [row.get(f'AthleteJdg{x}', np.nan) for x in 'ABCDE']
            valid = [s for s in judge_scores if pd.notna(s)]
            
            # Style score: sum of middle 3 judges
            if len(valid) >= 5:
                style = sum(sorted(valid)[1:4])
            elif len(valid) >= 3:
                style = sum(sorted(valid)[:3])
            else:
                style = np.nan
            
            # Physical score: total minus style
            athlete_score = row.get('AthleteScore', np.nan)
            physical = athlete_score - style if pd.notna(athlete_score) and pd.notna(style) else np.nan
            
            scores.append({
                'jump_id': row['ID'],
                'AthleteScore': athlete_score,
                'Style_Score': style,
                'Physical_Score': physical,
                'AthleteDistance': row.get('AthleteDistance', np.nan),
                'HillHS': row.get('HillHS', np.nan),
                'HillK': row.get('HillK', np.nan),
                'AthleteGender': 1 if row.get('AthleteGender') == 'M' else 0
            })
        
        df_scores = pd.DataFrame(scores)
        
        # -----------------------------------------------------------------
        # LOAD TIME-SERIES METRICS
        # -----------------------------------------------------------------
        if self.timeseries_metrics_file.exists():
            df_ts = pd.read_csv(self.timeseries_metrics_file)
            print(f"✅ Loaded time-series metrics: {len(df_ts)} jumps")
        else:
            df_ts = pd.DataFrame({'jump_id': []})
            print("⚠️ Time-series metrics not found")
        
        # -----------------------------------------------------------------
        # LOAD ADVANCED METRICS
        # -----------------------------------------------------------------
        if self.advanced_metrics_file.exists():
            df_adv = pd.read_csv(self.advanced_metrics_file)
            print(f"✅ Loaded advanced metrics: {len(df_adv)} jumps")
        else:
            df_adv = pd.DataFrame({'jump_id': []})
            print("⚠️ Advanced metrics not found")
        
        # -----------------------------------------------------------------
        # LOAD OLD METRICS
        # -----------------------------------------------------------------
        if self.old_metrics_file.exists():
            df_old = pd.read_csv(self.old_metrics_file)
            print(f"✅ Loaded old metrics: {len(df_old)} jumps")
        else:
            df_old = pd.DataFrame({'jump_id': []})
            print("⚠️ Old metrics not found")
        
        # -----------------------------------------------------------------
        # MERGE ALL DATA
        # -----------------------------------------------------------------
        df = df_scores.copy()
        
        # Merge time-series metrics
        if not df_ts.empty and 'jump_id' in df_ts.columns:
            df = df.merge(df_ts, on='jump_id', how='left')
        
        # Merge advanced metrics
        if not df_adv.empty and 'jump_id' in df_adv.columns:
            df = df.merge(df_adv, on='jump_id', how='left', suffixes=('', '_adv'))
        
        # Merge old metrics (select useful columns)
        if not df_old.empty and 'jump_id' in df_old.columns:
            old_cols = ['jump_id', 'avg_v_style_front', 'avg_v_style_back', 
                        'avg_body_ski_angle', 'flight_stability_std', 'takeoff_knee_angle']
            df_old_subset = df_old[[c for c in old_cols if c in df_old.columns]]
            df = df.merge(df_old_subset, on='jump_id', how='left', suffixes=('', '_old'))
        
        print(f"\n📊 Merged dataset: {len(df)} rows, {len(df.columns)} columns")
        
        # -----------------------------------------------------------------
        # FEATURE ENGINEERING
        # -----------------------------------------------------------------
        df = self._engineer_features(df)
        
        # -----------------------------------------------------------------
        # IDENTIFY FEATURE COLUMNS (with exclusions)
        # -----------------------------------------------------------------
        # Base exclusions
        exclude_cols = ['jump_id'] + self.target_columns + ['AthleteScore', 'HillHS', 'HillK']
        
        # Add global EXCLUDED_FEATURES
        all_exclusions = set(exclude_cols + EXCLUDED_FEATURES)
        
        # Select only numeric features not in exclusion list
        self.feature_columns = [
            c for c in df.columns 
            if c not in all_exclusions 
            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
        ]
        
        print(f"📊 Feature columns: {len(self.feature_columns)}")
        
        # Generate feature selection report
        self._generate_feature_report(df)
        
        return df
    
    def _generate_feature_report(self, df: pd.DataFrame):
        """Generate a report of selected features with availability stats."""
        
        report_file = self.output_dir / 'feature_selection_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("FEATURE SELECTION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Excluded features
            f.write("EXCLUDED FEATURES:\n")
            f.write("-" * 40 + "\n")
            for feat in EXCLUDED_FEATURES:
                if feat in ['normalized_distance', 'AthleteDistance', 'DistancePoints']:
                    reason = "data leakage (derives from target)"
                elif feat in ['AthleteGender', 'AthleteNat', 'HillLocation', 'CompetitionDate']:
                    reason = "demographic bias"
                elif feat in ['flight_range', 'knee_mean_velocity', 'knee_extension_range', 
                              'takeoff_acceleration_peak', 'landing_absorption_rate']:
                    reason = "redundant with better alternative"
                else:
                    reason = "identifier"
                f.write(f"  - {feat}: {reason}\n")
            
            f.write("\n\nSELECTED FEATURES:\n")
            f.write("-" * 40 + "\n")
            
            # Count availability for each feature
            feature_stats = []
            for feat in self.feature_columns:
                n_valid = df[feat].notna().sum()
                n_total = len(df)
                pct = 100 * n_valid / n_total
                feature_stats.append((feat, n_valid, n_total, pct))
            
            # Sort by availability
            feature_stats.sort(key=lambda x: x[3], reverse=True)
            
            for feat, n_valid, n_total, pct in feature_stats:
                f.write(f"  {feat}: {n_valid}/{n_total} jumps ({pct:.0f}%)\n")
            
            f.write(f"\n\nTOTAL FEATURES USED: {len(self.feature_columns)}\n")
        
        print(f"✅ Feature report saved: {report_file}")
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features from raw data.
        
        This creates:
        - Interaction terms
        - Ratio features
        - Normalized hill-adjusted features
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        # -----------------------------------------------------------------
        # HILL-NORMALIZED DISTANCE
        # -----------------------------------------------------------------
        # Normalize distance relative to hill size for fair comparison
        if 'AthleteDistance' in df.columns and 'HillHS' in df.columns:
            df['normalized_distance'] = (df['AthleteDistance'] - df['HillK']) / (df['HillHS'] - df['HillK'] + 1)
        
        # -----------------------------------------------------------------
        # STABILITY RATIO
        # -----------------------------------------------------------------
        # Combine multiple stability metrics
        if 'flight_std' in df.columns and 'flight_jitter' in df.columns:
            df['stability_ratio'] = df['flight_std'] / (df['flight_jitter'] + 0.001)
        
        # -----------------------------------------------------------------
        # EXPLOSIVE EFFICIENCY
        # -----------------------------------------------------------------
        # Takeoff power relative to timing
        if 'knee_peak_velocity' in df.columns and 'takeoff_timing_offset' in df.columns:
            df['explosive_efficiency'] = df['knee_peak_velocity'] / (abs(df['takeoff_timing_offset']) + 1)
        
        # -----------------------------------------------------------------
        # LANDING QUALITY INDEX
        # -----------------------------------------------------------------
        if 'landing_smoothness_score' in df.columns and 'telemark_scissor_mean' in df.columns:
            df['landing_quality_index'] = df['landing_smoothness_score'] * df['telemark_scissor_mean']
        
        return df
    
    # =========================================================================
    # MODEL TRAINING AND EVALUATION
    # =========================================================================
    
    def prepare_features_target(self, df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix X and target vector y.
        
        This method:
        1. Selects feature columns
        2. Removes rows with missing target
        3. Imputes missing feature values
        4. Scales features
        
        Args:
            df: Input dataframe
            target: Target column name
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Filter to rows with valid target
        mask = df[target].notna()
        df_valid = df[mask].copy()
        
        # Get feature columns that have at least some valid data
        valid_features = []
        for col in self.feature_columns:
            if col in df_valid.columns and df_valid[col].notna().sum() > 5:
                valid_features.append(col)
        
        if len(valid_features) == 0:
            return np.array([]), np.array([]), []
        
        X = df_valid[valid_features].values
        y = df_valid[target].values
        
        # Impute missing values with median
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, valid_features
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str]) -> Dict:
        """
        Train Random Forest and extract feature importance.
        
        Random Forest is ideal for this task because:
        1. Handles non-linear relationships
        2. Robust to outliers and noise
        3. Provides feature importance out-of-the-box
        4. Works well with small datasets
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            feature_names: List of feature names
            
        Returns:
            Dict with model performance and feature importance
        """
        if len(X) < 10:
            print(f"⚠️ Not enough samples: {len(X)}")
            return {}
        
        # Model configuration
        # Use small max_depth to prevent overfitting
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=4,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=1  # Sequential to avoid threading issues
        )
        
        # Cross-Validation: Use LOO for small datasets, k-fold for larger
        if len(X) <= 30:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        # Get predictions using CV
        y_pred = cross_val_predict(rf, X, y, cv=cv)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Fit on all data for feature importance
        rf.fit(X, y)
        
        # Built-in feature importance
        importance_builtin = rf.feature_importances_
        
        # Permutation importance (more reliable)
        perm_imp = permutation_importance(rf, X, y, n_repeats=10, random_state=self.random_state)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_builtin': importance_builtin,
            'importance_perm_mean': perm_imp.importances_mean,
            'importance_perm_std': perm_imp.importances_std
        }).sort_values('importance_perm_mean', ascending=False)
        
        return {
            'model': rf,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_true': y,
            'y_pred': y_pred,
            'importance': importance_df,
            'n_samples': len(y)
        }
    
    def train_gradient_boosting(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: List[str]) -> Dict:
        """
        Train Gradient Boosting model (XGBoost if available, else sklearn).
        
        Gradient Boosting often provides better performance than Random Forest
        for structured/tabular data.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dict with model performance and feature importance
        """
        if len(X) < 10:
            return {}
        
        # Choose model
        if HAS_XGB:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbosity=0,
                n_jobs=1  # Sequential to avoid issues
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )
        
        # Cross-Validation: Use LOO for small datasets, k-fold for larger
        if len(X) <= 30:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # Metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Fit and get importance
        model.fit(X, y)
        
        if HAS_XGB:
            importance_builtin = model.feature_importances_
        else:
            importance_builtin = model.feature_importances_
        
        perm_imp = permutation_importance(model, X, y, n_repeats=10, random_state=self.random_state)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_builtin': importance_builtin,
            'importance_perm_mean': perm_imp.importances_mean,
            'importance_perm_std': perm_imp.importances_std
        }).sort_values('importance_perm_mean', ascending=False)
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_true': y,
            'y_pred': y_pred,
            'importance': importance_df,
            'n_samples': len(y)
        }
    
    def train_linear_baseline(self, X: np.ndarray, y: np.ndarray,
                               feature_names: List[str]) -> Dict:
        """
        Train regularized linear model as baseline.
        
        Linear models help identify:
        1. Linear relationships in the data
        2. Feature coefficients (direct interpretation)
        3. Whether non-linear models provide improvement
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            
        Returns:
            Dict with model performance and coefficients
        """
        if len(X) < 10:
            return {}
        
        # Ridge regression (L2 regularization)
        model = Ridge(alpha=1.0)
        
        # Cross-Validation: Use LOO for small datasets, k-fold for larger
        if len(X) <= 30:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # Metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Fit for coefficients
        model.fit(X, y)
        
        # Coefficients as importance (absolute value)
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return {
            'model': model,
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'y_true': y,
            'y_pred': y_pred,
            'coefficients': coef_df,
            'n_samples': len(y)
        }
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                                 target_name: str,
                                 model_name: str,
                                 save_path: Optional[Path] = None):
        """
        Plot feature importance bar chart.
        
        Args:
            importance_df: DataFrame with feature importance
            target_name: Name of target variable
            model_name: Name of the model
            save_path: Path to save figure
        """
        # Top 15 features
        df_plot = importance_df.head(15).copy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_pos = np.arange(len(df_plot))
        
        # Use permutation importance if available
        if 'importance_perm_mean' in df_plot.columns:
            values = df_plot['importance_perm_mean']
            errors = df_plot['importance_perm_std']
            ax.barh(y_pos, values, xerr=errors, capsize=3, alpha=0.8, color='steelblue')
        else:
            values = df_plot['importance_builtin']
            ax.barh(y_pos, values, alpha=0.8, color='steelblue')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_plot['feature'])
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('Permutation Importance')
        ax.set_title(f'Feature Importance: {model_name} → {target_name}')
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.close()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                          target_name: str, model_name: str,
                          save_path: Optional[Path] = None):
        """
        Plot actual vs predicted values.
        
        This visualizes model accuracy and identifies outliers.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            target_name: Name of target
            model_name: Name of model
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Regression line
        z = np.polyfit(y_true, y_pred, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], [p(min_val), p(max_val)], 'b-', alpha=0.5, label='Fit Line')
        
        # R² annotation
        r2 = r2_score(y_true, y_pred)
        ax.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                    fontsize=12, fontweight='bold')
        
        ax.set_xlabel(f'Actual {target_name}')
        ax.set_ylabel(f'Predicted {target_name}')
        ax.set_title(f'{model_name}: Actual vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.close()
    
    def plot_model_comparison(self, results: Dict, target_name: str,
                               save_path: Optional[Path] = None):
        """
        Compare performance across different models.
        
        Args:
            results: Dict mapping model names to their results
            target_name: Name of target variable
            save_path: Path to save figure
        """
        model_names = list(results.keys())
        r2_scores = [results[m].get('r2', np.nan) for m in model_names]
        mae_scores = [results[m].get('mae', np.nan) for m in model_names]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # R² comparison
        ax1 = axes[0]
        colors = ['green' if r > 0 else 'red' for r in r2_scores]
        ax1.bar(model_names, r2_scores, color=colors, alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='--')
        ax1.set_ylabel('R² Score')
        ax1.set_title(f'Model Comparison: R² for {target_name}')
        ax1.tick_params(axis='x', rotation=45)
        
        # MAE comparison
        ax2 = axes[1]
        ax2.bar(model_names, mae_scores, color='steelblue', alpha=0.7)
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title(f'Model Comparison: MAE for {target_name}')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved: {save_path}")
        
        plt.close()
    
    # =========================================================================
    # SHAP ANALYSIS (EXPLAINABILITY)
    # =========================================================================
    
    def compute_shap_values(self, model, X: np.ndarray, feature_names: List[str],
                            save_path: Optional[Path] = None) -> Optional[np.ndarray]:
        """
        Compute SHAP values for model explainability.
        
        SHAP (SHapley Additive exPlanations) provides:
        1. Per-prediction feature contributions
        2. Global feature importance
        3. Interaction effects
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: List of feature names
            save_path: Path to save SHAP summary plot
            
        Returns:
            np.ndarray: SHAP values matrix, or None if unavailable
        """
        if not HAS_SHAP:
            print("⚠️ SHAP not installed - skipping explainability analysis")
            return None
        
        try:
            # Create explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Summary plot
            if save_path:
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"✅ SHAP plot saved: {save_path}")
                plt.close()
            
            return shap_values
            
        except Exception as e:
            print(f"⚠️ SHAP computation failed: {e}")
            return None
    
    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================
    
    def run_full_analysis(self):
        """
        Run complete ML analysis pipeline.
        
        This method:
        1. Loads and prepares data
        2. Trains multiple models for each target
        3. Extracts and compares feature importance
        4. Generates visualizations
        5. Saves results
        """
        print("\n" + "=" * 70)
        print("MACHINE LEARNING ANALYSIS")
        print("=" * 70)
        
        # -----------------------------------------------------------------
        # LOAD DATA
        # -----------------------------------------------------------------
        df = self.load_and_prepare_data()
        
        if df is None or df.empty:
            print("❌ Failed to load data")
            return False
        
        # -----------------------------------------------------------------
        # TRAIN MODELS FOR EACH TARGET
        # -----------------------------------------------------------------
        all_results = {}
        
        for target in ['Style_Score', 'Physical_Score']:
            print(f"\n{'='*50}")
            print(f"TARGET: {target}")
            print('='*50)
            
            # Prepare data
            X, y, feature_names = self.prepare_features_target(df, target)
            
            if len(X) < 10:
                print(f"⚠️ Insufficient data for {target}: {len(X)} samples")
                continue
            
            print(f"📊 Data shape: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Train models
            results = {}
            
            # 1. Random Forest
            print("\n🌲 Training Random Forest...")
            rf_result = self.train_random_forest(X, y, feature_names)
            if rf_result:
                results['Random Forest'] = rf_result
                print(f"   R² = {rf_result['r2']:.3f}, MAE = {rf_result['mae']:.3f}")
            
            # 2. Gradient Boosting
            print("\n🚀 Training Gradient Boosting...")
            gb_result = self.train_gradient_boosting(X, y, feature_names)
            if gb_result:
                results['Gradient Boosting'] = gb_result
                print(f"   R² = {gb_result['r2']:.3f}, MAE = {gb_result['mae']:.3f}")
            
            # 3. Linear Baseline
            print("\n📈 Training Linear Baseline...")
            lin_result = self.train_linear_baseline(X, y, feature_names)
            if lin_result:
                results['Ridge Regression'] = lin_result
                print(f"   R² = {lin_result['r2']:.3f}, MAE = {lin_result['mae']:.3f}")
            
            all_results[target] = results
            
            # -----------------------------------------------------------------
            # VISUALIZATIONS
            # -----------------------------------------------------------------
            
            # Feature importance plots
            for model_name, res in results.items():
                if 'importance' in res:
                    plot_path = self.output_dir / f'importance_{target}_{model_name.replace(" ", "_")}.png'
                    self.plot_feature_importance(res['importance'], target, model_name, plot_path)
                
                # Prediction plot
                if 'y_true' in res and 'y_pred' in res:
                    plot_path = self.output_dir / f'predictions_{target}_{model_name.replace(" ", "_")}.png'
                    self.plot_predictions(res['y_true'], res['y_pred'], target, model_name, plot_path)
            
            # Model comparison
            if results:
                plot_path = self.output_dir / f'model_comparison_{target}.png'
                self.plot_model_comparison(results, target, plot_path)
            
            # SHAP analysis (for best model)
            if rf_result and HAS_SHAP:
                shap_path = self.output_dir / f'shap_{target}.png'
                self.compute_shap_values(rf_result['model'], X, feature_names, shap_path)
        
        # -----------------------------------------------------------------
        # SAVE SUMMARY REPORT
        # -----------------------------------------------------------------
        self._save_summary_report(all_results)
        
        print("\n" + "=" * 70)
        print("ML ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)
        
        return True
    
    def _save_summary_report(self, all_results: Dict):
        """
        Save summary report of all models.
        
        Args:
            all_results: Dict mapping targets to model results
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("MACHINE LEARNING ANALYSIS SUMMARY")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        for target, results in all_results.items():
            report_lines.append(f"\nTARGET: {target}")
            report_lines.append("-" * 50)
            
            for model_name, res in results.items():
                report_lines.append(f"\n  {model_name}:")
                report_lines.append(f"    Samples: {res.get('n_samples', 'N/A')}")
                report_lines.append(f"    R²:  {res.get('r2', np.nan):.4f}")
                report_lines.append(f"    MAE: {res.get('mae', np.nan):.4f}")
                report_lines.append(f"    MSE: {res.get('mse', np.nan):.4f}")
                
                # Top 5 features
                if 'importance' in res:
                    report_lines.append("\n    Top 5 Features:")
                    for i, row in res['importance'].head(5).iterrows():
                        report_lines.append(f"      {row['feature']}: {row['importance_perm_mean']:.4f}")
        
        # Save report
        report_path = self.output_dir / 'ml_summary_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Summary report saved: {report_path}")
        
        # Also save all feature importance to CSV
        for target, results in all_results.items():
            for model_name, res in results.items():
                if 'importance' in res:
                    csv_path = self.output_dir / f'importance_{target}_{model_name.replace(" ", "_")}.csv'
                    res['importance'].to_csv(csv_path, index=False)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MACHINE LEARNING MODELS FOR SKI JUMPING")
    print("=" * 70)
    
    if not HAS_SKLEARN:
        print("❌ scikit-learn is required. Install with: pip install scikit-learn")
    else:
        ml = SkiJumpingMLModels()
        ml.run_full_analysis()
