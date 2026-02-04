"""
ML Predictions - Detailed Output Per Jump
==========================================

This script generates detailed prediction files showing:
- Actual scores vs Predicted scores for each jump
- Prediction error per jump
- Model comparison (RF, GB, Ridge, Ensemble)

Output:
- predictions_Style_Score.csv
- predictions_Physical_Score.csv

Author: SkiProject-SportTech
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class DetailedMLPredictor:
    """
    Creates detailed prediction files showing actual vs predicted for each jump.
    """
    
    def __init__(self):
        """Initialize paths and output directory."""
        self.base_path = Path(__file__).parent.parent
        self.metrics_path = Path(__file__).parent
        
        # Output to models subfolder
        self.output_dir = self.metrics_path / 'models'
        self.output_dir.mkdir(exist_ok=True)
        
        # Random state for reproducibility
        self.random_state = 42
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and merge all data sources.
        
        Returns:
            DataFrame with merged metrics and scores
        """
        print("\n📂 Loading data sources...")
        
        # Load JP data (scores)
        jp_file = self.base_path / 'JP_data.csv'
        if not jp_file.exists():
            raise FileNotFoundError(f"JP_data.csv not found at {jp_file}")
        
        df_jp = pd.read_csv(jp_file)
        df_jp['jump_id'] = df_jp['ID']
        
        # Compute Style_Score and Physical_Score
        scores = []
        for _, row in df_jp.iterrows():
            judges = [row.get(f'AthleteJdg{x}', np.nan) for x in 'ABCDE']
            valid = [s for s in judges if pd.notna(s)]
            
            if len(valid) >= 5:
                style = sum(sorted(valid)[1:4])  # Middle 3 of 5
            elif len(valid) >= 3:
                style = sum(sorted(valid)[:3])
            else:
                style = np.nan
            
            athlete_score = row.get('AthleteScore', np.nan)
            physical = athlete_score - style if pd.notna(athlete_score) and pd.notna(style) else np.nan
            
            scores.append({
                'jump_id': row['ID'],
                'Style_Score': style,
                'Physical_Score': physical,
                'AthleteScore': athlete_score,
                'AthleteDistance': row.get('AthleteDistance', np.nan),
                'AthleteName': f"{row.get('AthleteName', '')} {row.get('AthleteSurname', '')}"
            })
        
        df_scores = pd.DataFrame(scores)
        print(f"   ✓ JP data: {len(df_scores)} jumps with scores")
        
        # Load time-series metrics (from timeseries_metrics subfolder)
        ts_file = self.metrics_path / 'timeseries_metrics' / 'timeseries_summary.csv'
        if ts_file.exists():
            df_ts = pd.read_csv(ts_file)
            df_scores = df_scores.merge(df_ts, on='jump_id', how='left')
            print(f"   ✓ Time-series metrics merged")
        
        # Load advanced metrics (from advanced_metrics subfolder)
        adv_file = self.metrics_path / 'advanced_metrics' / 'advanced_metrics_summary.csv'
        if adv_file.exists():
            df_adv = pd.read_csv(adv_file)
            df_scores = df_scores.merge(df_adv, on='jump_id', how='left')
            print(f"   ✓ Advanced metrics merged")
        
        # Clean data
        df_scores = df_scores.replace([np.inf, -np.inf], np.nan)
        
        print(f"\n📊 Final dataset: {len(df_scores)} rows, {len(df_scores.columns)} columns")
        
        return df_scores
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of valid feature columns (exclude metadata and targets).
        """
        exclude_cols = [
            'jump_id', 'AthleteName', 'AthleteScore', 'AthleteDistance',
            'Style_Score', 'Physical_Score', 'ID', 'AthleteSurname',
            'AthleteNat', 'HillLocation', 'AthleteJdgA', 'AthleteJdgB',
            'AthleteJdgC', 'AthleteJdgD', 'AthleteJdgE'
        ]
        
        # Check all numeric columns
        numeric_dtypes = ['float64', 'int64', 'float32', 'int32', 'float', 'int']
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        
        print(f"\n📊 Found {len(feature_cols)} numeric feature columns")
        
        return feature_cols
    
    def prepare_features(self, df: pd.DataFrame, feature_cols: list) -> tuple:
        """
        Prepare feature matrix by removing high-NaN columns and imputing.
        
        Returns:
            X (cleaned features), valid_features (list of feature names)
        """
        X = df[feature_cols].copy()
        
        # First, filter to only rows that have ANY metric data
        # (rows where at least one feature column has a value)
        rows_with_data = X.notna().any(axis=1)
        print(f"   Rows with at least some feature data: {rows_with_data.sum()}/{len(X)}")
        
        # Remove columns with >80% NaN (within rows that have data)
        nan_ratio = X.isna().sum() / len(X)
        valid_features = nan_ratio[nan_ratio < 0.8].index.tolist()
        X = X[valid_features]
        
        print(f"   Features after NaN filter: {len(valid_features)}")
        # Impute remaining NaNs with median
        X = X.fillna(X.median())
        
        return X, valid_features
    
    def predict_with_cv(self, X: pd.DataFrame, y: pd.Series, model, model_name: str) -> dict:
        """
        Make predictions using cross-validation.
        
        Uses LOO for small datasets (<40) and 10-fold for larger.
        
        Returns:
            dict with predictions, actuals, errors
        """
        n_samples = len(X)
        
        # Choose CV strategy
        if n_samples <= 40:
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        predictions = np.zeros(n_samples)
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and predict
            model.fit(X_train_scaled, y_train)
            predictions[test_idx] = model.predict(X_test_scaled)
        
        return {
            'predictions': predictions,
            'actuals': y.values,
            'errors': predictions - y.values,
            'abs_errors': np.abs(predictions - y.values)
        }
    
    def analyze_target(self, df: pd.DataFrame, target_name: str, feature_cols: list) -> pd.DataFrame:
        """
        Run full analysis for one target variable.
        
        Returns:
            DataFrame with predictions from all models
        """
        print(f"\n{'='*60}")
        print(f"TARGET: {target_name}")
        print(f"{'='*60}")
        
        # Filter to rows with valid target
        df_valid = df.dropna(subset=[target_name]).copy()
        
        # CRITICAL: Filter to rows that have metrics data
        # Check if at least one metric column has data
        metric_cols = [col for col in feature_cols if col in df_valid.columns]
        if not metric_cols:
            print(f"⚠️ No feature columns found in data")
            return pd.DataFrame(), {}
            
        # Find rows that have at least some metric data
        has_metrics = df_valid[metric_cols].notna().any(axis=1)
        df_valid = df_valid[has_metrics].copy()
        
        if len(df_valid) < 10:
            print(f"⚠️ Not enough samples with metrics: {len(df_valid)}")
            return pd.DataFrame(), {}
        
        # Prepare features
        X, valid_features = self.prepare_features(df_valid, feature_cols)
        y = df_valid[target_name]
        
        print(f"📊 Samples: {len(X)}, Features: {len(valid_features)}")
        
        # Initialize results DataFrame
        results = pd.DataFrame({
            'jump_id': df_valid['jump_id'].values,
            'AthleteName': df_valid['AthleteName'].values,
            'actual': y.values
        })
        
        # Define models
        models = {
            'Random_Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=1
            ),
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            'Ridge': Ridge(alpha=1.0)
        }
        
        # Run each model
        model_results = {}
        
        for model_name, model in models.items():
            print(f"\n🔄 Running {model_name}...")
            
            cv_results = self.predict_with_cv(X, y, model, model_name)
            
            results[f'pred_{model_name}'] = cv_results['predictions']
            results[f'error_{model_name}'] = cv_results['errors']
            results[f'abs_error_{model_name}'] = cv_results['abs_errors']
            
            mae = np.mean(cv_results['abs_errors'])
            rmse = np.sqrt(np.mean(cv_results['errors']**2))
            r2 = 1 - np.sum(cv_results['errors']**2) / np.sum((y - y.mean())**2)
            
            model_results[model_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
            print(f"   R² = {r2:.3f}, MAE = {mae:.2f}, RMSE = {rmse:.2f}")
        
        # Ensemble prediction (average of all models)
        pred_cols = [f'pred_{name}' for name in models.keys()]
        results['pred_Ensemble'] = results[pred_cols].mean(axis=1)
        results['error_Ensemble'] = results['pred_Ensemble'] - results['actual']
        results['abs_error_Ensemble'] = np.abs(results['error_Ensemble'])
        
        # Ensemble metrics
        mae_ens = results['abs_error_Ensemble'].mean()
        rmse_ens = np.sqrt((results['error_Ensemble']**2).mean())
        r2_ens = 1 - (results['error_Ensemble']**2).sum() / ((y - y.mean())**2).sum()
        
        print(f"\n🎯 Ensemble: R² = {r2_ens:.3f}, MAE = {mae_ens:.2f}, RMSE = {rmse_ens:.2f}")
        
        # Sort by error (worst predictions first)
        results = results.sort_values('abs_error_Ensemble', ascending=False)
        
        # Print worst predictions
        print(f"\n⚠️ Worst 5 Predictions:")
        print("-" * 60)
        worst = results[['jump_id', 'AthleteName', 'actual', 'pred_Ensemble', 'error_Ensemble']].head(5)
        for _, row in worst.iterrows():
            print(f"  {row['AthleteName'][:20]:20s} | Actual: {row['actual']:.1f} | "
                  f"Pred: {row['pred_Ensemble']:.1f} | Error: {row['error_Ensemble']:+.1f}")
        
        # Print best predictions
        print(f"\n✅ Best 5 Predictions:")
        print("-" * 60)
        best = results.nsmallest(5, 'abs_error_Ensemble')[['jump_id', 'AthleteName', 'actual', 'pred_Ensemble', 'error_Ensemble']]
        for _, row in best.iterrows():
            print(f"  {row['AthleteName'][:20]:20s} | Actual: {row['actual']:.1f} | "
                  f"Pred: {row['pred_Ensemble']:.1f} | Error: {row['error_Ensemble']:+.1f}")
        
        return results, model_results
    
    def create_visualization(self, results: pd.DataFrame, target_name: str):
        """Create scatter plot of actual vs predicted."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = ['Random_Forest', 'Gradient_Boosting', 'Ridge', 'Ensemble']
        
        for ax, model_name in zip(axes.flatten(), models):
            pred_col = f'pred_{model_name}'
            
            ax.scatter(results['actual'], results[pred_col], alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(results['actual'].min(), results[pred_col].min())
            max_val = max(results['actual'].max(), results[pred_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
            
            # Calculate R²
            ss_res = ((results['actual'] - results[pred_col])**2).sum()
            ss_tot = ((results['actual'] - results['actual'].mean())**2).sum()
            r2 = 1 - ss_res / ss_tot
            
            ax.set_xlabel(f'Actual {target_name}')
            ax.set_ylabel(f'Predicted {target_name}')
            ax.set_title(f'{model_name}\nR² = {r2:.3f}')
            ax.grid(alpha=0.3)
            ax.legend()
        
        plt.suptitle(f'Predictions for {target_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / f'predictions_plot_{target_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Plot saved: {save_path}")
        plt.close()
    
    def run_analysis(self):
        """Main analysis pipeline."""
        print("=" * 60)
        print("DETAILED ML PREDICTIONS")
        print("=" * 60)
        
        # Load data
        df = self.load_data()
        feature_cols = self.get_feature_columns(df)
        
        # Analyze each target
        targets = ['Style_Score', 'Physical_Score']
        
        for target in targets:
            results, model_results = self.analyze_target(df, target, feature_cols)
            
            if not results.empty:
                # Save predictions
                save_path = self.output_dir / f'predictions_{target}.csv'
                results.to_csv(save_path, index=False)
                print(f"\n✅ Predictions saved: {save_path}")
                
                # Create visualization
                self.create_visualization(results, target)
        
        print(f"\n{'='*60}")
        print(f"✅ Analysis complete! Results in: {self.output_dir}")
        print(f"{'='*60}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    predictor = DetailedMLPredictor()
    predictor.run_analysis()
