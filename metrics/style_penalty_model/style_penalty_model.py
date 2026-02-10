"""
Style Penalty Model - Manual Feature Selection
==============================================

This model predicts STYLE SCORE LOSS based on 3 carefully selected features:

SELECTED FEATURES:
------------------
1. flight_std: Flight stability (body-ski angle variation)
   - Rationale: Judges reward stable, controlled flight positions
   - Source: timeseries_metrics/timeseries_summary.csv
   
2. landing_hip_velocity: Impact hardness at landing
   - Rationale: Judges heavily penalize hard landings (major deduction)
   - Source: timeseries_metrics/timeseries_summary.csv
   
3. telemark_scissor_mean: Telemark leg separation quality
   - Rationale: Proper telemark position is required for style points
   - Source: advanced_metrics/advanced_metrics_summary.csv

WHY THESE 3 FEATURES:
---------------------
- They are NOT correlated with each other (no multicollinearity)
- They cover different phases: Flight (std), Landing (hip_velocity, scissor)
- They have the strongest validated correlations with Style_Score
- They are interpretable for coaches

Output:
- models/style_penalty_predictions.csv
- models/STYLE_PENALTY_FORMULA.txt
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class StylePenaltyModel:
    """
    Transparent model predicting style score penalties using 3 manual features.
    """
    
    # Manual feature selection - these are the 3 key features
    MANUAL_FEATURES = {
        'flight_std': {
            'name': 'Flight Stability',
            'description': 'Body-ski angle variation during flight (lower = more stable)',
            'expected_effect': 'positive',  # More variation = more penalty
            'category': 'Flight',
            'source': 'timeseries'
        },
        'landing_hip_velocity': {
            'name': 'Landing Impact',
            'description': 'Hip descent velocity at landing (higher = harder landing)',
            'expected_effect': 'positive',  # Harder landing = more penalty
            'category': 'Landing',
            'source': 'timeseries'
        },
        'telemark_scissor_mean': {
            'name': 'Telemark Quality',
            'description': 'Leg separation in telemark position (normalized)',
            'expected_effect': 'context',  # Optimal range exists
            'category': 'Landing',
            'source': 'advanced'
        }
    }
    
    def __init__(self):
        """Initialize paths and configurations."""
        self.base_path = Path(__file__).parent.parent.parent
        self.metrics_path = self.base_path / 'metrics'
        
        # Output directory
        self.output_dir = self.metrics_path / 'style_penalty_model'
        self.output_dir.mkdir(exist_ok=True)
        
        # Model storage
        self.model = None
        self.scaler = None
        self.coefficients = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and merge all required data."""
        print("\n📂 Loading data...")
        
        # Load JP data
        jp_file = self.base_path / 'dataset' / 'JP_data.csv'
        df_jp = pd.read_csv(jp_file)
        df_jp['jump_id'] = df_jp['ID']
        
        # Compute Style_Score (middle 3 of 5 judges)
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
            
            scores.append({
                'jump_id': row['ID'],
                'Style_Score': style,
                'AthleteName': f"{row.get('AthleteName', '')} {row.get('AthleteSurname', '')}"
            })
        
        df = pd.DataFrame(scores)
        print(f"   ✓ JP data: {len(df)} jumps")
        
        # Max style for penalty calculation
        self.max_style = df['Style_Score'].max()
        print(f"   ✓ Max Style Score: {self.max_style:.1f}")
        
        # Load time-series metrics
        ts_file = self.metrics_path / 'timeseries_metrics' / 'timeseries_summary.csv'
        if ts_file.exists():
            df_ts = pd.read_csv(ts_file)
            df = df.merge(df_ts, on='jump_id', how='left')
            print(f"   ✓ Time-series metrics: merged")
        else:
            print(f"   ⚠️ Time-series file not found: {ts_file}")
        
        # Load advanced metrics
        adv_file = self.metrics_path / 'core_metrics' / 'metrics_summary_per_jump.csv'
        if adv_file.exists():
            df_adv = pd.read_csv(adv_file)
            df = df.merge(df_adv, on='jump_id', how='left')
            print(f"   ✓ Advanced metrics: merged")
        else:
            print(f"   ⚠️ Advanced metrics file not found: {adv_file}")
        
        # Clean
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix using only manual features."""
        print("\n🔍 Preparing manual features...")
        
        available_features = []
        for feat_name in self.MANUAL_FEATURES.keys():
            if feat_name in df.columns:
                n_valid = df[feat_name].notna().sum()
                pct = 100 * n_valid / len(df)
                print(f"   ✓ {feat_name}: {n_valid} samples ({pct:.0f}%)")
                available_features.append(feat_name)
            else:
                print(f"   ❌ {feat_name}: NOT FOUND in data")
        
        if not available_features:
            print("   ⚠️ No features available!")
            return None, None, []
        
        # Filter to rows with at least 2 features available
        df_valid = df.dropna(subset=['Style_Score'])
        feature_mask = df_valid[available_features].notna().sum(axis=1) >= 2
        df_valid = df_valid[feature_mask].copy()
        
        # Compute penalty (points lost from max)
        df_valid['style_penalty'] = self.max_style - df_valid['Style_Score']
        
        # Impute missing with median
        X = df_valid[available_features].copy()
        X = X.fillna(X.median())
        
        y = df_valid['style_penalty']
        
        print(f"\n   📊 Final dataset: {len(df_valid)} samples, {len(available_features)} features")
        
        return X, y, df_valid
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train Ridge regression with LOO-CV."""
        print("\n🎯 Training Style Penalty Model...")
        print(f"   Training samples: {len(X)}")
        print(f"   Penalty range: {y.min():.1f} to {y.max():.1f} points")
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Ridge regression
        self.model = Ridge(alpha=1.0)
        self.model.fit(X_scaled, y)
        
        # Store coefficients
        for i, feat in enumerate(X.columns):
            self.coefficients[feat] = self.model.coef_[i]
        
        # LOO-CV for evaluation
        loo = LeaveOneOut()
        predictions = np.zeros(len(X))
        
        for train_idx, test_idx in loo.split(X):
            X_train = X_scaled[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_scaled[test_idx]
            
            model_temp = Ridge(alpha=1.0)
            model_temp.fit(X_train, y_train)
            predictions[test_idx] = model_temp.predict(X_test)
        
        # Metrics
        errors = predictions - y.values
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        ss_res = np.sum(errors**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"\n   📈 Model Performance (LOO-CV):")
        print(f"      R² = {r2:.3f}")
        print(f"      MAE = {mae:.2f} style points")
        print(f"      RMSE = {rmse:.2f} style points")
        
        return {
            'predictions': predictions,
            'actuals': y.values,
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
    
    def generate_report(self, df_valid: pd.DataFrame, results: dict):
        """Generate human-readable report."""
        
        print("\n" + "=" * 70)
        print("📋 STYLE PENALTY MODEL - COACH REPORT")
        print("=" * 70)
        
        # Formula
        print("\n🎯 STYLE PENALTY FORMULA:")
        print("-" * 50)
        print("Style_Penalty (points lost) =")
        
        # Sort by absolute coefficient
        sorted_feats = sorted(self.coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        total_abs = sum(abs(c) for _, c in sorted_feats)
        
        for feat, coef in sorted_feats:
            feat_info = self.MANUAL_FEATURES.get(feat, {})
            name = feat_info.get('name', feat)
            importance = 100 * abs(coef) / total_abs if total_abs > 0 else 0
            sign = '+' if coef >= 0 else ''
            print(f"    {sign}{coef:6.2f} × {name} ({importance:.0f}%)")
        
        print(f"    + {self.model.intercept_:6.2f} (base penalty)")
        
        # Interpretation
        print("\n📖 INTERPRETATION:")
        print("-" * 50)
        
        for feat, coef in sorted_feats:
            feat_info = self.MANUAL_FEATURES.get(feat, {})
            name = feat_info.get('name', feat)
            desc = feat_info.get('description', '')
            
            print(f"\n  {name}:")
            print(f"    → {desc}")
            if coef > 0:
                print(f"    → Higher value = MORE penalty (worse)")
                print(f"    💡 Coach tip: REDUCE this metric")
            else:
                print(f"    → Higher value = LESS penalty (better)")
                print(f"    💡 Coach tip: INCREASE this metric")
        
        # Athlete analysis
        if 'AthleteName' in df_valid.columns:
            df_valid = df_valid.copy()
            df_valid['predicted_penalty'] = results['predictions']
            df_valid['prediction_error'] = results['predictions'] - df_valid['style_penalty']
            
            print("\n📊 ATHLETE ANALYSIS:")
            print("-" * 50)
            
            # Worst predictions (highest penalty)
            worst = df_valid.nlargest(5, 'style_penalty')
            print("\n  ⚠️ JUMPS WITH HIGHEST STYLE PENALTY:")
            for _, row in worst.iterrows():
                print(f"\n    {row['AthleteName']}")
                print(f"    Style Score: {row['Style_Score']:.1f} (Best: {self.max_style})")
                print(f"    Penalty: {row['style_penalty']:.1f} points")
        
        print("\n" + "=" * 70)
    
    def save_results(self, df_valid: pd.DataFrame, results: dict):
        """Save predictions and formula to files."""
        
        # Predictions CSV
        df_out = df_valid[['jump_id', 'AthleteName', 'Style_Score']].copy()
        df_out['style_penalty_actual'] = df_valid['style_penalty']
        df_out['style_penalty_predicted'] = results['predictions']
        df_out['prediction_error'] = results['predictions'] - df_valid['style_penalty'].values
        
        pred_file = self.output_dir / 'style_penalty_predictions.csv'
        df_out.to_csv(pred_file, index=False)
        print(f"✅ Saved: {pred_file}")
        
        # Formula text file
        formula_file = self.output_dir / 'STYLE_PENALTY_FORMULA.txt'
        with open(formula_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("STYLE PENALTY MODEL\n")
            f.write("Predicts Style Points Lost Due to Biomechanical Flaws\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("FORMULA:\n")
            f.write("-" * 40 + "\n")
            f.write("Style_Penalty (points) =\n")
            
            sorted_feats = sorted(self.coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
            total_abs = sum(abs(c) for _, c in sorted_feats)
            
            for feat, coef in sorted_feats:
                feat_info = self.MANUAL_FEATURES.get(feat, {})
                name = feat_info.get('name', feat)
                importance = 100 * abs(coef) / total_abs if total_abs > 0 else 0
                sign = '+' if coef >= 0 else ''
                f.write(f"    {sign}{coef:7.4f} x {name} ({importance:.0f}%)\n")
            
            f.write(f"    +{self.model.intercept_:7.4f} (intercept)\n\n")
            
            f.write("FEATURE DETAILS:\n")
            f.write("-" * 40 + "\n\n")
            
            for feat, coef in sorted_feats:
                feat_info = self.MANUAL_FEATURES.get(feat, {})
                name = feat_info.get('name', feat)
                desc = feat_info.get('description', '')
                
                f.write(f"{name}:\n")
                f.write(f"  Original metric: {feat}\n")
                f.write(f"  Coefficient: {coef:.4f}\n")
                f.write(f"  Description: {desc}\n")
                if coef > 0:
                    f.write(f"  Effect: Higher value -> MORE penalty\n")
                else:
                    f.write(f"  Effect: Higher value -> LESS penalty\n")
                f.write("\n")
            
            f.write("\nMODEL PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  R² = {results['r2']:.3f}\n")
            f.write(f"  MAE = {results['mae']:.2f} points\n")
            f.write(f"  RMSE = {results['rmse']:.2f} points\n")
        
        print(f"✅ Saved: {formula_file}")
    
    def run(self):
        """Main execution."""
        print("=" * 70)
        print("STYLE PENALTY MODEL")
        print("Using Manual Feature Selection")
        print("=" * 70)
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        X, y, df_valid = self.prepare_features(df)
        
        if X is None or len(X) < 10:
            print("❌ Insufficient data for model training")
            return False
        
        # Train model
        results = self.train_model(X, y)
        
        # Generate report
        self.generate_report(df_valid, results)
        
        # Save results
        self.save_results(df_valid, results)
        
        print("\n" + "=" * 70)
        print("✅ STYLE PENALTY MODEL COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 70)
        
        return True


if __name__ == "__main__":
    model = StylePenaltyModel()
    model.run()
