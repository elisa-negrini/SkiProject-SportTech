import pandas as pd
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

@dataclass
class JumpPhase:
    name: str
    start_frame: int
    end_frame: int
    duration_seconds: float

class SkiJumpAnalyst:
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent.parent
        self.data_dir = self.base_path / 'dataset'
        self.output_dir = self.base_path / 'metrics' / 'profile_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.keypoints_file = self.data_dir / 'keypoints_dataset.csv'
        self.phases_file = self.data_dir / 'jump_phases_SkiTB.csv'
        self.jp_data_file = self.data_dir / 'JP_data.csv'
        
        self.fps = 30
        self.normalized_length = 100  
        
        self.kpt_map = {
            'head': '1', 'neck': '2', 'pelvis': '9',
            'r_shoulder': '3', 'l_shoulder': '6',
            'r_hip': '17', 'l_hip': '10',
            'r_knee': '18', 'l_knee': '11',
            'r_ankle': '19', 'l_ankle': '12',
            'r_tip': '23', 'r_tail': '22',
            'l_tip': '16', 'l_tail': '15'
        }

    def load_data(self) -> bool:
        if not self.keypoints_file.exists():
            print(f"❌ Missing: {self.keypoints_file}")
            return False
            
        print("Loading datasets...")
        self.df_kpts = pd.read_csv(self.keypoints_file)
        self.df_kpts['jump_id'] = self.df_kpts['jump_id'].apply(self._normalize_jid)
        self.df_kpts['frame_idx'] = self.df_kpts['frame_name'].apply(self._extract_frame_num)
        self.df_kpts = self.df_kpts.drop_duplicates(subset=['jump_id', 'frame_idx'])
        
        if self.phases_file.exists():
            self.df_phases = pd.read_csv(self.phases_file)
            self.df_phases['jump_id'] = self.df_phases['jump_id'].apply(self._normalize_jid)
        
        if self.jp_data_file.exists():
            self.df_jp = pd.read_csv(self.jp_data_file)
            self._compute_scores()
        else:
            self.df_scores = None
            print("⚠️ JP_data.csv not found (Scores unavailable)")
            
        print(f" Loaded: {len(self.df_kpts['jump_id'].unique())} jumps")
        return True

    def _normalize_jid(self, val) -> str:
        s = str(val).strip()
        if s.lower().startswith('jump'):
            return f"JP{int(s[4:]):04d}"
        return s

    def _extract_frame_num(self, fname: str) -> int:
        import re
        match = re.match(r'^(\d+)', str(fname))
        return int(match.group(1)) if match else -1

    def _compute_scores(self):
        scores = []
        for _, row in self.df_jp.iterrows():
            judges = [row.get(f'AthleteJdg{x}', np.nan) for x in 'ABCDE']
            valid = [s for s in judges if pd.notna(s)]
            style = sum(sorted(valid)[1:4]) if len(valid) >= 5 else np.nan
            total = row.get('AthleteScore', np.nan)
            physical = total - style if (pd.notna(total) and pd.notna(style)) else np.nan
            
            scores.append({
                'jump_id': row['ID'],
                'Style_Score': style,
                'Physical_Score': physical,
                'AthleteName': f"{row.get('AthleteName','')} {row.get('AthleteSurname','')}"
            })
        self.df_scores = pd.DataFrame(scores)

    def get_point(self, row, name: str):
        try:
            kid = self.kpt_map[name]
            x, y = row[f'kpt_{kid}_x'], row[f'kpt_{kid}_y']
            if pd.isna(x) or pd.isna(y): return None
            return np.array([x, y])
        except: return None

    def segment_phases(self, phase_row) -> Dict[str, JumpPhase]:
        phases = {}
        def get_val(col): 
            v = phase_row.get(col)
            return int(v) if pd.notna(v) else None

        takeoff = get_val('take_off_frame')
        bsa_start = get_val('bsa_start')
        bsa_end = get_val('bsa_end')
        landing = get_val('landing')
        tele_end = get_val('telemark_end')

        if takeoff:
            phases['take_off'] = JumpPhase('take_off', max(0, takeoff-10), takeoff+5, 15/self.fps)
        if takeoff and bsa_start:
            phases['early_flight'] = JumpPhase('early_flight', takeoff+5, bsa_start, (bsa_start-takeoff-5)/self.fps)
        if bsa_start and bsa_end:
            phases['mid_flight'] = JumpPhase('mid_flight', bsa_start, bsa_end, (bsa_end-bsa_start)/self.fps)
        if bsa_end and landing:
            phases['late_flight'] = JumpPhase('late_flight', bsa_end, landing, (landing-bsa_end)/self.fps)
        if landing:
            end = tele_end if tele_end else landing + 15
            phases['landing'] = JumpPhase('landing', landing, end, (end-landing)/self.fps)
        return phases

    def compute_phase_metrics(self, jump_df, phases) -> Dict:
        metrics = {}
        if 'take_off' in phases:
            p = phases['take_off']
            df_p = jump_df[(jump_df['frame_idx'] >= p.start_frame) & (jump_df['frame_idx'] <= p.end_frame)]
            knee_angles = []
            for _, row in df_p.iterrows():
                angles = []
                for s in ['r', 'l']:
                    h, k, a = self.get_point(row, f'{s}_hip'), self.get_point(row, f'{s}_knee'), self.get_point(row, f'{s}_ankle')
                    if h is not None and k is not None and a is not None:
                        v1, v2 = h-k, a-k
                        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1, 1)))
                        angles.append(angle)
                if angles: knee_angles.append(np.mean(angles))
            if len(knee_angles) > 2:
                vel = np.diff(knee_angles) * self.fps
                metrics['takeoff_explosiveness'] = np.max(vel)
        
        if 'mid_flight' in phases:
            p = phases['mid_flight']
            df_p = jump_df[(jump_df['frame_idx'] >= p.start_frame) & (jump_df['frame_idx'] <= p.end_frame)]
            bsa_values = []
            for _, row in df_p.iterrows():
                neck, pelvis = self.get_point(row, 'neck'), self.get_point(row, 'pelvis')
                r_tip, r_tail = self.get_point(row, 'r_tip'), self.get_point(row, 'r_tail')
                if all(x is not None for x in [neck, pelvis, r_tip, r_tail]):
                    body, ski = neck - pelvis, r_tip - r_tail
                    angle = np.degrees(np.arccos(np.clip(np.dot(body, ski)/(np.linalg.norm(body)*np.linalg.norm(ski)), -1, 1)))
                    bsa_values.append(angle)
            if len(bsa_values) > 5:
                metrics['flight_mean_bsa'] = np.mean(bsa_values)
                metrics['flight_stability_std'] = np.std(bsa_values)
        
        if 'landing' in phases:
            p = phases['landing']
            df_p = jump_df[(jump_df['frame_idx'] >= p.start_frame) & (jump_df['frame_idx'] <= p.end_frame)]
            scissor_diffs = []
            for _, row in df_p.iterrows():
                ra, la = self.get_point(row, 'r_ankle'), self.get_point(row, 'l_ankle')
                if ra is not None and la is not None: scissor_diffs.append(abs(ra[1] - la[1]))
            if scissor_diffs:
                metrics['landing_telemark_quality'] = np.mean(scissor_diffs)
        return metrics

    def normalize_trajectory(self, series: pd.Series) -> np.ndarray:
        clean = series.dropna()
        if len(clean) < 5: return np.full(self.normalized_length, np.nan)
        x_old = np.linspace(0, 1, len(clean))
        f = interp1d(x_old, clean.values, kind='linear', fill_value="extrapolate")
        return f(np.linspace(0, 1, self.normalized_length))

    def extract_flight_curve(self, jump_id):
        row = self.df_phases[self.df_phases['jump_id'] == jump_id]
        if row.empty: return None
        start, end = row.iloc[0]['bsa_start'], row.iloc[0]['landing']
        if pd.isna(start) or pd.isna(end): return None
        
        df_j = self.df_kpts[(self.df_kpts['jump_id'] == jump_id) & (self.df_kpts['frame_idx'] >= start) & (self.df_kpts['frame_idx'] <= end)].sort_values('frame_idx')
        angles, frames = [], []
        for _, row in df_j.iterrows():
            neck, pelvis = self.get_point(row, 'neck'), self.get_point(row, 'pelvis')
            r_tip, r_tail = self.get_point(row, 'r_tip'), self.get_point(row, 'r_tail')
            if all(x is not None for x in [neck, pelvis, r_tip, r_tail]):
                body, ski = neck - pelvis, r_tip - r_tail
                angle = np.degrees(np.arccos(np.clip(np.dot(body, ski)/(np.linalg.norm(body)*np.linalg.norm(ski)), -1, 1)))
                angles.append(angle)
                frames.append(row['frame_idx'])
        return pd.Series(data=angles, index=frames)

    def analyze_top_vs_flop(self, top_n=5):
        if self.df_scores is None: return None
        valid_jumps = self.df_kpts['jump_id'].unique()
        df_s = self.df_scores[self.df_scores['jump_id'].isin(valid_jumps)].copy()
        
        if len(df_s) < top_n * 2:
            print("⚠️ Not enough data for Top/Flop analysis")
            return None
            
        df_s = df_s.sort_values('Style_Score', ascending=False)
        top_group = df_s.head(top_n)
        flop_group = df_s.tail(top_n)
        
        print(f"\n🏆 TOP {top_n} ATLETS (High Style):")
        for _, r in top_group.iterrows(): print(f"   {r['jump_id']}: {r['AthleteName']} ({r['Style_Score']})")
        print(f"\n📉 FLOP {top_n} ATLETS (Low Style):")
        for _, r in flop_group.iterrows(): print(f"   {r['jump_id']}: {r['AthleteName']} ({r['Style_Score']})")
        
        top_ids = top_group['jump_id'].tolist()
        flop_ids = flop_group['jump_id'].tolist()
        
        def get_avg_curve(ids):
            curves = []
            for jid in ids:
                raw = self.extract_flight_curve(jid)
                if raw is not None:
                    norm = self.normalize_trajectory(raw)
                    if not np.all(np.isnan(norm)): curves.append(norm)
            return np.mean(curves, axis=0) if curves else None, np.std(curves, axis=0) if curves else None

        top_mean, top_std = get_avg_curve(top_ids)
        flop_mean, flop_std = get_avg_curve(flop_ids)
        
        return {
            'top_mean': top_mean, 'top_std': top_std,
            'flop_mean': flop_mean, 'flop_std': flop_std
        }

    def plot_top_vs_flop(self, data):
        if data['top_mean'] is None: return
        x = np.linspace(0, 100, self.normalized_length)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, data['top_mean'], 'g-', linewidth=2, label='Top Performers')
        plt.fill_between(x, data['top_mean']-data['top_std'], data['top_mean']+data['top_std'], color='green', alpha=0.2)
        plt.plot(x, data['flop_mean'], 'r--', linewidth=2, label='Low Performers')
        plt.fill_between(x, data['flop_mean']-data['flop_std'], data['flop_mean']+data['flop_std'], color='red', alpha=0.2)
        
        plt.title("Body-Ski Angle: Top 5 vs Bottom 5", fontsize=14)
        plt.xlabel("Flight Phase %")
        plt.ylabel("Angle (deg)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_file = self.output_dir / 'top_vs_flop_comparison.png'
        plt.savefig(out_file, dpi=150)
        print(f" Plot saved: {out_file}")
        plt.close()

    def save_trend_csv(self, data):
        if data['top_mean'] is None: return
        
        df = pd.DataFrame({
            'Flight_Percent': np.linspace(0, 100, self.normalized_length),
            'Top_Mean_BSA': data['top_mean'],
            'Top_Std_BSA': data['top_std'],
            'Flop_Mean_BSA': data['flop_mean'],
            'Flop_Std_BSA': data['flop_std']
        })
        
        out_file = self.output_dir / 'top_vs_flop_trends.csv'
        df.to_csv(out_file, index=False)
        print(f" Curve data saved: {out_file}")

    def run(self):
        print("🚀 SKI JUMP UNIFIED ANALYSIS")
        
        if not self.load_data(): return
        
        results = []
        for jid in self.df_kpts['jump_id'].unique():
            p_row = self.df_phases[self.df_phases['jump_id'] == jid]
            if p_row.empty: continue
            phases = self.segment_phases(p_row.iloc[0])
            if not phases: continue
            metrics = self.compute_phase_metrics(self.df_kpts[self.df_kpts['jump_id'] == jid], phases)
            metrics['jump_id'] = jid
            results.append(metrics)
            
        df_res = pd.DataFrame(results)
        df_res.to_csv(self.output_dir / 'comprehensive_metrics.csv', index=False)
        print(f"\n Phase metrics saved: {self.output_dir / 'comprehensive_metrics.csv'}")
        
        tvf_data = self.analyze_top_vs_flop()
        if tvf_data:
            self.plot_top_vs_flop(tvf_data)
            self.save_trend_csv(tvf_data)
            
        print("\n" + "="*60)
        print(" DONE")

if __name__ == "__main__":
    SkiJumpAnalyst().run()