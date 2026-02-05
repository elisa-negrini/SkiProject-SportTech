# 📊 Guida all'Interpretazione degli Output

Questa guida spiega come leggere e interpretare tutti i file generati dal sistema di analisi delle metriche.

---

## 🏛️ Struttura delle Cartelle

```
metrics/
├── core_metrics/           # Metriche geometriche di base
├── advanced_metrics/       # Metriche dinamiche avanzate
├── timeseries_metrics/     # Statistiche delle serie temporali
├── correlations/           # Analisi delle correlazioni
├── models/                 # Output dei modelli ML
├── data_quality/           # Report di qualità dei dati
└── visualizations/         # Grafici e visualizzazioni
```

---

## 📁 1. Core Metrics (`core_metrics/`)

### `metrics_summary_per_jump.csv`
**Che cos'è:** Metriche geometriche aggregate per ogni salto (1 riga = 1 salto).

| Colonna | Significato | Range Atteso | Buono/Cattivo |
|---------|-------------|--------------|---------------|
| `avg_v_style_front` | Angolo V tra gli sci (vista frontale) | 10-60° | ~35-45° ottimale |
| `avg_v_style_back` | Angolo V tra gli sci (vista posteriore) | 10-60° | ~35-45° ottimale |
| `takeoff_knee_angle` | Angolo del ginocchio allo stacco | 100-180° | >160° = buona estensione |
| `avg_body_ski_angle` | Inclinazione corpo-sci | 0-40° | Basso = corpo parallelo sci |
| `avg_symmetry_index_back` | Simmetria delle gambe | 0-30° | <10° = simmetrico |
| `avg_telemark_proj_ski` | Profondità telemark (proiez. sci) | 0-1 | >0.2 = buon telemark |
| `avg_telemark_depth_ratio` | Profondità telemark normalizzata | 0-0.5 | >0.1 = buon telemark |
| `avg_telemark_leg_angle` | Angolo apertura gambe | 0-90° | 15-40° tipico |

### `metrics_per_frame.csv`
**Che cos'è:** Metriche frame-by-frame per analisi dettagliata.
**Quando usarlo:** Per studiare l'evoluzione di una metrica durante un salto specifico.

---

## 📁 2. Advanced Metrics (`advanced_metrics/`)

### `advanced_metrics_summary.csv`
**Che cos'è:** Metriche dinamiche calcolate da derivate dei keypoint.

| Colonna | Significato | Range Atteso | Interpretazione |
|---------|-------------|--------------|-----------------|
| `takeoff_timing_offset` | Frame di anticipo/ritardo stacco | -10 a +10 | 0 = timing perfetto |
| `takeoff_peak_velocity` | Velocità picco estensione ginocchio | 50-800 °/s | Alto = stacco esplosivo |
| `telemark_scissor_mean` | Distanza media caviglie (normalizzata) | 0-0.30 | >0.15 = buon telemark |
| `telemark_stability` | Stabilità del telemark | 0-50 | Alto = posizione stabile |
| `landing_absorption_rate` | Velocità discesa bacino | -2 a +2 | Positivo = buon ammortizzamento |

---

## 📁 3. Timeseries Metrics (`timeseries_metrics/`)

### `timeseries_summary.csv`
**Che cos'è:** Statistiche estratte dalle serie temporali delle fasi (takeoff, flight, landing).

| Colonna | Significato | Range Atteso |
|---------|-------------|--------------|
| `knee_peak_velocity` | Picco velocità estensione ginocchio | 50-800 °/s |
| `knee_angle_at_takeoff` | Angolo ginocchio allo stacco | 100-180° |
| `flight_std` | Variabilità angolo corpo-sci in volo | 0-15° |
| `flight_jitter` | Variazione frame-to-frame in volo | 0-10° |
| `flight_mean_bsa` | Media inclinazione in volo | 0-45° |
| `landing_hip_velocity` | Velocità discesa fianchi all'atterraggio | 0-3 u/s |
| `landing_knee_compression` | Flessione ginocchio all'atterraggio | 0-90° |

---

## 📁 4. Correlations (`correlations/`)

### `correlations.csv`
**Che cos'è:** Matrice delle correlazioni tra metriche e punteggi.

| Colonna | Significato |
|---------|-------------|
| `metric` | Nome della metrica |
| `target` | Punteggio target (Style_Score, Physical_Score, etc.) |
| `r` | Coefficiente di correlazione Pearson (-1 a +1) |
| `p` | P-value del test (significativo se < 0.05) |
| `n` | Numero di campioni usati |

**Come leggerlo:**
- **r > 0.3** e **p < 0.05**: Correlazione positiva significativa ↑
- **r < -0.3** e **p < 0.05**: Correlazione negativa significativa ↓
- **|r| < 0.3** o **p > 0.05**: Nessuna correlazione significativa

### `merged_scores_metrics.csv`
**Che cos'è:** Dataset unificato con tutti i punteggi e tutte le metriche.
**Quando usarlo:** Per analisi statistiche personalizzate.

---

## 📁 5. Models (`models/`)

### `feature_selection_report.txt`
**Che cos'è:** Lista delle feature usate e escluse nel training.

### `model_results.csv`
**Che cos'è:** Performance di tutti i modelli testati.

| Colonna | Significato | Valore Buono |
|---------|-------------|--------------|
| `r2` | R² (varianza spiegata) | >0.5 = buono |
| `mae` | Errore medio assoluto | <1.5 punti |
| `rmse` | Root mean squared error | <2.0 punti |

### `importance_*.png`
**Che cos'è:** Grafici dell'importanza delle feature per ogni modello.
**Come leggerlo:** Barre più lunghe = feature più importanti per il modello.

### `predictions_*.png`
**Che cos'è:** Scatterplot predizioni vs valori reali.
**Come leggerlo:** Punti vicini alla diagonale = buone predizioni.

---

## 📁 6. Data Quality (`data_quality/`)

### `outliers_report.csv`
**Che cos'è:** Valori fuori dai range fisicamente plausibili.

| Colonna | Significato |
|---------|-------------|
| `jump_id` | ID del salto problematico |
| `metric` | Metrica con valore anomalo |
| `value` | Valore registrato |
| `expected_range` | Range atteso |

**Azione:** Investigare manualmente i salti elencati - potrebbero avere errori di annotazione.

### `warnings_report.csv`
**Che cos'è:** Valori statisticamente estremi (>3 deviazioni standard).
**Azione:** Valori da verificare ma non necessariamente sbagliati.

### `data_quality_summary.txt`
**Che cos'è:** Riassunto testuale dei problemi trovati.

---

## 📊 Workflow Consigliato

### Per analisi di un nuovo dataset:
1. **Esegui gli script in ordine:**
   ```
   python utils/metrics_calculator.py     # Core metrics
   python metrics/advanced_metrics.py     # Advanced metrics
   python metrics/test_timeseries_metrics.py  # Timeseries
   python metrics/test_correlation_analysis.py  # Correlazioni
   python metrics/data_quality_check.py   # Quality check
   python metrics/ml_models.py            # ML (opzionale)
   ```

2. **Controlla prima la qualità:**
   - Apri `data_quality/data_quality_summary.txt`
   - Se ci sono molti outlier, investiga le annotazioni

3. **Analizza le correlazioni:**
   - Cerca in `correlations.csv` le righe con p < 0.05
   - Le metriche con |r| > 0.4 sono predittori promettenti

4. **Valuta i modelli ML:**
   - R² > 0.3 indica che le metriche catturano qualcosa
   - R² < 0.1 con questo dataset (20 salti) è normale

---

## ⚠️ Limitazioni Note

1. **Sample size piccolo**: Solo ~20 salti con metriche complete
2. **R² negativo**: Comune con LOO-CV su dataset piccoli
3. **Errori tkinter**: Ignorabili, i plot vengono comunque salvati
4. **NaN values**: Alcuni salti mancano di alcune fasi annotate

---

## 📈 Feature Set Finale (v2.0)

Dopo la pulizia, il sistema usa queste **15 feature core**:

### Core Geometric (8):
- `avg_v_style_front`, `avg_v_style_back`
- `takeoff_knee_angle`, `avg_body_ski_angle`
- `avg_symmetry_index_back`
- `avg_telemark_proj_ski`, `avg_telemark_depth_ratio`, `avg_telemark_leg_angle`

### Advanced Dynamic (5):
- `takeoff_timing_offset`, `takeoff_peak_velocity`
- `telemark_scissor_mean`, `telemark_stability`, `landing_absorption_rate`

### Timeseries Stats (6):
- `knee_peak_velocity`, `knee_angle_at_takeoff`
- `flight_std`, `flight_jitter`, `flight_mean_bsa`
- `landing_hip_velocity`, `landing_knee_compression`

### Feature Rimosse (10):
- `flight_stability_std` → redundant con flight_std
- `avg_telemark_offset_x` → redundant con telemark_proj_ski
- `takeoff_acceleration_peak` → derivata di peak_velocity
- `takeoff_smoothness` → non interpretabile
- `knee_mean_velocity` → redundant con peak_velocity
- `knee_extension_range` → redundant con peak_velocity
- `flight_range` → redundant con flight_std
- `flight_trend` → basso potere predittivo
- `landing_hip_drop` → dipende da altezza salto
- `landing_smoothness_score` → feature ingegnerizzata

---

*Guida aggiornata: $(date)*
*Versione: 2.0*
