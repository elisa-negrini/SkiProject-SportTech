# 📊 Riepilogo Analisi Correlazioni

**Data:** 5 Febbraio 2026  
**Dataset:** 100 salti, 22 con metriche complete  
**Feature analizzate:** 19 metriche  

---

## ✅ PROBLEMA RISOLTO

**Prima:** Solo 7 metriche analizzate (mancavano advanced metrics)  
**Dopo:** 19 metriche complete (timeseries + core + advanced)

### Diagnostici Merge:
- **Timeseries**: 20/100 salti con dati
- **Core**: 20/100 salti con dati  
- **Advanced**: 15/100 salti con dati
- **Overlap totale**: 22 salti con almeno una metrica

---

## 🎯 CORRELAZIONI SIGNIFICATIVE (p < 0.05)

### 🔝 TOP PREDITTORI (|r| > 0.6)

| Metrica | Target | r | p | n | Interpretazione |
|---------|--------|---|---|---|-----------------|
| **avg_v_style_front** | AthleteScore | **+0.78** | 0.003 | 12 | V-style ampio = punteggio alto |
| **avg_v_style_front** | AthleteDistance | **+0.77** | 0.003 | 12 | V-style ampio = distanza maggiore |
| **avg_v_style_front** | Physical_Score | **+0.75** | 0.005 | 12 | V-style ampio = fisica migliore |
| **landing_knee_compression** | Style_Score | **-0.71** | 0.0003 | 21 | Flessione maggiore = voto tecnico basso |
| **telemark_scissor_mean** | Style_Score | **+0.64** | 0.045 | 10 | Telemark più profondo = voto migliore |

### 📈 PREDITTORI MODERATI (0.4 < |r| < 0.6)

| Metrica | Target | r | p | n | Interpretazione |
|---------|--------|---|---|---|-----------------|
| **telemark_stability** | AthleteScore | **+0.47** | 0.030 | 21 | Posizione stabile = punteggio alto |
| **telemark_stability** | Physical_Score | **+0.47** | 0.031 | 21 | Stabilità = prestazione fisica |
| **flight_mean_bsa** | AthleteScore | **-0.47** | 0.039 | 20 | Inclinazione minore = meglio |
| **landing_hip_velocity** | Style_Score | **-0.44** | 0.044 | 21 | Discesa lenta = tecnica migliore |

---

## 📉 METRICHE NON SIGNIFICATIVE (p > 0.05)

Queste metriche **NON** hanno correlazione statisticamente significativa:

### Timeseries:
- `knee_peak_velocity` (r~0.1)
- `knee_angle_at_takeoff` (r~-0.16)
- `flight_std` (r~-0.33, p=0.14)
- `flight_jitter` (r~-0.26)

### Core Geometric:
- `avg_body_ski_angle` (r~-0.32, p=0.17)
- `takeoff_knee_angle` (r~-0.65, ma solo n=8!)
- `avg_symmetry_index_back` (r~0.56, ma solo n=5!)
- `avg_telemark_proj_ski` (r~-0.10)
- `avg_telemark_depth_ratio` (r~-0.01)
- `avg_telemark_leg_angle` (r~-0.39, p=0.15)

### Advanced:
- `takeoff_timing_offset` (r~0.50, p=0.06 - quasi significativo!)
- `takeoff_peak_velocity` (r~0.50, p=0.06 - quasi significativo!)
- `landing_absorption_rate` (r~-0.07)

---

## 🔬 ANALISI APPROFONDITA

### 1. V-Style (avg_v_style_front)
**Correlazione più forte del dataset (r=0.78)**

- **Perché è importante:** Apertura sci ottimale = più portanza aerodinamica
- **Campione piccolo:** Solo 12 salti con vista frontale
- **Azione:** Priorità #1 per annotazione future

### 2. Landing Knee Compression
**Correlazione negativa forte (r=-0.71)**

- **Interpretazione:** Flessione eccessiva = tecnica difettosa
- **Range problematico:** >90° considerato outlier
- **Azione:** Usare come indicatore qualità atterraggio

### 3. Telemark Metrics
**Scissor depth (r=+0.64) vs Stability (r=+0.47)**

- **Trade-off:** Profondità telemark ≠ stabilità
- **Sample size:** Solo 10-21 salti con telemark
- **Azione:** Servono più dati laterali per confermare

### 4. Takeoff Dynamics
**Quasi significativi (p~0.06)**

- `takeoff_timing_offset`: r=0.50, p=0.06
- `takeoff_peak_velocity`: r=0.50, p=0.06
- **Problema:** Sample piccolo (n=15), servono più dati

---

## ⚠️ LIMITAZIONI

### 1. Sample Size Piccolo
- Solo 22 salti con metriche complete
- Alcune metriche <15 salti (telemark, v-style)
- p-value influenzati da n piccolo

### 2. Missing Data Pattern
- **Frontal view:** 12/100 salti (avg_v_style_front)
- **Lateral view:** 20/100 salti (telemark metrics)
- **Takeoff phase:** 15/100 salti (advanced metrics)

### 3. Confounding Variables
- Condizioni meteo (non registrate)
- Livello atleta (non normalizzato)
- Hill size diversity (HS=90-185m)

---

## 📊 RACCOMANDAZIONI

### Per Futuri Esperimenti:
1. **Priorità annotazioni:**
   - Vista frontale (V-style front)
   - Vista laterale (telemark)
   - Fase stacco (takeoff dynamics)

2. **Metriche da enfatizzare:**
   - ✅ `avg_v_style_front` (r=0.78)
   - ✅ `landing_knee_compression` (r=-0.71)
   - ✅ `telemark_scissor_mean` (r=0.64)
   - ⚠️ `takeoff_timing_offset` (r=0.50, quasi significativo)

3. **Metriche da rimuovere/ridurre:**
   - ❌ `flight_jitter` (r=-0.26)
   - ❌ `avg_telemark_depth_ratio` (r=-0.01)
   - ❌ `landing_absorption_rate` (r=-0.07)

### Per Modelli ML:
- Focus su **top 5 feature** (r>0.4)
- Considerare interazioni V-style × landing
- Feature engineering su ratios (e.g., v_style/bsa)

---

## 📈 CONFRONTO PRE/POST FIX

| Aspetto | Prima | Dopo |
|---------|-------|------|
| Metriche analizzate | 7 | **19** |
| Correlazioni calcolate | 28 | **76** |
| Correlazioni significative | 4 | **9** |
| Advanced metrics | ❌ Mancanti | ✅ Incluse |

---

*Generato automaticamente da test_correlation_analysis.py*
