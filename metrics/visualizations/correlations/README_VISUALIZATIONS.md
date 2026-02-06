# 📊 Guida Rapida alle Visualizzazioni delle Correlazioni

**Cartella:** `metrics/visualizations/correlations/`  
**Generato da:** `python metrics/visualize_correlations.py`

---

## 🎯 File Generati

### 1️⃣ **1_heatmap_full.png** - Matrice Completa
**Cosa mostra:** Tutte le 19 metriche × 4 target in una heatmap colorata

- **Rosso** = Correlazione negativa forte (-1)
- **Blu** = Correlazione positiva forte (+1)
- **Bianco** = Nessuna correlazione (0)

**Quando usarlo:** Per avere un overview completo di tutte le relazioni

---

### 2️⃣ **2_heatmap_significant.png** - Solo Correlazioni Significative
**Cosa mostra:** Solo le 9 correlazioni con p < 0.05

- Rimuove il "rumore" delle correlazioni non significative
- Valori annotati con 3 decimali per precisione

**Quando usarlo:** Per focus sulle relazioni statisticamente robuste

---

### 3️⃣ **3_top_15_correlations.png** - Top 15 Classifica
**Cosa mostra:** Bar chart orizzontale delle correlazioni più forti

- **Verde scuro** = Positiva significativa (p<0.05)
- **Rosso scuro** = Negativa significativa (p<0.05)
- **Verde chiaro** = Positiva non significativa
- **Rosso chiaro** = Negativa non significativa

**Quando usarlo:** Per identificare rapidamente le feature più predittive

**Top 3 assolute:**
1. `avg_v_style_front` → AthleteScore: **r=0.79** ⭐⭐⭐
2. `avg_v_style_front` → AthleteDistance: **r=0.77** ⭐⭐⭐
3. `avg_v_style_front` → Physical_Score: **r=0.75** ⭐⭐⭐

---

### 4️⃣ **4_scatter_top_6.png** - Scatter Plots con Regressione
**Cosa mostra:** 6 grafici scatter delle correlazioni più forti

- Ogni punto = 1 salto
- Linea tratteggiata = regressione lineare
- Titolo include: r, p-value, sample size

**Quando usarlo:** Per vedere la distribuzione effettiva dei dati

**Cosa cercare:**
- ✅ Punti vicini alla linea = buona correlazione
- ❌ Punti sparsi = correlazione debole/spuria
- ⚠️ Outliers = salti da investigare

---

### 5️⃣ **5_correlations_by_target.png** - Raggruppato per Target
**Cosa mostra:** 4 subplot (uno per target) con tutte le correlazioni significative

**Utile per:**
- Vedere quali metriche predicono meglio ciascun target
- Comparare Pattern tra Physical vs Style score

**Insight chiave:**
- **Style_Score**: dominato da landing metrics (knee compression, hip velocity)
- **Physical_Score**: dominato da V-style e telemark stability
- **AthleteScore/Distance**: V-style front è predittore #1

---

### 6️⃣ **VISUALIZATION_SUMMARY.txt** - Report Testuale
**Cosa contiene:**
- Statistiche generali (76 correlazioni, 9 significative)
- Top 10 positive
- Top 10 negative
- Lista completa per target

**Quando usarlo:** Per citazioni in report o paper

---

## 🔍 Come Interpretare le Visualizzazioni

### Heatmap (1-2)
```
       Physical  Style  Distance  Score
metric1   0.75    0.20    0.68    0.72   ← Alta correlazione con 3/4 target
metric2  -0.02    0.10   -0.05    0.03   ← Nessuna correlazione (rumore)
metric3  -0.71   -0.44   -0.50   -0.55   ← Correlazione negativa forte
```

### Bar Chart (3)
- **Lunghezza barra** = Forza della correlazione
- **Direzione** = Segno (destra=+, sinistra=-)
- **Colore** = Significatività statistica

### Scatter Plots (4)
```
  🟢 Linea verde (positiva): y aumenta con x
  🔴 Linea rossa (negativa): y diminuisce con x
  
  Punteggio ↑
            |    🔵
            |  🔵  🔵
            | 🔵 🔵  🔵   ← Buona correlazione
            |🔵  🔵
            +----------→ Metrica
```

---

## 📈 Key Findings dalle Visualizzazioni

### 🥇 Feature Champions (|r| > 0.7)
1. **avg_v_style_front** (r=0.78): Predice tutto (distance, style, physical)
2. **landing_knee_compression** (r=-0.71): Predice style score (inverso)

### 🥈 Runner-Ups (0.6 < |r| < 0.7)
3. **telemark_scissor_mean** (r=0.64): Predice style score
4. **avg_symmetry_index_back** (r=0.57): Correlato ma non significativo (n=5)

### 🥉 Emerging Contenders (0.4 < |r| < 0.6)
5. **telemark_stability** (r=0.47): Predice physical score
6. **flight_mean_bsa** (r=-0.47): Correlazione negativa con score
7. **landing_hip_velocity** (r=-0.44): Predice style (inverso)

---

## ⚠️ Limitazioni Visibili nei Grafici

### Sample Size Issues
Guardando gli scatter plots, noterai:
- **avg_v_style_front**: Solo 12 punti (vista frontale rara)
- **telemark_scissor_mean**: Solo 10 punti (vista laterale)
- **takeoff_knee_angle**: Solo 8 punti (fase stacco incompleta)

### Outliers
Nel grafico scatter di `landing_knee_compression`:
- 1-2 salti con valori >100° (outliers fisici)
- Influenzano la regressione

### Non-linearità
Alcuni scatter mostrano relazioni non-lineari:
- `flight_std` potrebbe avere relazione quadratica
- `telemark_stability` sembra avere plateau effect

---

## 🎨 Customizzazione

Per modificare le visualizzazioni, edita `visualize_correlations.py`:

```python
# Cambia numero di correlazioni mostrate
self.plot_top_correlations_bar(top_n=20)  # Default: 15

# Cambia numero di scatter plots
self.plot_scatter_top_correlations(top_n=9)  # Default: 6

# Cambia colormap heatmap
cmap='coolwarm'  # Invece di 'RdBu_r'

# Cambia soglia significatività
df_sig = self.df_corr[self.df_corr['pearson_p'] < 0.01]  # Più strict
```

---

## 📊 Workflow Consigliato

### Per Presentazioni:
1. Usa **2_heatmap_significant.png** come slide principale
2. Aggiungi **3_top_15_correlations.png** per ranking
3. Mostra **4_scatter_top_6.png** per i top 3 findings

### Per Paper/Thesis:
1. Includi **1_heatmap_full.png** in appendice
2. **2_heatmap_significant.png** nel corpo principale
3. **4_scatter_top_6.png** per validazione visiva
4. Cita valori da **VISUALIZATION_SUMMARY.txt**

### Per Debugging:
1. Controlla scatter plots per outliers
2. Verifica sample size nei titoli
3. Confronta pearson vs spearman in `correlations.csv`

---

## 🔄 Aggiornamento Visualizzazioni

Quando aggiungi nuovi salti o metriche:

```bash
# 1. Rigenera correlazioni
python metrics/test_correlation_analysis.py

# 2. Rigenera visualizzazioni
python metrics/visualize_correlations.py
```

Le immagini vengono sovrascritte automaticamente.

---

## 📧 Export per Report

### PDF Multi-pagina:
```python
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('correlations_report.pdf') as pdf:
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    # etc...
```

### PowerPoint:
1. Apri immagini in PowerPoint
2. Dimensioni consigliate: 1920×1080 (16:9)

### LaTeX:
```latex
\begin{figure}
  \includegraphics[width=\textwidth]{metrics/visualizations/correlations/2_heatmap_significant.png}
  \caption{Significant correlations between biomechanical metrics and competition scores.}
\end{figure}
```

---

*Generato: 5 Febbraio 2026*  
*Versione: 1.0*
