# 📊 Analisi Dettagliata - Scalabilità e Ottimizzazione del Modello

---

## 1. FUNZIONERANNO CON PIÙ SALTI ANNOTATI?

### ✅ **SÌ, ma con benefici progressivi**

#### Scalabilità Tecnica:
Tutti gli script sono progettati per essere **data-agnostic**:
- Leggono i CSV dinamicamente (non c'è hardcoding di "23 salti")
- I loop iterano su `df.iterrows()` o `for jump_id in jump_ids`
- Leave-One-Out si adatta automaticamente a N samples

**Quindi tecnicamente: sì, funzioneranno con 50, 100 o 200 salti.**

---

#### Scalabilità Statistica (la parte critica):

| N Salti | Cosa Succede | Qualità Modello |
|---------|--------------|-----------------|
| **23 (ora)** | Modelli instabili, R² basso, alta varianza | ⚠️ Risultati indicativi |
| **50** | Modelli iniziano a stabilizzarsi, pattern emergono | ✅ Risultati affidabili |
| **100** | Random Forest robusto, correlazioni chiare | ✅✅ Ottimo |
| **200+** | Possiamo usare modelli più complessi (GB, NN) | ✅✅✅ Eccellente |

#### Cosa Migliorerà con Più Dati:

**A. Feature Importance diventerà stabile:**
Ora con 23 salti, la top feature può cambiare drasticamente se rimuoviamo 2-3 outlier. Con 100 salti, l'ordine sarà molto più robusto.

**B. R² aumenterà:**
- Attualmente: R² ≈ 0.0 - 0.27 (quasi casuale)
- Con 50 salti: R² ≈ 0.3 - 0.5 (moderato)
- Con 100 salti: R² ≈ 0.5 - 0.7 (buono)

**C. Potremo fare validation set separato:**
Ora usiamo LOO per necessità (troppo pochi dati per split train/test). Con 100+ salti, potremo fare:
```
Train: 70 salti
Validation: 15 salti  
Test: 15 salti (MAI visti dal modello)
```

**D. Feature Selection diventerà più affidabile:**
Con 23 salti, una feature può sembrare importante per caso. Con 100, possiamo usare tecniche come:
- Permutation Importance (più robusto)
- Recursive Feature Elimination
- Cross-validated feature selection

---

#### Problemi che PERSISTERANNO anche con più dati:

**1. Variabilità della prospettiva diagonale:**
- `body_rotation_velocity` continuerà a essere rumorosa se la camera cambia tra eventi
- Soluzione: filtrare salti per "stesso evento" o normalizzare per camera angle

**2. Qualità keypoints:**
- Mani e sci rimangono difficili da tracciare accuratamente
- Soluzione: usare confidence scores e droppare frame con bassa confidence

**3. Fattori esterni non catturati:**
- Condizioni vento (non misurate perfettamente)
- Bias giudici (soggettività residua)
- Questi aggiungono "rumore ineliminabile" → R² massimo teorico potrebbe essere 0.7-0.8, non 1.0

---

## 2. MODELLO CON MENO PARAMETRI È MEGLIO?

### ✅ **SÌ, assolutamente - per questi motivi:**

#### A. Il Problema del "Curse of Dimensionality"

Con 23 salti e 15+ features:
```
Ratio samples/features = 23/15 ≈ 1.5

Regola empirica: serve almeno 10 samples per feature per evitare overfitting
Ideale per 23 salti: MAX 2-3 features!
```

**Cosa succede con troppe features:**
- Il modello "memorizza" invece di generalizzare
- Cattura rumore invece di pattern reali
- R² sembra buono in training, pessimo in test
- Coefficienti diventano instabili (cambiano drasticamente tra fold)

---

#### B. Interpretabilità per gli Allenatori

Un modello con **3-4 features chiave** è:
- ✅ Comprensibile: "Lavora su questi 3 aspetti"
- ✅ Actionable: Puoi misurare miglioramenti specifici
- ✅ Trustable: Non è una "black box"

Un modello con **15 features**:
- ❌ Confuso: "Migliora... tutto?"
- ❌ Contraddittorio: Feature correlate si "cannibalizzano"
- ❌ Instabile: I coefficienti non hanno senso fisico

---

#### C. Evidenza dai tuoi Risultati Attuali

Guardando `style_penalty_model/feature_importance.csv`:

| Feature | Importance | Problemi Evidenti |
|---------|-----------|-------------------|
| body_rotation_velocity_max | 21% | Rumorosa, sensibile a camera |
| flight_range | 19% | Ridondante con flight_std |
| flight_jitter | 15% | Correlata con flight_std |
| ski_symmetry_score | 10% | Molti NaN, dati corrotti |

**Il modello sta usando features "di riempimento"** perché non ha abbastanza dati per discriminare quali sono veramente importanti.

---

## 3. DOVREMMO DROPPARE CERTE VARIABILI?

### ✅ **SÌ, con criterio strategico**

#### Strategia di Selezione: **3-Tier System**

**TIER 1 - Features Robuste (SEMPRE includere):**
```
✅ flight_std - Stabilità volo (ben validata, r = -0.556)
✅ landing_hip_velocity - Impatto atterraggio (r = -0.650, causalmente corretta)
✅ flight_jitter - Oscillazioni frame-to-frame (complementare a flight_std)
```
**Perché queste:**
- Correlazioni significative con Style_Score
- Teoricamente fondate (aerodinamica + giudizi estetici)
- Robuste alla prospettiva (misurano variazioni, non assoluti)

---

**TIER 2 - Features Da Valutare (includere SE passano threshold):**
```
⚠️ vstyle_final_angle - SE la vista è consistente tra salti
⚠️ telemark_scissor_mean - SE l'atterraggio è ben visibile
⚠️ knee_peak_velocity - SE abbiamo vista laterale decente
```
**Threshold da applicare:**
1. **Correlation test**: |r| > 0.3 con target (significativo)
2. **Missing data**: < 30% NaN
3. **Multicollinearity**: VIF < 5 (non troppo correlata con altre)

---

**TIER 3 - Features Da ESCLUDERE (troppo rumorose):**
```
❌ body_rotation_velocity_max - Falsi positivi da cambio camera
❌ arm_stability_std - Keypoints mani inaccurati
❌ ski_jitter_range - Valori impossibili (357°) indicano dati corrotti
❌ compactness_mean - Definizione ambigua, dipende da come calcoli "bounding box"
```

---

#### Threshold Concrete da Implementare:

**1. Correlation Threshold:**
```python
# Mantieni solo feature con correlazione significativa
threshold_r = 0.25  # Almeno "weak correlation"
threshold_p = 0.10  # p-value < 0.10 (90% confidence)

valid_features = correlations[
    (abs(correlations['pearson_r']) > threshold_r) &
    (correlations['pearson_p'] < threshold_p)
]['metric'].tolist()
```

**2. Multicollinearity Check:**
```python
# Rimuovi feature ridondanti
# Se flight_range e flight_std correlano a 0.85+, tieni solo la più importante
```

**3. Data Quality Check:**
```python
# Rimuovi feature con troppi dati mancanti
missing_threshold = 0.30  # Max 30% NaN
valid_features = [f for f in features if df[f].isna().sum() / len(df) < missing_threshold]
```

**4. Physical Plausibility Check:**
```python
# Rimuovi outlier impossibili
# Es: ski_jitter_range > 90° è impossibile fisicamente
df.loc[df['ski_jitter_range'] > 90, 'ski_jitter_range'] = np.nan
```

---

## 4. MODELLO OTTIMALE CON DATI ATTUALI

### Raccomandazione: **3-Feature Model**

**Formula Proposta:**
```
Style_Penalty = α × flight_std + β × landing_hip_velocity + γ × flight_jitter + δ
```

**Vantaggi:**
1. **Copertura completa del salto:**
   - `flight_std`: Fase di volo (stabilità globale)
   - `flight_jitter`: Fase di volo (micro-correzioni)
   - `landing_hip_velocity`: Fase di atterraggio

2. **Basso rischio overfitting:**
   - Ratio 23/3 ≈ 7.7 (vicino alla regola dei 10)

3. **Interpretabile:**
   - Ogni coefficiente ha significato chiaro
   - Allenatore sa dove intervenire

4. **Robusto:**
   - Tutte e 3 hanno correlazioni validate
   - Meno sensibili alla prospettiva

---

### Con 50+ Salti, Espandere a **5-Feature Model:**

```
Aggiungere:
+ vstyle_final_angle (se vista consistente)
+ telemark_scissor_mean (per valutare atterraggio)
```

---

## 5. COSA FARE QUANDO ANNOTERETE PIÙ SALTI

### Roadmap Incrementale:

**A. Con 50 Salti:**
1. ✅ Ri-eseguire tutti gli script (funzioneranno automaticamente)
2. ✅ Feature Selection con threshold (correlation > 0.30, p < 0.05)
3. ✅ Usare 5-Feature Model
4. ✅ Passare da LOO a 5-Fold Cross-Validation
5. ✅ Aspettarsi R² ≈ 0.35-0.50

**B. Con 100 Salti:**
1. ✅ Gradient Boosting diventa affidabile
2. ✅ Split Train/Validation/Test (70/15/15)
3. ✅ Permutation Importance invece di coefficienti lineari
4. ✅ Possibile espandere a 7-8 features
5. ✅ Aspettarsi R² ≈ 0.50-0.65

**C. Con 200+ Salti:**
1. ✅ Modelli ensemble (stacking)
2. ✅ Neural Network semplice (3 layer)
3. ✅ Analisi per sottogruppi (uomini vs donne, HS vs K-point)
4. ✅ Aspettarsi R² ≈ 0.65-0.75

---

## 6. RISPOSTA ALLE TUE PREOCCUPAZIONI SPECIFICHE

### "Body Rotation non ci convince"
**Hai ragione al 100%.** Motivi:
- Sensibile a movimenti camera
- Keypoints spalle/anche sono i meno stabili in diagonale
- Valori come 198°/sec sono irrealistici (nessun atleta ruota così tanto in volo)

**Azione:** ❌ **Escludere** finché non hai:
- Camera fissa (non pan/zoom)
- Confidence scores sui keypoints > 0.8
- Vista più frontale (per vedere effettivamente la rotazione)

---

### "Flight Jitter non ci convince"
**Qui è più sfumato.** Pro e contro:

**PRO:**
- Correlazione moderata con Style_Score (r = -0.468)
- Misura micro-oscillazioni (complementare a flight_std)
- Meno sensibile a prospettiva (misura delta frame-to-frame)

**CONTRO:**
- Parzialmente correlata con flight_std (r ≈ 0.6-0.7)
- Dipende dalla framerate (30 fps vs 60 fps darebbe valori diversi)

**Azione:** ⚠️ **Mantenere SE flight_std da sola non basta**
- In modello a 3 feature: flight_std + landing + (flight_jitter O vstyle_angle)
- Test: prova modello con/senza, vedi quale ha R² migliore in validation

---

### "Se droppassimo variabili sarebbe un problema?"
**NO, anzi migliorerebbe il modello.**

**Evidenza matematica:**
Con 23 salti, un modello con 3 features BATTE un modello con 15 features perché:
- Meno overfitting
- Coefficienti più stabili
- Errore di generalizzazione più basso

**Esperimento che puoi fare:**
```
Modello A (15 features): R² train = 0.80, R² test = 0.10 (overfitting!)
Modello B (3 features): R² train = 0.35, R² test = 0.32 (generalizza!)
```

---

## 🎯 RACCOMANDAZIONE FINALE

### Azione Immediata (con 23 salti):

**1. Crea versione "Lite" dello Style Penalty Model:**
- Solo 3 features: flight_std, landing_hip_velocity, flight_jitter
- Salva come `style_penalty_model_lite.py`
- Confronta R² con versione completa

**2. Implementa Data Quality Checks:**
- Flag per salti con `ski_jitter_range > 90°` (dati corrotti)
- Escludi salti con > 30% keypoints mancanti
- Aggiungi colonna "data_quality_score" nel CSV

**3. Documenta Assumptions:**
- Crea file `METRICS_RELIABILITY.md` che lista:
  - Features robuste (Tier 1)
  - Features da validare (Tier 2)
  - Features da escludere (Tier 3)

---

### Piano con Più Dati:

| N Salti | N Features Raccomandate | Modello | R² Atteso |
|---------|-------------------------|---------|-----------|
| 23 | 3 | Ridge/RF | 0.25-0.35 |
| 50 | 5 | RF + Validation | 0.40-0.55 |
| 100 | 7-8 | GB + Ensemble | 0.55-0.70 |
| 200+ | 10-12 | Stacking/NN | 0.65-0.80 |

---

**Vuoi che implementi la versione "Lite" del modello con solo le 3 feature più robuste?**




INDAGARE ASSOLUTAMENTE SU SE E COME DROPPARE LE VARIABILI SENZA FARE DANNI AL CODICE