# 📊 Report: Metriche Biomeccaniche per l'Analisi del Salto con gli Sci

---

## 🎯 Introduzione

Questo report descrive le **metriche biomeccaniche** sviluppate per analizzare la performance nel salto con gli sci. Le metriche sono estratte da **keypoints pose estimation** (23 punti anatomici tracciati frame-by-frame) e misurano aspetti tecnici del salto che influenzano i punteggi degli atleti.

---

## 📍 Struttura del Salto: Le 4 Fasi

Ogni salto è suddiviso in fasi temporali definite da frame chiave:

| Fase | Frame di Riferimento | Durata Tipica | Cosa Succede |
|------|---------------------|---------------|--------------|
| **1. Take-off** | `take_off_frame` ± 10 | ~0.3 sec | Stacco dal trampolino |
| **2. Flight** | `bsa_start` → `landing` | 1.5-3 sec | Volo in aria |
| **3. Landing** | `landing` → `landing+20` | ~0.6 sec | Atterraggio e assorbimento |
| **4. Telemark** | `telemark_start` → `telemark_end` | ~0.2 sec | Posizione finale piedi a forbice |

---

## 🔵 METRICHE CORE (metrics_calculator.py)

### 1. **V-Style Angle** (Angolo a V degli Sci)

**Cosa misura:**  
L'apertura angolare tra i due sci durante il volo (vista frontale o posteriore).

**Come si calcola:**
```
Vettore sci sinistro: punta_sci_L → tallone_sci_L
Vettore sci destro: punta_sci_R → tallone_sci_R
V-Style = angolo tra i due vettori
```

**Range normale:** 25-40°  
**Fase:** Volo (calcolato su `v_style_front_start` → `v_style_front_end`)

**Colonne CSV:**
- `avg_v_style_front`: Media angolo V-style (vista frontale)
- `avg_v_style_back`: Media angolo V-style (vista posteriore)

**Correlazione con Score:**
- ✅ **vs Physical_Score: r = +0.883** (correlazione fortissima!)
  - V-style più aperto → maggiore portanza → più distanza

---

### 2. **Body-Ski Angle (BSA)** (Angolo Corpo-Sci)

**Cosa misura:**  
L'angolo tra l'asse del corpo e l'asse degli sci (quanto l'atleta è "disteso" sugli sci).

**Come si calcola:**
```
Vettore corpo: regressione lineare su keypoints spalle-anche-ginocchia
Vettore sci: best-fit tra keypoints degli sci
BSA = angolo tra i due vettori
```

**Range normale:** 155-170°  
**Fase:** Volo (calcolato su `bsa_start` → `bsa_end`)

**Colonne CSV:**
- `avg_body_ski_angle`: Media BSA durante il volo

**Interpretazione:**
- BSA alto (~165°) = posizione compatta, aerodinamica
- BSA basso (~140°) = posizione "aperta", meno efficiente

---

### 3. **Takeoff Knee Angle** (Angolo Ginocchio allo Stacco)

**Cosa misura:**  
L'estensione del ginocchio al momento dello stacco dal trampolino.

**Come si calcola:**
```
Al frame = take_off_frame:
  Vettore coscia: anca → ginocchio
  Vettore gamba: ginocchio → caviglia
  Knee_angle = angolo tra i due vettori
```

**Range normale:** 160-175°  
**Fase:** Take-off (frame singolo: `take_off_frame`)

**Colonne CSV:**
- `takeoff_knee_angle`: Angolo al momento dello stacco

**Interpretazione:**
- ~170° = estensione quasi completa (esplosività massima)
- ~140° = stacco "seduto" (subottimale)

---

### 4. **Symmetry Index** (Indice di Simmetria Sci)

**Cosa misura:**  
Quanto simmetricamente sono posizionati gli sci (differenza angoli sci sinistro vs destro).

**Come si calcola:**
```
Per ogni frame in volo:
  angle_L = angolo sci sinistro
  angle_R = angolo sci destro
  symmetry_index = |angle_L - angle_R|
```

**Range normale:** 0-10°  
**Fase:** Volo

**Colonne CSV:**
- `avg_symmetry_index_back`: Media differenza angoli (vista posteriore)

**Interpretazione:**
- Valore basso (~2°) = sci paralleli (buono)
- Valore alto (~15°) = sci asimmetrici (perdita controllo)

---

### 5. **Telemark Offset** (Distanza Piedi Atterraggio)

**Cosa misura:**  
La distanza orizzontale tra piede anteriore e posteriore nella posizione Telemark.

**Come si calcola:**
```
telemark_offset_x = |x_ankle_front - x_ankle_back|
```

**Range normale:** 0.02-0.06 (normalizzato)  
**Fase:** Landing (calcolato su `telemark_start` → `telemark_end`)

**Colonne CSV:**
- `avg_telemark_offset_x`: Distanza media piedi
- `avg_telemark_proj_ski`: Proiezione della forbice lungo sci
- `avg_telemark_leg_angle`: Angolo tra le gambe

**Interpretazione:**
- Valore ottimale: ~0.04 (forbice visibile ma controllata)
- Troppo basso: atterraggio "a due piedi" (penalità)
- Troppo alto: instabile

---

### 6. **Flight Stability** (Stabilità Volo)

**Cosa misura:**  
Quanto stabile è la posizione durante il volo (quanto "tremano" angoli corpo/sci).

**Come si calcola:**
```
flight_stability_std = std(body_ski_angle[bsa_start : landing])
```

**Range normale:** 0.3-2.0°  
**Fase:** Volo

**Colonne CSV:**
- `flight_stability_std`: Deviazione standard BSA

**Correlazione con Score:**
- ✅ **vs Style_Score: r = -0.556** (correlazione negativa forte!)
  - Più stabile (std bassa) → più punti stile

---

## 🟢 METRICHE TIME-SERIES (test_timeseries_metrics.py)

Queste metriche misurano **dinamiche temporali** (velocità, accelerazioni) invece di posizioni statiche.

---

### 7. **Knee Peak Velocity** (Velocità Esplosiva Ginocchio)

**Cosa misura:**  
La velocità angolare massima del ginocchio durante lo stacco (quanto velocemente si estende).

**Come si calcola:**
```
Per ogni frame in finestra [takeoff-5 : takeoff+5]:
  knee_velocity = (angle[t+1] - angle[t-1]) / (2 * dt)
  dove dt = 1/30 sec (framerate)
  
knee_peak_velocity = max(knee_velocity)
```

**Unità:** gradi/secondo  
**Range normale:** 300-600 deg/sec  
**Fase:** Take-off (finestra ±5 frame)

**Colonne CSV:**
- `knee_peak_velocity`: Velocità massima esplosione

**Interpretazione:**
- Alto (~500 deg/sec) = stacco esplosivo
- Basso (~200 deg/sec) = stacco "lento"

**⚠️ Limitazioni:** Sensibile alla prospettiva camera (vista laterale vs frontale).

---

### 8. **Flight Stability (Extended)** (Stabilità Completa Volo)

**Cosa misura:**  
Tre aspetti della stabilità durante il volo:

**A. flight_std** (deviazione standard BSA)
```
flight_std = std(BSA durante tutto il volo)
```
- Misura: Oscillazioni lente/grandi
- Range normale: 0.5-2.5°

**B. flight_range** (range BSA)
```
flight_range = max(BSA) - min(BSA)
```
- Misura: Escursione totale angolo
- Range normale: 2-10°

**C. flight_jitter** (tremolio ad alta frequenza)
```
frame_to_frame_changes = |BSA[t] - BSA[t-1]|
flight_jitter = std(frame_to_frame_changes)
```
- Misura: Micro-oscillazioni rapide
- Range normale: 0.1-0.8°

**Correlazioni con Style_Score:**
- flight_std: r = -0.556 ✅
- flight_range: r = -0.611 ✅
- flight_jitter: r = -0.468 ✅

**Interpretazione:** Tutte negative = più stabilità → più punti.

---

### 9. **Landing Hip Velocity** (Velocità Impatto Atterraggio)

**Cosa misura:**  
Quanto velocemente il bacino scende dopo l'atterraggio (durezza impatto).

**Come si calcola:**
```
Per frame in [landing : landing+15]:
  hip_velocity = gradient(hip_y_position) * fps
  
landing_hip_velocity = mean(|hip_velocity|)
```

**Unità:** unità_normalizzate/secondo  
**Range normale:** 0.3-1.2  
**Fase:** Landing (15 frame post-atterraggio)

**Colonne CSV:**
- `landing_hip_velocity`: Velocità media discesa bacino

**Correlazione con Score:**
- ✅ **vs Style_Score: r = -0.650** (LA CORRELAZIONE PIÙ FORTE!)
  - Atterraggio morbido (valore basso) → più punti
  - Atterraggio duro (valore alto) → penalità

**Interpretazione:**
- < 0.5: Assorbimento graduale (eccellente)
- 0.5-0.8: Normale
- \> 1.0: Impatto secco (penalità)

---

### 10. **Flight Trend** (Tendenza Apertura/Chiusura)

**Cosa misura:**  
Se l'atleta tende ad aprire o chiudere l'angolo corpo-sci durante il volo.

**Come si calcola:**
```
Regressione lineare di BSA nel tempo:
  BSA(t) = slope × t + intercept
  
flight_trend = slope
```

**Unità:** gradi/frame  
**Range normale:** -0.5 a +0.5  
**Fase:** Volo

**Colonne CSV:**
- `flight_trend`: Coefficiente angolare

**Interpretazione:**
- Positivo: "Opener" (apre durante il volo)
- Negativo: "Closer" (chiude durante il volo)
- ~0: "Stable" (mantiene posizione)

---

## 🟡 METRICHE AVANZATE (advanced_metrics.py)

### 11. **Ski Jitter** (Oscillazione Sci)

**Cosa misura:**  
Quanto "ballano" gli sci (differenza angolare tra sci L e R che varia nel tempo).

**Come si calcola:**
```
Per ogni frame:
  angle_diff = |angle_ski_L - angle_ski_R|
  
ski_jitter_std = std(angle_diff durante volo)
ski_jitter_range = max(angle_diff) - min(angle_diff)
```

**Range normale (std):** 5-30°  
**Fase:** Volo

**Colonne CSV:**
- `ski_jitter_std`: Deviazione standard differenza
- `ski_jitter_range`: Escursione totale
- `ski_angle_diff_mean`: Differenza media

**⚠️ Limitazioni:** 
- Valori > 90° indicano dati corrotti (sci occlusi/interpolati male)
- Vista diagonale nasconde uno sci

---

### 12. **Ski Symmetry Score** (Sincronizzazione Sci)

**Cosa misura:**  
Quanto si muovono sincronizzati i due sci (correlazione temporale).

**Come si calcola:**
```
Serie temporale A = angoli sci sinistro nel tempo
Serie temporale B = angoli sci destro nel tempo

ski_symmetry_score = pearson_correlation(A, B)
```

**Range:** -1 a +1  
**Fase:** Volo

**Colonne CSV:**
- `ski_symmetry_score`: Correlazione movimenti

**Interpretazione:**
- +1.0: Sci perfettamente sincronizzati (si muovono insieme)
- 0.0: Sci indipendenti
- -1.0: Sci si muovono in opposizione (anomalo)

---

### 13. **Body Rotation Velocity** (Velocità Rotazione Torso)

**Cosa misura:**  
Quanto velocemente ruota il torso durante il volo (movimento indesiderato).

**Come si calcola:**
```
torso_angle = angolo tra vettore_spalle e vettore_anche
rotation_velocity = gradient(torso_angle) * fps
```

**Unità:** gradi/secondo  
**Range atteso:** < 50 deg/sec  
**Fase:** Volo

**Colonne CSV:**
- `body_rotation_velocity_max`: Picco velocità rotazione
- `body_rotation_velocity_mean`: Media
- `body_rotation_total`: Rotazione totale accumulata

**⚠️ ATTENZIONE:** 
- Valori > 100 deg/sec sono probabilmente artefatti (camera che si muove, keypoints instabili)
- Rotazione totale > 90° è fisicamente impossibile

---

### 14. **Arm Stability** (Stabilità Braccia)

**Cosa misura:**  
Quanto si muovono le braccia rispetto al corpo durante il volo.

**Come si calcola:**
```
arm_distance = |posizione_mano - posizione_spalla|
arm_stability_std = std(arm_distance durante volo)
```

**Range normale:** 0.01-0.05 (normalizzato)  
**Fase:** Volo

**Colonne CSV:**
- `arm_stability_std`: Variabilità posizione
- `arm_distance_mean`: Distanza media
- `arm_distance_range`: Escursione

**⚠️ Limitazioni:** Keypoints mani spesso inaccurati (occlusi, piccoli).

---

### 15. **Takeoff Timing** (Precisione Timing Stacco)

**Cosa misura:**  
Quanto l'atleta anticipa o ritarda l'estensione massima rispetto al bordo trampolino.

**Come si calcola:**
```
frame_peak_velocity = frame dove knee_velocity è massima
frame_takeoff_annotato = take_off_frame da annotazioni

takeoff_timing_offset = frame_peak_velocity - frame_takeoff_annotato
```

**Unità:** frame (@ 30 fps = 0.033 sec/frame)  
**Range normale:** -3 a +3 frame  
**Fase:** Take-off

**Colonne CSV:**
- `takeoff_timing_offset`: Errore timing

**Interpretazione:**
- 0: Timing perfetto (esplosione esattamente al bordo)
- Negativo: Anticipo (esplode prima del bordo)
- Positivo: Ritardo (esplode dopo il bordo)

---

### 16. **Takeoff Dynamics** (Dinamica Stacco Completa)

**Cosa misura:**  
Tre aspetti della qualità dello stacco:

**A. takeoff_peak_velocity** (velocità massima)
- Unità: deg/sec
- Range: 300-600

**B. takeoff_acceleration_peak** (accelerazione massima)
- Come si calcola: `gradient(knee_velocity)`
- Unità: deg/sec²
- Range: 10000-30000

**C. takeoff_smoothness** (fluidità)
- Come si calcola: `ratio velocità_media / velocità_picco`
- Range: 0.3-0.7 (più alto = più fluido)

**Colonne CSV:**
- `takeoff_peak_velocity`
- `takeoff_acceleration_peak`
- `takeoff_smoothness`

---

### 17. **Telemark Quality** (Qualità Atterraggio Dinamico)

**Cosa misura:**  
Tre aspetti dinamici del Telemark:

**A. telemark_scissor_mean** (forbice media)
```
scissor = |y_ankle_front - y_ankle_back| / hip_height
```
- Normalizzato per altezza atleta
- Range: 0.02-0.06

**B. telemark_stability** (stabilità forbice)
```
telemark_stability = std(scissor durante landing)
```
- Più basso = forbice mantenuta costante

**C. landing_absorption_rate** (velocità assorbimento)
```
absorption_rate = 1 / frames_to_max_compression
```
- Misura quanto rapidamente l'atleta "crolla"

**Colonne CSV:**
- `telemark_scissor_mean`
- `telemark_stability`
- `landing_absorption_rate`

---

### 18. **V-Style Dynamics** (Dinamica Apertura V)

**Cosa misura:**  
Come l'atleta apre il V-style:

**A. vstyle_opening_rate** (velocità apertura)
```
opening_rate = (vstyle_final - vstyle_initial) / duration
```
- Unità: deg/sec

**B. vstyle_final_angle** (angolo finale)
- L'angolo V raggiunto e mantenuto

**C. vstyle_overshoot** (superamento)
- Se l'atleta "apre troppo" e poi chiude

**Colonne CSV:**
- `vstyle_opening_rate`
- `vstyle_final_angle`
- `vstyle_overshoot`

---

### 19. **Body Compactness** (Compattezza Corporea)

**Cosa misura:**  
Quanto l'atleta è "compatto" (bounding box area).

**Come si calcola:**
```
bbox_area = (max_x - min_x) × (max_y - min_y) di tutti i keypoints
compactness = 1 / bbox_area (normalizzato)
```

**Range:** 70-85%  
**Fase:** Volo

**Colonne CSV:**
- `compactness_mean`: Media durante volo
- `compactness_trend`: Se diventa più/meno compatto

**Interpretazione:**
- Alto: Posizione "raccolta", aerodinamica
- Basso: Posizione "aperta"

---

## 📊 TABELLA RIASSUNTIVA CORRELAZIONI

### Con Style_Score (Punteggio Stile):

| Metrica | Correlazione (r) | Significato | Interpretazione |
|---------|------------------|-------------|-----------------|
| **landing_hip_velocity** | -0.650 | Forte negativa | Atterraggio morbido → +punti |
| **flight_range** | -0.611 | Forte negativa | Meno oscillazione → +punti |
| **flight_std** | -0.556 | Forte negativa | Più stabilità → +punti |
| **flight_jitter** | -0.468 | Moderata negativa | Meno tremolio → +punti |

### Con Physical_Score (Punteggio Fisico/Distanza):

| Metrica | Correlazione (r) | Significato | Interpretazione |
|---------|------------------|-------------|-----------------|
| **avg_v_style_front** | +0.883 | Fortissima positiva | V più aperto → +distanza |

---

## 🎯 Metriche per Fase: Riepilogo

| Fase | Metriche Chiave | Cosa Misuri |
|------|----------------|-------------|
| **Take-off** | `takeoff_knee_angle`, `knee_peak_velocity`, `takeoff_timing` | Esplosività, timing |
| **Flight** | `flight_std`, `flight_jitter`, `vstyle_angle`, `body_ski_angle` | Stabilità, aerodinamica |
| **Landing** | `landing_hip_velocity`, `telemark_offset`, `telemark_stability` | Controllo impatto |

---

## ⭐ Metriche Raccomandate (Robuste)

Basate su validità tecnica e correlazioni significative:

**TIER 1 (Sempre usare):**
1. ✅ `flight_std` - Stabilità volo
2. ✅ `landing_hip_velocity` - Impatto atterraggio
3. ✅ `avg_v_style_front` - Apertura V (distanza)

**TIER 2 (Usare con cautela):**
4. ⚠️ `flight_jitter` - Tremolio
5. ⚠️ `vstyle_final_angle` - V-style finale
6. ⚠️ `knee_peak_velocity` - Esplosività stacco

**TIER 3 (Evitare o validare manualmente):**
7. ❌ `body_rotation_velocity` - Sensibile a camera
8. ❌ `ski_jitter` - Dati spesso corrotti
9. ❌ `arm_stability` - Keypoints inaccurati

---

## 📐 Note Tecniche

**Normalizzazione:**
- Tutte le coordinate keypoints sono normalizzate rispetto alle dimensioni video (0-1 range)
- Angoli in gradi (0-360°)
- Velocità in gradi/secondo o unità_normalizzate/secondo

**Framerate:**
- 30 fps (dt = 0.033 sec tra frame)

**Missing Data:**
- Alcune metriche hanno NaN per:
  - Vista non adatta (es: V-style in vista laterale)
  - Keypoints non rilevati
  - Fase non annotata (es: take_off_frame mancante)




# 📊 Report Completo e Dettagliato: Analisi Metriche Ski Jumping

Rispondo a tutte le tue domande in modo strutturato e approfondito.

---

## PARTE 2: test_timeseries_metrics.py

### A. Knee Velocity (Esplosività Stacco)

#### Cos'è:
```
Velocità angolare del ginocchio = quanto velocemente si estende il ginocchio al momento dello stacco
Formula: ω = Δθ / Δt (gradi/secondo)
```

#### Validità della Metrica:

| Pro | Contro |
|-----|--------|
| La **velocità** è più robusta dell'angolo assoluto | La prospettiva diagonale **comprime** comunque i movimenti |
| Un movimento esplosivo rimane "veloce" da ogni angolazione | Camera non standardizzata tra salti |
| Misura la **dinamica**, non la posizione statica | Richiede keypoints accurati su hip-knee-ankle |

#### Mia Valutazione Onesta:
**6/10** - È una metrica **teoricamente valida** ma **praticamente limitata** nel nostro contesto:

1. **Problema prospettiva**: Se la camera è più laterale, vedo meglio l'estensione. Se è più frontale, il movimento è "schiacciato".

2. **Problema varianza**: Tra JP0005 e JP0032 la camera potrebbe essere posizionata diversamente → velocità non comparabili direttamente.

3. **Quando funziona**: Se confrontiamo salti dello **stesso evento** (stessa camera), la metrica è valida per ranking relativo.

#### Come Ottimizzare:
```python
# Normalizzare per la "visibilità" del movimento
# Se hip-knee-ankle sono quasi allineati orizzontalmente → vista laterale buona
# Se sono quasi verticali → vista frontale, metrica meno affidabile

def compute_view_quality(hip, knee, ankle):
    """Stima quanto bene vediamo il piano sagittale"""
    horizontal_spread = abs(hip[0] - ankle[0])  # Spread orizzontale
    vertical_spread = abs(hip[1] - ankle[1])    # Spread verticale
    return horizontal_spread / (vertical_spread + 0.001)  # Alto = vista laterale
```

---

### B. Flight Stability (Frozenness)

#### Cos'è:
```
Deviazione standard dell'angolo corpo-sci durante il volo
STD bassa = atleta "congelato" in posizione = buono
STD alta = atleta che fa correzioni/trema = cattivo
```

#### Le Tue Domande:

**"Non è poco farla solo su BSA?"**

Hai ragione. Attualmente calcoliamo la stabilità SOLO sull'angolo corpo-sci. Potremmo estenderla a:
- Stabilità degli sci (angolo tra sci destro e sinistro)
- Stabilità delle braccia
- Stabilità della testa

**"Non è limitante farla su 10 frame?"**

Nel codice attuale NON è limitata a 10 frame. Guarda:

```python
# Nel codice:
def compute_flight_frozenness(self, bsa_series, bsa_start, bsa_end):
    # Usa TUTTA la finestra bsa_start → landing (non solo 10 frame)
    extended_end = int(landing_frame)  # Estende fino all'atterraggio
```

La finestra tipica è **50-100 frame** (1.5-3 secondi di volo).

**"La prospettiva cambia tra salti"**

Esatto, questo è il problema principale. **MA**: la STD misura la **variazione interna** al salto, non il valore assoluto.

| Scenario | Angolo Vero | Angolo Visto (diagonale) | STD Vera | STD Vista |
|----------|-------------|--------------------------|----------|-----------|
| Atleta stabile | 165° ± 2° | 155° ± 2° | 2° | ~2° ✅ |
| Atleta instabile | 165° ± 8° | 155° ± 7° | 8° | ~7° ✅ |

La **proporzione** della variazione si mantiene! È per questo che la STD è più robusta del valore medio.

#### Validità:
**7.5/10** - Metrica abbastanza robusta perché:
- Misura variazione relativa, non assoluta
- Il "tremolio" si vede da ogni angolazione
- Correlazione significativa con Style_Score (r = -0.556, p = 0.009)

---

### C. Landing Smoothness

#### Cos'è:
```
landing_hip_velocity = Velocità media di discesa del bacino dopo l'atterraggio
                     = Quanto velocemente il bacino scende nei 15 frame post-landing

Velocità ALTA = impatto secco, atterraggio duro
Velocità BASSA = assorbimento graduale, atterraggio morbido
```

#### Calcolo Dettagliato:
```python
def compute_landing_smoothness(self, hip_series, landing_frame, window=15):
    # 1. Prendi altezza bacino dal frame landing al frame landing+15
    landing_hip = hip_series[landing_frame : landing_frame + 15]
    
    # 2. Calcola derivata (velocità di discesa)
    hip_velocity = np.gradient(landing_hip.values) * fps  # unità/secondo
    
    # 3. Media del valore assoluto della velocità
    landing_hip_velocity = np.abs(hip_velocity).mean()
```

#### Perché È Valida:
1. **Misura il movimento verticale**: La componente Y (verticale) è meno distorta dalla prospettiva diagonale rispetto alla X.

2. **Indicatore di controllo**: Un atleta che "crolla" dopo l'atterraggio ha velocità alta. Un atleta che assorbe gradualmente ha velocità bassa.

3. **Correlazione forte**: r = -0.650 con Style_Score (la più alta tra tutte le metriche!)

#### Limitazioni:
- Dipende dalla qualità del keypoint `center_pelvis`
- Se l'atleta atterra e subito si rialza vs. rimane piegato, la metrica cambia
- Non cattura la **simmetria** dell'atterraggio (Telemark)

#### Ottimizzazione Possibile:
```python
# Aggiungere la "curva di assorbimento"
# Un buon Telemark ha: discesa rapida iniziale → plateau → risalita
def analyze_absorption_curve(hip_series, landing_frame):
    curve = hip_series[landing_frame : landing_frame + 30]
    
    # Trova il punto di minimo (massima compressione)
    min_idx = curve.idxmin()
    time_to_min = min_idx - landing_frame  # Frame per raggiungere massima compressione
    
    # Buon atterraggio: time_to_min ~ 10-15 frame (graduale)
    # Cattivo atterraggio: time_to_min ~ 3-5 frame (impatto secco)
    return time_to_min
```

---

## PARTE 4: test_correlation_analysis.py

### Spiegazione Correlazioni

#### Pearson vs Spearman:

| Tipo | Cosa Misura | Quando Usarlo |
|------|-------------|---------------|
| **Pearson** | Relazione **lineare** | Dati normalmente distribuiti |
| **Spearman** | Relazione **monotonica** (anche non lineare) | Dati con outlier, non normali |

```
Esempio:
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]  → Pearson = 1.0, Spearman = 1.0 (perfetta linearità)

X = [1, 2, 3, 4, 5]
Y = [1, 4, 9, 16, 25]  → Pearson = 0.98, Spearman = 1.0 (monotonica ma non lineare)
```

Nel nostro caso usiamo **entrambi** perché:
- Pearson ci dice se c'è relazione lineare diretta
- Spearman è più robusto agli outlier (utile con pochi dati)

#### Perché Correlazione NEGATIVA = Più Stabilità = Più Punti?

Leggi così:
```
flight_std vs Style_Score: r = -0.556

Significa:
- Quando flight_std AUMENTA (più instabilità) → Style_Score DIMINUISCE
- Quando flight_std DIMINUISCE (più stabilità) → Style_Score AUMENTA

Quindi: correlazione NEGATIVA tra instabilità e punteggio
      = correlazione POSITIVA tra stabilità e punteggio


#### Interpretazione Risultati Chiave:

```csv
metric,target,pearson_r,significant
flight_std,Style_Score,-0.556,True      # ✅ Più stabile = più punti stile
flight_range,Style_Score,-0.611,True    # ✅ Meno oscillazione = più punti
landing_hip_velocity,Style_Score,-0.650,True  # ✅ Atterraggio morbido = più punti
avg_v_style_front,Physical_Score,0.883,True   # ✅ V-style più aperto = più distanza
```

**Cosa significano questi numeri?**

| r | Interpretazione |
|---|-----------------|
| 0.0 - 0.3 | Correlazione debole |
| 0.3 - 0.5 | Correlazione moderata |
| 0.5 - 0.7 | Correlazione forte |
| 0.7 - 1.0 | Correlazione molto forte |

- `landing_hip_velocity vs Style_Score: r = -0.650` → **Forte correlazione negativa**
  - Il 42% della varianza di Style_Score è spiegato da questa metrica (r² = 0.42)

- `avg_v_style_front vs Physical_Score: r = 0.883` → **Correlazione molto forte**
  - Il 78% della varianza di Physical_Score è spiegato dal V-style!

---

## PARTE 3: `advanced_metrics.py`

### 1. Ski Jitter

#### Cos'è:
```python
ski_jitter_std = STD dell'angolo TRA i due sci durante il volo
ski_jitter_range = MAX - MIN di questo angolo
ski_angle_diff_mean = Media della differenza angolare tra sci L e R
```

#### Quando Lo Calcoliamo:
Nel codice attuale, lo calcoliamo su **tutta la fase di volo** (bsa_start → landing).

#### Validità:
**5/10** - Problematica perché:

1. **Prospettiva**: In vista diagonale, uno sci potrebbe essere "nascosto" dietro l'altro
2. **Valori strani**: Guarda `ski_jitter_range = 357°` per JP0005 → Questo è impossibile fisicamente!

Il problema è che quando uno sci non è visibile, i keypoints vengono interpolati male, creando angoli assurdi.

#### Come Interpretare i Risultati:

```csv
jump_id,ski_jitter_std,ski_jitter_range,ski_symmetry_score
JP0011,6.51,25.6,0.987   # ✅ Buono: poca variazione, alta simmetria
JP0005,172.1,357.4,0.277  # ❌ Dati corrotti: range impossibile
```

**Regola pratica**: Se `ski_jitter_range > 90°`, i dati sono probabilmente corrotti.

---

### 2-3. Body Rotation & Arm Stability

#### Body Rotation:
```python
body_rotation_velocity_max = Velocità massima di rotazione del torso (deg/sec)
body_rotation_velocity_mean = Velocità media
body_rotation_total = Rotazione totale accumulata
```

**Perché valori così alti (198°/sec per JP0011)?**

Due possibili cause:
1. **Cambio camera**: Se la camera si muove/zooma, il torso "sembra" ruotare
2. **Keypoints instabili**: Spalle che "saltano" tra frame → rotazione fittizia

**Come verificare**:
```python
# Se body_rotation_total > 90° in un salto, probabilmente è un artefatto
# Un atleta NON ruota di 90° durante il volo!
```

#### Arm Stability:
```python
arm_stability_std = STD della distanza mano-spalla durante il volo
arm_distance_mean = Distanza media mano-spalla
arm_distance_range = Variazione della lunghezza del "braccio"
```

**Cosa significa**: Se `arm_stability_std` è alto, le braccia si muovono molto. Un buon saltatore tiene le braccia ferme lungo il corpo.

**Problema**: I keypoints delle mani sono spesso i meno accurati (piccoli, spesso occlusi).

---

### 4. Takeoff Timing

#### Cos'è:
```python
takeoff_timing_offset = frame_picco_velocità - frame_takeoff_annotato
```

**Output**:
- `offset = 0`: L'atleta raggiunge la massima estensione esattamente al momento dello stacco ✅
- `offset = -5`: L'atleta si estende 5 frame PRIMA dello stacco (anticipato)
- `offset = +5`: L'atleta si estende 5 frame DOPO lo stacco (ritardato)

**Interpretazione**:
- Valori vicini a 0 = timing perfetto
- Valori negativi alti = atleta "esplode" troppo presto
- Valori positivi alti = atleta "esplode" troppo tardi

---

### 5. Landing Symmetry vs Telemark (metrics_calculator)

#### In `metrics_calculator.py`:
```python
telemark_offset_x = |x_ankle_front - x_ankle_back|  # Distanza orizzontale piedi
telemark_leg_angle = angolo tra le gambe
```
Misura la **posizione statica** del Telemark.

#### In `advanced_metrics.py`:
```python
telemark_scissor_mean = Media della "forbice" (distanza piedi normalizzata per altezza)
telemark_stability = STD della scissor durante la fase di atterraggio
landing_absorption_rate = Velocità di compressione post-landing
```
Misura la **dinamica** del Telemark.

**Differenza chiave**: `metrics_calculator` guarda una foto istantanea, `advanced_metrics` guarda il video.

---

### 6-7. Ski Symmetry & Flight Trend

#### Ski Symmetry Score:
```python
ski_symmetry_score = correlazione tra movimenti sci destro e sinistro
```
- `1.0` = Sci perfettamente sincronizzati (si muovono insieme)
- `0.0` = Sci indipendenti
- `-1.0` = Sci si muovono in direzioni opposte

**Risultati**:
```csv
JP0011: 0.987  # Sci quasi perfettamente sincronizzati
JP0019: -0.442 # Sci si muovono in modo opposto (strano, possibile errore dati)
```

#### Flight Trend:
```python
flight_trend = slope della regressione lineare di BSA durante il volo
```
- `> 0` = Atleta "Opener" (apre l'angolo durante il volo)
- `< 0` = Atleta "Closer" (chiude l'angolo durante il volo)
- `≈ 0` = Atleta stabile

---

### Spiegazione Colonne advanced_metrics_summary.csv

| Colonna | Significato | Unità | Buono se... |
|---------|-------------|-------|-------------|
| `ski_jitter_std` | Instabilità angolo tra sci | gradi | Basso (<20°) |
| `ski_jitter_range` | Range oscillazione sci | gradi | Basso (<50°) |
| `ski_angle_diff_mean` | Differenza media angoli sci | gradi | Vicino a 0° |
| `ski_symmetry_score` | Correlazione movimenti sci | -1 to 1 | Alto (>0.8) |
| `body_rotation_velocity_max` | Picco rotazione torso | deg/sec | Basso (<50) |
| `body_rotation_total` | Rotazione totale accumulata | gradi | Basso (<30°) |
| `arm_stability_std` | Variazione posizione braccia | norm. | Basso |
| `takeoff_timing_offset` | Errore timing stacco | frames | Vicino a 0 |
| `takeoff_peak_velocity` | Velocità picco estensione | deg/sec | Alto (>400) |
| `takeoff_acceleration_peak` | Accelerazione picco | deg/sec² | Alto |
| `takeoff_smoothness` | Fluidità estensione | ratio | Alto (>0.5) |
| `telemark_scissor_mean` | Forbice media atterraggio | norm. | 0.02-0.05 |
| `telemark_stability` | Stabilità Telemark | norm. | Basso |
| `landing_absorption_rate` | Velocità assorbimento | 1/frame | Basso (graduale) |
| `vstyle_opening_rate` | Velocità apertura V-style | deg/sec | Moderato |
| `vstyle_final_angle` | Angolo V-style finale | gradi | 25-40° |
| `vstyle_overshoot` | Superamento angolo target | gradi | Basso |
| `compactness_mean` | Quanto è "compatto" l'atleta | % | 70-85% |
| `compactness_trend` | Trend compattezza nel volo | slope | Stabile (~0) |

---

## PARTE 4: timeseries_analysis.py

### Cosa Analizziamo:

#### 1. DTW Clustering
```
Dynamic Time Warping = Algoritmo che allinea due serie temporali
                       anche se hanno durate diverse
```

**Come funziona**:
```
Salto A: [160°, 162°, 165°, 163°]  (4 frame)
Salto B: [158°, 160°, 164°, 166°, 165°, 162°]  (6 frame)

DTW trova il miglior "allineamento" tra le due curve,
stirando/comprimendo il tempo per minimizzare la differenza.
```

**Come aiuta il progetto**:
1. **Identifica stili di salto**: Cluster 1 = "openers", Cluster 2 = "closers", Cluster 3 = "stabili"
2. **Trova salti simili**: "JP0005 è simile a JP0012" → stesso allenatore? stessa tecnica?
3. **Anomaly detection**: Salti che non appartengono a nessun cluster = possibili errori o tecniche insolite

**Output `cluster_assignments.csv`**:
```csv
jump_id,cluster
JP0005,1
JP0006,2
JP0007,1
```

#### 2. Top vs Flop Comparison
```
Confronta le curve medie dei migliori 10 atleti vs peggiori 10
```

**Output**: Grafici che mostrano:
- Linea verde = curva media Top 10
- Linea rossa = curva media Flop 10
- Zone blu = differenze statisticamente significative

**Come interpretare**:
Se vedi che nella fase 20-40% del volo i Top hanno un angolo più stabile, quello è un "momento critico" su cui lavorare in allenamento.

#### 3. Frequency Analysis (FFT)
```
Fast Fourier Transform = Scompone una curva nelle sue frequenze
```

**Output `frequency_analysis.csv`**:
```csv
jump_id,dominant_frequency,high_freq_power_ratio
JP0005,1.2,0.15
JP0011,0.8,0.45
```

**Interpretazione**:
- `dominant_frequency = 1.2 Hz` = L'atleta oscilla circa 1.2 volte al secondo
- `high_freq_power_ratio = 0.45` = 45% dell'energia è in oscillazioni rapide (tremolio)

**Buono**: `high_freq_power_ratio` basso (< 0.2) = posizione stabile
**Cattivo**: `high_freq_power_ratio` alto (> 0.3) = troppo tremolio

---

## PARTE 5: `ml_models.py`

### Perché Questi Modelli?

| Modello | Perché Lo Usiamo | Pro | Contro |
|---------|------------------|-----|--------|
| **Random Forest** | Robusto, gestisce non-linearità | Feature Importance, no scaling richiesto | Può overfittare |
| **Gradient Boosting** | Più accurato di RF | Migliori predizioni | Più sensibile a iperparametri |
| **Ridge Regression** | Baseline lineare | Interpretabile, veloce | Assume linearità |

### LOO con 50 Salti

**Leave-One-Out** funziona così:
```
Per ogni salto i:
    1. Allena su tutti gli altri 49 salti
    2. Testa sul salto i
    3. Registra errore

Errore finale = media di tutti gli errori
```

**È appropriato per 50 salti?** SÌ, per questi motivi:
- Con k-fold (k=5), avresti solo 10 salti per test fold → alta varianza
- LOO massimizza i dati di training (49 vs 1)
- Con dataset piccoli, LOO dà stime più stabili

**Contro**: Computazionalmente costoso (50 iterazioni), ma con RF è comunque veloce.

### Perché Neural Network è Commentato?

```python
# Neural Network non implementato perché:
# 1. Richiede MOLTI più dati (>500 samples per risultati affidabili)
# 2. Con 25-50 salti, overfitterebbe immediatamente
# 3. Non dà Feature Importance interpretabile
```

### Come Interpretare i Risultati

**Da `ml_summary_report.txt`**:
```
Random Forest - Style_Score:
  R² = 0.24
  MAE = 2.1
  RMSE = 2.8
```

**Interpretazione**:
- **R² = 0.24**: Il modello spiega il 24% della varianza di Style_Score
  - Non è altissimo, ma con 50 samples e features rumorose è accettabile
  - Significa che il 76% della varianza dipende da altri fattori (giudici, condizioni, ecc.)

- **MAE = 2.1**: In media, sbagliamo di 2.1 punti stile
  - Se Style_Score medio è ~50, l'errore è ~4%

- **RMSE = 2.8**: Penalizza di più gli errori grandi

**Feature Importance** (da `importance_Style_Score_Random_Forest.csv`):
```csv
feature,importance
flight_std,0.25
landing_hip_velocity,0.20
flight_jitter,0.15
...
```

Significa: `flight_std` contribuisce per il 25% alla predizione di Style_Score.

---

## PARTE 6: `penalty_predictor.py`

### Abbiamo Scelto le Feature Giuste?

Le feature usate sono:
1. `flight_jitter` - Instabilità durante il volo
2. `landing_hip_velocity` - Impatto atterraggio
3. `symmetry_index_back` - Simmetria (quando disponibile)

**Sono le migliori?** Dipende dalla disponibilità dei dati. Il modello seleziona automaticamente le feature con più dati validi.

### Come Interpretare Output

**Da `PENALTY_FORMULA.txt`**:
```
PENALTY PREDICTOR MODEL
=======================

Formula:
Penalty (m) = 13.21 × LandingSmooth + 11.86 × FlightJitter + -9.01 × SkiSymmetry + 111.17

Component Importance:
  LandingSmooth: 39%
  FlightJitter: 35%
  SkiSymmetry: 26%
```

**Come leggerlo**:
- Se `LandingSmooth` aumenta di 1 unità → Penalità aumenta di 13.21 metri
- Se `FlightJitter` aumenta di 1 unità → Penalità aumenta di 11.86 metri
- Se `SkiSymmetry` aumenta di 1 unità → Penalità **diminuisce** di 9.01 metri (segno negativo!)

**Attenzione**: L'R² è negativo (-0.03), il che significa che il modello non è predittivo. Questo succede perché:
1. Pochi dati (21 samples)
2. Features non sufficientemente correlate con la distanza
3. Troppo rumore

**Cosa fare**: Questo modello è più **illustrativo** che predittivo. Mostra quali fattori *potrebbero* influenzare la penalità, ma non può fare predizioni affidabili.

---

## PARTE 7: `phase_analysis.py`

### Cosa Fa

Divide ogni salto in **5 fasi** e calcola metriche separate per ognuna:

```
1. TAKE-OFF    (frame takeoff ± 10)
2. EARLY FLIGHT (0-30% del volo)
3. MID FLIGHT   (30-70% del volo)
4. LATE FLIGHT  (70-100% del volo)
5. LANDING      (frame landing → landing+20)
```

### Come Interpretare `phase_metrics.csv`

```csv
jump_id,phase,bsa_mean,bsa_std,bsa_slope,knee_velocity_max
JP0005,takeoff,165.7,2.3,+12.5,450.2
JP0005,early_flight,160.1,0.8,-0.5,30.1
JP0005,mid_flight,159.5,0.5,-0.2,15.0
JP0005,late_flight,158.8,1.2,+0.8,25.0
JP0005,landing,155.0,3.5,-2.0,120.0
```

**Interpretazione per JP0005**:
- **Take-off**: BSA = 165.7°, variabilità 2.3°, sta aumentando (+12.5 deg/frame)
- **Early flight**: Si stabilizza (std = 0.8°), leggera chiusura (-0.5)
- **Mid flight**: Molto stabile (std = 0.5°) ← **Questa è la fase critica!**
- **Late flight**: Inizia a prepararsi per l'atterraggio (std aumenta)
- **Landing**: Alta variabilità (std = 3.5°), veloce compressione

### Come Interpretare `athlete_profiles.csv`

```csv
AthleteName,bsa_mean_flight,bsa_std_flight,knee_velocity_takeoff,landing_smoothness
Sara Takanashi,162.5,1.2,480.5,0.85
Ryoyu Kobayashi,165.0,0.8,520.3,0.92
```

**Uso pratico per allenatori**:
- "Kobayashi ha velocità di stacco più alta (+40 deg/sec vs Takanashi)"
- "Kobayashi è più stabile in volo (std 0.8 vs 1.2)"
- "Kobayashi ha atterraggio più morbido (0.92 vs 0.85)"

---

## 🎯 Riepilogo Validità Metriche

| Metrica | Validità | Motivo |
|---------|----------|--------|
| `flight_std` | ⭐⭐⭐⭐ | Robusta: misura variazione, non assoluto |
| `landing_hip_velocity` | ⭐⭐⭐⭐ | Robusta: componente Y meno distorta |
| `flight_jitter` | ⭐⭐⭐ | Buona: frame-to-frame è locale |
| `knee_velocity` | ⭐⭐ | Problematica: dipende dalla vista |
| `ski_jitter` | ⭐⭐ | Problematica: sci spesso occlusi |
| `body_rotation` | ⭐ | Molto problematica: sensibile a camera |
| `arm_stability` | ⭐ | Molto problematica: keypoints mani inaccurati |

---

## 📝 Raccomandazioni Finali

1. **Usa principalmente**: `flight_std`, `landing_hip_velocity`, `flight_jitter`
2. **Usa con cautela**: `knee_velocity`, `ski_jitter`
3. **Evita o verifica manualmente**: `body_rotation`, `arm_stability`
4. **Per i modelli ML**: Con 50 salti, usa Random Forest + LOO, non aspettarti R² > 0.3
5. **Penalty Predictor**: Usalo come strumento illustrativo, non predittivo

Hai altre domande specifiche?
