import os
import glob
from pathlib import Path # Aggiunto per una migliore gestione dei percorsi

def filtra_boxes_txt(base_dir, file_extension="*.jpg"):
    """
    Rimuove le righe dal file boxes.txt che corrispondono ai fotogrammi
    mancanti nella directory principale rispetto al totale.

    Args:
        base_dir (str): Il percorso della directory base (es. '.../JP0006').
        file_extension (str): L'estensione dei file immagine (es. "*.jpg").
    """
    
    # --- Definizione dei Percorsi ---
    dir_principale = os.path.join(base_dir)
    dir_removed = os.path.join(base_dir, 'removed')
    dir_occluded = os.path.join(base_dir, 'occluded')
    
    # Il file boxes.txt originale si trova nella directory principale
    input_file_path = os.path.join(dir_principale, 'boxes.txt')
    output_file_path = os.path.join(dir_principale, 'boxes_filtered.txt')
    
    if not os.path.exists(input_file_path):
        print(f"ERRORE: File boxes.txt non trovato al percorso: {input_file_path}")
        return

    # --- Funzioni Helper (rimane invariata) ---
    def get_indices_from_dir(directory):
        indices = set()
        pattern = os.path.join(directory, file_extension)
        # Sostituiamo il glob precedente con un glob più robusto per i soli .jpg
        file_paths = glob.glob(pattern)
        for file_path in file_paths:
            nome_file_completo = os.path.basename(file_path)
            nome_senza_estensione, _ = os.path.splitext(nome_file_completo)
            indices.add(nome_senza_estensione)
        return indices

    # 1. Calcolo degli Indici e delle Posizioni di Eliminazione
    
    indici_principale = get_indices_from_dir(dir_principale)
    indici_removed = get_indices_from_dir(dir_removed)
    indici_occluded = get_indices_from_dir(dir_occluded)
    
    # 1a. Calcolo tutti gli indici univoci
    tutti_gli_indici_set = indici_principale.union(indici_removed).union(indici_occluded)
    tutti_gli_indici_ordinati = sorted(list(tutti_gli_indici_set), key=int)
    
    # 1b. Calcolo gli indici da rimuovere (quelli che non sono in 'principale')
    indici_da_rimuovere = tutti_gli_indici_set.difference(indici_principale)
    
    # 1c. Mappa l'indice mancante alla POSIZIONE 1-based (numero di riga)
    posizioni_da_saltare = set()
    for posizione_0based, indice in enumerate(tutti_gli_indici_ordinati):
        if indice in indici_da_rimuovere:
            posizioni_da_saltare.add(posizione_0based + 1)

    print(f"Indici da rimuovere: {sorted(list(indici_da_rimuovere), key=int)}")
    print(f"Totale righe da saltare: {len(posizioni_da_saltare)}")
    print(f"Posizioni (numero riga) da saltare nel boxes.txt: {sorted(list(posizioni_da_saltare))}")
    print("-" * 70)
    
    # 2. Lettura del file e Scrittura del file filtrato
    
    righe_mantenute_count = 0
    righe_saltate_count = 0
    
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line_number, line in enumerate(infile, 1):
                # line_number è il numero di riga (1, 2, 3, ...)
                
                if line_number not in posizioni_da_saltare:
                    outfile.write(line)
                    righe_mantenute_count += 1
                else:
                    righe_saltate_count += 1
                    
        print(f"✅ Filtro completato con successo!")
        print(f"File originale: {input_file_path}")
        print(f"Nuovo file creato: {output_file_path}")
        print(f"   - Righe mantenute: {righe_mantenute_count}")
        print(f"   - Righe saltate (rimosse): {righe_saltate_count}")
        print(f"   - Totale righe originali (stimato): {righe_mantenute_count + righe_saltate_count}")

    except Exception as e:
        print(f"ERRORE durante l'apertura/scrittura dei file: {e}")


# --- Parte Principale: Esecuzione AGGIORNATA ---

# Percorso base fisso fino alla directory 'frames'
ROOT_PATH = Path(r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\dataset\frames")
FILE_EXT = "*.jpg"

while True:
    try:
        # 1. Chiedi il numero del salto
        jump_number = int(input("Inserisci il numero del Salto (es. 6 per JP0006): "))
        if jump_number > 0:
            break
        else:
            print("Inserisci un numero positivo.")
    except ValueError:
        print("Input non valido. Inserisci un numero.")

# 2. Costruisci il JUMP ID formattato (es. 6 -> JP0006)
JUMP_ID = f"JP{jump_number:04d}"

# 3. Costruisci il percorso completo della directory di base
BASE_PATH = ROOT_PATH / JUMP_ID

if not BASE_PATH.exists():
    print(f"ERRORE: La directory base per il salto {JUMP_ID} non esiste: {BASE_PATH}")
    sys.exit()

# 4. Chiama la funzione di filtraggio
print("=" * 70)
print(f"INIZIO FILTRAGGIO FILE BOXES.TXT per {JUMP_ID}")
print("=" * 70)
filtra_boxes_txt(BASE_PATH, FILE_EXT)