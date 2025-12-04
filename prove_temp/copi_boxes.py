import os
import shutil
from pathlib import Path

# --- Percorsi Base ---
# Percorso base della sorgente (fino a SkiTB\)
source_base = Path(r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB")

# Percorso base della destinazione (fino a dataset\frames\)
destination_base = Path(r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\dataset\frames")

# --- Configurazione Nomi File/Cartelle ---
# La cartella da cui copiare (es. JP0005)
JUMP_FOLDER_PREFIX = "JP" 
# Il nome della sottocartella che contiene boxes.txt (nel caso JP0005, è MC)
SUBFOLDER_NAME = "MC"
# Il nome del file da copiare
FILE_NAME = "boxes.txt"

# --- Iterazione e Copia ---
print("Inizio la copia dei file boxes.txt da JP0001 a JP0100...")

# Itera da 1 a 100 (incluso)
for i in range(9, 101):
    # Formatta il nome della cartella del salto (es. JP0001, JP0010, JP0100)
    jump_id = f"{JUMP_FOLDER_PREFIX}{i:04d}" # :04d garantisce 4 cifre con zeri iniziali

    # 1. Costruisci il percorso del file sorgente
    # Esempio: C:\...\SkiTB\JP\JP0005\MC\boxes.txt
    source_path = source_base / "JP" / jump_id / SUBFOLDER_NAME / FILE_NAME
    
    # 2. Costruisci il percorso del file di destinazione
    # Esempio: C:\...\frames\JP0005\boxes.txt
    # Nota: Stiamo creando una cartella con jump_id (JP0005) all'interno di frames
    destination_folder = destination_base / jump_id
    destination_path = destination_folder / FILE_NAME
    
    # --- Esegui la Copia ---
    try:
        # Verifica se il file sorgente esiste
        if not source_path.exists():
            print(f"⚠️ ATTENZIONE: File sorgente non trovato per {jump_id} a {source_path}")
            continue # Passa al salto successivo

        # Crea la cartella di destinazione se non esiste
        destination_folder.mkdir(parents=True, exist_ok=True)
        
        # Esegue la copia del file
        shutil.copy2(source_path, destination_path)
        
        print(f"✅ Copiato: {jump_id} -> {destination_path.name}")
        
    except Exception as e:
        print(f"❌ ERRORE durante la copia di {jump_id}: {e}")

print("---")
print("Copia completata per l'intervallo specificato.")