import os
import shutil
import glob
from tqdm import tqdm

# --- Variabili di Configurazione ---

# Directory base dove si trovano le cartelle JPXXXX originali
BASE_SOURCE_DIR = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB\JP"

# Directory base dove verranno create le cartelle JPXXXX di destinazione
BASE_DEST_DIR = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\dataset\frames"

# Prefisso e intervallo per l'iterazione delle cartelle JP
PREFIX = "JP"
START_INDEX = 1
END_INDEX = 100 # Includerà JP0001 fino a JP0100

# Estensioni dei file immagine da copiare (aggiungi/rimuovi se necessario)
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg']

# --- Funzione di Copia ---

def copy_images_in_batches(start_index, end_index, prefix, base_source, base_dest, extensions):
    """
    Copia i file immagine da una serie di cartelle di origine a quelle di destinazione.
    """
    total_files_copied = 0
    
    # tqdm fornisce una bella barra di progresso per l'iterazione
    for i in tqdm(range(start_index, end_index + 1), desc="Elaborazione cartelle JP"):
        # Formatta il nome della cartella (es. JP0001, JP0002, ..., JP0100)
        folder_name = f"{prefix}{i:04d}"
        
        # Percorsi completi
        source_frames_dir = os.path.join(base_source, folder_name, "frames")
        dest_folder_dir = os.path.join(base_dest, folder_name)
        
        print(f"\nTentativo di elaborare: {folder_name}")
        
        # 1. Verifica se la cartella di origine esiste
        if not os.path.exists(source_frames_dir):
            print(f"  ⚠️ Cartella di origine non trovata: {source_frames_dir}. Saltato.")
            continue # Passa alla cartella successiva
            
        # 2. Crea la cartella di destinazione (se non esiste)
        try:
            os.makedirs(dest_folder_dir, exist_ok=True)
        except OSError as e:
            print(f"  ❌ Errore nella creazione della cartella di destinazione {dest_folder_dir}: {e}")
            continue

        # 3. Trova e copia i file
        files_copied_in_batch = 0
        
        for ext in extensions:
            # Crea un percorso di ricerca, es. C:\...\JP0001\frames\*.png
            search_path = os.path.join(source_frames_dir, ext)
            
            # glob.glob trova tutti i file corrispondenti al pattern
            image_files = glob.glob(search_path)
            
            for file_path in image_files:
                file_name = os.path.basename(file_path)
                dest_path = os.path.join(dest_folder_dir, file_name)
                
                try:
                    # Copia il file
                    shutil.copy2(file_path, dest_path)
                    files_copied_in_batch += 1
                except Exception as e:
                    print(f"  ❌ Errore nella copia del file {file_name}: {e}")
        
        if files_copied_in_batch > 0:
            print(f"  ✅ Copia completata per {folder_name}. Copiati {files_copied_in_batch} file.")
        else:
            print(f"  ℹ️ Nessun file immagine trovato in {source_frames_dir} con le estensioni specificate. Saltato.")
            
        total_files_copied += files_copied_in_batch

    print("\n--- Riassunto ---")
    print(f"Processo completato.")
    print(f"Totale file copiati: **{total_files_copied}**")


# --- Esegui la Funzione ---

if __name__ == "__main__":
    copy_images_in_batches(
        start_index=START_INDEX,
        end_index=END_INDEX,
        prefix=PREFIX,
        base_source=BASE_SOURCE_DIR,
        base_dest=BASE_DEST_DIR,
        extensions=IMAGE_EXTENSIONS
    )