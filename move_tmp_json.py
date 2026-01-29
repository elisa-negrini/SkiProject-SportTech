import os
import shutil

# --- CONFIGURAZIONE PERCORSI ---
# Percorso BASE dove si trovano le cartelle JPxxxx
# Nota: Uso r"..." per indicare una raw string e non avere problemi con i backslash di Windows
SOURCE_ROOT = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\dataset\annotations"

# Percorso di DESTINAZIONE dove copiare i file
DEST_DIR = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\temp_jsons"

def copy_json_files():
    # 1. Chiedo l'intervallo di salti
    try:
        start_jump = int(input("Inserisci il numero del primo salto (es. 6): "))
        end_jump = int(input("Inserisci il numero dell'ultimo salto (es. 10): "))
    except ValueError:
        print("Errore: Inserisci solo numeri interi.")
        return

    # 2. Creo la cartella di destinazione se non esiste
    if not os.path.exists(DEST_DIR):
        try:
            os.makedirs(DEST_DIR)
            print(f"Creata cartella di destinazione: {DEST_DIR}")
        except OSError as e:
            print(f"Errore nella creazione della cartella: {e}")
            return

    print(f"\n--- Inizio copia dal salto {start_jump} al {end_jump} ---\n")

    # 3. Ciclo attraverso i numeri dei salti
    count_copied = 0
    
    for i in range(start_jump, end_jump + 1):
        # Costruisco il nome della cartella (es. JP0006)
        # :04d significa "numero intero a 4 cifre con zeri iniziali"
        folder_name = f"JP{i:04d}"
        
        # Costruisco il nome del file (es. annotations_normalized_jump6.coco.json)
        # Qui il numero solitamente non ha zeri iniziali nel nome del file, basandomi sul tuo esempio
        file_name = f"annotations_normalized_jump{i}.coco.json"

        # Costruisco il percorso completo sorgente
        # Struttura: ...\dataset\annotations\JP0006\train\annotations_normalized_jump6.coco.json
        source_path = os.path.join(SOURCE_ROOT, folder_name, "train", file_name)
        
        # Costruisco il percorso destinazione
        dest_path = os.path.join(DEST_DIR, file_name)

        # 4. Controllo ed esecuzione copia
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path) # copy2 preserva i metadati del file
                print(f"✅ Copiato: {file_name}")
                count_copied += 1
            except Exception as e:
                print(f"❌ Errore durante la copia di {file_name}: {e}")
        else:
            print(f"⚠️  File non trovato: {source_path}")

    print(f"\n--- Operazione completata. Copiati {count_copied} file su {end_jump - start_jump + 1} richiesti. ---")
    print(f"I file sono qui: {DEST_DIR}")

if __name__ == "__main__":
    copy_json_files()