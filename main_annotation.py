import subprocess
import sys
import os
import glob
from pathlib import Path # Aggiunto per Path
import json
import re

# --- Configurazione dei percorsi ---

# Script nella root del progetto
EXTRACT_SCRIPT = "extract_annotations.py"
INTERPOLATE_SCRIPT = "interpolation.py"

# Script nella sottocartella (Assumo sia 'visualize_annotations' basato sul contesto)
VISUALIZE_DIR = "visualize_annotations"
VISUALIZE_SCRIPT = os.path.join(VISUALIZE_DIR, "visualize_interpolation.py")
CREATE_VIDEO_SCRIPT = os.path.join(VISUALIZE_DIR, "create_video.py")

# Percorso base fisso per la directory 'frames' (DIPENDENZA DA filter_boxes)
ROOT_FRAMES_PATH = Path(r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\dataset\frames")
FILE_EXT = "*.jpg" # DIPENDENZA DA filter_boxes

# --- Funzioni di supporto per il FILTRAGGIO BOXES ---

def get_indices_from_dir(directory, file_extension="*.jpg"):
    """ Ottiene i numeri di frame da una directory usando glob e estrazione del nome. """
    indices = set()
    try:
        pattern = os.path.join(directory, file_extension)
        file_paths = glob.glob(pattern)
        for file_path in file_paths:
            nome_file_completo = os.path.basename(file_path)
            # Assumiamo che il nome sia solo il numero senza estensione
            nome_senza_estensione, _ = os.path.splitext(nome_file_completo)
            # Usiamo int(nome_senza_estensione) per l'ordinamento, ma lo teniamo come stringa qui
            indices.add(nome_senza_estensione)
        return indices
    except Exception as e:
        # In questo contesto, un errore qui è critico
        print(f"⚠️ Avviso durante il recupero degli indici da {directory}: {e}")
        return indices

def filtra_boxes_txt_local(jump_number):
    """
    Esegue la logica di filtraggio del boxes.txt, adattata per l'uso locale in main_annotation.py.
    Ritorna True se il filtraggio è andato a buon fine, False altrimenti.
    """
    JUMP_ID = f"JP{jump_number:04d}"
    BASE_PATH = ROOT_FRAMES_PATH / JUMP_ID
    
    if not BASE_PATH.exists():
        print(f"--- ❌ Errore: La directory frames per il salto {JUMP_ID} non esiste: {BASE_PATH} ---")
        return False

    # --- Definizione dei Percorsi (Locale) ---
    dir_principale = BASE_PATH
    dir_removed = BASE_PATH / 'removed'
    dir_occluded = BASE_PATH / 'occluded'
    
    input_file_path = dir_principale / 'boxes.txt'
    output_file_path = dir_principale / 'boxes_filtered.txt'
    
    if not input_file_path.exists():
        print(f"--- ❌ Errore: File boxes.txt non trovato al percorso: {input_file_path} ---")
        return False

    # 1. Calcolo degli Indici e delle Posizioni di Eliminazione
    indici_principale = get_indices_from_dir(str(dir_principale), FILE_EXT)
    indici_removed = get_indices_from_dir(str(dir_removed), FILE_EXT)
    indici_occluded = get_indices_from_dir(str(dir_occluded), FILE_EXT)
    
    tutti_gli_indici_set = indici_principale.union(indici_removed).union(indici_occluded)
    tutti_gli_indici_ordinati = sorted(list(tutti_gli_indici_set), key=int)
    indici_da_rimuovere = tutti_gli_indici_set.difference(indici_principale)
    
    posizioni_da_saltare = set()
    for posizione_0based, indice in enumerate(tutti_gli_indici_ordinati):
        if indice in indici_da_rimuovere:
            posizioni_da_saltare.add(posizione_0based + 1)
            
    # Stampa i dettagli chiave
    print(f"   Indici totali trovati: {len(tutti_gli_indici_set)}")
    print(f"   Righe da saltare (corrispondenti a frame non in 'principale'): {len(posizioni_da_saltare)}")

    # 2. Lettura del file e Scrittura del file filtrato
    righe_mantenute_count = 0
    righe_saltate_count = 0
    
    try:
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            for line_number, line in enumerate(infile, 1):
                if line_number not in posizioni_da_saltare:
                    outfile.write(line)
                    righe_mantenute_count += 1
                else:
                    righe_saltate_count += 1
                    
        print(f"   ✅ Filtro completato. Righe mantenute: {righe_mantenute_count}, Rimosse: {righe_saltate_count}.")
        return True

    except Exception as e:
        print(f"--- ❌ Errore durante il filtraggio del boxes.txt: {e} ---")
        return False


# --- Funzioni di supporto per l'Esecuzione (Invariate) ---

def run_script(script_path, jump_number):
    """
    Esegue uno script Python esterno, indipendentemente dalla sua posizione,
    passando il numero del salto come input.
    """
    
    print(f"\n--- Esecuzione: {script_path} per il Salto {jump_number} ---")
    
    if not os.path.exists(script_path):
        print(f"--- ❌ Errore: Lo script non è stato trovato al percorso: {script_path} ---")
        return False
        
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            input=str(jump_number).encode('utf-8'),
            capture_output=False, 
            check=True,          
            text=False           
        )
        print(f"--- ✅ {script_path} completato con successo. ---")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"--- ❌ Errore in {script_path} ---")
        print(f"Il processo ha restituito un codice di errore: {e.returncode}")
        return False
    except Exception as e:
        print(f"--- ❌ Errore sconosciuto durante l'esecuzione di {script_path}: {e} ---")
        return False


def get_jump_number():
    """Chiede all'utente il numero del salto da analizzare."""
    while True:
        try:
            jump_number = int(input("\nQuale SALTO vuoi analizzare (es. 6)? "))
            if jump_number > 0:
                return jump_number
            else:
                print("Inserisci un numero positivo.")
        except ValueError:
            print("Input non valido. Inserisci un numero.")

def get_yes_no_input(prompt):
    """Chiede una risposta Y/N all'utente."""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes', 'n', 'no']:
            return response.startswith('y')
        else:
            print("Risposta non valida. Inserisci 'y' o 'n'.")

# --- Flusso di lavoro principale AGGIORNATO ---

def main_workflow():
    """Il flusso di lavoro principale del programma."""
    
    print("\n" + "="*50)
    print("🎬 WORKFLOW ANALISI SALTI SCI")
    print("="*50)

    while True:
        # 1. Ottieni il numero del Salto
        jump_number = get_jump_number()
        
        # 2. Esegui Extract Annotations (Root)
        print("\nPasso 1/5: Estrazione Annotazioni...")
        if not run_script(EXTRACT_SCRIPT, jump_number):
            if get_yes_no_input("Estrazione fallita. Vuoi riprovare/analizzare un altro salto?"):
                continue
            else:
                break
        
        # 3. Esegui Filter Boxes (Nuovo Passo Locale)
        print("\nPasso 2/5: Filtro Bounding Boxes (boxes.txt -> boxes_filtered.txt)...")
        if not filtra_boxes_txt_local(jump_number):
            print("--- ❌ Filtraggio Boxes fallito. Interruzione del workflow. ---")
            break
        
        # 4. Chiedi per l'Interpolazione (Root)
        do_interpolate = get_yes_no_input("Vuoi INTERPOLARE i frame mancanti?")
        if do_interpolate:
            print("\nPasso 3/5: Interpolazione Keypoints...")
            if not run_script(INTERPOLATE_SCRIPT, jump_number):
                print("⚠️ Attenzione: Interpolazione fallita. Procedo con la visualizzazione/video solo dei frame estratti.")
            else:
                 print("Interpolazione completata. Il file COCO interpolato è ora disponibile.")
        else:
            print("Interpolazione saltata.")

        # 5. Chiedi per la Visualizzazione (Sottocartella)
        do_visualize = get_yes_no_input("Vuoi VISUALIZZARE e creare il Video?")
        
        if do_visualize:
            # 5a. Esegui Visualizzazione (Sottocartella)
            print("\nPasso 4/5: Creazione delle Visualizzazioni...")
            if not run_script(VISUALIZE_SCRIPT, jump_number):
                print("❌ Visualizzazione fallita. Non posso procedere alla creazione del video.")
            else:
                # 5b. Esegui Creazione Video (Sottocartella)
                print("\nPasso 5/5: Creazione del Video...")
                run_script(CREATE_VIDEO_SCRIPT, jump_number)
        else:
            print("Visualizzazione e creazione video saltate.")

        # 6. Chiedi se continuare
        if not get_yes_no_input("\nAnalisi completata. Vuoi analizzare un altro salto?"):
            break

    print("\n" + "="*50)
    print("Programma terminato. Arrivederci!")
    print("="*50)


if __name__ == "__main__":
    main_workflow()