import subprocess
import sys
import os

# --- Configurazione dei percorsi ---

# Script nella root del progetto
EXTRACT_SCRIPT = "extract_annotations.py"
INTERPOLATE_SCRIPT = "interpolation.py"

# Script nella sottocartella (Assumo sia 'visualize_annotations' basato sul contesto)
VISUALIZE_DIR = "visualize_annotations"
VISUALIZE_SCRIPT = os.path.join(VISUALIZE_DIR, "visualize_interpolation.py")
CREATE_VIDEO_SCRIPT = os.path.join(VISUALIZE_DIR, "create_video.py")

# --- Funzioni di supporto ---

def run_script(script_path, jump_number):
    """
    Esegue uno script Python esterno, indipendentemente dalla sua posizione,
    passando il numero del salto come input.
    """
    
    print(f"\n--- Esecuzione: {script_path} per il Salto {jump_number} ---")
    
    # Verifica che lo script esista
    if not os.path.exists(script_path):
        print(f"--- ❌ Errore: Lo script non è stato trovato al percorso: {script_path} ---")
        return False
        
    try:
        # sys.executable è il percorso dell'interprete Python corrente
        # L'input viene codificato e passato allo script come input standard
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

# --- Flusso di lavoro principale ---

def main_workflow():
    """Il flusso di lavoro principale del programma."""
    
    print("\n" + "="*50)
    print("🎬 WORKFLOW ANALISI SALTI SCI")
    print(f"   Root scripts: {EXTRACT_SCRIPT}, {INTERPOLATE_SCRIPT}")
    print(f"   Visualize scripts directory: {VISUALIZE_DIR}")
    print("="*50)

    while True:
        # 1. Ottieni il numero del Salto
        jump_number = get_jump_number()
        
        # 2. Esegui Extract Annotations (Root)
        print("\nPasso 1/4: Estrazione Annotazioni...")
        if not run_script(EXTRACT_SCRIPT, jump_number):
            if get_yes_no_input("Estrazione fallita. Vuoi riprovare/analizzare un altro salto?"):
                continue
            else:
                break

        # 3. Chiedi per l'Interpolazione (Root)
        do_interpolate = get_yes_no_input("Vuoi INTERPOLARE i frame mancanti?")
        if do_interpolate:
            print("\nPasso 2/4: Interpolazione Keypoints...")
            if not run_script(INTERPOLATE_SCRIPT, jump_number):
                print("⚠️ Attenzione: Interpolazione fallita. Procedo con la visualizzazione/video solo dei frame estratti.")
            else:
                 print("Interpolazione completata. Il file COCO interpolato è ora disponibile.")
        else:
            print("Interpolazione saltata.")

        # 4. Chiedi per la Visualizzazione (Sottocartella)
        do_visualize = get_yes_no_input("Vuoi VISUALIZZARE e creare il Video?")
        
        if do_visualize:
            # 4a. Esegui Visualizzazione (Sottocartella)
            print("\nPasso 3/4: Creazione delle Visualizzazioni...")
            if not run_script(VISUALIZE_SCRIPT, jump_number):
                print("❌ Visualizzazione fallita. Non posso procedere alla creazione del video.")
            else:
                # 4b. Esegui Creazione Video (Sottocartella)
                print("\nPasso 4/4: Creazione del Video...")
                run_script(CREATE_VIDEO_SCRIPT, jump_number)
        else:
            print("Visualizzazione e creazione video saltate.")

        # 5. Chiedi se continuare
        if not get_yes_no_input("\nAnalisi completata. Vuoi analizzare un altro salto?"):
            break

    print("\n" + "="*50)
    print("Programma terminato. Arrivederci!")
    print("="*50)


if __name__ == "__main__":
    main_workflow()