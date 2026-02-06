import os
import re
from pathlib import Path

def extract_frame_number(filename):
    """Extract number from filename (e.g. frame_00123.jpg -> 123)"""
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

def check_jump_files():
    # 1. Ask jump ID
    raw_input = input("\nEnter jump number to check (e.g. 31): ").strip()
    try:
        jump_num = int(raw_input)
    except ValueError:
        print("❌ Enter a valid number.")
        return

    jump_id = f"JP{jump_num:04d}"
    
    # Costruisci i percorsi
    root_dir = Path("dataset/frames") / jump_id
    removed_dir = root_dir / "removed"
    occluded_dir = root_dir / "occluded"

    if not root_dir.exists():
        print(f"❌ La cartella {root_dir} non esiste!")
        return

    print(f"\n--- CONTROLLO FILE PER {jump_id} ---")
    print(f"Percorso base: {root_dir}")

    # 2. Raccogli tutti i file trovati
    found_frames = {} # Dizionario: numero_frame -> (nome_file, status)

    # Scansione MAIN (cartella principale)
    if root_dir.exists():
        for f in root_dir.glob("*.jpg"):
            num = extract_frame_number(f.name)
            if num != -1: found_frames[num] = (f.name, "✅ MAIN")

    # Scansione REMOVED
    if removed_dir.exists():
        for f in removed_dir.glob("*.jpg"):
            num = extract_frame_number(f.name)
            if num != -1: found_frames[num] = (f.name, "🗑️  REMOVED")

    # Scansione OCCLUDED
    if occluded_dir.exists():
        for f in occluded_dir.glob("*.jpg"):
            num = extract_frame_number(f.name)
            if num != -1: found_frames[num] = (f.name, "👁️  OCCLUDED")

    if not found_frames:
        print("❌ Nessun file JPG trovato in nessuna cartella!")
        return

    # 3. Analisi Sequenza
    sorted_nums = sorted(found_frames.keys())
    min_frame = sorted_nums[0]
    max_frame = sorted_nums[-1]
    
    total_physical = len(sorted_nums)
    expected_range = max_frame - min_frame + 1
    
    print(f"\nRange rilevato: dal frame {min_frame} al {max_frame}")
    print(f"File fisici trovati: {total_physical}")
    
    # 4. Stampa Lista Dettagliata
    print("\n{:<10} {:<25} {:<15}".format("FRAME", "FILENAME", "STATUS"))
    print("-" * 50)

    missing_count = 0
    
    # Itera da min a max per trovare i buchi
    for i in range(min_frame, max_frame + 1):
        if i in found_frames:
            fname, status = found_frames[i]
            print("{:<10} {:<25} {:<15}".format(i, fname, status))
        else:
            # FRAME MANCANTE (Il colpevole dell'errore Sync!)
            print("{:<10} {:<25} {:<15}".format(i, "---", "❌ MISSING (CANCELLATO?)"))
            missing_count += 1

    print("-" * 50)
    print(f"\nRIEPILOGO:")
    print(f"- Totale file presenti: {total_physical}")
    print(f"- Totale buchi (MISSING): {missing_count}")
    
    if missing_count > 0:
        print("\n⚠️  ATTENZIONE: Hai dei file 'MISSING'.")
        print("   Il 'box_filter.py' si aspetta che questi file esistano o siano in 'removed'.")
        print("   Poiché sono spariti, il conteggio delle righe nel file boxes.txt non combacia più.")
    else:
        print("\n✅ La sequenza è completa (nessun file mancante nel mezzo).")

if __name__ == "__main__":
    check_jump_files()