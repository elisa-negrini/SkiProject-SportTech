import os
from pathlib import Path

def debug_jump_frames(jump_number, frames_root="dataset/frames"):
    jump_id = f"JP{jump_number:04d}"
    base_path = Path(frames_root) / jump_id
    
    if not base_path.exists():
        print(f"❌ Percorso non trovato: {base_path}")
        return

    # Sottocartelle da controllare
    subs = {
        "main": base_path,
        "removed": base_path / 'removed',
        "occluded": base_path / 'occluded'
    }

    # Dizionario per mappare frame -> posizione
    frame_map = {}
    
    for label, path in subs.items():
        if path.exists():
            for file in path.glob("*.jpg"):
                try:
                    idx = int(file.stem)
                    frame_map[idx] = label
                except ValueError:
                    continue

    if not frame_map:
        print(f"❓ Nessun frame .jpg trovato in {base_path}")
        return

    # Trova il range completo
    min_f = min(frame_map.keys())
    max_f = max(frame_map.keys())
    
    # Verifica anche quante righe ha il file boxes.txt originale
    boxes_file = base_path / 'boxes.txt'
    total_boxes = 0
    if boxes_file.exists():
        with open(boxes_file, 'r') as f:
            total_boxes = sum(1 for _ in f)

    print(f"\n=== DIAGNOSI {jump_id} ===")
    print(f"Range frame: {min_f} -> {max_f}")
    print(f"Totale righe in boxes.txt: {total_boxes}")
    print(f"{'Frame':<10} | {'Stato':<15}")
    print("-" * 30)

    for i in range(min_f, max_f + 1):
        status = frame_map.get(i, "MISSING (nessuna cartella)")
        print(f"{i:<10} | {status}")

    # Conteggio riassuntivo
    main_count = list(frame_map.values()).count("main")
    print(f"\nRiassunto:")
    print(f"- Frame in 'main': {main_count}")
    print(f"- Frame totali trovati (main+occ+rem): {len(frame_map)}")
    
    if total_boxes != len(frame_map):
        print(f"⚠️ DISALLINEAMENTO CRITICO: boxes.txt ha {total_boxes} righe, ma abbiamo trovato {len(frame_map)} file totali.")

# Esempio di utilizzo per il salto 39
if __name__ == "__main__":
    debug_jump_frames(39)