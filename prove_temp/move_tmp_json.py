import os
import shutil

# --- PATH CONFIG ---
# Base path where JPxxxx folders are located
# Use a raw string (r"...") to avoid Windows backslash issues
SOURCE_ROOT = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\dataset\annotations"

# Destination path where files will be copied
DEST_DIR = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiProject-SportTech\temp_jsons"

def copy_json_files():
    # 1. Ask for jump range
    try:
        start_jump = int(input("Enter first jump number (e.g. 6): "))
        end_jump = int(input("Enter last jump number (e.g. 10): "))
    except ValueError:
        print("Error: enter integers only.")
        return

    # 2. Create destination folder if missing
    if not os.path.exists(DEST_DIR):
        try:
            os.makedirs(DEST_DIR)
            print(f"Created destination folder: {DEST_DIR}")
        except OSError as e:
            print(f"Error creating folder: {e}")
            return

    print(f"\n--- Starting copy from jump {start_jump} to {end_jump} ---\n")

    # 3. Ciclo attraverso i numeri dei salti
    count_copied = 0
    
    for i in range(start_jump, end_jump + 1):
        # Build folder name (e.g. JP0006)
        # :04d means 4-digit zero-padded integer
        folder_name = f"JP{i:04d}"
        
        # Build file name (e.g. annotations_normalized_jump6.coco.json)
        file_name = f"annotations_normalized_jump{i}.coco.json"

        # Costruisco il percorso completo sorgente
        # Struttura: ...\dataset\annotations\JP0006\train\annotations_normalized_jump6.coco.json
        source_path = os.path.join(SOURCE_ROOT, folder_name, "train", file_name)
        
        # Costruisco il percorso destinazione
        dest_path = os.path.join(DEST_DIR, file_name)

        # 4. Controllo ed esecuzione copia
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path) # copy2 preserves file metadata
                print(f"✅ Copied: {file_name}")
                count_copied += 1
            except Exception as e:
                print(f"❌ Error copying {file_name}: {e}")
        else:
            print(f"⚠️ File not found: {source_path}")

    print(f"\n--- Operation complete. Copied {count_copied} files out of {end_jump - start_jump + 1} requested. ---")
    print(f"Files are here: {DEST_DIR}")

if __name__ == "__main__":
    copy_json_files()