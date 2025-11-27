import cv2
import os
from pathlib import Path
import re

# ====== CONFIGURATION ======

# --- DYNAMIC INPUT FOR JUMP NUMBER ---
while True:
    try:
        jump_number = int(input("Enter the Jump number you want to create video for (e.g., 7): "))
        if jump_number > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

JUMP_ID = f"JP{jump_number:04d}"  # Format as JP0007, JP0012, etc.

# --- FILE PATHS (Dynamic) ---
# 1. Definisci il percorso della cartella che contiene le immagini (la destinazione)
images_folder = f"dataset/annotations/{JUMP_ID}/visualizations"

# 2. Definisci il nome del file di output
output_filename = f"output_video_{JUMP_ID}.mp4"

# 3. COMBINA: Il percorso del video sarà nella cartella delle immagini
output_video = os.path.join(images_folder, output_filename)

# Video settings
FPS = 10  # Frames per second (modifica questo valore se vuoi velocizzare o rallentare)
CODEC = 'mp4v'  # Codec video

# ====== FUNCTIONS ======

def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename like 'frame_00001.jpg'"""
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def create_video_from_images(images_folder, output_video, fps=10):
    """Create a video from a folder of images"""
    
    images_path = Path(images_folder)
    
    # Get all jpg files and sort them by frame number
    image_files = [f for f in images_path.glob("*.jpg")]
    image_files.sort(key=lambda x: extract_frame_number(x.name))
    
    if not image_files:
        print(f"❌ No images found in {images_folder}")
        return False
    
    print(f"Found {len(image_files)} images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print(f"❌ Could not read first image: {image_files[0]}")
        return False
    
    height, width, _ = first_image.shape
    print(f"Video resolution: {width}x{height}")
    print(f"FPS: {fps}")
    
    # Create video writer
    # cv2.VideoWriter_fourcc(*CODEC) richiede una stringa di 4 caratteri per il codec
    fourcc = cv2.VideoWriter_fourcc(*CODEC) 
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"❌ Could not create video writer for path: {output_video}")
        print("   Check if the codec is supported or if the folder exists.")
        return False
    
    # Process each image
    print("Creating video...")
    for i, img_file in enumerate(image_files):
        img = cv2.imread(str(img_file))
        
        if img is None:
            print(f"⚠️ Could not read {img_file.name}, skipping...")
            continue
        
        # Ensure image has correct dimensions
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
        
        video_writer.write(img)
        
        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(image_files)} frames...")
    
    video_writer.release()
    print(f"✅ Video created successfully: {output_video}")
    print(f"   Total frames: {len(image_files)}")
    print(f"   Duration: {len(image_files)/fps:.2f} seconds")
    return True

# ====== MAIN ======

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Creating video for {JUMP_ID}")
    print(f"{'='*60}\n")
    
    # Check if folder exists
    if not os.path.exists(images_folder):
        print(f"❌ Error: Folder not found: {images_folder}")
        print("   Assicurati che il percorso sia corretto rispetto a dove esegui lo script.")
        exit(1)
    
    # Create video
    success = create_video_from_images(images_folder, output_video, FPS)
    
    if success:
        print(f"\n{'='*60}")
        print("VIDEO CREATION COMPLETED!")
        print(f"{'='*60}")
        print(f"\nOutput file saved to: **{output_video}**")
        print(f"\nTo change video speed, modify the FPS value in the script:")
        print(f" - Current FPS: {FPS}")
        print(f" - Slower video: decrease FPS (e.g., 5)")
        print(f" - Faster video: increase FPS (e.g., 20)")
    else:
        print("\n❌ Video creation failed")
        exit(1)