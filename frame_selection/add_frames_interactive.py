import os
import shutil
import glob

# --- CONFIGURATION ---
BASE_DATASET_PATH = r"C:\Users\utente\Desktop\UNITN secondo anno\Sport Tech\ski project\SkiTB dataset\SkiTB\JP"
DESTINATION_FOLDER_NAME = "filtered_frames"
FRAME_OFFSET = 2  # Number of frames before and after the target frame


def get_available_jumps():
    """Returns a list of available JPXXXX folders in the base dataset path."""
    pattern = os.path.join(BASE_DATASET_PATH, "JP*")
    folders = [os.path.basename(f) for f in glob.glob(pattern) if os.path.isdir(f)]
    # Filter only JPXXXX format
    jumps = [f for f in folders if f.startswith("JP") and len(f) == 6 and f[2:].isdigit()]
    return sorted(jumps)


def get_frame_files(source_folder):
    """Returns a sorted list of all frame files in the source folder."""
    search_path_jpg = os.path.join(source_folder, '*.jpg')
    search_path_jpeg = os.path.join(source_folder, '*.jpeg')
    all_files = sorted(glob.glob(search_path_jpg) + glob.glob(search_path_jpeg))
    return all_files


def find_frame_by_number(frame_files, frame_number):
    """Finds a frame file by its number (e.g., 00290)."""
    target_frame = f"{frame_number:05d}"
    for file_path in frame_files:
        file_name = os.path.basename(file_path)
        # Extract the numeric part from the filename (assumes format like 00290.jpg)
        if file_name.startswith(target_frame):
            return file_path
    return None


def get_frame_index(frame_files, frame_path):
    """Returns the index of a frame in the sorted list."""
    try:
        return frame_files.index(frame_path)
    except ValueError:
        return -1


def copy_frame(source_path, destination_folder):
    """Copies a single frame to the destination folder."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    file_name = os.path.basename(source_path)
    destination_path = os.path.join(destination_folder, file_name)
    
    # Check if file already exists
    if os.path.exists(destination_path):
        print(f"   ⚠️  Frame {file_name} already exists in destination. Skipping.")
        return False
    
    try:
        shutil.copy2(source_path, destination_path)
        print(f"   ✅ Copied: {file_name}")
        return True
    except Exception as e:
        print(f"   ❌ Error copying {file_name}: {e}")
        return False


def add_frames_around_target(jump_code, frame_number, offset=FRAME_OFFSET):
    """
    Adds only the frames at -offset and +offset from the target frame.
    
    Args:
        jump_code: The JPXXXX code (e.g., "JP0001")
        frame_number: The target frame number (e.g., 290)
        offset: Number of frames before and after to add (default: 3)
              For example, with frame 290 and offset 3, adds only 287 and 293
    """
    # Define paths
    source_folder = os.path.join(BASE_DATASET_PATH, jump_code, "frames")
    destination_folder = os.path.join(BASE_DATASET_PATH, DESTINATION_FOLDER_NAME, jump_code)
    
    # Check if source folder exists
    if not os.path.isdir(source_folder):
        print(f"❌ Source folder not found: {source_folder}")
        return
    
    # Get all frame files
    frame_files = get_frame_files(source_folder)
    
    if not frame_files:
        print(f"❌ No frame files found in {source_folder}")
        return
    
    # Find the target frame
    target_frame_path = find_frame_by_number(frame_files, frame_number)
    
    if not target_frame_path:
        print(f"❌ Frame {frame_number:05d} not found in {jump_code}")
        print(f"   Available range: {os.path.basename(frame_files[0])} to {os.path.basename(frame_files[-1])}")
        return
    
    # Get the index of the target frame
    target_index = get_frame_index(frame_files, target_frame_path)
    
    if target_index == -1:
        print(f"❌ Error finding frame index")
        return
    
    print(f"\n📸 Adding frames around {os.path.basename(target_frame_path)} (index {target_index})")
    print(f"   Will add: frame at -{offset} and frame at +{offset}")
    
    # Calculate indices for only the frames at -offset and +offset
    indices_to_copy = []
    
    # Frame before (target - offset)
    before_index = target_index - offset
    if before_index >= 0:
        indices_to_copy.append(before_index)
    else:
        print(f"   ⚠️  Frame at -{offset} is out of range (would be index {before_index})")
    
    # Frame after (target + offset)
    after_index = target_index + offset
    if after_index < len(frame_files):
        indices_to_copy.append(after_index)
    else:
        print(f"   ⚠️  Frame at +{offset} is out of range (would be index {after_index})")
    
    # Copy frames
    copied_count = 0
    for i in indices_to_copy:
        if copy_frame(frame_files[i], destination_folder):
            copied_count += 1
    
    print(f"\n✨ Successfully copied {copied_count} new frame(s) to {jump_code}")


def main():
    """Main interactive function."""
    print("=" * 60)
    print("   📹 FRAME ADDITION TOOL FOR FILTERED FRAMES")
    print("=" * 60)
    
    # Get available jumps
    available_jumps = get_available_jumps()
    
    if not available_jumps:
        print("❌ No JPXXXX folders found in the dataset path.")
        return
    
    print(f"\n📁 Available jumps: {len(available_jumps)}")
    print(f"   Range: {available_jumps[0]} to {available_jumps[-1]}")
    
    # Ask for jump code
    while True:
        jump_input = input("\n🎿 Enter the jump code (e.g., JP0021 or just 21): ").strip()
        
        # Handle different input formats
        if jump_input.upper().startswith("JP"):
            jump_code = jump_input.upper()
        elif jump_input.isdigit():
            jump_code = f"JP{int(jump_input):04d}"
        else:
            print("❌ Invalid format. Please enter a valid jump code.")
            continue
        
        if jump_code in available_jumps:
            break
        else:
            print(f"❌ Jump {jump_code} not found. Available: {available_jumps[0]} to {available_jumps[-1]}")
            retry = input("   Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return
    
    # Ask for frame number
    while True:
        frame_input = input("\n🖼️  Enter the target frame number (e.g., 290 or 00290): ").strip()
        
        try:
            frame_number = int(frame_input)
            if frame_number < 0:
                print("❌ Frame number must be positive.")
                continue
            break
        except ValueError:
            print("❌ Invalid number. Please enter a valid frame number.")
    
    # Confirm action
    print(f"\n📋 Summary:")
    print(f"   Jump: {jump_code}")
    print(f"   Target frame: {frame_number:05d}")
    print(f"   Will add frames: {frame_number - FRAME_OFFSET:05d} and {frame_number + FRAME_OFFSET:05d}")
    
    confirm = input("\n✓ Proceed? (y/n): ").strip().lower()
    
    if confirm == 'y':
        add_frames_around_target(jump_code, frame_number)
    else:
        print("❌ Operation cancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")