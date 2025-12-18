from utils.annotation_manager import AnnotationManager
from utils.box_filter import filter_boxes
from utils.interpolator import Interpolator
from utils.visualizer import Visualizer

def get_input_range():
    val = input("\nEnter Jump range (e.g. '1-5') or single number ('6'): ").strip()
    if '-' in val:
        s, e = map(int, val.split('-'))
        return list(range(s, e + 1))
    return [int(val)]

def main():
    print("=== SKI JUMP ANALYSIS PIPELINE ===")
    
    jumps = get_input_range()
    do_interp = input("Interpolate? (y/n): ").lower().startswith('y')
    do_vis = input("Visualize & Video? (y/n): ").lower().startswith('y')

    # Initialize Tools
    extractor = AnnotationManager()
    interpolator = Interpolator()
    viz = Visualizer()

    for jn in jumps:
        print(f"\n--- Processing JP{jn:04d} ---")
        
        # 1. Extract
        if not extractor.extract_jump(jn): continue
        
        # 2. Filter Boxes
        if not filter_boxes(jn): continue
        
        # 3. Interpolate
        if do_interp:
            if not interpolator.process(jn):
                print("⚠️ Interpolation failed.")
        
        # 4. Visualize
        if do_vis:
            viz.generate_images(jn)
            viz.create_video(jn)

    print("\n=== BATCH COMPLETE ===")

if __name__ == "__main__":
    main()