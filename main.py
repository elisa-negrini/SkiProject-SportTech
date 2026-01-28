from old_scripts import normalizer_elisa
from utils.annotation_manager import AnnotationManager
from utils.box_filter import filter_boxes
from utils.interpolator import Interpolator
from utils.visualizer import Visualizer
from utils.normalizer import Normalizer
from utils.metrics_calculator import MetricsCalculator

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
    do_vis_video = input("Visualize Interpolation (Video)? (y/n): ").lower().startswith('y')
    do_norm = input("Normalize Data? (y/n): ").lower().startswith('y')
    do_metrics = input("Calculate Metrics? (y/n): ").lower().startswith('y')
    
    # Initialize Tools
    extractor = AnnotationManager()
    interpolator = Interpolator()
    normalizer = Normalizer()
    metrics_calc = MetricsCalculator()
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
                continue
        
        # 4. Visualize
        if do_vis_video:
            print("   -> Generating interpolation video...")
            viz.generate_images(jn)
            viz.create_video(jn)

        # 5. Normalization
        # Reads 'annotations_interpolated' and creates 'annotations_normalized'
        if do_norm:
            print(f"   -> Normalizing JP{jn:04d}...")
            if normalizer.process(jn):
                # 6. Visualize Normalization (Check Overlay)
                # Uses 'annotations_normalized'
                normalizer.visualize_normalization(jn)
            else:
                print("⚠️ Normalization failed.")

    print("\n=== BATCH COMPLETE ===" )

    # --- POST-PROCESSING ---
    
    # Dataset CSV Creation (uses normalized annotations)
    if do_norm:
        print("\n--- Generating Normalized Dataset CSV ---")
        normalizer.create_dataset_csv()
    
    # Metrics Calculation (requires dataset CSV to exist)
    if do_metrics:
        print("\n--- Calculating Biomechanical Metrics ---")
        if metrics_calc.load_data():
            metrics_calc.process()
        else:
            print("⚠️ Metrics calculation failed. Ensure keypoints_dataset.csv exists.")

if __name__ == "__main__":
    main()