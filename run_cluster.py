import os
import sys
from datetime import datetime
from pathlib import Path

def get_latest_output_dir():
    # Get the base outputs directory
    outputs_dir = Path("outputs")
    
    if not outputs_dir.exists():
        return None
    
    # Get the latest date directory
    date_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    if not date_dirs:
        return None
    latest_date_dir = max(date_dirs)
    
    # Get the latest time directory
    time_dirs = [d for d in latest_date_dir.iterdir() if d.is_dir()]
    if not time_dirs:
        return None
    latest_time_dir = max(time_dirs)
    
    return latest_time_dir

def main():
    # # Check if JIGSAW dataset exists
    # jigsaw_path = Path("/JIGSAW_data/combined_dataset1.zarr")
    # if not jigsaw_path.is_dir():
    #     print("Combined JIGSAW dataset not found. Running combine.py...")
    #     combine_result = os.system("python /JIGSAW_data/combine.py")
        
    #     # Check if combine.py executed successfully
    #     if combine_result != 0:
    #         print("Error: Failed to create combined JIGSAW dataset")
    #         sys.exit(1)
        
    #     # Verify the dataset was created
    #     if not jigsaw_path.is_dir():
    #         print("Error: Combined JIGSAW dataset still not found after running combine.py")
    #         sys.exit(1)
    
    # Check if there's any existing output directory
    latest_dir = get_latest_output_dir()
    
    if latest_dir is None:
        # No previous run found, start fresh training
        print("Starting fresh training run...")
        os.system("python src/main.py env.train.id=BreakoutNoFrameskip-v4")
    else:
        # Previous run found, execute resume.sh from the latest directory
        print(f"Resuming from latest checkpoint in: {latest_dir}")
        resume_script = latest_dir / "scripts" / "resume.sh"
        
        if not resume_script.exists():
            print(f"Error: resume.sh not found in {latest_dir}/scripts/")
            sys.exit(1)
            
        # Change to the directory containing resume.sh and execute it
        os.chdir(latest_dir / "scripts")
        os.system("bash resume.sh")

if __name__ == "__main__":
    main()