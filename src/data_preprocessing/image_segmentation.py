import os
import re
import subprocess
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
PYTHON_IO_ENCODING = os.environ.get("PYTHON_IO_ENCODING")

def segment_all_images():
    input_folder = os.path.join(PROJECT_ROOT, "data", "raw", "original_manuscript", "reproduction14453_100")
    output_folder = os.path.join(PROJECT_ROOT, "data", "processed", "segmented_images")
    os.makedirs(output_folder, exist_ok=True)
    KRAKEN_BIN = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "kraken.exe")

    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found at {input_folder}")
        return

    # To avoid Unicode/Checkmark errors
    _env = os.environ.copy()
    _env["PYTHONIOENCODING"] = PYTHON_IO_ENCODING

    print(f"Starting segmentation for files in: {input_folder}")
    
    for image_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, image_file)
        
        if not os.path.isfile(input_path):
            continue
            
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png", ".tif")):
            continue

        base_name = os.path.splitext(image_file)[0]
        padded_name = re.sub(r'^\d+', lambda m: m.group().zfill(2), base_name)
        processed_name = padded_name.replace(" ", "").replace("-", "_")
        output_filename = processed_name + ".json"
        output_path = os.path.join(output_folder, output_filename)

        # Convert paths to forward slashes for the command line
        input_path_cmd = input_path.replace(os.sep, '/')
        output_path_cmd = output_path.replace(os.sep, '/')

        print(f"Processing: {image_file}")

        try:
            subprocess.run(
                [KRAKEN_BIN, "-i", input_path_cmd, output_path_cmd, "segment", "-bl"],
                check=True,
                capture_output=True,
                text=True,
                env=_env
            )
            print(f"  -> Success: Saved {output_filename}")
            
        except subprocess.CalledProcessError as e:
            print(f"  -> Failed: {e.stderr}")
        except Exception as e:
            print(f"  -> Unexpected Error: {e}")

if __name__ == "__main__":
    segment_all_images()