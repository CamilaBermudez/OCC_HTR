import os
import re
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
PYTHON_IO_ENCODING = os.environ.get("PYTHON_IO_ENCODING")

def plot_images_w_bounds():
    input_folder = os.path.join(PROJECT_ROOT, "data", "raw", "original_manuscript", "reproduction14453_100")
    output_kraken_folder = os.path.join(PROJECT_ROOT, "data", "processed", "segmented_images")
    output_folder = os.path.join(PROJECT_ROOT, "results", "image_segmentation")
    
    for image_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, image_file)
        
        base_name = os.path.splitext(image_file)[0]
        padded_name = re.sub(r'^\d+', lambda m: m.group().zfill(2), base_name)
        processed_name = padded_name.replace(" ", "").replace("-", "_").replace("f.", "f_")
        json_path = processed_name + ".json"
        json_path = os.path.join(output_kraken_folder, json_path)
        output_img = processed_name + ".png"
        output_path = os.path.join(output_folder, output_img)

        print(f"Processing: {image_file}")

        with open(json_path, "r", encoding="utf-8") as f:
            kraken_data = json.load(f)

        img = Image.open(input_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for i, line in enumerate(kraken_data.get("lines", []), start=1):
            # Baseline (red)
            baseline = line["baseline"]
            draw.line([tuple(baseline[0]), tuple(baseline[1])], fill="red", width=2)

            # Boundary polygon (blue)
            boundary = [tuple(pt) for pt in line.get("boundary", [])]
            if boundary:
                draw.polygon(boundary, outline="blue")

            # ID (green)
            x, y = baseline[0]
            draw.text((x-20, y-20), str(i-1), fill="green", font=font)

        img.save(output_path)

if __name__ == "__main__":
    plot_images_w_bounds()