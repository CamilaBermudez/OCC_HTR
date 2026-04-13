import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
PYTHON_IO_ENCODING = os.environ.get("PYTHON_IO_ENCODING")

# Load image
input_path = os.path.join(PROJECT_ROOT, "data", "raw", "original_manuscript", "reproduction14453_100", "8 - f. 003v - 004.jpg")
img = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(img_rgb, extent=[0, w, h, 0])
ax.set_title("CLICK to get coordinates | Press 'q' in console to stop", fontsize=12)
ax.set_xlabel("X (pixels)")
ax.set_ylabel("Y (pixels)")

# Add grid for reference
tick_step = max(50, int(max(w, h) / 8))
ax.set_xticks(np.arange(0, w + tick_step, tick_step))
ax.set_yticks(np.arange(0, h + tick_step, tick_step))
ax.grid(True, linestyle=':', alpha=0.5)

# Store clicked points
clicked_points = []

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append((x, y))
        
        # Visual feedback: mark the click
        ax.plot(x, y, 'r+', markersize=15, markeredgewidth=2)
        fig.canvas.draw()
        
        # Print coordinates
        print(f"Point {len(clicked_points)}: ({x}, {y})")
        
        # If we have 2 points, suggest rectangle
        if len(clicked_points) == 2:
            x1, y1 = clicked_points[0]
            x2, y2 = clicked_points[1]
            # Ensure proper ordering
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            print(f"\n📐 Suggested rectangle:")
            print(f"   cv2.rectangle(mask, ({x_min}, {y_min}), ({x_max}, {y_max}), 0, -1)")
            print(f"   Area: {(x_max-x_min) * (y_max-y_min)} pixels\n")

# Connect the click handler
fig.canvas.mpl_connect('button_press_event', on_click)

print("Click on the image to mark points (top-left and bottom-right of rectangle)")
print("Click twice to get a complete rectangle definition\n")

plt.tight_layout()
plt.show()

# After closing/finishing, show all collected points
print("\n" + "="*50)
print("ALL CLICKED POINTS:")
for i, (x, y) in enumerate(clicked_points, 1):
    print(f"   {i}. ({x}, {y})")
print("="*50)