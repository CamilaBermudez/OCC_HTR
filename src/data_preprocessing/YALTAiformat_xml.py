import os
import glob
import xml.etree.ElementTree as ET
import sys
from dotenv import load_dotenv
load_dotenv()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT")

# Official YALTAi Tag Dictionary
YALTAI_TAGS = [
    ("BT000", "DamageZone"), ("BT001", "DigitizationArtefactZone"),
    ("BT002", "DropCapitalZone"), ("BT003", "GraphicZone"),
    ("BT004", "MainZone"), ("BT005", "MarginTextZone"),
    ("BT006", "MusicZone"), ("BT007", "NumberingZone"),
    ("BT008", "QuireMarksZone"), ("BT009", "RunningTitleZone"),
    ("BT010", "SealZone"), ("BT011", "StampZone"),
    ("BT012", "TableZone"), ("BT013", "TitlePageZone")
]

# ALTO v4 Default Namespace
NS = "http://www.loc.gov/standards/alto/ns-v4#"
ET.register_namespace("", NS)  # Prevents ns0: prefixes in output

def close_polygon(points_str: str) -> str:
    """Ensures polygon is closed (5 points for rectangles)."""
    pts = points_str.strip().split()
    if len(pts) < 4 or len(pts) % 2 != 0:
        return points_str 
    if pts[:2] != pts[-2:]:
        pts.extend(pts[:2])
    return " ".join(pts)

def process_single_xml(xml_path: str, output_dir: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Handle namespace - try multiple ways to find Page
    NS = "http://www.loc.gov/standards/alto/ns-v4#"
    
    # Try to find Page element with namespace
    page = root.find(f"{{{NS}}}Layout/{{{NS}}}Page")
    if page is None:
        # Fallback: search without namespace
        page = root.find(".//Page")
    
    if page is None:
        print(f"⚠️ Warning: No Page element found in {os.path.basename(xml_path)}")
        # Use default dimensions
        img_w, img_h = 1.0, 1.0
    else:
        img_w = float(page.get("WIDTH", 1))
        img_h = float(page.get("HEIGHT", 1))

    #Replace <Tags> section with YALTAi standards
    tags_elem = root.find(f"{{{NS}}}Tags")
    if tags_elem is None:
        tags_elem = root.find(".//Tags")
    
    if tags_elem is not None:
        tags_elem.clear()
        for tag_id, label in YALTAI_TAGS:
            ET.SubElement(tags_elem, "OtherTag", {
                "ID": tag_id,
                "LABEL": label,
                "DESCRIPTION": f"block type {label}"
            })
    else:
        print(f"No Tags element found in {os.path.basename(xml_path)}")

    yolo_lines = []

    # 2Process all <TextBlock> elements (with namespace fallback)
    textblocks = root.findall(f"{{{NS}}}Layout/{{{NS}}}Page/{{{NS}}}PrintSpace/{{{NS}}}TextBlock")
    if not textblocks:
        textblocks = root.findall(".//TextBlock")
    
    for tb in textblocks:
        hpos = tb.get("HPOS", "0")
        vpos = tb.get("VPOS", "0")
        width = tb.get("WIDTH", "0")
        height = tb.get("HEIGHT", "0")
        tb_id = tb.get("ID", "")

        tb.attrib.clear()
        tb.set("HPOS", hpos)
        tb.set("VPOS", vpos)
        tb.set("WIDTH", width)
        tb.set("HEIGHT", height)
        tb.set("ID", tb_id)
        tb.set("TAGREFS", "BT004")  # All mapped to MainZone (Class 4)

        # Fix polygon closure
        shape = tb.find(f"{{{NS}}}Shape")
        if shape is None:
            shape = tb.find(".//Shape")
        
        if shape is not None:
            poly = shape.find(f"{{{NS}}}Polygon")
            if poly is None:
                poly = shape.find(".//Polygon")
            
            if poly is not None:
                poly.set("POINTS", close_polygon(poly.get("POINTS", "")))

        # Generate YOLO line (Class 4 = MainZone)
        if img_w > 0 and img_h > 0:
            cx = (float(hpos) + float(width) / 2) / img_w
            cy = (float(vpos) + float(height) / 2) / img_h
            nw = float(width) / img_w
            nh = float(height) / img_h
            yolo_lines.append(f"4 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    #Save formatted XML
    out_xml = os.path.join(output_dir, os.path.basename(xml_path))
    try:
        ET.indent(tree, space="  ", level=0)  # Python 3.9+
    except AttributeError:
        pass  # Skip indenting for Python < 3.9
    
    tree.write(out_xml, encoding="UTF-8", xml_declaration=True)

    #Save corresponding YOLO .txt
    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    out_txt = os.path.join(output_dir, f"{base_name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines))

def batch_convert(input_dir: str, output_dir: str):
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' not found.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
    
    if not xml_files:
        print("No .xml files found in the input directory.")
        return

    print(f"Found {len(xml_files)} XML files. Starting conversion...\n")
    for f in xml_files:
        print(f"Processing: {os.path.basename(f)}")
        process_single_xml(f, output_dir)
        
    print(f"\n Formatted files saved to: {output_dir}")

if __name__ == "__main__":
    INPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "processed", "annotated_samples","retrain","annotations")
    OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "processed", "annotated_samples","retrain","yaltai_formatted")
    
    batch_convert(INPUT_FOLDER, OUTPUT_FOLDER)