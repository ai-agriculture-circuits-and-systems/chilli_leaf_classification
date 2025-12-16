#!/usr/bin/env python3
"""
Convert Chilli Leaf Disease Classification dataset annotations to COCO JSON format.
Based on the standardized dataset structure specification.

This dataset uses a subcategory organization structure for classification tasks.
Each subcategory (healthy, cercospora, etc.) contains images and annotations.

License: CC BY 4.0 (see LICENSE). This script is distributed alongside the
dataset and follows the same license terms. Cite the original dataset in publications.

Usage examples:
    python scripts/convert_to_coco.py --root . --out annotations \
        --category chillies --splits train val test
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image


def _lower_keys(mapping: Dict[str, str]) -> Dict[str, str]:
    """Return a case-insensitive mapping by lowering keys."""
    return {k.lower(): v for k, v in mapping.items()}


def _read_split_list(split_file: Path) -> List[str]:
    """Read image base names (without extension) from a split file."""
    if not split_file.exists():
        return []
    lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _image_size(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) for an image path using PIL."""
    with Image.open(image_path) as img:
        return img.width, img.height


def _parse_csv_boxes(csv_path: Path) -> List[Dict]:
    """Parse a single per-image CSV file and return COCO-style bboxes.
    
    The parser is resilient to header variants by using case-insensitive
    lookups. Supported schemas:
      - Rectangle: x, y, w/h or dx/dy or width/height
      - Circle: x, y, r (converted to rectangle)
    """
    if not csv_path.exists():
        return []
    
    boxes: List[Dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return boxes
        
        header = _lower_keys({k: k for k in reader.fieldnames})
        
        def get(row: Dict[str, str], *keys: str) -> Optional[float]:
            for key in keys:
                if key in row and row[key] not in (None, ""):
                    try:
                        return float(row[key])
                    except ValueError:
                        continue
            return None
        
        for raw_row in reader:
            row = {k.lower(): v for k, v in raw_row.items()}
            x = get(row, "x", "xc", "x_center")
            y = get(row, "y", "yc", "y_center")
            # Circle
            r = get(row, "r", "radius")
            # Rectangle sizes
            w = get(row, "w", "width", "dx")
            h = get(row, "h", "height", "dy")
            label = get(row, "label", "class", "category_id")
            
            if x is None or y is None:
                continue
            
            category_id = int(label) if label is not None else 1
            
            if r is not None:
                # Convert circle to rectangle
                bbox = [x - r, y - r, 2 * r, 2 * r]
                area = (2 * r) * (2 * r)
            elif w is not None and h is not None:
                bbox = [x, y, w, h]
                area = w * h
            else:
                continue
            
            boxes.append({
                "bbox": bbox,
                "area": area,
                "category_id": category_id,
            })
    
    return boxes


def _load_labelmap(labelmap_path: Path) -> Dict[str, int]:
    """Load labelmap.json and return a mapping from subcategory name to category_id."""
    if not labelmap_path.exists():
        return {}
    
    with open(labelmap_path, 'r', encoding='utf-8') as f:
        labelmap = json.load(f)
    
    # Create mapping: subcategory name -> category_id
    mapping = {}
    for item in labelmap:
        if item.get('object_id', 0) > 0:  # Skip background (id=0)
            mapping[item['object_name']] = item['object_id']
    
    return mapping


def _collect_annotations_for_split(
    category_root: Path,
    split: str,
    labelmap: Dict[str, int],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Collect COCO dictionaries for images, annotations, and categories.
    
    This function handles subcategory organization structure where each
    subcategory directory contains images and CSV annotations.
    Each subcategory has its own sets directory.
    """
    # Collect image stems from all subcategories' sets directories
    image_stems = set()
    for subcat_dir in category_root.iterdir():
        if not subcat_dir.is_dir() or subcat_dir.name == 'sets':
            continue
        subcat_sets_dir = subcat_dir / "sets"
        split_file = subcat_sets_dir / f"{split}.txt"
        if split_file.exists():
            image_stems.update(_read_split_list(split_file))
    
    if not image_stems:
        # Fall back to all images if no split file
        for subcat_dir in category_root.iterdir():
            if subcat_dir.is_dir() and subcat_dir.name != 'sets':
                images_dir = subcat_dir / "images"
                image_stems.update({p.stem for p in images_dir.glob("*.jpg")})
                image_stems.update({p.stem for p in images_dir.glob("*.JPG")})
                image_stems.update({p.stem for p in images_dir.glob("*.png")})
                image_stems.update({p.stem for p in images_dir.glob("*.bmp")})
    
    images: List[Dict] = []
    anns: List[Dict] = []
    
    # Build categories list from labelmap
    categories: List[Dict] = []
    for subcat_name, cat_id in sorted(labelmap.items(), key=lambda x: x[1]):
        categories.append({
            "id": cat_id,
            "name": subcat_name,
            "supercategory": "chilli_leaf"
        })
    
    image_id_counter = 1
    ann_id_counter = 1
    
    # Find images across all subcategories
    for subcat_dir in category_root.iterdir():
        if not subcat_dir.is_dir() or subcat_dir.name == 'sets':
            continue
        
        subcat_name = subcat_dir.name
        if subcat_name not in labelmap:
            continue
        
        images_dir = subcat_dir / "images"
        csv_dir = subcat_dir / "csv"
        category_id = labelmap[subcat_name]
        
        for stem in sorted(image_stems):
            # Check if this image exists in this subcategory
            img_path = None
            for ext in ['.jpg', '.JPG', '.png', '.PNG', '.bmp']:
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
            
            if img_path is None:
                continue
            
            width, height = _image_size(img_path)
            images.append({
                "id": image_id_counter,
                "file_name": f"{category_root.name}/{subcat_name}/images/{img_path.name}",
                "width": width,
                "height": height,
            })
            
            csv_path = csv_dir / f"{stem}.csv"
            boxes = _parse_csv_boxes(csv_path)
            
            # If no boxes found, create a full-image box for classification task
            if not boxes:
                boxes = [{
                    "bbox": [0, 0, width, height],
                    "area": width * height,
                    "category_id": category_id,
                }]
            
            for box in boxes:
                # Override category_id with the subcategory's category_id
                anns.append({
                    "id": ann_id_counter,
                    "image_id": image_id_counter,
                    "category_id": category_id,  # Use subcategory's category_id
                    "bbox": box["bbox"],
                    "area": box["area"],
                    "iscrowd": 0,
                })
                ann_id_counter += 1
            
            image_id_counter += 1
    
    return images, anns, categories


def _build_coco_dict(
    images: List[Dict],
    anns: List[Dict],
    categories: List[Dict],
    description: str,
) -> Dict:
    """Build a complete COCO dict from components."""
    return {
        "info": {
            "year": 2025,
            "version": "1.0.0",
            "description": description,
            "url": "https://data.mendeley.com/public-files/datasets/tf9dtfz9m6/files/71452a49-947d-4b1f-a898-d51c8e351a53/file_downloaded",
        },
        "images": images,
        "annotations": anns,
        "categories": categories,
        "licenses": [],
    }


def convert(
    root: Path,
    out_dir: Path,
    category: str,
    splits: List[str],
) -> None:
    """Convert selected category and splits to COCO JSON files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    category_root = root / category
    labelmap_path = category_root / "labelmap.json"
    labelmap = _load_labelmap(labelmap_path)
    
    if not labelmap:
        print(f"Warning: Could not load labelmap from {labelmap_path}")
        return
    
    for split in splits:
        images, anns, categories = _collect_annotations_for_split(
            category_root, split, labelmap
        )
        desc = f"Chilli Leaf Disease Classification {category} {split} split"
        coco = _build_coco_dict(images, anns, categories, desc)
        out_path = out_dir / f"{category}_instances_{split}.json"
        out_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
        print(f"Generated {out_path} with {len(images)} images and {len(anns)} annotations")


def main() -> int:
    """Entry point for the converter CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Dataset root containing category subfolders (default: dataset root)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "annotations",
        help="Output directory for COCO JSON files (default: <root>/annotations)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="chillies",
        help="Category to convert (default: chillies)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=str,
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to generate (default: train val test)",
    )
    
    args = parser.parse_args()
    
    convert(
        root=Path(args.root),
        out_dir=Path(args.out),
        category=args.category,
        splits=args.splits,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

