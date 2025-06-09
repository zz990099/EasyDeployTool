import argparse
import json
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

COCO_LABEL_MAP = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90
]

def filename_to_imageid(coco_gt, img_suffix):
    """
    构建 filename -> image_id 的映射
    """
    return {img['file_name']: img['id'] for img in coco_gt.dataset['images']}

def merge_detections(det_folder, filename2id, img_suffix):
    """
    合并所有检测结果为COCO检测格式
    """
    merged = []
    image_ids = []
    not_found = 0
    for fname in os.listdir(det_folder):
        if not fname.endswith('.json'):
            continue
        img_base = os.path.splitext(fname)[0]
        img_name = img_base + img_suffix
        image_id = filename2id.get(img_name)
        if image_id is None:
            print(f"Warning: {img_name} not found in GT. Skipped.")
            not_found += 1
            continue
        with open(os.path.join(det_folder, fname), 'r') as f:
            boxes = json.load(f)
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            w, h = x2 - x1, y2 - y1
            merged.append({
                'image_id': image_id,
                'category_id': COCO_LABEL_MAP[int(box['label'])],  # 要保证label是COCO category_id
                'bbox': [x1, y1, w, h],
                'score': box['conf']
            })
        image_ids.append(image_id)

    return merged, not_found, image_ids

def evaluate_with_coco(gt_json, detections_json, image_ids, iou_type='bbox'):
    """
    用COCO API评估
    """
    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(detections_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval

def parse_args():
    parser = argparse.ArgumentParser(description="COCO Detection Evaluation Tool")
    parser.add_argument("--gt_json", required=True, help="COCO annotations file (e.g. instances_val2017.json)")
    parser.add_argument("--det_folder", required=True, help="Folder with detection result json files")
    parser.add_argument("--img_suffix", default=".jpg", help="Image filename suffix (default: .jpg)")
    parser.add_argument("--output_json", default="result_coco_format.json", help="Merged detection output JSON path")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Loading COCO annotations...")
    try:
        from pycocotools.coco import COCO
        coco_gt = COCO(args.gt_json)
    except ImportError:
        print("Please install pycocotools! (pip install pycocotools)")
        return
    filename2id = filename_to_imageid(coco_gt, args.img_suffix)

    print("Merging detections...")
    merged, not_found, image_ids = merge_detections(args.det_folder, filename2id, args.img_suffix)
    print(f"Total boxes merged: {len(merged)} | Files unmatched: {not_found}")

    print("Saving merged detections file:", args.output_json)
    with open(args.output_json, 'w') as f:
        json.dump(merged, f)

    print("Evaluating with COCO API...")
    evaluate_with_coco(args.gt_json, args.output_json, image_ids)
    print("Done.")

if __name__ == "__main__":
    main()
