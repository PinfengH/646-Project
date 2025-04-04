import pickle
import json
import os

def convert_refs_to_json(
    refs_path='data/refcoco/refs(unc).p',
    instances_path='data/refcoco/instances.json',
    coco_ann_path='data/annotations/instances_train2014.json',
    save_path='data/refcoco_mini.json',
    #max_samples=50000
):
    # Load .p file (list of refs)
    with open(refs_path, 'rb') as f:
        refs = pickle.load(f)

    # Load object-level annotation (object_id → bbox, image_id)
    with open(instances_path, 'r') as f:
        inst_data = json.load(f)
        ann_id_map = {ann['id']: ann for ann in inst_data['annotations']}

    # Load image-level annotation (image_id → file_name)
    with open(coco_ann_path, 'r') as f:
        coco_data = json.load(f)
        image_id_map = {img['id']: img['file_name'] for img in coco_data['images']}

    results = []
    for i, ref in enumerate(refs[:len(refs)]):
        ann_id = ref['ann_id']
        sent_list = ref['sentences']
        image_id = ref['image_id']

        if ann_id not in ann_id_map or image_id not in image_id_map:
            continue  # Skip invalid

        bbox = ann_id_map[ann_id]['bbox']
        file_name = image_id_map[image_id]

        for sent in sent_list:
            results.append({
                "file_name": file_name,
                "sentence": sent['sent'],
                "bbox": bbox
            })

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} samples to {save_path}")


if __name__ == "__main__":
    convert_refs_to_json()
