from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence


JsonDict = Dict[str, object]


@dataclass(frozen=True)
class COCOIndex:
    images_by_id: Mapping[int, JsonDict]
    annotations_by_id: Mapping[int, JsonDict]
    annotations_by_image: Mapping[int, Sequence[JsonDict]]
    categories_by_id: Mapping[int, JsonDict]
    captions_by_image: Mapping[int, Sequence[JsonDict]]

    def iter_images(self) -> Iterable[JsonDict]:
        for image_id in sorted(self.images_by_id):
            yield self.images_by_id[image_id]

    def annotations_for_image(self, image_id: int) -> Sequence[JsonDict]:
        return self.annotations_by_image.get(image_id, ())

    def captions_for_image(self, image_id: int) -> Sequence[JsonDict]:
        return self.captions_by_image.get(image_id, ())

    def category_name(self, category_id: int) -> str:
        category = self.categories_by_id[category_id]
        return str(category["name"])


def read_json(path: Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_coco_index(instances_json: Path, captions_json: Path) -> COCOIndex:
    instances = read_json(Path(instances_json))
    captions = read_json(Path(captions_json))

    images_by_id = {int(img["id"]): img for img in instances.get("images", [])}
    categories_by_id = {int(cat["id"]): cat for cat in instances.get("categories", [])}

    annotations_by_id: Dict[int, JsonDict] = {}
    annotations_by_image: DefaultDict[int, List[JsonDict]] = defaultdict(list)
    for ann in instances.get("annotations", []):
        ann_id = int(ann["id"])
        image_id = int(ann["image_id"])
        annotations_by_id[ann_id] = ann
        annotations_by_image[image_id].append(ann)

    for anns in annotations_by_image.values():
        anns.sort(key=lambda item: int(item["id"]))

    captions_by_image: DefaultDict[int, List[JsonDict]] = defaultdict(list)
    for caption in captions.get("annotations", []):
        image_id = int(caption["image_id"])
        captions_by_image[image_id].append(caption)

    for items in captions_by_image.values():
        items.sort(key=lambda item: int(item["id"]))

    return COCOIndex(
        images_by_id=images_by_id,
        annotations_by_id=annotations_by_id,
        annotations_by_image=annotations_by_image,
        categories_by_id=categories_by_id,
        captions_by_image=captions_by_image,
    )


def load_manifest(path: Path) -> List[JsonDict]:
    records: List[JsonDict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def save_manifest(records: Sequence[JsonDict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def manifest_exists(path: Path) -> bool:
    return Path(path).is_file()

