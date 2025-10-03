"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from deim._engine.data.dataset.coco_dataset import CocoDetection
from deim._engine.data.dataset.coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from deim._engine.data.dataset.coco_eval import CocoEvaluator
from deim._engine.data.dataset.coco_utils import get_coco_api_from_dataset
from deim._engine.data.dataset.voc_detection import VOCDetection
from deim._engine.data.dataset.voc_eval import VOCEvaluator
