"""
Annotation package for object detection visualization.
Provides utilities for annotating frames with detection results.
Supports batch processing for improved performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import supervision as sv
    from supervision.annotators.core import BoxAnnotator, RichLabelAnnotator
    from supervision.detection.core import Detections
    from supervision.geometry.core import Position
    # Import base Tracker class for type hinting purposes if needed for clarity
    # from supervision.tracker.byte_track.byte_track import ByteTrack # Example import
except ImportError:
    raise ImportError(
        "Supervision package is required. Install it with 'pip install supervision'"
    )

# Module-level variable to hold the global tracker instance
_global_byte_track_tracker: Optional[sv.ByteTrack] = None

@dataclass(frozen=True)
class ResizeInfo:
    """Stores information about the resizing and padding process."""

    ratio: float
    pad_w: int
    pad_h: int
    original_height: int
    original_width: int

    @classmethod
    def batch_from_arrays(
        cls,
        ratios: List[float],
        pad_ws: List[int],
        pad_hs: List[int],
        orig_heights: List[int],
        orig_widths: List[int],
    ) -> List["ResizeInfo"]:
        """Create a batch of ResizeInfo objects from parallel arrays."""
        batch_size = len(ratios)
        return [
            cls(
                ratio=ratios[i],
                pad_w=pad_ws[i],
                pad_h=pad_hs[i],
                original_height=orig_heights[i],
                original_width=orig_widths[i],
            )
            for i in range(batch_size)
        ]


@dataclass(frozen=True)
class DetectionResult:
    """Stores a single detection result after post-processing."""

    box: np.ndarray  # Shape (4,) representing xyxy
    score: float
    label: int


def _get_or_create_global_tracker() -> sv.ByteTrack:
    """
    Gets the global ByteTrack tracker instance, initializing it if necessary.
    """
    global _global_byte_track_tracker
    if _global_byte_track_tracker is None:
        # Initialize the tracker with default parameters
        # Adjust parameters like track_activation_threshold and lost_track_buffer
        # as needed for your specific application.
        _global_byte_track_tracker = sv.ByteTrack(
            track_activation_threshold=0.4, lost_track_buffer=8
        )
    return _global_byte_track_tracker


def _annotate_single_image_with_sv_detections(
    image: np.ndarray,
    sv_detections: Detections,
    class_names: Dict[int, str],
    box_annotator: BoxAnnotator,
    text_annotator: RichLabelAnnotator,
    tracker: Optional[sv.ByteTrack] = None,  # Use ByteTrack for specific type hint
    # Alternatively, use a more general Tracker base class if available in sv
    # tracker: Optional[sv.tracker.Tracker] = None,
) -> np.ndarray:
    """
    Helper function to apply supervision annotation to a single image
    given detections in supervision.Detections format.
    Handles tracking and label generation based on tracker presence.
    """
    current_detections = sv_detections

    # Apply tracking if a tracker instance is provided
    if tracker is not None:
        # Use the tracker to update detections.
        # ByteTrack might filter detections or add/remove tracks.
        # mypy type ignore needed because ByteTrack update method signature
        # might not perfectly align with generic tracker definition expected by mypy.
        # This is where detections might be "missed" if the tracker's logic
        # decides they don't form a stable track.
        current_detections = tracker.update_with_detections(current_detections)  # type: ignore

    # Prepare labels for annotation
    labels = []
    # Determine if tracker_id is available and should be used in labels
    use_tracker_id_in_labels = (
        tracker is not None
        and hasattr(current_detections, "tracker_id")
        and current_detections.tracker_id is not None
        and len(current_detections.tracker_id) == len(current_detections)
    )

    if use_tracker_id_in_labels:
        # Use tracker_id, label, and confidence for labels
        # Iterate using detections attributes directly for clarity and potential robustness
        # mypy type ignore needed as tracker_id is added dynamically and confidence is Optional
        for i in range(len(current_detections.class_id)):
            class_id = current_detections.class_id[i]
            confidence = current_detections.confidence[i]  # type: ignore
            tracker_id = current_detections.tracker_id[i]  # type: ignore

            label_text = class_names.get(int(class_id), "unknown")
            # Ensure tracker_id is not None for f-string formatting
            tracker_id_str = str(tracker_id) if tracker_id is not None else "N/A"
            confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
            labels.append(f"{tracker_id_str}: {label_text} - {confidence_str}")
    else:
        # Use only label and confidence for labels (no tracking)
        # Iterate using detections attributes directly
        # mypy type ignore needed as confidence is Optional
        for i in range(len(current_detections.class_id)):
            class_id = current_detections.class_id[i]
            confidence = current_detections.confidence[i]  # type: ignore

            label_text = class_names.get(int(class_id), "unknown")
            confidence_str = f"{confidence:.2f}" if confidence is not None else "N/A"
            labels.append(f"{label_text}: {confidence_str}")

    # Annotate the image (operates on a copy passed from the caller)
    # Ensure image is not None, although type hint suggests it won't be.
    if image is None:
        # Depending on source, handle this appropriately - e.g., return placeholder or raise error
        # Assuming valid image based on type hint, proceeding with annotation.
        # If you expect None, add robust error handling or return a default.
        pass  # Should not happen with correct usage/type hints

    annotated_image = box_annotator.annotate(
        scene=image,
        detections=current_detections,
    )
    annotated_image = text_annotator.annotate(
        scene=annotated_image,
        detections=current_detections,
        labels=labels,
    )

    return annotated_image


def annotate_frames(
    frames: List[np.ndarray],  # Original frames (RGB format)
    processed_detections: List[
        List[DetectionResult]
    ],  # Detections per frame (already filtered/processed as needed)
    class_names: Dict[int, str],
    thickness: int = 2,
    font_size: int = 12,
    is_tracking: bool = False,  # New parameter to enable tracking
) -> List[np.ndarray]:
    """
    Annotate frames with processed detections using Supervision.
    Optionally enables object tracking across frames.

    Args:
        frames: List of original frames (RGB format).
        processed_detections: List of detection results per frame.
                                These are expected to be the final detections
                                to be considered for annotation/tracking.
        class_names: Dictionary mapping class IDs to names for labels.
        thickness: Line thickness for boxes.
        font_size: Font size for labels.
        is_tracking: If True, applies ByteTrack tracking across the frames.

    Returns:
        List[np.ndarray]: List of annotated frames.
    """
    # Create annotators (reused for all frames)
    box_annotator = BoxAnnotator(thickness=thickness)
    text_annotator = RichLabelAnnotator(
        font_size=font_size,
        border_radius=5,
        text_padding=5,
        text_position=Position.TOP_CENTER,
        ensure_in_frame=True,
    )

    # Initialize tracker only if tracking is enabled
    tracker = (
        
        _get_or_create_global_tracker() 
        if is_tracking
        else None
    )  # type: ignore

    annotated_frames = []

    for frame, detections in zip(frames, processed_detections):
        # Prepare data for supervision.Detections from DetectionResult dataclass
        # These detections are assumed to be the input for tracking/annotation
        # after any necessary filtering/processing.
        if not detections:
            # Handle frames with no detections
            sv_detections = sv.Detections.empty()
        else:
            boxes_np = np.array([det.box for det in detections])
            scores_np = np.array([det.score for det in detections])
            labels_np = np.array([det.label for det in detections]).astype(int)

            # Create Supervision Detections object
            sv_detections = Detections(
                xyxy=boxes_np,
                confidence=scores_np,
                class_id=labels_np,
            )

        # Annotate the frame using the helper function
        # Pass the tracker (or None) initialized before the loop
        annotated_frame = _annotate_single_image_with_sv_detections(
            image=frame.copy(),  # Pass a copy to keep original frame untouched
            sv_detections=sv_detections,
            class_names=class_names,
            box_annotator=box_annotator,
            text_annotator=text_annotator,
            tracker=tracker,
        )

        annotated_frames.append(annotated_frame)

    return annotated_frames


def annotate_detections(
    image: np.ndarray,  # RGB format image
    boxes: np.ndarray,  # xyxy format boxes
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Dict[int, str],
    conf_threshold: float = 0.5,
    thickness: int = 2,
    font_size: int = 12,
    is_tracking: bool = False,  # New parameter to enable tracking (transient)
) -> np.ndarray:
    """
    Annotate a single image with detection results.
    Optionally enables tracking for this specific image frame.
    Note: If is_tracking is True, a NEW tracker is initialized for this call.
    This is intended for annotating individual, potentially unrelated images,
    or for use within an external loop that manages tracker state.
    Using this for sequential frames repeatedly will NOT provide persistent tracking IDs
    across frames unless the tracker object is managed externally.

    Args:
        image: Input image (RGB format)
        boxes: Bounding boxes in xyxy format
        scores: Confidence scores
        labels: Class labels
        class_names: Dictionary mapping class IDs to names for labels
        conf_threshold: Confidence threshold for filtering detections.
                        Applied BEFORE tracking if enabled.
        thickness: Line thickness for boxes
        font_size: Font size for labels
        is_tracking: If True, applies ByteTrack tracking for this image frame.

    Returns:
        np.ndarray: Annotated image
    """
    # Filter by confidence BEFORE creating sv_detections
    mask = scores > conf_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    # If no detections above threshold, return a copy of the original image
    if not np.any(mask):
        # Initialize a tracker with empty detections if tracking is enabled,
        # to potentially handle state if this is part of a sequence (though unlikely for this function's typical use)
        if is_tracking:
            tracker = _get_or_create_global_tracker()  # type: ignore
            tracker.update_with_detections(sv.Detections.empty())  # type: ignore
        return image.copy()

    # Create annotators
    box_annotator = BoxAnnotator(thickness=thickness)
    text_annotator = RichLabelAnnotator(
        font_size=font_size,
        border_radius=5,
        text_padding=5,
        text_position=Position.TOP_CENTER,
        ensure_in_frame=True,
    )

    # Initialize a tracker only if tracking is enabled.
    # Note: This tracker's state is lost after the function returns.
    tracker = (
        _get_or_create_global_tracker()
        if is_tracking
        else None
    )  # type: ignore

    # Create Supervision Detections object from filtered data
    sv_detections = Detections(
        xyxy=filtered_boxes,
        confidence=filtered_scores,
        class_id=filtered_labels,
    )

    # Annotate the image using the helper function, passing the tracker (or None)
    annotated_image = _annotate_single_image_with_sv_detections(
        image=image.copy(),  # Pass a copy to keep original image untouched
        sv_detections=sv_detections,
        class_names=class_names,
        box_annotator=box_annotator,
        text_annotator=text_annotator,
        tracker=tracker,  # Pass the tracker initialized for this call (or None)
    )

    return annotated_image


def annotate_batch(
    images: List[np.ndarray],  # RGB format images
    boxes_batch: List[np.ndarray],  # xyxy format boxes for each image
    scores_batch: List[np.ndarray],  # scores for each image
    labels_batch: List[np.ndarray],  # labels for each image
    class_names: Dict[int, str],
    conf_threshold: float = 0.5,
    thickness: int = 2,
    font_size: int = 12,
    is_tracking: bool = False,  # New parameter to enable tracking (treating batch as sequence)
) -> List[np.ndarray]:
    """
    Annotate a batch of images with detection results.
    Optionally enables object tracking across the images in the batch.
    WARNING: If is_tracking is True, this function treats the list of images
    as a TEMPORAL SEQUENCE and uses a single tracker instance updated
    iteratively. This will produce incorrect tracking results if the images
    are not a strict sequence of frames. If you just need to annotate
    independent images in a batch, set is_tracking=False (default).

    Args:
        images: List of input images (RGB format).
        boxes_batch: List of bounding boxes arrays (one per image).
        scores_batch: List of score arrays (one per image).
        labels_batch: List of label arrays (one per image).
        class_names: Dictionary mapping class IDs to names for labels.
        conf_threshold: Confidence threshold for filtering detections.
                        Applied BEFORE tracking if enabled.
        thickness: Line thickness for boxes.
        font_size: Font size for labels.
        is_tracking: If True, applies ByteTrack tracking across the images
                     in the batch, treating the batch as a sequence. Use
                     with caution for non-sequential image batches.

    Returns:
        List[np.ndarray]: List of annotated images.
    """
    # Create annotators (reused for all images in the batch)
    box_annotator = BoxAnnotator(thickness=thickness)
    text_annotator = RichLabelAnnotator(
        font_size=font_size,
        border_radius=5,
        text_padding=5,
        text_position=Position.TOP_CENTER,
        ensure_in_frame=True,
    )

    # Initialize a tracker only if tracking is enabled.
    # This single tracker is updated iteratively for each image in the batch.
    # WARNING: This assumes the batch is a temporal sequence if tracking is True.
    tracker = (
        _get_or_create_global_tracker()
        if is_tracking
        else None
    )  # type: ignore

    annotated_images = []

    for image, boxes, scores, labels in zip(
        images, boxes_batch, scores_batch, labels_batch
    ):
        # Filter by confidence BEFORE creating sv_detections
        mask = scores > conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]

        # If no detections above threshold, append a copy of the original image
        # If tracking is enabled, update the tracker with empty detections.
        if not np.any(mask):
            annotated_images.append(image.copy())
            if tracker is not None:
                tracker.update_with_detections(sv.Detections.empty())  # type: ignore
            continue

        # Create Supervision Detections object from filtered data
        sv_detections = Detections(
            xyxy=filtered_boxes,
            confidence=filtered_scores,
            class_id=filtered_labels,
        )

        # Annotate the image using the helper function.
        # Pass the tracker (or None) initialized before the loop.
        annotated_image = _annotate_single_image_with_sv_detections(
            image=image.copy(),  # Pass a copy to keep original image untouched
            sv_detections=sv_detections,
            class_names=class_names,
            box_annotator=box_annotator,
            text_annotator=text_annotator,
            tracker=tracker,  # Pass the tracker (or None)
        )

        annotated_images.append(annotated_image)

    return annotated_images


def apply_detection_preprocessing(
    boxes: np.ndarray,  # xyxy format
    ratio: float,
    pad_w: int,
    pad_h: int,
    original_width: int,
    original_height: int,
) -> np.ndarray:
    """
    Apply preprocessing to detection boxes to account for resizing and padding.
    Maps boxes from model output space back to original image space.

    Args:
        boxes: Boxes in xyxy format from model. Can be empty.
        ratio: Resize ratio used during preprocessing.
        pad_w: Padding width.
        pad_h: Padding height.
        original_width: Original image width.
        original_height: Original image height.

    Returns:
        np.ndarray: Boxes adjusted to original image coordinates. Returns an empty
                    array of the same shape if input boxes is empty.
    """
    # Handle None or empty boxes array gracefully
    if boxes is None or boxes.size == 0:
        # Return an empty array with shape (0, 4) if input was valid but empty
        return np.empty((0, 4), dtype=np.float32) if boxes is not None else np.array([])

    adjusted_boxes = boxes.copy().astype(
        np.float32
    )  # Ensure float type for calculations

    # Adjust bounding boxes according to the resizing and padding
    adjusted_boxes[:, 0] = (adjusted_boxes[:, 0] - pad_w) / ratio
    adjusted_boxes[:, 1] = (adjusted_boxes[:, 1] - pad_h) / ratio
    adjusted_boxes[:, 2] = (adjusted_boxes[:, 2] - pad_w) / ratio
    adjusted_boxes[:, 3] = (adjusted_boxes[:, 3] - pad_h) / ratio

    # Clip to image boundaries (0 to original dimensions)
    adjusted_boxes[:, 0] = np.clip(adjusted_boxes[:, 0], 0, original_width)
    adjusted_boxes[:, 1] = np.clip(adjusted_boxes[:, 1], 0, original_height)
    adjusted_boxes[:, 2] = np.clip(adjusted_boxes[:, 2], 0, original_width)
    adjusted_boxes[:, 3] = np.clip(adjusted_boxes[:, 3], 0, original_height)

    return adjusted_boxes


def apply_detection_preprocessing_batch(
    boxes_batch: List[np.ndarray],  # List of boxes arrays (one per image)
    resize_infos: List[ResizeInfo],  # Resize info for each image
) -> List[np.ndarray]:
    """
    Apply preprocessing to a batch of detection boxes.

    Args:
        boxes_batch: List of boxes arrays (one per image).
        resize_infos: List of resize info objects (one per image).

    Returns:
        List[np.ndarray]: List of adjusted boxes arrays.
    """
    adjusted_boxes_batch = []

    for boxes, resize_info in zip(boxes_batch, resize_infos):
        adjusted_boxes = apply_detection_preprocessing(
            boxes=boxes,
            ratio=resize_info.ratio,
            pad_w=resize_info.pad_w,
            pad_h=resize_info.pad_h,
            original_width=resize_info.original_width,
            original_height=resize_info.original_height,
        )
        adjusted_boxes_batch.append(adjusted_boxes)

    return adjusted_boxes_batch


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int,
) -> Tuple[np.ndarray, ResizeInfo]:
    """
    Resizes an image while maintaining aspect ratio and pads it to a square target size.

    Args:
        image: Input image (NumPy array, HWC format).
        target_size: Target square size (width and height) for resizing and padding.

    Returns:
        Tuple[np.ndarray, ResizeInfo]: Resized and padded image, and resize information.
    """
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions while preserving aspect ratio
    ratio = min(target_size / original_width, target_size / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize image
    # Ensure new dimensions are at least 1 pixel to avoid errors with tiny images
    new_width = max(1, new_width)
    new_height = max(1, new_height)

    resized_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )  # Using INTER_AREA for shrinking

    # Calculate padding
    pad_w = (target_size - new_width) // 2
    pad_h = (target_size - new_height) // 2

    # Create padded image
    # Ensure padded_image has correct dimensions based on target_size
    padded_image = np.full(
        (target_size, target_size, image.shape[2]),
        114,
        dtype=image.dtype,  # Use a common padding color like 114 for models
    )
    # Place the resized image onto the padded background
    padded_image[pad_h : pad_h + new_height, pad_w : pad_w + new_width] = resized_image

    resize_info = ResizeInfo(
        ratio=ratio,
        pad_w=pad_w,
        pad_h=pad_h,
        original_height=original_height,
        original_width=original_width,
    )

    return padded_image, resize_info


def resize_batch_with_aspect_ratio(
    images: List[np.ndarray], target_size: int
) -> Tuple[List[np.ndarray], List[ResizeInfo]]:
    """
    Resizes a batch of images while maintaining aspect ratio and pads them.

    Args:
        images: List of input images (NumPy arrays, HWC format).
        target_size: Target size for resizing.

    Returns:
        Tuple[List[np.ndarray], List[ResizeInfo]]: List of resized images and their resize information.
    """
    resized_images = []
    resize_infos = []

    for image in images:
        resized_image, resize_info = resize_with_aspect_ratio(image, target_size)
        resized_images.append(resized_image)
        resize_infos.append(resize_info)

    return resized_images, resize_infos
