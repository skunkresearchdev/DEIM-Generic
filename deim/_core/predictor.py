"""
Inference module for DEIM
Handles prediction on images, videos, and batches
"""

import sys
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2


class Predictor:
    """
    Inference handler for DEIM models

    Handles:
    - Single image inference
    - Batch image inference
    - Video inference
    - Visualization with supervision
    """

    def __init__(self, config: Dict[str, Any], checkpoint_path: str, device: torch.device):
        """
        Initialize predictor with model

        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
            device: PyTorch device
        """
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # Set default image size
        self.img_size = config.get('img_size', 640)

        # Initialize supervision for visualization if available
        try:
            import supervision as sv
            self.supervision_available = True
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
        except ImportError:
            self.supervision_available = False
            print("⚠️  Supervision not installed. Visualization disabled.")

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint"""

        # Add engine to path
        engine_path = Path(__file__).parent.parent / "_engine"
        if str(engine_path) not in sys.path:
            sys.path.insert(0, str(engine_path))

        try:
            # Import DEIM model
            from deim import DEIM as DEIMModel
            from core import YAMLConfig

            # Create model from config
            model_config = self.config.get('DEIM', {})

            # Initialize model
            model = DEIMModel(**model_config)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume checkpoint is the state dict
                model.load_state_dict(checkpoint)

            model = model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Attempting alternative loading method...")

            # Alternative: Load as generic PyTorch model
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Try to reconstruct model from checkpoint
            if 'model' in checkpoint:
                model = checkpoint['model']
            else:
                raise RuntimeError(f"Could not load model from checkpoint: {checkpoint_path}")

            return model.to(self.device)

    def predict(self,
                sources: Union[str, List[str]],
                conf_threshold: float = 0.4,
                visualize: bool = False,
                save_path: Optional[str] = None,
                save_dir: Optional[str] = None) -> Union[Dict, List[Dict]]:
        """
        Run inference on sources

        Args:
            sources: Image path(s) or video path
            conf_threshold: Confidence threshold
            visualize: Whether to visualize results
            save_path: Path to save single output
            save_dir: Directory to save batch outputs

        Returns:
            Detection results
        """

        # Handle different source types
        if isinstance(sources, str):
            source_path = Path(sources)

            if source_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                # Video inference
                return self._predict_video(
                    sources, conf_threshold, visualize, save_path
                )
            else:
                # Single image
                results = self._predict_image(
                    sources, conf_threshold, visualize
                )

                if visualize and save_path:
                    self._save_image(results['visualization'], save_path)

                return results

        elif isinstance(sources, list):
            # Batch inference
            return self._predict_batch(
                sources, conf_threshold, visualize, save_dir
            )

    def _predict_image(self,
                      image_path: str,
                      conf_threshold: float,
                      visualize: bool) -> Dict[str, Any]:
        """Predict on single image"""

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)

        # Resize image
        resized, scale = self._resize_image(image_np, self.img_size)

        # Convert to tensor
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(resized).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Process outputs
        results = self._process_outputs(outputs, scale, conf_threshold)

        # Add visualization if requested
        if visualize and self.supervision_available:
            results['visualization'] = self._visualize_detections(
                image_np, results
            )

        results['image_path'] = image_path
        results['image_size'] = image_np.shape[:2]

        return results

    def _predict_batch(self,
                      image_paths: List[str],
                      conf_threshold: float,
                      visualize: bool,
                      save_dir: Optional[str]) -> List[Dict[str, Any]]:
        """Predict on batch of images"""

        results = []

        for idx, image_path in enumerate(image_paths):
            print(f"  Processing {idx + 1}/{len(image_paths)}: {image_path}")

            result = self._predict_image(image_path, conf_threshold, visualize)
            results.append(result)

            if visualize and save_dir and 'visualization' in result:
                save_path = Path(save_dir) / f"{Path(image_path).stem}_pred.jpg"
                self._save_image(result['visualization'], str(save_path))

        return results

    def _predict_video(self,
                      video_path: str,
                      conf_threshold: float,
                      visualize: bool,
                      save_path: Optional[str]) -> Dict[str, Any]:
        """Predict on video"""

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = {
            'video_path': video_path,
            'fps': fps,
            'resolution': (width, height),
            'total_frames': total_frames,
            'frame_results': []
        }

        # Setup video writer if saving
        if visualize and save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            frame_result = self._process_frame(
                frame_rgb, conf_threshold, visualize
            )
            frame_result['frame_idx'] = frame_idx
            results['frame_results'].append(frame_result)

            # Write visualized frame if requested
            if visualize and save_path and 'visualization' in frame_result:
                vis_frame = frame_result['visualization']
                vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
                out.write(vis_frame_bgr)

            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"    Processed {frame_idx}/{total_frames} frames")

        cap.release()
        if visualize and save_path:
            out.release()

        print(f"  ✓ Processed {frame_idx} frames")

        return results

    def _process_frame(self,
                      frame: np.ndarray,
                      conf_threshold: float,
                      visualize: bool) -> Dict[str, Any]:
        """Process single video frame"""

        # Resize frame
        resized, scale = self._resize_image(frame, self.img_size)

        # Convert to tensor
        transform = T.Compose([T.ToTensor()])
        frame_tensor = transform(resized).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(frame_tensor)

        # Process outputs
        results = self._process_outputs(outputs, scale, conf_threshold)

        # Add visualization if requested
        if visualize and self.supervision_available:
            results['visualization'] = self._visualize_detections(frame, results)

        return results

    def _resize_image(self, image: np.ndarray, target_size: int):
        """Resize image maintaining aspect ratio"""

        h, w = image.shape[:2]
        scale = target_size / max(h, w)

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to square
        pad_w = target_size - new_w
        pad_h = target_size - new_h

        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return padded, scale

    def _process_outputs(self,
                        outputs: torch.Tensor,
                        scale: float,
                        conf_threshold: float) -> Dict[str, Any]:
        """Process model outputs to detection format"""

        # This depends on the exact output format of DEIM model
        # Assuming outputs are (labels, boxes, scores) or similar

        if isinstance(outputs, tuple) and len(outputs) == 3:
            labels, boxes, scores = outputs
        else:
            # Handle different output formats
            # This may need adjustment based on actual DEIM output
            boxes = outputs[..., :4]
            scores = outputs[..., 4]
            labels = outputs[..., 5:].argmax(-1)

        # Convert to numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

        # Apply confidence threshold
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

        # Rescale boxes
        boxes = boxes / scale

        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'num_detections': len(boxes)
        }

    def _visualize_detections(self,
                            image: np.ndarray,
                            results: Dict[str, Any]) -> np.ndarray:
        """Visualize detections using supervision"""

        if not self.supervision_available:
            return image

        try:
            import supervision as sv

            # Create detections object
            detections = sv.Detections(
                xyxy=results['boxes'],
                confidence=results['scores'],
                class_id=results['labels'].astype(int)
            )

            # Get class names (if available in config)
            class_names = self.config.get('class_names', {})
            labels = [
                f"{class_names.get(int(class_id), f'Class {class_id}')} {score:.2f}"
                for class_id, score in zip(results['labels'], results['scores'])
            ]

            # Annotate image
            annotated = self.box_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            annotated = self.label_annotator.annotate(
                scene=annotated, detections=detections, labels=labels
            )

            return annotated

        except Exception as e:
            print(f"Visualization error: {str(e)}")
            return image

    def _save_image(self, image: np.ndarray, save_path: str):
        """Save image to file"""
        Image.fromarray(image).save(save_path)
        print(f"  Saved: {save_path}")