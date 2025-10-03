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
            self.label_annotator = sv.LabelAnnotator(smart_position=True)
        except ImportError:
            self.supervision_available = False
            print("⚠️  Supervision not installed. Visualization disabled.")

    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model from checkpoint"""

        # By importing these, we are registering the modules in the workspace
        import deim._engine.backbone
        import deim._engine.deim

        try:
            from deim._engine.core.yaml_config import YAMLConfig
            import yaml
            import tempfile
            import os
            from collections import OrderedDict

            # YAMLConfig needs a config file path. We have a dict.
            # So, we write the dict to a temporary file.
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yml') as f:
                yaml.dump(self.config, f)
                temp_config_path = f.name
            
            # The YAMLConfig class modifies the output_dir by adding a timestamp.
            # To avoid creating unwanted directories, we can point it to a temp dir.
            # However, let's first try without this and see if it's a problem.
            # The config from the API should have 'output_dir' set.
            
            # Create YAMLConfig object. This will also handle model creation.
            cfg = YAMLConfig(temp_config_path)
            model = cfg.model
            
            os.remove(temp_config_path)

            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract state dict from checkpoint
            if 'ema' in checkpoint and checkpoint['ema'] is not None:
                state_dict = checkpoint['ema']['module']
                print("INFO: Loading EMA weights for inference.")
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Clean state dict keys if they are from a parallelized model
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)

            model = model.to(self.device)
            model.eval()

            return model

        except Exception as e:
            print(f"❌ Failed to load model dynamically: {e}")
            print("   Please ensure your checkpoint and config are compatible.")
            raise e

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
        resized, scale, pad_info = self._resize_image(image_np, self.img_size)

        # Convert to tensor
        transform = T.Compose([T.ToTensor()])
        image_tensor = transform(resized).unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Process outputs - pass original image shape and padding info
        orig_h, orig_w = image_np.shape[:2]
        results = self._process_outputs(outputs, (orig_w, orig_h), scale, pad_info, conf_threshold)

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
        """Resize image maintaining aspect ratio

        Returns:
            padded: Padded square image
            scale: Scaling factor applied
            pad_info: Dict with padding offsets {'top', 'left'}
        """

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

        pad_info = {'top': top, 'left': left}
        return padded, scale, pad_info

    def _process_outputs(self,
                        outputs: Dict[str, torch.Tensor],
                        orig_size: tuple,
                        scale: float,
                        pad_info: dict,
                        conf_threshold: float) -> Dict[str, Any]:
        """Process model outputs to detection format

        Args:
            outputs: Model outputs dict with 'pred_logits' and 'pred_boxes'
            orig_size: Original image size as (width, height) - COCO format
            scale: Scaling factor used during resize
            pad_info: Padding information {'top': int, 'left': int}
            conf_threshold: Confidence threshold for filtering
        """

        # DEIM returns dict with 'pred_logits' and 'pred_boxes'
        # Boxes are relative to the padded 640x640 image
        # Need to: 1) scale to 640x640, 2) remove padding, 3) scale to original

        # Import postprocessor
        from deim._engine.deim.postprocessor import PostProcessor

        # Initialize postprocessor if not already done
        if not hasattr(self, 'postprocessor'):
            self.postprocessor = PostProcessor(
                num_classes=self.config.get('num_classes', 80),
                use_focal_loss=self.config.get('use_focal_loss', True),
                num_top_queries=self.config.get('num_top_queries', 300)
            )
            # Enable deploy mode for tuple output
            self.postprocessor.deploy()

        # Use padded image size (640x640) for postprocessor
        batch_size = outputs['pred_logits'].shape[0]
        padded_size = torch.tensor([[self.img_size, self.img_size]] * batch_size,
                                   dtype=torch.float32,
                                   device=self.device)

        # Apply postprocessor - returns (labels, boxes, scores) when in deploy mode
        # Boxes are now in pixel coordinates relative to 640x640 padded image
        labels, boxes, scores = self.postprocessor(outputs, padded_size)

        # Convert to numpy and get first batch element
        labels = labels[0].cpu().numpy()
        boxes = boxes[0].cpu().numpy()
        scores = scores[0].cpu().numpy()

        # Remove padding offset - boxes are in xyxy format
        # [x1, y1, x2, y2] relative to padded image
        boxes[:, [0, 2]] -= pad_info['left']  # x coordinates
        boxes[:, [1, 3]] -= pad_info['top']   # y coordinates

        # Scale back to original image size
        # Boxes are currently relative to resized image (after removing padding)
        boxes /= scale

        # Apply confidence threshold
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]

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