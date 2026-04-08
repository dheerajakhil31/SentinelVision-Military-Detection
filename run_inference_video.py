#!/usr/bin/env python
"""Run a model on a video and save annotated output.

Supports:
- Ultralytics YOLOv8 `.pt` models (recommended for object detection)
- Keras `.h5` image-classifier models (frame-level classification)

Usage:
    python run_inference_video.py --model yolov8n.pt --source input.mp4 --output out.mp4

Requirements (see `requirements_video.txt`): ultralytics, opencv-python, numpy, tensorflow (if using .h5)
"""

import argparse
import time
import os
import sys
import cv2
import numpy as np
import json
from collections import Counter

def run_yolo(model_path, source, output, conf=0.25, device='cpu', show=False):
    detection_counts = Counter()
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('Ultralytics not available:', e)
        print('Install with: pip install ultralytics opencv-python')
        sys.exit(1)

    model = YOLO(model_path)
    # model.names may be dict or list
    try:
        names = model.names
    except Exception:
        try:
            names = model.model.names
        except Exception:
            names = {}

    is_video = source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open source: {source}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if is_video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_idx = 0
    t0 = time.time()
    print(f'Starting inference (YOLO) on {"video" if is_video else "image"}. Press Ctrl+C to stop.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ultralytics accepts RGB images
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = model(img, conf=conf, device=device)  # returns Results
            res = results[0]
        except Exception as e:
            # Fallback: try predict API
            results = model.predict(img, conf=conf, device=device)
            res = results[0]

        boxes = getattr(res, 'boxes', None)
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                # boxes may already be numpy
                xyxy = np.array(boxes.xyxy)
                confs = np.array(boxes.conf)
                cls = np.array(boxes.cls).astype(int)

            # Tactical naming
            CLASS_MAP = {
                "military_tank": "Tank",
                "military_truck": "Truck",
                "military_vehicle": "Vehicle",
                "military_aircraft": "Aircraft",
                "military_artillery": "Artillery",
                "soldier": "Soldier",
                "camouflage_soldier": "Camouflage Soldier",
                "weapon": "Weapon",
                "civilian": "Civilian",
                "civilian_vehicle": "Civilian Vehicle",
                "trench": "Trench",
                "military_warship": "Warship"
            }
            for (x1, y1, x2, y2), c, cl in zip(xyxy, confs, cls):
                if c < 0.4:
                    continue
                # Using strategic tactical colors
                class_name = names.get(cl, cl) if isinstance(names, dict) else names[cl]
                class_name = CLASS_MAP.get(class_name, class_name)
                
                # Signal color: Green for personnel, Blue for hardware
                color = (0, 255, 65) if "Soldier" in class_name else (255, 122, 0)
                
                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                
                # Draw sharp tactical box
                cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, 1)
                
                # Draw label with semi-transparent background effect
                lbl = f"{class_name.upper()} | {c:.2f}"
                t_size = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1i, y1i - t_size[1] - 10), (x1i + t_size[0], y1i), color, -1)
                cv2.putText(frame, lbl, (x1i, y1i - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Track detections per frame
                if class_name in ["Tank", "Vehicle", "Truck"]:
                    detection_counts["Vehicles"] += 1
                elif "Soldier" in class_name:
                    detection_counts["Personnel"] += 1
                elif class_name in ["Weapon", "Artillery"]:
                    detection_counts["Weapons"] += 1
                elif class_name == "Aircraft":
                    detection_counts["Aircraft"] += 1
                else:
                    detection_counts[class_name] += 1

        if writer:
            writer.write(frame)
        else:
            # Support for single image save
            cv2.imwrite(output, frame)
            
        if show:
            cv2.imshow('Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    t1 = time.time()
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = t1 - t0
    print(f'Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)')
    
    # Output detection metrics as JSON for the dashboard
    metrics = {
        "summary": dict(detection_counts),
        "total_frames": frame_idx,
        "elapsed_time": elapsed
    }
    print(f"DETECTION_METRICS_START{json.dumps(metrics)}DETECTION_METRICS_END")


def run_keras_classifier(model_path, source, output, input_size=(224,224), class_names=None, show=False):
    try:
        import tensorflow as tf
    except Exception as e:
        print('TensorFlow not available:', e)
        print('Install with: pip install tensorflow opencv-python')
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    # try to infer input size
    try:
        ishape = model.input_shape
        if ishape and len(ishape) >= 3:
            input_size = (ishape[1], ishape[2])
    except Exception:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {source}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    frame_idx = 0
    t0 = time.time()
    print('Starting inference (Keras classifier). Press Ctrl+C to stop.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        preds = model.predict(np.expand_dims(img, 0))
        idx = int(np.argmax(preds[0]))
        label = (class_names[idx] if class_names and idx < len(class_names) else str(idx))
        cv2.putText(frame, f'{label} {preds[0][idx]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,220,50), 2)

        writer.write(frame)
        if show:
            cv2.imshow('Inference', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    t1 = time.time()
    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = t1 - t0
    print(f'Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model file (.pt for YOLOv8 or .h5 for Keras)')
    parser.add_argument('--source', required=True, help='Path to input video file')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold (YOLO)')
    parser.add_argument('--device', default='cpu', help='Device for inference (cpu or gpu:0)')
    parser.add_argument('--show', action='store_true', help='Show frames while processing')
    parser.add_argument('--class-names', required=False, help='Optional path to a text file with class names for classifier (one per line)')
    parser.add_argument('--output', required=True, help='Path to output file')
    args = parser.parse_args()

    model_path = args.model
    source = args.source
    output = args.output

    class_names = None
    if args.class_names:
        if os.path.exists(args.class_names):
            with open(args.class_names, 'r', encoding='utf-8') as f:
                class_names = [l.strip() for l in f.readlines() if l.strip()]

    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.pt' or 'yolov8' in os.path.basename(model_path).lower():
        run_yolo(model_path, source, output, conf=args.conf, device=args.device, show=args.show)
    elif ext in ('.h5', '.keras'):
        run_keras_classifier(model_path, source, output, class_names=class_names, show=args.show)
    else:
        print('Unsupported model type. Provide a .pt (YOLOv8) or .h5 (Keras) model.')


if __name__ == '__main__':
    main()
