import argparse
import cv2
import time
import numpy as np
import supervision as sv
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
import os
from improved_reid import DeepPersonReID

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced person tracking with Deep Re-ID")
    parser.add_argument("--input", "-i", type=str, default="0", 
                        help="Path to video file or camera index (default: 0 for webcam)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output video file (default: None)")
    parser.add_argument("--appearance-weight", type=float, default=0.6, 
                        help="Weight for appearance feature (default: 0.6)")
    parser.add_argument("--pose-weight", type=float, default=0.2,
                        help="Weight for pose feature (default: 0.2)")
    parser.add_argument("--height-weight", type=float, default=0.2,
                        help="Weight for height feature (default: 0.2)")
    parser.add_argument("--similarity-threshold", type=float, default=0.6,
                        help="Threshold to consider persons as the same (default: 0.6)")
    parser.add_argument("--save-features", type=str, default="deep_person_features.pkl",
                        help="Path to save person features (default: deep_person_features.pkl)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections (default: 0.5)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for model inference if available")
    args = parser.parse_args()
    
    # Initialize video source
    if args.input.isdigit():
        video_source = int(args.input)
    else:
        video_source = args.input
    
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.input}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize video writer if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Use TorchVision's Faster R-CNN model for detection
    print("Loading Faster R-CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.to(device).eval()
    
    # COCO class labels
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Person class index
    person_class_idx = COCO_INSTANCE_CATEGORY_NAMES.index('person')
    
    # Initialize ByteTrack tracker
    byte_tracker = sv.ByteTrack()
    
    # Initialize our advanced Deep Person Re-ID system
    reid = DeepPersonReID(
        appearance_weight=args.appearance_weight,
        pose_weight=args.pose_weight,
        height_weight=args.height_weight,
        similarity_threshold=args.similarity_threshold,
        use_gpu=args.use_gpu
    )
    
    # Try to load previously saved features
    try:
        reid.load_features(args.save_features)
    except Exception as e:
        print(f"No saved features found, starting fresh: {e}")
    
    # Initialize annotation tools
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    
    # Function to transform image for the model
    def transform_image(image):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to tensor and normalize
        image_tensor = F.to_tensor(image_rgb)
        return image_tensor.to(device)
    
    # Function to convert model output to supervision Detections
    def process_predictions(predictions, confidence_threshold=0.5):
        boxes = []
        scores = []
        class_ids = []
        
        pred_boxes = predictions[0]['boxes'].cpu().detach().numpy()
        pred_scores = predictions[0]['scores'].cpu().detach().numpy()
        pred_labels = predictions[0]['labels'].cpu().detach().numpy()
        
        # Filter predictions by confidence and class (person)
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > confidence_threshold and label == person_class_idx:
                boxes.append(box)
                scores.append(score)
                class_ids.append(0)  # Using 0 for person to match supervision convention
        
        if not boxes:
            return sv.Detections.empty()
            
        return sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(scores),
            class_id=np.array(class_ids),
        )
    
    # Statistics
    frame_count = 0
    start_time = time.time()
    
    # Main loop
    try:
        print("Starting video capture...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Prepare image for model
            image_tensor = transform_image(frame)
            
            # Run inference
            with torch.no_grad():
                predictions = model([image_tensor])
            
            # Process predictions
            detections = process_predictions(predictions, args.confidence)
            
            # Track objects using ByteTrack
            detections = byte_tracker.update_with_detections(detections)
            
            # Apply advanced person re-identification
            detections = reid.update(frame, detections)
            
            # Create annotations with colored IDs
            labels = []
            
            if detections.tracker_id is not None:
                for tracker_id in detections.tracker_id:
                    if tracker_id is not None:
                        # Create label with person ID
                        person_id = int(tracker_id)
                        labels.append(f"Person {person_id}")
                    else:
                        labels.append("")
            
            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            fps_text = f"FPS: {frame_count / elapsed_time:.1f}"
            
            # Draw bounding boxes with custom colors
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )
            
            # Add FPS counter and tracking info
            cv2.putText(
                frame, fps_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Add count of unique people seen
            unique_count = len(reid.person_features)
            cv2.putText(
                frame, f"Unique people: {unique_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Write frame to output video if specified
            if writer:
                writer.write(frame)
            
            # Display the frame
            cv2.imshow("Advanced Person Tracking with Deep Re-ID", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Save person features for future use
        reid.save_features(args.save_features)
        
        # Print statistics
        elapsed_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")
        print(f"Tracked {len(reid.person_features)} unique people")
        
        # Release resources
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 