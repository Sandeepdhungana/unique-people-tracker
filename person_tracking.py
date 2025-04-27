import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Use TorchVision models instead of YOLO
def main():
    # Use TorchVision's Faster R-CNN model
    print("Loading Faster R-CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    model.to(device).eval()
    
    # COCO class labels (only need person which is index 1 in COCO)
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
    
    # Initialize Person Re-ID system
    reid = PersonReID(height_weight=0.3, appearance_weight=0.7, similarity_threshold=0.6)
    
    # Try to load previously saved features
    try:
        reid.load_features()
        print("Loaded saved person features")
    except:
        print("No saved features found, starting fresh")
    
    # Initialize video source (0 for webcam, or provide video path)
    video_source = "test.mp4"  # Change this to video file path if needed
    cap = cv2.VideoCapture(video_source)
    
    # Initialize supervision annotators
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
    
    # Main loop
    try:
        print("Starting video capture...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prepare image for model
            image_tensor = transform_image(frame)
            
            # Run inference
            with torch.no_grad():
                predictions = model([image_tensor])
            
            # Process predictions
            detections = process_predictions(predictions)
            
            # Track objects using ByteTrack
            detections = byte_tracker.update_with_detections(detections)
            
            # Apply person re-identification
            detections = reid.update(frame, detections)
            
            # Create annotations
            labels = [
                f"Person {tracker_id}" 
                for tracker_id in detections.tracker_id
            ]
            
            # Draw bounding boxes
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections,
                labels=labels
            )
            
            # Display the frame
            cv2.imshow("Person Tracking with Re-ID", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Save person features for future use
        reid.save_features()
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

class PersonReID:
    def __init__(self, height_weight=0.3, appearance_weight=0.7, similarity_threshold=0.6):
        """
        Initialize Person Re-identification system
        Args:
            height_weight: Weight for height feature in similarity calculation
            appearance_weight: Weight for appearance feature in similarity calculation
            similarity_threshold: Threshold to consider persons as the same
        """
        self.person_features = {}  # Store features for each tracked ID
        self.height_weight = height_weight
        self.appearance_weight = appearance_weight
        self.similarity_threshold = similarity_threshold
        self.id_mapping = {}  # Maps ByteTrack IDs to persistent IDs
        self.next_persistent_id = 1
        
    def extract_appearance_features(self, image, box):
        """Extract color histogram features from person crop"""
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:  # Invalid box
            return np.zeros(3*30)  # Return zeros if box is invalid
            
        person_img = image[y1:y2, x1:x2]
        
        # Split into 3 vertical parts: upper body, middle, lower body
        height = person_img.shape[0]
        upper = person_img[:height//3]
        middle = person_img[height//3:2*height//3]
        lower = person_img[2*height//3:]
        
        # Extract color histograms for each section
        features = []
        for section in [upper, middle, lower]:
            if section.size == 0:
                hist = np.zeros(30)
            else:
                hist = cv2.calcHist([section], [0, 1, 2], None, [10, 10, 10], 
                                   [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()[:30]  # Use only first 30 bins
            features.extend(hist)
            
        return np.array(features)
    
    def calculate_similarity(self, features1, height1, features2, height2):
        """Calculate similarity between two persons based on appearance and height"""
        appearance_sim = cosine_similarity([features1], [features2])[0][0]
        
        # Normalize height difference (0 = identical, 1 = very different)
        height_diff = abs(height1 - height2) / max(height1, height2)
        height_sim = 1 - min(1.0, height_diff)
        
        # Combined similarity
        similarity = (self.appearance_weight * appearance_sim + 
                     self.height_weight * height_sim)
        
        return similarity
    
    def update(self, image, detections):
        """
        Update person re-identification with new detections
        Returns updated detections with persistent IDs
        """
        # Create a copy of the detections to modify
        # Since supervision Detections doesn't have a copy method, we need to create a new one
        updated_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if detections.confidence is not None else None,
            class_id=detections.class_id.copy() if detections.class_id is not None else None,
            tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None,
            mask=detections.mask.copy() if detections.mask is not None else None
        )
        
        if updated_detections.tracker_id is None:
            return updated_detections
            
        updated_tracker_ids = updated_detections.tracker_id.copy()
        
        for i, (xyxy, tracker_id, confidence, class_id) in enumerate(
            zip(updated_detections.xyxy, updated_detections.tracker_id, 
                updated_detections.confidence, updated_detections.class_id)):
            
            if tracker_id is None or class_id != 0:  # Skip if not a person or no tracker ID
                continue
                
            # Calculate person height (in pixels)
            height = xyxy[3] - xyxy[1]
            
            # Extract appearance features
            features = self.extract_appearance_features(image, xyxy)
            
            # Check if we've seen this tracker_id before
            if tracker_id in self.id_mapping:
                persistent_id = self.id_mapping[tracker_id]
                # Update features for this persistent ID
                old_features, old_height = self.person_features[persistent_id]
                
                # Update with running average
                alpha = 0.7  # Weight for new features
                updated_features = (1-alpha) * old_features + alpha * features
                updated_height = (1-alpha) * old_height + alpha * height
                
                self.person_features[persistent_id] = (updated_features, updated_height)
            else:
                # New tracker_id, check if it matches any known person who left the frame
                best_match_id = None
                best_similarity = 0
                
                for p_id, (p_features, p_height) in self.person_features.items():
                    if p_id not in self.id_mapping.values():  # Only consider IDs not currently tracked
                        similarity = self.calculate_similarity(features, height, p_features, p_height)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_id = p_id
                
                # If similarity is above threshold, consider it the same person
                if best_match_id is not None and best_similarity > self.similarity_threshold:
                    persistent_id = best_match_id
                    
                    # Update features with running average
                    old_features, old_height = self.person_features[persistent_id]
                    alpha = 0.3  # Weight for new features (lower for re-identified persons)
                    updated_features = (1-alpha) * old_features + alpha * features
                    updated_height = (1-alpha) * old_height + alpha * height
                    
                    self.person_features[persistent_id] = (updated_features, updated_height)
                else:
                    # Create a new persistent ID
                    persistent_id = self.next_persistent_id
                    self.next_persistent_id += 1
                    self.person_features[persistent_id] = (features, height)
                
                # Map tracker_id to persistent_id
                self.id_mapping[tracker_id] = persistent_id
            
            # Update detection with persistent ID
            updated_tracker_ids[i] = self.id_mapping[tracker_id]
        
        # Update detection object with persistent IDs
        updated_detections.tracker_id = updated_tracker_ids
        return updated_detections
    
    def save_features(self, filename="person_features.pkl"):
        """Save person features to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'person_features': self.person_features,
                'id_mapping': self.id_mapping,
                'next_persistent_id': self.next_persistent_id
            }, f)
    
    def load_features(self, filename="person_features.pkl"):
        """Load person features from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.person_features = data['person_features']
                self.id_mapping = data['id_mapping']
                self.next_persistent_id = data['next_persistent_id']

if __name__ == "__main__":
    main() 