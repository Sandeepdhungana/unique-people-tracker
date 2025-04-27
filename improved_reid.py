import cv2
import numpy as np
import torch
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torchvision
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

class DeepPersonReID:
    def __init__(self, 
                 appearance_weight=0.6, 
                 pose_weight=0.2,
                 height_weight=0.2, 
                 similarity_threshold=0.6,
                 use_gpu=True):
        """
        Initialize Deep Person Re-identification system
        Args:
            appearance_weight: Weight for appearance feature in similarity calculation
            pose_weight: Weight for pose feature in similarity calculation
            height_weight: Weight for height feature in similarity calculation
            similarity_threshold: Threshold to consider persons as the same
            use_gpu: Whether to use GPU for inference
        """
        self.person_features = {}  # Store features for each tracked ID
        self.appearance_weight = appearance_weight
        self.pose_weight = pose_weight
        self.height_weight = height_weight
        self.similarity_threshold = similarity_threshold
        self.id_mapping = {}  # Maps tracker IDs to persistent IDs
        self.next_persistent_id = 1
        self.recent_trajectories = {}  # Store recent positions for trajectory analysis
        self.max_trajectory_length = 30  # Number of frames to keep trajectory history
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize Deep feature extraction model (using ResNet50 instead of OSNet)
        self.setup_model()
        
        # Initialize image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard size for person ReID
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def setup_model(self):
        """Initialize the deep learning model for feature extraction"""
        print("Setting up ResNet50 model for feature extraction...")
        # Use ResNet50 with pretrained weights
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        
        # Remove the classification layer to get features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device).eval()
        print("Model initialized successfully")

    def extract_deep_features(self, image, box):
        """Extract deep features from person crop using ReID model"""
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:  # Skip tiny or invalid boxes
            return np.zeros(512)  # Return zeros with expected feature dimension
            
        # Crop person from image
        person_img = image[y1:y2, x1:x2]
        
        # Convert from BGR to RGB (OpenCV uses BGR, but our model expects RGB)
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        try:
            tensor = self.transform(person_img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(tensor)
                
            # Return as numpy array
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(512)  # Return zeros with expected feature dimension
    
    def extract_pose_features(self, image, box):
        """
        Extract pose-based features from the person
        This is a placeholder - in a real implementation you would use a pose estimation model
        """
        # TODO: Implement actual pose estimation
        # For now, we'll use a simple proxy based on aspect ratio and size
        x1, y1, x2, y2 = [int(coord) for coord in box]
        width = x2 - x1
        height = y2 - y1
        
        # Aspect ratio can capture some pose information (standing vs. sitting)
        aspect_ratio = width / height if height > 0 else 0
        
        # Simple placeholder for pose features
        return np.array([aspect_ratio, width, height])
    
    def update_trajectory(self, tracker_id, box):
        """Update trajectory information for a person"""
        if tracker_id not in self.recent_trajectories:
            self.recent_trajectories[tracker_id] = []
            
        # Get center point of bounding box
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Add to trajectory history
        self.recent_trajectories[tracker_id].append((center_x, center_y))
        
        # Limit trajectory length
        if len(self.recent_trajectories[tracker_id]) > self.max_trajectory_length:
            self.recent_trajectories[tracker_id].pop(0)
    
    def calculate_trajectory_similarity(self, trajectory1, trajectory2):
        """
        Calculate similarity between two trajectories
        Returns 0 if either trajectory is too short
        """
        if len(trajectory1) < 5 or len(trajectory2) < 5:
            return 0
            
        # Use the last 5 points of each trajectory
        t1 = trajectory1[-5:]
        t2 = trajectory2[-5:]
        
        # Calculate direction vectors between consecutive points
        def get_directions(traj):
            directions = []
            for i in range(1, len(traj)):
                dx = traj[i][0] - traj[i-1][0]
                dy = traj[i][1] - traj[i-1][1]
                # Normalize
                mag = np.sqrt(dx*dx + dy*dy)
                if mag > 0:
                    directions.append((dx/mag, dy/mag))
                else:
                    directions.append((0, 0))
            return directions
            
        dir1 = get_directions(t1)
        dir2 = get_directions(t2)
        
        # Calculate cosine similarity between direction vectors
        if not dir1 or not dir2:
            return 0
            
        similarity_sum = 0
        for d1, d2 in zip(dir1, dir2):
            dot_product = d1[0]*d2[0] + d1[1]*d2[1]
            similarity_sum += (dot_product + 1) / 2  # Convert from [-1,1] to [0,1]
            
        return similarity_sum / len(dir1)
    
    def calculate_combined_similarity(self, appearance1, pose1, height1, trajectory1,
                                      appearance2, pose2, height2, trajectory2):
        """Calculate overall similarity between two persons using multiple features"""
        # Appearance similarity (deep features)
        appearance_sim = cosine_similarity([appearance1], [appearance2])[0][0]
        
        # Pose similarity
        if pose1 is not None and pose2 is not None:
            # Normalize pose features
            pose_sim = 1 - np.linalg.norm(pose1 - pose2) / (np.linalg.norm(pose1) + np.linalg.norm(pose2) + 1e-6)
        else:
            pose_sim = 0
        
        # Height similarity
        height_diff = abs(height1 - height2) / max(height1, height2)
        height_sim = 1 - min(1.0, height_diff)
        
        # Trajectory similarity
        traj_sim = self.calculate_trajectory_similarity(trajectory1, trajectory2)
        
        # Combined similarity with weights
        combined_sim = (
            self.appearance_weight * appearance_sim +
            self.pose_weight * pose_sim + 
            self.height_weight * height_sim
        )
        
        # If trajectories are available, boost similarity for matching trajectories
        if traj_sim > 0.8:  # High trajectory match
            combined_sim = combined_sim * 1.2  # Boost similarity
            combined_sim = min(1.0, combined_sim)  # Cap at 1.0
        
        return combined_sim
    
    def update(self, image, detections):
        """
        Update person re-identification with new detections
        Returns updated detections with persistent IDs
        """
        # Create a copy of the detections
        updated_detections = detections
        
        if updated_detections.tracker_id is None:
            return updated_detections
            
        updated_tracker_ids = updated_detections.tracker_id.copy()
        
        for i, (xyxy, tracker_id, confidence, class_id) in enumerate(
            zip(updated_detections.xyxy, updated_detections.tracker_id, 
                updated_detections.confidence, updated_detections.class_id)):
            
            if tracker_id is None or class_id != 0:  # Skip if not a person or no tracker ID
                continue
                
            # Calculate person height
            height = xyxy[3] - xyxy[1]
            
            # Extract features
            deep_features = self.extract_deep_features(image, xyxy)
            pose_features = self.extract_pose_features(image, xyxy)
            
            # Update trajectory
            self.update_trajectory(tracker_id, xyxy)
            trajectory = self.recent_trajectories.get(tracker_id, [])
            
            # Check if we've seen this tracker_id before
            if tracker_id in self.id_mapping:
                persistent_id = self.id_mapping[tracker_id]
                
                # Update features for this persistent ID with running average
                old_features = self.person_features[persistent_id]
                alpha = 0.7  # Weight for new features
                
                # Update all features
                updated_features = {
                    'appearance': (1-alpha) * old_features['appearance'] + alpha * deep_features,
                    'pose': pose_features,  # Just replace pose
                    'height': (1-alpha) * old_features['height'] + alpha * height,
                    'last_seen': 0,  # Frame counter since last seen
                    'trajectory': trajectory
                }
                
                self.person_features[persistent_id] = updated_features
            else:
                # New tracker_id, check if it matches any known person who left the frame
                best_match_id = None
                best_similarity = 0
                
                # Consider only IDs not currently tracked (not in id_mapping values)
                untracked_ids = [p_id for p_id in self.person_features.keys() 
                                 if p_id not in self.id_mapping.values()]
                
                for p_id in untracked_ids:
                    p_features = self.person_features[p_id]
                    
                    # Calculate similarity between current person and stored person
                    similarity = self.calculate_combined_similarity(
                        deep_features, pose_features, height, trajectory,
                        p_features['appearance'], p_features['pose'], 
                        p_features['height'], p_features['trajectory']
                    )
                    
                    # Adjust similarity based on how long the person has been gone
                    frames_gone = p_features['last_seen']
                    time_factor = max(0, 1 - (frames_gone / 100))  # Decay factor
                    adjusted_similarity = similarity * time_factor
                    
                    if adjusted_similarity > best_similarity:
                        best_similarity = adjusted_similarity
                        best_match_id = p_id
                
                # If similarity is above threshold, consider it the same person
                if best_match_id is not None and best_similarity > self.similarity_threshold:
                    persistent_id = best_match_id
                    
                    # Update features with running average
                    old_features = self.person_features[persistent_id]
                    alpha = 0.3  # Lower weight for re-identified persons
                    
                    # Update features
                    updated_features = {
                        'appearance': (1-alpha) * old_features['appearance'] + alpha * deep_features,
                        'pose': pose_features,  # Just replace pose
                        'height': (1-alpha) * old_features['height'] + alpha * height,
                        'last_seen': 0,  # Reset frame counter
                        'trajectory': trajectory
                    }
                    
                    self.person_features[persistent_id] = updated_features
                    print(f"Re-identified person {persistent_id} with similarity {best_similarity:.2f}")
                else:
                    # Create a new persistent ID
                    persistent_id = self.next_persistent_id
                    self.next_persistent_id += 1
                    
                    # Store all features
                    self.person_features[persistent_id] = {
                        'appearance': deep_features,
                        'pose': pose_features,
                        'height': height,
                        'last_seen': 0,
                        'trajectory': trajectory
                    }
                    
                    print(f"New person detected with ID {persistent_id}")
                
                # Map tracker_id to persistent_id
                self.id_mapping[tracker_id] = persistent_id
            
            # Update detection with persistent ID
            updated_tracker_ids[i] = self.id_mapping[tracker_id]
        
        # Increment last_seen counter for persons not in current frame
        current_p_ids = set(self.id_mapping.values())
        for p_id in self.person_features:
            if p_id not in current_p_ids:
                self.person_features[p_id]['last_seen'] += 1
        
        # Remove tracker IDs that are no longer tracked
        current_tracker_ids = set(updated_detections.tracker_id)
        self.id_mapping = {t_id: p_id for t_id, p_id in self.id_mapping.items() 
                           if t_id in current_tracker_ids}
        
        # Update detection object with persistent IDs
        updated_detections.tracker_id = updated_tracker_ids
        return updated_detections
    
    def save_features(self, filename="deep_person_features.pkl"):
        """Save person features to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'person_features': self.person_features,
                'id_mapping': self.id_mapping,
                'next_persistent_id': self.next_persistent_id
            }, f)
        print(f"Saved {len(self.person_features)} person profiles to {filename}")
    
    def load_features(self, filename="deep_person_features.pkl"):
        """Load person features from file"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.person_features = data['person_features']
                self.id_mapping = data['id_mapping']
                self.next_persistent_id = data['next_persistent_id']
            print(f"Loaded {len(self.person_features)} person profiles from {filename}") 