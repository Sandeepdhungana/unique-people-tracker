import cv2
import numpy as np
import torch
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torchvision
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

class DeepPersonReID:
    def __init__(self, 
                 appearance_weight=0.6, 
                 pose_weight=0.2,
                 height_weight=0.2, 
                 similarity_threshold=0.6,
                 feature_memory_frames=200,
                 use_gpu=True):
        """
        Initialize Deep Person Re-identification system
        Args:
            appearance_weight: Weight for appearance feature in similarity calculation
            pose_weight: Weight for pose feature in similarity calculation
            height_weight: Weight for height feature in similarity calculation
            similarity_threshold: Threshold to consider persons as the same
            feature_memory_frames: Number of frames to remember a person after they exit
            use_gpu: Whether to use GPU for inference
        """
        self.person_features = {}  # Store features for each tracked ID
        self.appearance_weight = appearance_weight
        self.pose_weight = pose_weight
        self.height_weight = height_weight
        self.similarity_threshold = similarity_threshold
        self.feature_memory_frames = feature_memory_frames  # Longer memory for better reidentification
        self.id_mapping = {}  # Maps tracker IDs to persistent IDs
        self.next_persistent_id = 1
        self.recent_trajectories = {}  # Store recent positions for trajectory analysis
        self.max_trajectory_length = 60  # Increased trajectory history for more robust tracking
        self.feature_dim = 768  # ViT-B/16 outputs 768-dim features
        
        # Keep track of person positions for occlusion handling
        self.last_positions = {}  # Store last known positions of each persistent ID
        self.velocity_vectors = {}  # Store velocity vectors for better prediction during occlusion
        
        # Short-term memory for handling brief occlusions (tracker_id: last_persistent_id)
        self.recent_tracker_ids = {}  # Recently seen tracker IDs and their persistent IDs
        self.short_term_memory = 10  # Frames to remember disappeared tracker IDs
        self.last_seen_frame = {}  # Last frame where each tracker ID was seen
        self.current_frame = 0  # Frame counter
        
        # Set up device
        try:
            # First check if CUDA is available
            if torch.cuda.is_available() and use_gpu:
                # Try a small tensor operation to verify CUDA works correctly
                test_tensor = torch.zeros(1, device='cuda')
                _ = test_tensor + 1  # Simple operation to test CUDA
                self.device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                print("CUDA not available, using CPU")
        except Exception as e:
            print(f"CUDA error: {e}")
            print("Falling back to CPU")
            self.device = torch.device('cpu')
            
        print(f"Using device: {self.device}")
        
        # Initialize Deep feature extraction model (using Vision Transformer)
        self.setup_model()
        
        # Initialize image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ViT expects 224x224 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def setup_model(self):
        """Initialize the deep learning model for feature extraction"""
        print("Setting up Vision Transformer (ViT-B/16) model for feature extraction...")
        # Use the latest ViT model with pre-trained weights
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
        
        # Create a feature extractor that gets the CLS token output
        class ViTFeatureExtractor(torch.nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit = vit_model
                # Remove the classification head
                self.vit.heads = torch.nn.Identity()
                
            def forward(self, x):
                return self.vit(x)
        
        self.feature_extractor = ViTFeatureExtractor(model)
        self.feature_extractor = self.feature_extractor.to(self.device).eval()
        print("Vision Transformer model initialized successfully")
    
    def extract_deep_features(self, image, box):
        """Extract deep features from person crop using ViT model"""
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:  # Skip tiny or invalid boxes
            return np.zeros(self.feature_dim)
            
        # Crop person from image
        person_img = image[y1:y2, x1:x2]
        
        # Convert from BGR to RGB (OpenCV uses BGR, but our model expects RGB)
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        try:
            tensor = self.transform(person_img).unsqueeze(0)
            
            # Handle CUDA errors by falling back to CPU if needed
            try:
                tensor = tensor.to(self.device)
                with torch.no_grad():
                    features = self.feature_extractor(tensor)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"CUDA error during inference, falling back to CPU: {e}")
                    # Try again on CPU
                    self.device = torch.device('cpu')
                    self.feature_extractor = self.feature_extractor.to(self.device)
                    tensor = tensor.to(self.device)
                    with torch.no_grad():
                        features = self.feature_extractor(tensor)
                else:
                    raise e
                
            # Return as numpy array
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting features: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(self.feature_dim)
    
    def extract_pose_features(self, image, box):
        """
        Extract pose-based features from the person
        This is a placeholder - in a real implementation you would use a pose estimation model
        """
        # For now, we'll use a simple proxy based on aspect ratio and size
        x1, y1, x2, y2 = [int(coord) for coord in box]
        width = x2 - x1
        height = y2 - y1
        
        # Aspect ratio can capture some pose information (standing vs. sitting)
        aspect_ratio = width / height if height > 0 else 0
        
        # Simple placeholder for pose features
        return np.array([aspect_ratio, width, height])
    
    def extract_clothing_features(self, image, box):
        """
        Extract clothing-specific features from a person detection
        This separates the person into regions (upper body, lower body) and extracts detailed features
        """
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1 or (x2-x1)*(y2-y1) < 100:  # Skip tiny or invalid boxes
            return np.zeros(256)  # Return zeros for clothing features
            
        # Crop person from image
        person_img = image[y1:y2, x1:x2]
        
        # Convert from BGR to RGB 
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Divide the person into upper and lower body
        height, width = person_img.shape[:2]
        upper_body = person_img[:int(height * 0.5), :]  # Upper 50%
        lower_body = person_img[int(height * 0.5):, :]  # Lower 50%
        
        # Extract color histograms for both parts (more bins for better discrimination)
        def extract_color_hist(img):
            if img.size == 0:
                return np.zeros(128)
                
            # Extract histogram in HSV space for better color representation
            img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Calculate histograms with more bins for better discrimination
            h_hist = cv2.calcHist([img_hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([img_hsv], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([img_hsv], [2], None, [8], [0, 256])
            
            # Normalize histograms
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            v_hist = cv2.normalize(v_hist, v_hist).flatten()
            
            # Concatenate
            return np.concatenate([h_hist, s_hist, v_hist])
            
        # Extract histograms for upper and lower body
        upper_hist = extract_color_hist(upper_body)
        lower_hist = extract_color_hist(lower_body)
        
        # Combine features (128 features per body part = 256 total)
        return np.concatenate([upper_hist, lower_hist])
    
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
        
        # Calculate velocity if we have enough points
        if len(self.recent_trajectories[tracker_id]) >= 3:
            # Get the last 3 points
            pts = self.recent_trajectories[tracker_id][-3:]
            
            # Calculate the average velocity vector from the last few frames
            vx1 = pts[1][0] - pts[0][0]
            vy1 = pts[1][1] - pts[0][1]
            vx2 = pts[2][0] - pts[1][0]
            vy2 = pts[2][1] - pts[1][1]
            
            # Average velocity
            vx = (vx1 + vx2) / 2
            vy = (vy1 + vy2) / 2
            
            # Store velocity vector for future occlusion prediction
            self.velocity_vectors[tracker_id] = (vx, vy)
        
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
    
    def calculate_combined_similarity(self, appearance1, pose1, height1, trajectory1, position1,
                                      appearance2, pose2, height2, trajectory2, position2,
                                      clothing1=None, clothing2=None):
        """Calculate overall similarity between two persons using multiple features"""
        # Appearance similarity (deep features)
        appearance_sim = cosine_similarity([appearance1], [appearance2])[0][0]
        
        # Clothing similarity (if available)
        clothing_sim = 0
        if clothing1 is not None and clothing2 is not None:
            clothing_sim = cosine_similarity([clothing1], [clothing2])[0][0]
        
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
        
        # Spatial-temporal similarity - how close the person is to where we expect them to be
        spatial_sim = 0
        if position1 is not None and position2 is not None:
            # Calculate distance between positions
            dist = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
            # Convert to similarity (closer = more similar)
            max_dist = 300  # Maximum reasonable distance a person might move while occluded
            spatial_sim = max(0, 1 - dist / max_dist)
        
        # Combined similarity with weights
        # Add clothing features with weight of 0.2 (reduce appearance weight to compensate)
        appearance_weight = self.appearance_weight * 0.8
        clothing_weight = 0.2
        
        combined_sim = (
            appearance_weight * appearance_sim +
            clothing_weight * clothing_sim +
            self.pose_weight * pose_sim + 
            self.height_weight * height_sim
        )
        
        # Boost similarity for matching trajectories and positions
        if traj_sim > 0.8:  # High trajectory match
            combined_sim = combined_sim * 1.2  # Boost similarity
        
        if spatial_sim > 0.7:  # High spatial match
            combined_sim = combined_sim * 1.1  # Boost similarity
            
        # Cap at 1.0
        combined_sim = min(1.0, combined_sim)
        
        return combined_sim
    
    def predict_position(self, p_id, frames_gone):
        """Predict the current position of a person based on their last known position and velocity"""
        if p_id not in self.last_positions:
            return None
            
        # Get last known position and velocity
        last_pos = self.last_positions[p_id]
        velocity = self.velocity_vectors.get(p_id, (0, 0))
        
        # Apply velocity for the number of frames the person has been gone
        # Add some decay to the velocity over time
        decay_factor = max(0.1, 1.0 - (frames_gone / 30.0))
        pred_x = last_pos[0] + velocity[0] * frames_gone * decay_factor
        pred_y = last_pos[1] + velocity[1] * frames_gone * decay_factor
        
        return (pred_x, pred_y)
    
    def update(self, image, detections):
        """
        Update person re-identification with new detections
        Returns updated detections with persistent IDs
        """
        # Increment frame counter
        self.current_frame += 1
        
        # Create a copy of the detections
        updated_detections = detections
        
        if updated_detections.tracker_id is None:
            return updated_detections
            
        updated_tracker_ids = updated_detections.tracker_id.copy()
        
        # Current frame's persistent IDs - used to track who's visible
        current_p_ids = set()
        
        # First pass: check for short-term occlusions and direct matches
        for i, (xyxy, tracker_id, confidence, class_id) in enumerate(
            zip(updated_detections.xyxy, updated_detections.tracker_id, 
                updated_detections.confidence, updated_detections.class_id)):
            
            if tracker_id is None or class_id != 0:  # Skip if not a person or no tracker ID
                continue
                
            # Try to match with recently disappeared tracker IDs first (for short occlusions)
            if tracker_id in self.recent_tracker_ids:
                # This is a reappearing tracker ID we've seen recently
                persistent_id = self.recent_tracker_ids[tracker_id]
                # Check if this persistent ID is already assigned to another tracker in this frame
                if persistent_id not in current_p_ids:
                    # We can reuse the previous persistent ID
                    self.id_mapping[tracker_id] = persistent_id
                    updated_tracker_ids[i] = persistent_id
                    current_p_ids.add(persistent_id)
                    print(f"Recovered tracker ID {tracker_id} -> person {persistent_id} after short occlusion")
                    # Update the last seen frame for this tracker ID
                    self.last_seen_frame[tracker_id] = self.current_frame
                    continue
            
            # Update the last seen frame for all current tracker IDs
            self.last_seen_frame[tracker_id] = self.current_frame
            
            # Rest of the processing will happen in the second pass
        
        # Second pass: process remaining detections with feature matching
        for i, (xyxy, tracker_id, confidence, class_id) in enumerate(
            zip(updated_detections.xyxy, updated_detections.tracker_id, 
                updated_detections.confidence, updated_detections.class_id)):
            
            if tracker_id is None or class_id != 0:  # Skip if not a person or no tracker ID
                continue
                
            # Skip if already processed in the first pass
            if tracker_id in self.id_mapping and self.id_mapping[tracker_id] in current_p_ids:
                continue
                
            # Calculate person height
            height = xyxy[3] - xyxy[1]
            
            # Extract features
            deep_features = self.extract_deep_features(image, xyxy)
            pose_features = self.extract_pose_features(image, xyxy)
            clothing_features = self.extract_clothing_features(image, xyxy)
            
            # Update trajectory
            self.update_trajectory(tracker_id, xyxy)
            trajectory = self.recent_trajectories.get(tracker_id, [])
            
            # Get current position (center point)
            current_pos = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
            
            # Check if we've seen this tracker_id before
            if tracker_id in self.id_mapping:
                persistent_id = self.id_mapping[tracker_id]
                current_p_ids.add(persistent_id)
                
                # Update features for this persistent ID with running average
                old_features = self.person_features[persistent_id]
                alpha = 0.7  # Weight for new features
                
                # Update all features
                updated_features = {
                    'appearance': (1-alpha) * old_features['appearance'] + alpha * deep_features,
                    'pose': pose_features,  # Just replace pose
                    'height': (1-alpha) * old_features['height'] + alpha * height,
                    'last_seen': 0,  # Frame counter since last seen
                    'trajectory': trajectory,
                    'quality': min(1.0, old_features.get('quality', 0.5) + 0.01),  # Safely access quality with a default
                    'clothing': (1-alpha) * old_features['clothing'] + alpha * clothing_features
                }
                
                self.person_features[persistent_id] = updated_features
                
                # Update position tracking
                self.last_positions[persistent_id] = current_pos
                
                # Add to recent tracker IDs memory
                self.recent_tracker_ids[tracker_id] = persistent_id
            else:
                # New tracker_id, check if it matches any known person who left the frame
                best_match_id = None
                best_similarity = 0
                
                # Consider only IDs not currently tracked (not in id_mapping values)
                untracked_ids = [p_id for p_id in self.person_features.keys() 
                                 if p_id not in current_p_ids]
                
                for p_id in untracked_ids:
                    p_features = self.person_features[p_id]
                    
                    # Predict current position based on last known position and velocity
                    predicted_pos = self.predict_position(p_id, p_features['last_seen'])
                    
                    # Calculate similarity between current person and stored person
                    similarity = self.calculate_combined_similarity(
                        deep_features, pose_features, height, trajectory, current_pos,
                        p_features['appearance'], p_features['pose'], 
                        p_features['height'], p_features['trajectory'], predicted_pos,
                        clothing_features, p_features.get('clothing')
                    )
                    
                    # Adjust similarity based on how long the person has been gone
                    # More lenient decay to help with reidentification after longer absences
                    frames_gone = p_features['last_seen']
                    
                    # Use a more forgiving time factor for brief occlusions
                    if frames_gone < 10:  # Brief occlusion
                        time_factor = 0.95  # Almost no penalty for very brief occlusions
                    else:
                        time_factor = max(0.2, 1 - (frames_gone / self.feature_memory_frames))
                    
                    # Also consider feature quality - more tracked frames = higher confidence
                    quality_factor = p_features.get('quality', 0.5)  # Safely access with default value
                    
                    adjusted_similarity = similarity * time_factor * quality_factor
                    
                    if adjusted_similarity > best_similarity:
                        best_similarity = adjusted_similarity
                        best_match_id = p_id
                
                # Lower threshold for brief occlusions
                actual_threshold = self.similarity_threshold
                if best_match_id is not None:
                    frames_gone = self.person_features[best_match_id]['last_seen']
                    if frames_gone < 10:  # Brief occlusion
                        # Use lower threshold for brief occlusions
                        actual_threshold = max(0.3, self.similarity_threshold - 0.2)
                
                # If similarity is above threshold, consider it the same person
                if best_match_id is not None and best_similarity > actual_threshold:
                    persistent_id = best_match_id
                    current_p_ids.add(persistent_id)
                    
                    # Update features with running average
                    old_features = self.person_features[persistent_id]
                    alpha = 0.3  # Lower weight for re-identified persons
                    
                    # Update features
                    updated_features = {
                        'appearance': (1-alpha) * old_features['appearance'] + alpha * deep_features,
                        'pose': pose_features,  # Just replace pose
                        'height': (1-alpha) * old_features['height'] + alpha * height,
                        'last_seen': 0,  # Reset frame counter
                        'trajectory': trajectory,
                        'quality': old_features.get('quality', 0.5),  # Safely access quality with default
                        'clothing': (1-alpha) * old_features['clothing'] + alpha * clothing_features
                    }
                    
                    self.person_features[persistent_id] = updated_features
                    self.last_positions[persistent_id] = current_pos
                    
                    print(f"Re-identified person {persistent_id} with similarity {best_similarity:.2f}")
                else:
                    # Create a new persistent ID
                    
                    # First try to match with permanent features database
                    permanent_match, match_sim = self.match_with_permanent_features(
                        deep_features, clothing_features, height)
                    
                    if permanent_match is not None:
                        # Use the permanent ID
                        persistent_id = permanent_match
                        print(f"Using permanent ID {persistent_id} for new person (sim: {match_sim:.2f})")
                    else:
                        # Create a new ID
                        persistent_id = self.next_persistent_id
                        self.next_persistent_id += 1
                        print(f"New person detected with ID {persistent_id}")
                    
                    current_p_ids.add(persistent_id)
                    
                    # Store all features
                    self.person_features[persistent_id] = {
                        'appearance': deep_features,
                        'pose': pose_features,
                        'height': height,
                        'last_seen': 0,
                        'trajectory': trajectory,
                        'quality': 0.5,  # Initial quality rating (medium confidence)
                        'clothing': clothing_features
                    }
                    
                    self.last_positions[persistent_id] = current_pos
                
                # Map tracker_id to persistent_id
                self.id_mapping[tracker_id] = persistent_id
                # Add to recent tracker IDs memory
                self.recent_tracker_ids[tracker_id] = persistent_id
            
            # Update detection with persistent ID
            updated_tracker_ids[i] = self.id_mapping[tracker_id]
        
        # Increment last_seen counter for persons not in current frame
        for p_id in self.person_features:
            if p_id not in current_p_ids:
                self.person_features[p_id]['last_seen'] += 1
        
        # Clean up old tracker IDs from short-term memory
        for t_id in list(self.last_seen_frame.keys()):
            frames_gone = self.current_frame - self.last_seen_frame[t_id]
            if frames_gone > self.short_term_memory:
                if t_id in self.recent_tracker_ids:
                    del self.recent_tracker_ids[t_id]
                del self.last_seen_frame[t_id]
        
        # Remove tracker IDs that are no longer tracked
        current_tracker_ids = set(updated_detections.tracker_id)
        self.id_mapping = {t_id: p_id for t_id, p_id in self.id_mapping.items() 
                           if t_id in current_tracker_ids}
        
        # Remove person features that haven't been seen for too long
        ids_to_remove = [p_id for p_id, features in self.person_features.items() 
                         if features['last_seen'] > self.feature_memory_frames]
        for p_id in ids_to_remove:
            del self.person_features[p_id]
            if p_id in self.last_positions:
                del self.last_positions[p_id]
            if p_id in self.velocity_vectors:
                del self.velocity_vectors[p_id]
        
        # Update detection object with persistent IDs
        updated_detections.tracker_id = updated_tracker_ids
        return updated_detections
    
    def save_features(self, filename="deep_person_features.pkl"):
        """Save person features to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'person_features': self.person_features,
                'id_mapping': self.id_mapping,
                'next_persistent_id': self.next_persistent_id,
                'last_positions': self.last_positions,
                'velocity_vectors': self.velocity_vectors,
                'recent_tracker_ids': self.recent_tracker_ids,
                'last_seen_frame': self.last_seen_frame,
                'current_frame': self.current_frame
            }, f)
        print(f"Saved {len(self.person_features)} person profiles to {filename}")
        
        # Also save to permanent database for future video sessions
        self.save_permanent_features()
    
    def load_features(self, filename="deep_person_features.pkl"):
        """Load person features from file"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                    
                    # Check if features have compatible dimensions
                    if data['person_features']:
                        # Get first feature vector to check dimensions
                        first_id = next(iter(data['person_features']))
                        feature_dim = data['person_features'][first_id]['appearance'].shape[0]
                        
                        # If dimensions match, load the data
                        if feature_dim == self.feature_dim:
                            self.person_features = data['person_features']
                            self.id_mapping = data['id_mapping']
                            self.next_persistent_id = data['next_persistent_id']
                            
                            # Load position and velocity data if available
                            self.last_positions = data.get('last_positions', {})
                            self.velocity_vectors = data.get('velocity_vectors', {})
                            
                            # Load short-term memory tracking data if available
                            self.recent_tracker_ids = data.get('recent_tracker_ids', {})
                            self.last_seen_frame = data.get('last_seen_frame', {})
                            self.current_frame = data.get('current_frame', 0)
                            
                            # Upgrade existing features if they're missing new fields
                            self._upgrade_loaded_features()
                            
                            print(f"Loaded {len(self.person_features)} person profiles from {filename}")
                        else:
                            print(f"Feature dimensions mismatch: expected {self.feature_dim}, got {feature_dim}")
                            print("Starting with empty features due to model change")
                            # Start fresh with new feature dimensions
                            self.person_features = {}
                            self.id_mapping = {}
                            self.next_persistent_id = 1
                    else:
                        # No features found, start fresh
                        self.person_features = {}
                        self.id_mapping = {}
                        self.next_persistent_id = 1
                        print(f"No valid features found in {filename}, starting fresh")
            except Exception as e:
                print(f"Error loading features: {e}")
                print("Starting with empty features")
                self.person_features = {}
                self.id_mapping = {}
                self.next_persistent_id = 1
                
    def _upgrade_loaded_features(self):
        """Add missing fields to loaded feature dictionaries to ensure compatibility with newer versions"""
        for pid, features in self.person_features.items():
            # Add quality field if missing
            if 'quality' not in features:
                features['quality'] = 0.5
                print(f"Upgraded features for person {pid}: added quality field")
            
            # Add clothing features if missing
            if 'clothing' not in features:
                # Create a default clothing feature vector
                features['clothing'] = np.zeros(256)  # 256-dimensional clothing features
                print(f"Upgraded features for person {pid}: added clothing field")
    
    def save_permanent_features(self, filename="permanent_person_features.pkl"):
        """
        Save high-quality person features to a permanent database that persists across videos
        Only saves persons with high feature quality (confidence)
        """
        # Filter for high-quality features only (persons that have been tracked reliably)
        permanent_features = {}
        for p_id, features in self.person_features.items():
            # Only keep persons with good quality tracking history (0.7+ quality)
            if features.get('quality', 0) >= 0.7:
                # Store a copy of the features
                permanent_features[p_id] = {
                    'appearance': features['appearance'].copy(),
                    'clothing': features['clothing'].copy(),
                    'height': features['height'],
                    # Don't save temporary data like trajectory and last_seen
                }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
        
        # Load existing database if it exists, to merge with new features
        existing_features = {}
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    existing_features = pickle.load(f)
            except Exception as e:
                print(f"Error loading permanent features: {e}")
        
        # Merge new features with existing ones
        existing_features.update(permanent_features)
        
        # Save merged features
        with open(filename, 'wb') as f:
            pickle.dump(existing_features, f)
            
        print(f"Saved {len(permanent_features)} high-quality person profiles to permanent database")
        print(f"Permanent database now contains {len(existing_features)} unique people")
    
    def match_with_permanent_features(self, deep_features, clothing_features, height):
        """
        Try to match a person with permanent features from previous videos
        Returns best matching ID or None if no good match
        """
        best_match_id = None
        best_similarity = 0
        
        # Path to permanent features database
        permanent_db = "permanent_person_features.pkl"
        
        if not os.path.exists(permanent_db):
            return None, 0  # Return tuple with None and 0 similarity
        
        try:
            # Load permanent features database
            with open(permanent_db, 'rb') as f:
                permanent_features = pickle.load(f)
                
            # Try to match with each person in the database
            for p_id, features in permanent_features.items():
                # Calculate similarity between current person and stored person
                appearance_sim = cosine_similarity([deep_features], [features['appearance']])[0][0]
                clothing_sim = cosine_similarity([clothing_features], [features['clothing']])[0][0]
                
                # Height similarity
                height_diff = abs(height - features['height']) / max(height, features['height'])
                height_sim = 1 - min(1.0, height_diff)
                
                # Combined similarity
                similarity = 0.5 * appearance_sim + 0.3 * clothing_sim + 0.2 * height_sim
                
                if similarity > best_similarity and similarity > 0.75:  # Higher threshold for permanent matches
                    best_similarity = similarity
                    best_match_id = p_id
                    
            if best_match_id is not None:
                print(f"Matched with permanent ID {best_match_id} (sim: {best_similarity:.2f})")
                return best_match_id, best_similarity
                
        except Exception as e:
            print(f"Error matching with permanent features: {e}")
            
        return None, 0  # Return tuple with None and 0 similarity 