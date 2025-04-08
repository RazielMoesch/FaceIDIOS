import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import logging
import os
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model definition matching the trained architecture
class FaceRecognitionModel(nn.Module):
    def __init__(self):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = models.efficientnet_b0(pretrained=False)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # These layers match the training script but won't be used for inference
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 105)  # 105 classes from your original training
        )
    
    def get_features(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        return self.get_features(x)  # Only use backbone for inference

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(checkpoint_path, device):
    """Load the model from checkpoint, handling mismatched keys"""
    model = FaceRecognitionModel()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Filter out classifier weights and only load backbone weights
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith(('fc1', 'bn1', 'fc2', 'classifier'))}
        
        # Load the filtered state dict
        model.backbone.load_state_dict(filtered_state_dict, strict=False)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully, using only backbone weights")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def extract_features(model, image, device):
    """Extract feature embedding from an image"""
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.get_features(image_tensor)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        return None

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    try:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except:
        return -1

class FaceDatabase:
    def __init__(self):
        self.faces = defaultdict(list)  # {name: [feature_vectors]}
    
    def add_face(self, name, feature_vector):
        if feature_vector is not None:
            self.faces[name].append(feature_vector)
            logger.info(f"Added face for {name}")
    
    def recognize_face(self, feature_vector, threshold=0.7):
        """Recognize face by comparing with stored features"""
        if feature_vector is None or not self.faces:
            return "Unknown", 0
        
        best_match = "Unknown"
        highest_similarity = -1
        
        for name, vectors in self.faces.items():
            for stored_vector in vectors:
                similarity = cosine_similarity(feature_vector, stored_vector)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    if similarity > threshold:
                        best_match = name
        
        return best_match, highest_similarity * 100

def process_frame(frame, model, face_db, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        features = extract_features(model, face_roi, device)
        
        if features is not None:
            name, confidence = face_db.recognize_face(features)
            
            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{name} ({confidence:.1f}%)" if name != "Unknown" else "Unknown"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.9, color, 2)
    
    return frame

def main():
    # Configuration
    CHECKPOINT_PATH = 'best_face_recognition_model.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(CHECKPOINT_PATH, DEVICE)
    face_db = FaceDatabase()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Cannot open camera. Please check your webcam connection.")
        return
    
    adding_mode = False
    new_name = ""
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Can't receive frame. Exiting...")
            break
        
        # Process frame for recognition
        processed_frame = process_frame(frame.copy(), model, face_db, DEVICE)
        
        # Display instructions
        cv2.putText(processed_frame, "Press 'a' to add new face, 'q' to quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if adding_mode:
            cv2.putText(processed_frame, f"Enter name: {new_name}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_frame, "Press Enter to save, Esc to cancel", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == ord('a') and not adding_mode:  # Start adding new face
            adding_mode = True
            new_name = ""
        elif adding_mode:
            if key == 27:  # Esc to cancel
                adding_mode = False
            elif key == 13:  # Enter to save
                if new_name:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]  # Take first detected face
                        face_roi = frame[y:y+h, x:x+w]
                        features = extract_features(model, face_roi, DEVICE)
                        face_db.add_face(new_name, features)
                    adding_mode = False
            elif key >= 32 and key <= 126:  # Printable characters
                new_name += chr(key)
            elif key == 8 and new_name:  # Backspace
                new_name = new_name[:-1]
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application closed successfully")

if __name__ == '__main__':
    try:
        # Check if model file exists
        if not os.path.exists('best_face_recognition_model.pth'):
            logger.error("Model file 'best_face_recognition_model.pth' not found in current directory")
            raise FileNotFoundError("Model file not found")
        
        main()
    except Exception as e:
        logger.error(f"Error in execution: {str(e)}")
        raise