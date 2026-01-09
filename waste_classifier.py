"""
Waste Classification System
A beginner-friendly AI system to classify waste into categories
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
import os

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 6
CLASS_NAMES = ['glass', 'metal', 'organic', 'paper', 'plastic', 'recyclable']

print("=" * 50)
print("WASTE CLASSIFICATION SYSTEM")
print("=" * 50)

class WasteClassifier:
    """Main class for waste classification"""
    
    def __init__(self):
        self.model = None
        self.class_names = CLASS_NAMES
        
    def build_model(self):
        """Build the AI model using MobileNetV2"""
        print("\n[Step 1] Building AI Model...")
        
        # Load pre-trained MobileNetV2 (this will download first time)
        print("  - Loading MobileNetV2 (may take time on first run)...")
        base_model = MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model
        base_model.trainable = False
        print("  - MobileNetV2 loaded successfully!")
        
        # Add custom layers for waste classification
        print("  - Adding classification layers...")
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        
        # Create final model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("  ✓ Model built successfully!")
        print(f"  - Total parameters: {self.model.count_params():,}")
        return self.model
    
    def prepare_data(self, train_dir, val_dir):
        """Load and prepare training data"""
        print("\n[Step 2] Loading Dataset...")
        
        # Check if directories exist
        if not os.path.exists(train_dir):
            print(f"  ✗ Error: Training directory not found: {train_dir}")
            print(f"  Please create the folder structure as described in the guide")
            return None, None
            
        if not os.path.exists(val_dir):
            print(f"  ✗ Error: Validation directory not found: {val_dir}")
            return None, None
        
        # Data augmentation for training (helps model learn better)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load images from directories
        print(f"  - Loading training images from: {train_dir}")
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        print(f"  - Loading validation images from: {val_dir}")
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )
        
        print("  ✓ Dataset loaded successfully!")
        print(f"  - Training images: {train_generator.samples}")
        print(f"  - Validation images: {val_generator.samples}")
        print(f"  - Categories found: {list(train_generator.class_indices.keys())}")
        
        return train_generator, val_generator
    
    def train(self, train_generator, val_generator):
        """Train the model"""
        print("\n[Step 3] Training Model...")
        print(f"  - Epochs: {EPOCHS}")
        print(f"  - Batch size: {BATCH_SIZE}")
        print("  - This may take 10-30 minutes depending on your computer")
        print("  - You'll see progress for each epoch\n")
        
        # Callbacks for better training
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        history = self.model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n  ✓ Training completed!")
        return history
    
    def save_model(self, filepath='waste_classifier_model.h5'):
        """Save trained model"""
        print(f"\n[Step 4] Saving model to {filepath}...")
        self.model.save(filepath)
        print(f"  ✓ Model saved successfully!")
    
    def load_model(self, filepath='waste_classifier_model.h5'):
        """Load trained model"""
        print(f"Loading model from {filepath}...")
        if not os.path.exists(filepath):
            print(f"  ✗ Error: Model file not found: {filepath}")
            print(f"  Please train the model first (Option 1)")
            return False
        self.model = tf.keras.models.load_model(filepath)
        print("  ✓ Model loaded successfully!")
        return True


class WasteDetector:
    """Class for detecting and classifying waste"""
    
    def __init__(self, model_path='waste_classifier_model.h5'):
        self.classifier = WasteClassifier()
        success = self.classifier.load_model(model_path)
        if not success:
            self.classifier = None
        
    def preprocess_image(self, image):
        """Prepare image for prediction"""
        img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    
    def classify_image(self, image):
        """Classify waste from image"""
        if self.classifier is None:
            return None, 0
        
        processed_img = self.preprocess_image(image)
        predictions = self.classifier.model.predict(processed_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        class_name = self.classifier.class_names[class_idx]
        
        return class_name, confidence
    
    def classify_from_file(self, image_path):
        """Classify waste from image file"""
        print(f"\nClassifying image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"  ✗ Error: Image file not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ✗ Error: Could not read image file")
            return
        
        waste_type, confidence = self.classify_image(image)
        
        if waste_type is None:
            return
        
        # Display result
        print(f"  ✓ Classification: {waste_type.upper()}")
        print(f"  ✓ Confidence: {confidence*100:.2f}%")
        
        # Show image with prediction
        label = f"{waste_type}: {confidence*100:.1f}%"
        color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        
        cv2.putText(image, label, (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        cv2.imshow('Waste Classification Result', image)
        print("\n  Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def detect_from_camera(self):
        """Real-time detection from webcam"""
        if self.classifier is None:
            return
        
        print("\nStarting camera detection...")
        print("  - Press 'Q' to quit")
        print("  - Press 'S' to save a screenshot")
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print("  ✗ Error: Could not open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  ✗ Error: Could not read frame from camera")
                break
            
            # Classify current frame
            waste_type, confidence = self.classify_image(frame)
            
            # Display results on frame
            label = f"{waste_type}: {confidence*100:.2f}%"
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            
            cv2.putText(frame, label, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Add instructions
            cv2.putText(frame, "Press 'Q' to quit", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Waste Classification - Live', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                filename = f'screenshot_{waste_type}.jpg'
                cv2.imwrite(filename, frame)
                print(f"  ✓ Screenshot saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("  ✓ Camera closed")


def main():
    """Main function - Menu system"""
    
    while True:
        print("\n" + "=" * 50)
        print("WASTE CLASSIFICATION SYSTEM - MAIN MENU")
        print("=" * 50)
        print("\n1. Train New Model (First time users)")
        print("2. Test with Camera (Real-time)")
        print("3. Test with Image File")
        print("4. Exit")
        print("\n" + "=" * 50)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            # Training mode
            print("\n" + "=" * 50)
            print("TRAINING MODE")
            print("=" * 50)
            
            classifier = WasteClassifier()
            classifier.build_model()
            
            # Set up data paths
            train_dir = 'dataset/train'
            val_dir = 'dataset/validation'
            
            train_gen, val_gen = classifier.prepare_data(train_dir, val_dir)
            
            if train_gen is None or val_gen is None:
                print("\n✗ Cannot proceed with training. Please check your dataset folders.")
                continue
            
            # Confirm before training
            confirm = input("\nStart training? (yes/no): ").strip().lower()
            if confirm == 'yes' or confirm == 'y':
                history = classifier.train(train_gen, val_gen)
                classifier.save_model()
                print("\n" + "=" * 50)
                print("TRAINING COMPLETED SUCCESSFULLY!")
                print("=" * 50)
            else:
                print("Training cancelled.")
        
        elif choice == "2":
            # Camera mode
            print("\n" + "=" * 50)
            print("CAMERA DETECTION MODE")
            print("=" * 50)
            detector = WasteDetector()
            if detector.classifier:
                detector.detect_from_camera()
        
        elif choice == "3":
            # Image file mode
            print("\n" + "=" * 50)
            print("IMAGE FILE DETECTION MODE")
            print("=" * 50)
            image_path = input("\nEnter image file path: ").strip()
            detector = WasteDetector()
            if detector.classifier:
                detector.classify_from_file(image_path)
        
        elif choice == "4":
            print("\nThank you for using Waste Classification System!")
            print("Goodbye! 👋")
            break
        
        else:
            print("\n✗ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n✗ An error occurred: {e}")
        print("Please check the error message above and try again.")