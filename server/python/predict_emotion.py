#!/usr/bin/env python3
"""
Quick fix for the predict_emotion.py script to suppress warnings
"""

import os
import sys
import json
import torch
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# Now load transformers
from transformers import HubertForSequenceClassification

# Global Constants
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 3  # in seconds
N_MFCC = 30  # Using the enhanced value from your training
EMOTIONS = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

# Check arguments
if len(sys.argv) < 3:
    print(json.dumps({"error": "Missing arguments. Usage: python predict_emotion.py <model_path> <audio_path> [ensemble] [preloaded]"}))
    sys.exit(1)

# Parse command line arguments
model_path = sys.argv[1]
audio_path = sys.argv[2]
use_ensemble = len(sys.argv) > 3 and sys.argv[3] == 'ensemble'
use_preloaded = len(sys.argv) > 4 and sys.argv[4] == 'preloaded'

# Global model instances for preloading
PRELOADED_MODELS = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a better dummy audio file for preloading
def create_better_dummy_audio(dummy_path):
    """Create a dummy audio file with actual content instead of silence"""
    if os.path.exists(dummy_path):
        return
        
    # Create a simple sine wave
    duration = 2  # seconds
    sample_rate = SAMPLE_RATE
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a sine wave at 440 Hz (A4 note)
    note_a4 = 440
    audio = 0.5 * np.sin(note_a4 * 2 * np.pi * t)
    
    # Save using librosa
    librosa.output.write_wav(dummy_path, audio, sample_rate)
    print(f"Created dummy audio file at {dummy_path}", file=sys.stderr)

# MFCC extraction function 
def extract_mfcc_with_deltas(y, sr=SAMPLE_RATE, normalize=True):
    """Extract MFCC features with delta and delta-delta"""
    # For the enhanced model, include additional features
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Add energy as an additional feature
    energy = np.mean(librosa.feature.rms(y=y).reshape(-1))
    energy_feature = np.ones((1, mfccs.shape[1])) * energy

    # Calculate delta and delta-delta
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    # Add spectral contrast features
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Add zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)

    # Add chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # Combine all features
    mfccs = np.concatenate([mfccs, delta, delta2, energy_feature, contrast, zcr, chroma], axis=0)

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        mfccs = scaler.fit_transform(mfccs)

    # Ensure fixed width (time dimension)
    target_width = 100
    if mfccs.shape[1] > target_width:
        mfccs = mfccs[:, :target_width]
    else:
        pad_width = target_width - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')

    return mfccs


# CNN-LSTM pathway
class CNN_LSTM(torch.nn.Module):
    def __init__(self, input_height=None, input_width=100, hidden_size=128):
        super(CNN_LSTM, self).__init__()

        # Calculate input height based on feature extraction
        if input_height is None:
            # Estimate based on MFCCs, deltas, energy, contrast, ZCR, and chroma
            input_height = N_MFCC * 3 + 1 + 7 + 1 + 12  # Approximation

        self.input_height = input_height
        self.input_width = input_width

        # Enhanced CNN layers
        self.cnn = torch.nn.Sequential(
            # First conv block
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),  # More filters (16->32)
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.3),

            # Second conv block
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),  # More filters (32->64)
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.3),

            # Third conv block (new)
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.3)
        )

        # Calculate CNN output dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_height, self.input_width)
            cnn_out = self.cnn(dummy)
            _, c, h, w = cnn_out.shape
            self.lstm_input_size = c * h

        # Enhanced LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=hidden_size,
            num_layers=2,  # Increased from 1 to 2 layers
            batch_first=True,
            bidirectional=True,
            dropout=0.2  # Add dropout between LSTM layers
        )

        # Output size will be 2*hidden_size due to bidirectional LSTM
        self.output_dim = hidden_size * 2

    def forward(self, x):
        # Add channel dimension [batch, height, width] -> [batch, 1, height, width]
        x = x.unsqueeze(1)

        # Pass through CNN
        x = self.cnn(x)

        # Reshape for LSTM: [batch, channels, height, width] -> [batch, width, channels*height]
        batch_size, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch_size, w, c * h)

        # Pass through LSTM and get final output
        x, _ = self.lstm(x)
        # Use attention-like pooling: consider all time steps, not just the last one
        x = torch.mean(x, dim=1)

        return x


# HuBERT pathway
class HuBERTModule(torch.nn.Module):
    def __init__(self, pretrained_model="superb/hubert-base-superb-er"):
        super(HuBERTModule, self).__init__()

        # Load pre-trained HuBERT model
        start_time = time.time()
        print(f"Loading HuBERT model from {pretrained_model}...", file=sys.stderr)
        
        # Set download options to avoid warnings
        self.hubert = HubertForSequenceClassification.from_pretrained(
            pretrained_model, 
            cache_dir="./hubert_cache",
            resume_download=None  # Avoids the deprecated parameter warning
        )
        
        end_time = time.time()
        print(f"HuBERT model loaded in {end_time - start_time:.2f} seconds", file=sys.stderr)

        # Freeze feature extractor and some of the encoder layers
        self._freeze_base_layers()

        # Add additional processing for HuBERT output
        self.output_processor = torch.nn.Sequential(
            torch.nn.Linear(self.hubert.config.hidden_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 256)
        )

        # Output dimension is now fixed at 256
        self.output_dim = 256

    def _freeze_base_layers(self):
        # Freeze feature extractor (always frozen)
        for param in self.hubert.hubert.feature_extractor.parameters():
            param.requires_grad = False

        # Freeze some encoder layers
        for i, layer in enumerate(self.hubert.hubert.encoder.layers):
            if i < 2:  # FREEZE_HUBERT_LAYERS = 2 in your enhanced model
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        # Use mixed precision where available
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = self.hubert.hubert(x)
            # Use mean pooling of the last hidden states
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1)
            # Apply additional processing
            processed_output = self.output_processor(pooled_output)

        return processed_output


# Hybrid Model with Attention
class HybridModel(torch.nn.Module):
    def __init__(self, num_emotions=len(EMOTIONS)):
        super(HybridModel, self).__init__()

        # Initialize the two pathways
        self.cnn_lstm = CNN_LSTM()
        self.hubert = HuBERTModule()

        # Determine the output dimensions of each pathway
        self.cnn_lstm_dim = self.cnn_lstm.output_dim
        self.hubert_dim = self.hubert.output_dim

        # Improved fusion layer
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(self.cnn_lstm_dim + self.hubert_dim, 256),  # Larger first layer
            torch.nn.BatchNorm1d(256),  # Add batch norm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128),  # Add another layer
            torch.nn.BatchNorm1d(128),  # Add batch norm
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64)  # Final fusion dimension
        )

        # Attention mechanism
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

        # Classifier layer
        self.classifier = torch.nn.Linear(64, num_emotions)

    def forward(self, mfcc, audio):
        # Process through CNN-LSTM pathway
        cnn_lstm_out = self.cnn_lstm(mfcc)

        # Process through HuBERT pathway
        hubert_out = self.hubert(audio)

        # Concatenate outputs
        combined = torch.cat((cnn_lstm_out, hubert_out), dim=1)

        # Apply fusion layer
        fused = self.fusion(combined)

        # Apply attention 
        attention_weights = torch.softmax(self.attention(fused), dim=1)
        attended_fused = fused * attention_weights

        # Final classification
        output = self.classifier(attended_fused)

        return output


def load_model(model_path):
    """Load a model from file"""
    try:
        # Check if model is already preloaded
        if use_preloaded and model_path in PRELOADED_MODELS:
            print(f"Using preloaded model: {model_path}", file=sys.stderr)
            return PRELOADED_MODELS[model_path], DEVICE
        
        print(f"Loading model from: {model_path}", file=sys.stderr)
        start_time = time.time()
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Create model
        model = HybridModel()
        
        # Check if we have a state dict inside a dictionary or direct state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try loading directly
            model.load_state_dict(checkpoint)
            
        model.to(DEVICE)
        model.eval()
        
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds", file=sys.stderr)
        
        # Store in preloaded models
        PRELOADED_MODELS[model_path] = model
        
        return model, DEVICE
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None, None


def process_audio(file_path):
    """Process audio file for prediction"""
    try:
        start_time = time.time()
        
        # Load audio
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Normalize audio
        audio = librosa.util.normalize(audio)
        
        # Ensure right length
        if len(audio) < SAMPLE_RATE * MAX_AUDIO_LENGTH:
            audio = np.pad(audio, (0, int(SAMPLE_RATE * MAX_AUDIO_LENGTH) - len(audio)))
        else:
            audio = audio[:int(SAMPLE_RATE * MAX_AUDIO_LENGTH)]
            
        # Extract features
        mfcc = extract_mfcc_with_deltas(audio)
        
        end_time = time.time()
        print(f"Audio processed in {end_time - start_time:.2f} seconds", file=sys.stderr)
        
        return audio, mfcc
    except Exception as e:
        print(f"Error processing audio: {e}", file=sys.stderr)
        return None, None


def predict_emotion(model, device, audio, mfcc):
    """Predict emotion from processed audio"""
    start_time = time.time()
    
    # Convert to tensors
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(mfcc_tensor, audio_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        
    # Get predicted emotion
    idx_to_emotion = {v: k for k, v in EMOTIONS.items()}
    predicted_idx = np.argmax(probs)
    predicted_emotion = idx_to_emotion[predicted_idx]
    
    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds", file=sys.stderr)
    
    # Create results dictionary in the format expected by the Express server
    results = {
        'primaryEmotion': predicted_emotion.capitalize(),
        'confidenceScores': {idx_to_emotion[i].capitalize(): float(prob) for i, prob in enumerate(probs)}
    }
    
    return results


def predict_with_ensemble(model_paths, audio, mfcc):
    """Predict emotion using ensemble of models"""
    all_probs = []
    
    for model_path in model_paths:
        start_time = time.time()
        
        model, device = load_model(model_path)
        if model is None:
            continue
            
        # Convert to tensors
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(mfcc_tensor, audio_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            all_probs.append(probs)
        
        end_time = time.time()
        print(f"Model {model_path} prediction completed in {end_time - start_time:.2f} seconds", file=sys.stderr)
    
    if not all_probs:
        return None
        
    # Average probabilities (soft voting)
    avg_probs = np.mean(all_probs, axis=0)
    
    # Hard voting - get most common prediction
    predictions = [np.argmax(probs) for probs in all_probs]
    prediction_counts = np.bincount(predictions)
    most_common = np.argmax(prediction_counts)
    
    # Use the emotion from hard voting but confidence scores from averaging
    idx_to_emotion = {v: k for k, v in EMOTIONS.items()}
    predicted_emotion = idx_to_emotion[most_common]
    
    # Create results dictionary
    results = {
        'primaryEmotion': predicted_emotion.capitalize(),
        'confidenceScores': {idx_to_emotion[i].capitalize(): float(prob) for i, prob in enumerate(avg_probs)}
    }
    
    return results


# Main execution
def main():
    overall_start = time.time()
    
    if use_ensemble:
        # For ensemble mode, model_path should be a directory
        print(f"Using ensemble mode with models from: {model_path}", file=sys.stderr)
        
        ensemble_models = []
        for i in range(1, 10):  # Look for up to 10 ensemble models
            candidate_path = os.path.join(model_path, f"ensemble_model_{i}.pth")
            if os.path.exists(candidate_path):
                ensemble_models.append(candidate_path)
                
        if not ensemble_models:
            print(json.dumps({"error": "No ensemble models found"}))
            sys.exit(1)
            
        # Process audio
        audio, mfcc = process_audio(audio_path)
        if audio is None or mfcc is None:
            print(json.dumps({"error": "Failed to process audio"}))
            sys.exit(1)
            
        # Predict with ensemble
        results = predict_with_ensemble(ensemble_models, audio, mfcc)
        if results is None:
            print(json.dumps({"error": "Prediction failed"}))
            sys.exit(1)
    else:
        # Use single model
        model, device = load_model(model_path)
        if model is None:
            print(json.dumps({"error": "Failed to load model"}))
            sys.exit(1)
            
        # Process audio
        audio, mfcc = process_audio(audio_path)
        if audio is None or mfcc is None:
            print(json.dumps({"error": "Failed to process audio"}))
            sys.exit(1)
            
        # Predict emotion
        results = predict_emotion(model, device, audio, mfcc)
    
    overall_end = time.time()
    print(f"Total prediction time: {overall_end - overall_start:.2f} seconds", file=sys.stderr)
    
    # Output JSON result
    print(json.dumps(results))


if __name__ == "__main__":
    main()