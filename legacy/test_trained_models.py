#!/usr/bin/env python3
"""
Test script to verify trained models are loading correctly
"""

import sys
import os
sys.path.append('src')

from medai_inference_system import MedicalInferenceEngine
from pathlib import Path
import json
import numpy as np

def test_model_loading():
    """Test loading of trained models"""
    print("=== Testing Trained Model Loading ===")
    
    with open('models/model_config.json', 'r') as f:
        config = json.load(f)
    
    models_to_test = [
        'chest_xray_efficientnetv2',
        'chest_xray_vision_transformer', 
        'chest_xray_convnext'
    ]
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name} ---")
        
        model_config = config['models'][model_name]
        model_path = Path(model_config['model_path'])
        
        print(f"Model path: {model_path}")
        print(f"Model exists: {model_path.exists()}")
        
        if model_path.exists():
            print(f"Model size: {model_path.stat().st_size} bytes")
            
            try:
                engine = MedicalInferenceEngine(model_path, model_config)
                print(f"✅ Model loaded successfully")
                print(f"Is dummy model: {getattr(engine, '_is_dummy_model', 'Unknown')}")
                
                dummy_image = np.random.rand(224, 224, 3).astype(np.float32)
                result = engine.predict_single(dummy_image)
                print(f"✅ Prediction test successful: {result.predicted_class}")
                
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"❌ Model file not found")

if __name__ == "__main__":
    test_model_loading()
