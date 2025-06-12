#!/usr/bin/env python3
"""
Script to inspect the structure and weights of trained models
"""

import tensorflow as tf
import numpy as np
from pathlib import Path

def inspect_model_structure():
    """Inspect the structure and weights of the trained models"""
    print("=== Model Structure Inspection ===")
    
    model_paths = [
        "models/chest_xray_efficientnetv2_model.h5",
        "models/chest_xray_visiontransformer_model.h5", 
        "models/chest_xray_convnext_model.h5"
    ]
    
    for model_path in model_paths:
        print(f"\n--- Inspecting {model_path} ---")
        
        if not Path(model_path).exists():
            print(f"❌ Model file not found: {model_path}")
            continue
            
        try:
            model = tf.keras.models.load_model(model_path)
            
            print(f"Model type: {type(model)}")
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            print(f"Number of layers: {len(model.layers)}")
            print(f"Total parameters: {model.count_params():,}")
            
            weights = model.get_weights()
            if weights:
                first_layer_weights = weights[0]
                weight_std = np.std(first_layer_weights)
                weight_mean = np.mean(first_layer_weights)
                
                print(f"First layer weight statistics:")
                print(f"  Mean: {weight_mean:.6f}")
                print(f"  Std: {weight_std:.6f}")
                print(f"  Min: {np.min(first_layer_weights):.6f}")
                print(f"  Max: {np.max(first_layer_weights):.6f}")
                
                if weight_std < 0.01:
                    print("⚠️ Weights appear to be very small - possibly untrained")
                elif 0.01 <= weight_std <= 0.5:
                    print("✅ Weights appear to be in normal range")
                else:
                    print("⚠️ Weights have unusual distribution")
            
            dummy_input = np.random.rand(1, *model.input_shape[1:]).astype(np.float32)
            prediction = model.predict(dummy_input, verbose=0)
            
            print(f"Test prediction shape: {prediction.shape}")
            print(f"Test prediction: {prediction[0]}")
            print(f"Prediction sum: {np.sum(prediction[0]):.6f}")
            
            predictions = []
            for i in range(3):
                test_input = np.random.rand(1, *model.input_shape[1:]).astype(np.float32)
                pred = model.predict(test_input, verbose=0)
                predictions.append(pred[0])
            
            all_same = all(np.allclose(predictions[0], pred, atol=1e-6) for pred in predictions)
            if all_same:
                print("❌ Model returns identical predictions for different inputs - likely dummy model")
            else:
                print("✅ Model returns different predictions for different inputs")
                
        except Exception as e:
            print(f"❌ Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_model_structure()
