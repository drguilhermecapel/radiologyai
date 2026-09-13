#!/usr/bin/env python3
"""
Test script to verify that trained models are loaded correctly
and provide specific diagnostic conclusions instead of generic responses
"""

import sys
import os
sys.path.append('src')

import numpy as np
import cv2
from pathlib import Path
from medai_inference_system import MedicalInferenceEngine

def test_model_loading():
    """Test that the trained model is loaded correctly"""
    print("=== Testing Model Loading ===")
    
    try:
        model_path = Path("models/advanced_ensemble_model.h5")
        model_config = {
            'classes': ['Normal', 'Pneumonia', 'Derrame Pleural', 'Fratura', 'Tumor/Massa'],
            'input_shape': (224, 224, 3),
            'num_classes': 5
        }
        engine = MedicalInferenceEngine(model_path=model_path, model_config=model_config)
        
        print(f"Model loaded: {engine.model is not None}")
        print(f"Is dummy model: {getattr(engine, '_is_dummy_model', True)}")
        
        if engine.model:
            print(f"Model input shape: {engine.model.input_shape}")
            print(f"Model output shape: {engine.model.output_shape}")
            print(f"Model layers: {len(engine.model.layers)}")
            return True
        else:
            print("ERROR: No model loaded!")
            return False
            
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

def test_prediction_with_sample_image():
    """Test prediction with a sample image"""
    print("\n=== Testing Prediction with Sample Image ===")
    
    try:
        model_path = Path("models/advanced_ensemble_model.h5")
        model_config = {
            'classes': ['Normal', 'Pneumonia', 'Derrame Pleural', 'Fratura', 'Tumor/Massa'],
            'input_shape': (224, 224, 3),
            'num_classes': 5
        }
        engine = MedicalInferenceEngine(model_path=model_path, model_config=model_config)
        
        if not engine.model:
            print("ERROR: No model available for prediction")
            return False
        
        sample_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        cv2.ellipse(sample_image, (80, 112), (60, 80), 0, 0, 360, 150, -1)
        cv2.ellipse(sample_image, (144, 112), (60, 80), 0, 0, 360, 150, -1)
        
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample image dtype: {sample_image.dtype}")
        
        result = engine.predict_single(sample_image, return_attention=False)
        
        print(f"Predicted class: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing time: {result.processing_time:.3f}s")
        
        if hasattr(result, 'predictions') and result.predictions:
            print("All class predictions:")
            for class_name, prob in result.predictions.items():
                print(f"  {class_name}: {prob:.3f}")
        
        generic_responses = ['Não determinado', 'Erro', 'unknown', 'Erro na análise']
        is_specific = result.predicted_class not in generic_responses
        
        print(f"Is specific diagnosis (not generic): {is_specific}")
        
        return is_specific
        
    except Exception as e:
        print(f"ERROR in prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_predictions():
    """Test multiple predictions to verify consistency"""
    print("\n=== Testing Multiple Predictions ===")
    
    try:
        model_path = Path("models/advanced_ensemble_model.h5")
        model_config = {
            'classes': ['Normal', 'Pneumonia', 'Derrame Pleural', 'Fratura', 'Tumor/Massa'],
            'input_shape': (224, 224, 3),
            'num_classes': 5
        }
        engine = MedicalInferenceEngine(model_path=model_path, model_config=model_config)
        
        if not engine.model:
            print("ERROR: No model available for prediction")
            return False
        
        predictions = []
        
        for i in range(3):
            sample_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            
            if i == 0:
                cv2.ellipse(sample_image, (80, 112), (60, 80), 0, 0, 360, 120, -1)
                cv2.ellipse(sample_image, (144, 112), (60, 80), 0, 0, 360, 120, -1)
            elif i == 1:
                cv2.ellipse(sample_image, (80, 112), (60, 80), 0, 0, 360, 180, -1)
                cv2.ellipse(sample_image, (144, 112), (60, 80), 0, 0, 360, 180, -1)
                noise = np.random.randint(0, 50, (224, 224), dtype=np.uint8)
                sample_image = cv2.add(sample_image, noise)
            else:
                cv2.ellipse(sample_image, (80, 112), (60, 80), 0, 0, 360, 100, -1)
                cv2.ellipse(sample_image, (144, 112), (60, 80), 0, 0, 360, 100, -1)
                cv2.rectangle(sample_image, (0, 180), (224, 224), 200, -1)
            
            result = engine.predict_single(sample_image, return_attention=False)
            predictions.append({
                'class': result.predicted_class,
                'confidence': result.confidence,
                'image_type': ['normal-like', 'pneumonia-like', 'effusion-like'][i]
            })
            
            print(f"Image {i+1} ({predictions[i]['image_type']}): {predictions[i]['class']} ({predictions[i]['confidence']:.3f})")
        
        unique_classes = set(p['class'] for p in predictions)
        print(f"Unique predicted classes: {len(unique_classes)}")
        print(f"Classes: {list(unique_classes)}")
        
        return len(unique_classes) > 1  # Should get different predictions for different patterns
        
    except Exception as e:
        print(f"ERROR in multiple predictions: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing Trained Model Integration")
    print("=" * 50)
    
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("model_*.h5"))
        print(f"Found {len(checkpoint_files)} checkpoint files:")
        for cp in sorted(checkpoint_files):
            print(f"  {cp}")
    else:
        print("WARNING: No checkpoints directory found")
    
    print()
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Single Prediction", test_prediction_with_sample_image),
        ("Multiple Predictions", test_multiple_predictions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name}: FAILED with exception: {e}")
        print()
    
    print("=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Trained model integration successful!")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
