#!/usr/bin/env python3
"""
Comprehensive test script to validate pathology detection across all classes
"""

import sys
import os
sys.path.append('src')

import numpy as np
from pathlib import Path
import json
from medai_inference_system import MedicalInferenceEngine
from medai_integration_manager import MedAIIntegrationManager

def create_synthetic_test_images():
    """Create synthetic test images for each pathology class"""
    test_images_dir = Path("/home/ubuntu/test_images")
    test_images_dir.mkdir(exist_ok=True)
    
    pathology_classes = ["normal", "pneumonia", "pleural_effusion", "fracture", "tumor"]
    
    for pathology in pathology_classes:
        pathology_dir = test_images_dir / pathology
        pathology_dir.mkdir(exist_ok=True)
        
        for i in range(3):
            if pathology == "normal":
                image = np.random.normal(0.3, 0.1, (512, 512, 3))
                image = np.clip(image, 0, 1)
            elif pathology == "pneumonia":
                image = np.random.normal(0.4, 0.15, (512, 512, 3))
                image[200:350, 150:400] += 0.3
                image = np.clip(image, 0, 1)
            elif pathology == "pleural_effusion":
                image = np.random.normal(0.3, 0.1, (512, 512, 3))
                image[400:512, :] += 0.4
                image = np.clip(image, 0, 1)
            elif pathology == "fracture":
                image = np.random.normal(0.2, 0.08, (512, 512, 3))
                image[250:260, 100:400] = 0.8
                image = np.clip(image, 0, 1)
            elif pathology == "tumor":
                image = np.random.normal(0.3, 0.1, (512, 512, 3))
                y, x = np.ogrid[:512, :512]
                mask = (x - 256)**2 + (y - 200)**2 < 50**2
                image[mask] += 0.5
                image = np.clip(image, 0, 1)
            
            image_uint8 = (image * 255).astype(np.uint8)
            
            image_path = pathology_dir / f"{pathology}_test_{i+1:03d}.npy"
            np.save(image_path, image_uint8)
            print(f"Created test image: {image_path}")

def test_pathology_detection(pathology_type, image_path):
    """Test pathology detection for a specific image"""
    print(f"\n--- Testing {pathology_type} image: {image_path.name} ---")
    
    try:
        image = np.load(image_path)
        print(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        with open('models/model_config.json', 'r') as f:
            config = json.load(f)
        
        model_config = config['models']['chest_xray_efficientnetv2']
        model_path = Path(model_config['model_path'])
        
        engine = MedicalInferenceEngine(model_path, model_config)
        
        result = engine.predict_single(image)
        
        print(f"Predicted class: {result.predicted_class}")
        print(f"Confidence: {result.confidence:.2%}")
        
        if hasattr(result, 'predictions') and result.predictions:
            print("All pathology scores:")
            for pathology, score in result.predictions.items():
                print(f"  {pathology}: {score:.4f}")
        
        expected_pathology = pathology_type.lower()
        predicted_pathology = result.predicted_class.lower()
        
        if expected_pathology == "pleural_effusion":
            is_correct = "pleural" in predicted_pathology or "derrame" in predicted_pathology
        elif expected_pathology == "fracture":
            is_correct = "fracture" in predicted_pathology or "fratura" in predicted_pathology
        else:
            is_correct = expected_pathology in predicted_pathology
        
        print(f"Expected: {expected_pathology}")
        print(f"Predicted: {predicted_pathology}")
        print(f"Correct prediction: {'✅' if is_correct else '❌'}")
        
        return is_correct, result
        
    except Exception as e:
        print(f"❌ Error testing {pathology_type} image: {e}")
        return False, None

def run_comprehensive_pathology_tests():
    """Run comprehensive tests across all pathology classes"""
    print("=== Comprehensive Pathology Detection Tests ===")
    
    create_synthetic_test_images()
    
    test_images_dir = Path("/home/ubuntu/test_images")
    pathology_classes = ["normal", "pneumonia", "pleural_effusion", "fracture", "tumor"]
    
    results = {}
    total_tests = 0
    correct_predictions = 0
    
    for pathology in pathology_classes:
        pathology_dir = test_images_dir / pathology
        if not pathology_dir.exists():
            print(f"⚠️ No test images found for {pathology}")
            continue
        
        pathology_results = []
        test_images = list(pathology_dir.glob("*.npy"))
        
        print(f"\n{'='*50}")
        print(f"Testing {pathology.upper()} ({len(test_images)} images)")
        print(f"{'='*50}")
        
        for image_path in test_images:
            is_correct, result = test_pathology_detection(pathology, image_path)
            pathology_results.append({
                'image': image_path.name,
                'correct': is_correct,
                'predicted_class': result.predicted_class if result else 'ERROR',
                'confidence': result.confidence if result else 0.0
            })
            
            total_tests += 1
            if is_correct:
                correct_predictions += 1
        
        results[pathology] = pathology_results
    
    print(f"\n{'='*60}")
    print("PATHOLOGY DETECTION TEST SUMMARY")
    print(f"{'='*60}")
    
    for pathology, pathology_results in results.items():
        correct_count = sum(1 for r in pathology_results if r['correct'])
        total_count = len(pathology_results)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        
        print(f"{pathology.upper()}: {correct_count}/{total_count} correct ({accuracy:.1f}%)")
        
        predictions = {}
        for r in pathology_results:
            pred_class = r['predicted_class']
            predictions[pred_class] = predictions.get(pred_class, 0) + 1
        
        print(f"  Prediction distribution: {predictions}")
    
    overall_accuracy = (correct_predictions / total_tests * 100) if total_tests > 0 else 0
    print(f"\nOVERALL ACCURACY: {correct_predictions}/{total_tests} ({overall_accuracy:.1f}%)")
    
    pneumonia_bias_detected = False
    for pathology, pathology_results in results.items():
        if pathology != "pneumonia":
            pneumonia_predictions = sum(1 for r in pathology_results 
                                      if "pneumonia" in r['predicted_class'].lower())
            if pneumonia_predictions > len(pathology_results) * 0.5:
                print(f"⚠️ PNEUMONIA BIAS DETECTED: {pathology} images predicted as pneumonia {pneumonia_predictions}/{len(pathology_results)} times")
                pneumonia_bias_detected = True
    
    if not pneumonia_bias_detected:
        print("✅ NO SIGNIFICANT PNEUMONIA BIAS DETECTED")
    
    return results, overall_accuracy

if __name__ == "__main__":
    results, accuracy = run_comprehensive_pathology_tests()
    
    results_file = Path("pathology_test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'overall_accuracy': accuracy,
            'test_timestamp': '2025-06-11T04:34:00Z'
        }, f, indent=2)
    
    print(f"\nTest results saved to: {results_file}")
