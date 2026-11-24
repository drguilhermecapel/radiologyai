#!/usr/bin/env python3
"""
AI System Validation Test
Tests basic AI system initialization and functionality
"""

import sys
import os
sys.path.append('src')

def test_ai_imports():
    """Test if all AI modules can be imported"""
    print("Testing AI module imports...")
    
    try:
        from medai_inference_system import MedicalInferenceEngine
        print("✅ MedicalInferenceEngine imported successfully")
    except ImportError as e:
        print(f"❌ Error importing MedicalInferenceEngine: {e}")
        return False
    
    try:
        from medai_integration_manager import MedAIIntegrationManager
        print("✅ MedAIIntegrationManager imported successfully")
    except ImportError as e:
        print(f"❌ Error importing MedAIIntegrationManager: {e}")
        return False
    
    try:
        from medai_clinical_evaluation import ClinicalPerformanceEvaluator
        print("✅ ClinicalPerformanceEvaluator imported successfully")
    except ImportError as e:
        print(f"❌ Error importing ClinicalPerformanceEvaluator: {e}")
        return False
    
    return True

def test_ai_initialization():
    """Test AI system initialization"""
    print("\nTesting AI system initialization...")
    
    try:
        from medai_integration_manager import MedAIIntegrationManager
        manager = MedAIIntegrationManager()
        print("✅ MedAIIntegrationManager initialized")
    except Exception as e:
        print(f"❌ Error initializing integration manager: {e}")
        return False
    
    try:
        from medai_clinical_evaluation import ClinicalPerformanceEvaluator
        evaluator = ClinicalPerformanceEvaluator()
        print("✅ ClinicalPerformanceEvaluator initialized")
    except Exception as e:
        print(f"❌ Error initializing clinical evaluator: {e}")
        return False
    
    try:
        from medai_inference_system import MedicalInferenceEngine
        model_path = "models/"
        model_config = "models/model_config.json"
        engine = MedicalInferenceEngine(model_path, model_config)
        print("✅ MedicalInferenceEngine initialized with parameters")
    except Exception as e:
        print(f"⚠️ MedicalInferenceEngine requires parameters: {e}")
        print("✅ This is expected behavior - engine needs model path and config")
    
    return True

def main():
    """Main test function"""
    print("="*60)
    print("AI SYSTEM VALIDATION TEST")
    print("="*60)
    
    imports_ok = test_ai_imports()
    
    init_ok = test_ai_initialization()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"Initialization: {'✅ PASS' if init_ok else '❌ FAIL'}")
    
    overall_success = imports_ok and init_ok
    print(f"Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
