#!/usr/bin/env python3
"""
Test SOTA model integration for MedAI Radiologia
Validates EfficientNetV2, Vision Transformer, and ConvNeXt implementations
"""

import sys
import os
sys.path.append('src')

def test_sota_imports():
    """Test that SOTA models can be imported successfully"""
    try:
        from medai_sota_models import SOTAModelManager, StateOfTheArtModels
        print('✅ SOTA models import successful')
        return True
    except Exception as e:
        print(f'❌ Import error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_model_manager():
    """Test SOTAModelManager functionality"""
    try:
        from medai_sota_models import SOTAModelManager
        
        manager = SOTAModelManager()
        available = manager.get_available_models()
        print(f'✅ Available models: {available}')
        
        expected_models = ['efficientnetv2', 'vision_transformer', 'convnext', 'ensemble_model']
        for model in expected_models:
            if model not in available:
                print(f'❌ Missing expected model: {model}')
                return False
        
        print('✅ All expected models are available')
        return True
        
    except Exception as e:
        print(f'❌ Model manager error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_model_builder():
    """Test StateOfTheArtModels initialization"""
    try:
        from medai_sota_models import StateOfTheArtModels
        
        builder = StateOfTheArtModels((384, 384, 3), 5)
        print('✅ StateOfTheArtModels initialization successful')
        
        import tensorflow as tf
        test_input = tf.random.normal((1, 384, 384, 3))
        processed = builder._medical_preprocessing(test_input)
        print(f'✅ Medical preprocessing works: {processed.shape}')
        
        test_shape = (3, 3, 3, 32)
        kernel = builder._conv_kernel_initializer(test_shape)
        print(f'✅ Conv kernel initializer works: {kernel.shape}')
        
        dense_shape = (512, 256)
        dense_kernel = builder._dense_kernel_initializer(dense_shape)
        print(f'✅ Dense kernel initializer works: {dense_kernel.shape}')
        
        return True
        
    except Exception as e:
        print(f'❌ Model builder error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_model_architectures():
    """Test individual model architecture building (structure only)"""
    try:
        from medai_sota_models import StateOfTheArtModels
        import tensorflow as tf
        
        builder = StateOfTheArtModels((224, 224, 3), 5)
        
        print('Testing Vision Transformer architecture...')
        vit_model = builder.build_real_vision_transformer()
        print(f'✅ ViT model created: {vit_model.name}, params: {vit_model.count_params():,}')
        
        print('Testing EfficientNetV2 architecture...')
        eff_model = builder.build_real_efficientnetv2()
        print(f'✅ EfficientNetV2 model created: {eff_model.name}, params: {eff_model.count_params():,}')
        
        print('Testing ConvNeXt architecture...')
        convnext_model = builder.build_real_convnext()
        print(f'✅ ConvNeXt model created: {convnext_model.name}, params: {convnext_model.count_params():,}')
        
        return True
        
    except Exception as e:
        print(f'❌ Model architecture error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_model_compilation():
    """Test model compilation with medical settings"""
    try:
        from medai_sota_models import StateOfTheArtModels
        import tensorflow as tf
        
        builder = StateOfTheArtModels((224, 224, 3), 5)
        
        vit_model = builder.build_real_vision_transformer()
        
        compiled_model = builder.compile_sota_model(vit_model)
        print('✅ Model compilation successful')
        
        print(f'✅ Optimizer: {type(compiled_model.optimizer).__name__}')
        print(f'✅ Loss function: {compiled_model.loss}')
        print(f'✅ Metrics: {[m.name for m in compiled_model.metrics]}')
        
        return True
        
    except Exception as e:
        print(f'❌ Model compilation error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all SOTA integration tests"""
    print("🧪 Starting SOTA Model Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_sota_imports),
        ("Model Manager Test", test_model_manager),
        ("Model Builder Test", test_model_builder),
        ("Architecture Test", test_model_architectures),
        ("Compilation Test", test_model_compilation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All SOTA model integration tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
