#!/usr/bin/env python3
"""
Test Enhanced Training Pipeline with SOTA Models
Validates that the training pipeline works with integrated EfficientNetV2, ViT, and ConvNeXt
"""

import sys
import os
sys.path.append('src')

def test_training_pipeline_imports():
    """Test that enhanced training pipeline imports work"""
    try:
        from medai_ml_pipeline import MLPipeline, DatasetConfig, ModelConfig, TrainingConfig
        from medai_sota_models import StateOfTheArtModels
        print('✅ Enhanced training pipeline imports successful')
        return True
    except Exception as e:
        print(f'❌ Import error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_model_building_integration():
    """Test that MLPipeline can build SOTA models"""
    try:
        from medai_ml_pipeline import MLPipeline, ModelConfig
        
        pipeline = MLPipeline(
            project_name="Test_SOTA_Integration",
            experiment_name="pipeline_test"
        )
        
        efficientnet_config = ModelConfig(
            architecture='EfficientNetV2',
            input_shape=(384, 384, 3),
            num_classes=5
        )
        
        efficientnet_model = pipeline.build_model(efficientnet_config)
        print(f'✅ EfficientNetV2 built via pipeline: {efficientnet_model.count_params():,} params')
        
        vit_config = ModelConfig(
            architecture='VisionTransformer',
            input_shape=(224, 224, 3),
            num_classes=5
        )
        
        vit_model = pipeline.build_model(vit_config)
        print(f'✅ Vision Transformer built via pipeline: {vit_model.count_params():,} params')
        
        convnext_config = ModelConfig(
            architecture='ConvNeXt',
            input_shape=(256, 256, 3),
            num_classes=5
        )
        
        convnext_model = pipeline.build_model(convnext_config)
        print(f'✅ ConvNeXt built via pipeline: {convnext_model.count_params():,} params')
        
        return True
        
    except Exception as e:
        print(f'❌ Model building error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_training_configuration():
    """Test enhanced training configuration"""
    try:
        import train_models
        
        eff_config = train_models.get_model_config('EfficientNetV2', (384, 384, 3), 5, 0.001)
        expected_keys = ['input_shape', 'batch_size', 'learning_rate', 'epochs_default', 'freeze_layers', 'fine_tuning_lr', 'preprocessing']
        
        for key in expected_keys:
            if key not in eff_config:
                print(f'❌ Missing key {key} in EfficientNetV2 config')
                return False
        
        print(f'✅ EfficientNetV2 config: {eff_config["input_shape"]}, batch_size={eff_config["batch_size"]}')
        
        vit_config = train_models.get_model_config('VisionTransformer', (224, 224, 3), 5, 0.001)
        print(f'✅ VisionTransformer config: {vit_config["input_shape"]}, batch_size={vit_config["batch_size"]}')
        
        convnext_config = train_models.get_model_config('ConvNeXt', (256, 256, 3), 5, 0.001)
        print(f'✅ ConvNeXt config: {convnext_config["input_shape"]}, batch_size={convnext_config["batch_size"]}')
        
        ensemble_config = train_models.get_model_config('Ensemble', (384, 384, 3), 5, 0.001)
        print(f'✅ Ensemble config: {ensemble_config["input_shape"]}, batch_size={ensemble_config["batch_size"]}')
        
        return True
        
    except Exception as e:
        print(f'❌ Training configuration error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_fallback_functionality():
    """Test that fallback models work when SOTA models fail"""
    try:
        from medai_ml_pipeline import MLPipeline, ModelConfig
        
        pipeline = MLPipeline(
            project_name="Test_Fallback",
            experiment_name="fallback_test"
        )
        
        unknown_config = ModelConfig(
            architecture='UnknownArchitecture',
            input_shape=(224, 224, 3),
            num_classes=5
        )
        
        fallback_model = pipeline.build_model(unknown_config)
        print(f'✅ Fallback model works: {fallback_model.count_params():,} params')
        
        return True
        
    except Exception as e:
        print(f'❌ Fallback functionality error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all enhanced training pipeline tests"""
    print("🧪 Testing Enhanced Training Pipeline with SOTA Models")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_training_pipeline_imports),
        ("Model Building Integration", test_model_building_integration),
        ("Training Configuration", test_training_configuration),
        ("Fallback Functionality", test_fallback_functionality)
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
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enhanced training pipeline tests passed!")
        print("✅ Training pipeline successfully enhanced with SOTA models")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
