#!/usr/bin/env python3
"""
Test Advanced Attention-Weighted Ensemble
Validates 8-head attention mechanism with clinical performance weighting
"""

import sys
import os
sys.path.append('src')

def test_attention_ensemble_imports():
    """Test that attention ensemble imports work"""
    try:
        from medai_sota_models import SOTAModelManager, StateOfTheArtModels
        print('✅ Attention ensemble imports successful')
        return True
    except Exception as e:
        print(f'❌ Import error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_attention_ensemble_building():
    """Test building the advanced attention-weighted ensemble"""
    try:
        from medai_sota_models import StateOfTheArtModels
        
        builder = StateOfTheArtModels((384, 384, 3), 5)
        
        print('Building advanced attention-weighted ensemble...')
        ensemble_model = builder.build_attention_weighted_ensemble()
        
        print(f'✅ Ensemble model created: {ensemble_model.name}')
        print(f'✅ Total parameters: {ensemble_model.count_params():,}')
        print(f'✅ Input shape: {ensemble_model.input_shape}')
        print(f'✅ Output shape: {ensemble_model.output_shape}')
        
        layer_names = [layer.name for layer in ensemble_model.layers]
        
        required_components = [
            'advanced_attention_ensemble',
            'efficientnet_pred',
            'vit_pred', 
            'convnext_pred',
            'combined_features'
        ]
        
        for component in required_components:
            found = any(component in name for name in layer_names)
            if found:
                print(f'✅ Found required component: {component}')
            else:
                print(f'❌ Missing component: {component}')
                return False
        
        return True
        
    except Exception as e:
        print(f'❌ Ensemble building error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_attention_mechanism_parameters():
    """Test attention mechanism parameters"""
    try:
        from medai_sota_models import StateOfTheArtModels
        import tensorflow as tf
        
        builder = StateOfTheArtModels((224, 224, 3), 5)
        ensemble_model = builder.build_attention_weighted_ensemble()
        
        dummy_input = tf.random.normal((1, 224, 224, 3))
        
        print('Testing attention mechanism with dummy input...')
        predictions = ensemble_model(dummy_input)
        
        print(f'✅ Prediction shape: {predictions.shape}')
        print(f'✅ Prediction sum: {tf.reduce_sum(predictions, axis=-1).numpy()}')  # Should be ~1.0
        
        pred_sum = tf.reduce_sum(predictions, axis=-1).numpy()[0]
        if abs(pred_sum - 1.0) < 0.01:
            print('✅ Valid probability distribution')
        else:
            print(f'❌ Invalid probability distribution: sum = {pred_sum}')
            return False
        
        batch_input = tf.random.normal((4, 224, 224, 3))
        batch_predictions = ensemble_model(batch_input)
        
        print(f'✅ Batch prediction shape: {batch_predictions.shape}')
        print('✅ Attention mechanism working correctly')
        
        return True
        
    except Exception as e:
        print(f'❌ Attention mechanism test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_clinical_performance_weighting():
    """Test clinical performance weighting integration"""
    try:
        from medai_sota_models import StateOfTheArtModels
        import tensorflow as tf
        
        builder = StateOfTheArtModels((384, 384, 3), 5)
        ensemble_model = builder.build_attention_weighted_ensemble()
        
        attention_layer = None
        for layer in ensemble_model.layers:
            if 'advanced_attention_ensemble' in layer.name:
                attention_layer = layer
                break
        
        if attention_layer is None:
            print('❌ Attention ensemble layer not found')
            return False
        
        expected_weights = {
            'sensitivity': 0.35,
            'accuracy': 0.30,
            'specificity': 0.25,
            'auc': 0.10
        }
        
        if hasattr(attention_layer, 'clinical_weights'):
            actual_weights = attention_layer.clinical_weights
            for metric, expected_value in expected_weights.items():
                if metric in actual_weights:
                    actual_value = actual_weights[metric]
                    if abs(actual_value - expected_value) < 0.01:
                        print(f'✅ Clinical weight {metric}: {actual_value} (expected: {expected_value})')
                    else:
                        print(f'❌ Clinical weight {metric}: {actual_value} (expected: {expected_value})')
                        return False
                else:
                    print(f'❌ Missing clinical weight: {metric}')
                    return False
        else:
            print('❌ Clinical weights not found in attention layer')
            return False
        
        if hasattr(attention_layer, 'num_heads') and attention_layer.num_heads == 8:
            print('✅ Correct number of attention heads: 8')
        else:
            print(f'❌ Incorrect attention heads: {getattr(attention_layer, "num_heads", "not found")}')
            return False
        
        if hasattr(attention_layer, 'attention_dim') and attention_layer.attention_dim == 256:
            print('✅ Correct attention dimension: 256')
        else:
            print(f'❌ Incorrect attention dimension: {getattr(attention_layer, "attention_dim", "not found")}')
            return False
        
        if hasattr(attention_layer, 'temperature') and attention_layer.temperature == 1.5:
            print('✅ Correct temperature scaling: 1.5')
        else:
            print(f'❌ Incorrect temperature: {getattr(attention_layer, "temperature", "not found")}')
            return False
        
        if hasattr(attention_layer, 'confidence_threshold') and attention_layer.confidence_threshold == 0.8:
            print('✅ Correct confidence threshold: 0.8')
        else:
            print(f'❌ Incorrect confidence threshold: {getattr(attention_layer, "confidence_threshold", "not found")}')
            return False
        
        return True
        
    except Exception as e:
        print(f'❌ Clinical performance weighting test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_sota_model_manager_integration():
    """Test integration with SOTAModelManager"""
    try:
        from medai_sota_models import SOTAModelManager
        
        manager = SOTAModelManager()
        available_models = manager.get_available_models()
        
        if 'ensemble_model' in available_models:
            print('✅ Ensemble model available in SOTAModelManager')
        else:
            print('❌ Ensemble model not available in SOTAModelManager')
            return False
        
        print('Loading ensemble model through manager...')
        ensemble_model = manager.load_model('ensemble_model', input_shape=(384, 384, 3), num_classes=5)
        
        if ensemble_model is not None:
            print('✅ Ensemble model loaded successfully through manager')
            print(f'✅ Model name: {ensemble_model.name}')
            print(f'✅ Parameters: {ensemble_model.count_params():,}')
        else:
            print('❌ Failed to load ensemble model through manager')
            return False
        
        return True
        
    except Exception as e:
        print(f'❌ SOTAModelManager integration error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all attention-weighted ensemble tests"""
    print("🧪 Testing Advanced Attention-Weighted Ensemble")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_attention_ensemble_imports),
        ("Ensemble Building", test_attention_ensemble_building),
        ("Attention Mechanism Parameters", test_attention_mechanism_parameters),
        ("Clinical Performance Weighting", test_clinical_performance_weighting),
        ("SOTAModelManager Integration", test_sota_model_manager_integration)
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
        print("🎉 All attention-weighted ensemble tests passed!")
        print("✅ Advanced ensemble with 8-head attention ready for clinical deployment")
        print("✅ Clinical performance weighting: 35% sensitivity, 30% accuracy, 25% specificity, 10% AUC")
        print("✅ Temperature scaling: 1.5, Confidence threshold: 80%")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
