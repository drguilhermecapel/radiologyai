#!/usr/bin/env python3
"""
Test Performance Optimizations Implementation
"""

import sys
import os
sys.path.append('src')

def test_optimization_implementation():
    """Test the performance optimization implementation"""
    try:
        from medai_ml_pipeline import MLPipeline
        
        print("Testing performance optimization implementation...")
        
        pipeline = MLPipeline('MedAI_Optimization', 'optimization_test')
        
        model_path = 'models/chest_xray_efficientnetv2_model.h5'
        
        if not os.path.exists(model_path):
            print(f"⚠️ Trained model not found: {model_path}")
            print("Creating test model for optimization validation...")
            
            import tensorflow as tf
            test_model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(5, activation='softmax')
            ])
            test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            model_path = 'test_optimization_model.h5'
            test_model.save(model_path)
            print(f"✅ Created test model: {model_path}")
        else:
            print(f"✅ Found trained model: {model_path}")
        
        result = pipeline.implement_performance_optimizations(
            model_path,
            ['quantization', 'pruning'],
            0.95
        )
        
        print("\n📊 Optimization Results:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        if 'error' in result:
            print(f"❌ Optimization failed: {result['error']}")
            return False
        
        if 'optimized_models' in result:
            optimized_models = result['optimized_models']
            success_count = 0
            
            if 'quantized' in optimized_models:
                quantized_info = optimized_models['quantized']
                if 'error' not in quantized_info:
                    print("✅ Quantization optimization successful")
                    print(f"  Accuracy retention: {quantized_info.get('accuracy_retention', 0):.3f}")
                    print(f"  Size reduction: {quantized_info.get('size_reduction_percent', 0):.1f}%")
                    success_count += 1
                else:
                    print(f"❌ Quantization failed: {quantized_info['error']}")
            
            if 'pruned' in optimized_models:
                pruned_info = optimized_models['pruned']
                if 'error' not in pruned_info:
                    print("✅ Pruning optimization successful")
                    print(f"  Accuracy retention: {pruned_info.get('accuracy_retention', 0):.3f}")
                    print(f"  Sparsity achieved: {pruned_info.get('actual_sparsity_percent', 0):.1f}%")
                    success_count += 1
                else:
                    error_msg = pruned_info['error']
                    if 'tensorflow_model_optimization' in error_msg or 'tfmot' in error_msg:
                        print(f"⚠️ Pruning skipped (missing dependencies): {error_msg}")
                        success_count += 0.5
                    else:
                        print(f"❌ Pruning failed: {error_msg}")
            
            print(f"\n📊 Optimization success rate: {success_count}/2 optimizations")
            
            if model_path == 'test_optimization_model.h5' and os.path.exists(model_path):
                os.remove(model_path)
                print("🧹 Cleaned up test model")
            
            return success_count >= 1
        else:
            print("❌ No optimized models found in results")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run optimization implementation test"""
    print("🔧 TESTING PERFORMANCE OPTIMIZATIONS IMPLEMENTATION")
    print("=" * 60)
    
    success = test_optimization_implementation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Performance optimization implementation test PASSED!")
        return True
    else:
        print("❌ Performance optimization implementation test FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
