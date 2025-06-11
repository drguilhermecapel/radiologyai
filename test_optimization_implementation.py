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
            print(f"❌ Model not found: {model_path}")
            return False
        
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
            
            if 'quantized' in optimized_models:
                quantized_info = optimized_models['quantized']
                if 'error' not in quantized_info:
                    print("✅ Quantization optimization successful")
                else:
                    print(f"❌ Quantization failed: {quantized_info['error']}")
            
            if 'pruned' in optimized_models:
                pruned_info = optimized_models['pruned']
                if 'error' not in pruned_info:
                    print("✅ Pruning optimization successful")
                else:
                    print(f"❌ Pruning failed: {pruned_info['error']}")
        
        return True
        
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
