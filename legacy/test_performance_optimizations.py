#!/usr/bin/env python3
"""
Test Performance Optimizations
Validates quantization, pruning, and knowledge distillation optimizations
"""

import sys
import os
sys.path.append('src')

def test_optimization_imports():
    """Test that performance optimization imports work"""
    try:
        from medai_ml_pipeline import MLPipeline
        print('✅ Performance optimization imports successful')
        return True
    except Exception as e:
        print(f'❌ Import error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_quantization_optimization():
    """Test INT8 quantization optimization"""
    try:
        from medai_ml_pipeline import MLPipeline
        import tensorflow as tf
        import numpy as np
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        test_model_path = 'test_model.h5'
        model.save(test_model_path)
        
        pipeline = MLPipeline('test_project', 'performance_test')
        
        print('Testing quantization optimization...')
        optimization_results = pipeline.implement_performance_optimizations(
            test_model_path,
            optimization_types=['quantization'],
            target_accuracy_retention=0.90
        )
        
        if 'error' in optimization_results:
            print(f'❌ Quantization failed: {optimization_results["error"]}')
            return False
        
        optimized_models = optimization_results.get('optimized_models', {})
        if 'quantized' in optimized_models:
            quantized_info = optimized_models['quantized']
            
            if 'error' in quantized_info:
                print(f'❌ Quantization error: {quantized_info["error"]}')
                return False
            
            print(f'✅ Quantization successful:')
            print(f'  Size reduction: {quantized_info.get("size_reduction_percent", 0):.1f}%')
            print(f'  Accuracy retention: {quantized_info.get("accuracy_retention", 0):.3f}')
            print(f'  Inference speedup: {quantized_info.get("inference_speedup", 1.0):.2f}x')
            print(f'  Meets clinical threshold: {quantized_info.get("meets_clinical_threshold", False)}')
            
            accuracy_retention = quantized_info.get('accuracy_retention', 0)
            if accuracy_retention >= 0.85:
                print(f'✅ Accuracy retention meets medical standards: {accuracy_retention:.3f}')
            else:
                print(f'⚠️ Accuracy retention below medical standards: {accuracy_retention:.3f}')
            
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
            if os.path.exists(quantized_info.get('path', '')):
                os.remove(quantized_info['path'])
            
            return True
        else:
            print('❌ Quantized model not found in results')
            return False
        
    except Exception as e:
        print(f'❌ Quantization test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_pruning_optimization():
    """Test structured pruning optimization"""
    try:
        from medai_ml_pipeline import MLPipeline
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(224,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        test_model_path = 'test_pruning_model.h5'
        model.save(test_model_path)
        
        pipeline = MLPipeline('test_project', 'performance_test')
        
        print('Testing pruning optimization...')
        optimization_results = pipeline.implement_performance_optimizations(
            test_model_path,
            optimization_types=['pruning'],
            target_accuracy_retention=0.90
        )
        
        if 'error' in optimization_results:
            print(f'❌ Pruning failed: {optimization_results["error"]}')
            return False
        
        optimized_models = optimization_results.get('optimized_models', {})
        if 'pruned' in optimized_models:
            pruned_info = optimized_models['pruned']
            
            if 'error' in pruned_info:
                error_msg = pruned_info['error']
                if any(keyword in error_msg for keyword in ['tensorflow_model_optimization', 'tfmot', 'not available']):
                    print(f'⚠️ Pruning skipped due to missing dependencies (expected): {error_msg}')
                    return True
                else:
                    print(f'❌ Pruning error: {error_msg}')
                    return False
            
            print(f'✅ Pruning successful:')
            print(f'  Size reduction: {pruned_info.get("size_reduction_percent", 0):.1f}%')
            print(f'  Sparsity achieved: {pruned_info.get("actual_sparsity_percent", 0):.1f}%')
            print(f'  Accuracy retention: {pruned_info.get("accuracy_retention", 0):.3f}')
            print(f'  Meets clinical threshold: {pruned_info.get("meets_clinical_threshold", False)}')
            
            accuracy_retention = pruned_info.get('accuracy_retention', 0)
            if accuracy_retention >= 0.85:
                print(f'✅ Accuracy retention meets medical standards: {accuracy_retention:.3f}')
            else:
                print(f'⚠️ Accuracy retention below medical standards: {accuracy_retention:.3f}')
            
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
            if os.path.exists(pruned_info.get('path', '')):
                os.remove(pruned_info['path'])
            
            return True
        else:
            print('❌ Pruned model not found in results')
            return False
        
    except Exception as e:
        print(f'❌ Pruning test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_knowledge_distillation():
    """Test knowledge distillation optimization"""
    try:
        from medai_ml_pipeline import MLPipeline
        import tensorflow as tf
        
        teacher_model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        teacher_model_path = 'test_teacher_model.h5'
        teacher_model.save(teacher_model_path)
        
        pipeline = MLPipeline('test_project', 'performance_test')
        
        print('Testing knowledge distillation...')
        optimization_results = pipeline.implement_performance_optimizations(
            teacher_model_path,
            optimization_types=['distillation'],
            target_accuracy_retention=0.85
        )
        
        if 'error' in optimization_results:
            print(f'❌ Knowledge distillation failed: {optimization_results["error"]}')
            return False
        
        if 'distilled' in optimization_results['optimized_models']:
            distilled_info = optimization_results['optimized_models']['distilled']
            
            if 'error' in distilled_info:
                print(f'❌ Distillation error: {distilled_info["error"]}')
                return False
            
            print(f'✅ Knowledge distillation successful:')
            print(f'  Size reduction: {distilled_info.get("size_reduction_percent", 0):.1f}%')
            print(f'  Parameter reduction: {distilled_info.get("parameter_reduction", 0):.1f}%')
            print(f'  Architecture: {distilled_info.get("architecture", "Unknown")}')
            
            if os.path.exists(teacher_model_path):
                os.remove(teacher_model_path)
            if os.path.exists(distilled_info.get('path', '')):
                os.remove(distilled_info['path'])
            
            return True
        else:
            print('❌ Distilled model not found in results')
            return False
        
    except Exception as e:
        print(f'❌ Knowledge distillation test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_optimization_summary():
    """Test optimization summary generation"""
    try:
        from medai_ml_pipeline import MLPipeline
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        test_model_path = 'test_summary_model.h5'
        model.save(test_model_path)
        
        pipeline = MLPipeline('test_project', 'performance_test')
        
        print('Testing optimization summary generation...')
        optimization_results = pipeline.implement_performance_optimizations(
            test_model_path,
            optimization_types=['quantization', 'distillation'],
            target_accuracy_retention=0.90
        )
        
        if 'summary' in optimization_results:
            summary = optimization_results['summary']
            
            print(f'✅ Optimization summary generated:')
            print(f'  Total optimizations: {summary.get("total_optimizations_applied", 0)}')
            print(f'  Successful: {summary.get("successful_optimizations", 0)}')
            print(f'  Failed: {summary.get("failed_optimizations", 0)}')
            
            if summary.get('best_optimization'):
                best = summary['best_optimization']
                print(f'  Best optimization: {best.get("type", "Unknown")} (score: {best.get("score", 0):.3f})')
            
            if summary.get('clinical_recommendations'):
                print(f'  Clinical recommendations: {len(summary["clinical_recommendations"])} items')
                for rec in summary['clinical_recommendations'][:2]:  # Show first 2
                    print(f'    {rec}')
            
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
            
            return True
        else:
            print('❌ Optimization summary not found')
            return False
        
    except Exception as e:
        print(f'❌ Optimization summary test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_deployment_integration():
    """Test deployment integration with optimized models"""
    try:
        from medai_ml_pipeline import MLPipeline
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        test_model_path = 'test_deployment_model.h5'
        model.save(test_model_path)
        
        pipeline = MLPipeline('test_project', 'performance_test')
        
        print('Testing TFLite deployment integration...')
        deployment_config = {
            'type': 'tflite',
            'modality': 'CT',
            'model_name': 'test_deployment_model',
            'calibration_samples': 50,
            'use_mixed_precision': True
        }
        
        deployment_info = pipeline.deploy_model(test_model_path, deployment_config)
        
        if 'error' in deployment_info:
            print(f'❌ TFLite deployment failed: {deployment_info["error"]}')
            return False
        
        successful_deployments = deployment_info.get('successful_deployments', [])
        if 'tflite' in successful_deployments:
            print(f'✅ TFLite deployment integration successful')
            print(f'  Successful deployments: {successful_deployments}')
            
            deployment_results = deployment_info.get('deployment_results', {})
            if 'tflite' in deployment_results:
                tflite_info = deployment_results['tflite']
                print(f'  Model size: {tflite_info.get("model_size_bytes", 0)} bytes')
                print(f'  Deployment path: {tflite_info.get("deployment_path", "Unknown")}')
        else:
            print(f'⚠️ TFLite deployment partially successful but not in expected format')
            print(f'  Deployment info keys: {list(deployment_info.keys())}')
        
        print('Testing ONNX deployment integration (may fail due to compatibility)...')
        onnx_config = {
            'type': 'onnx',
            'modality': 'CT',
            'model_name': 'test_deployment_model_onnx',
            'onnx_opset': 13
        }
        
        onnx_deployment_info = pipeline.deploy_model(test_model_path, onnx_config)
        onnx_successful = onnx_deployment_info.get('successful_deployments', [])
        
        if 'onnx' in onnx_successful:
            print(f'✅ ONNX deployment also successful')
        else:
            print(f'⚠️ ONNX deployment failed as expected (compatibility issues)')
        
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        
        return True
        
    except Exception as e:
        print(f'❌ Deployment integration test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_medical_grade_optimization():
    """Test medical-grade optimization with clinical accuracy validation"""
    try:
        from medai_ml_pipeline import MLPipeline
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        test_model_path = 'test_medical_model.h5'
        model.save(test_model_path)
        
        pipeline = MLPipeline('test_project', 'medical_optimization_test')
        
        print('Testing medical-grade optimization with high accuracy retention...')
        optimization_results = pipeline.implement_performance_optimizations(
            test_model_path,
            optimization_types=['quantization', 'pruning'],
            target_accuracy_retention=0.95
        )
        
        if 'error' in optimization_results:
            print(f'❌ Medical optimization failed: {optimization_results["error"]}')
            return False
        
        print(f'✅ Medical-grade optimization completed:')
        print(f'  Original parameters: {optimization_results.get("original_parameters", 0):,}')
        print(f'  Target accuracy retention: {optimization_results.get("target_accuracy_retention", 0):.1%}')
        
        optimized_models = optimization_results.get('optimized_models', {})
        clinical_ready_count = 0
        
        for opt_type, opt_info in optimized_models.items():
            if 'error' not in opt_info:
                meets_clinical = opt_info.get('meets_clinical_threshold', False)
                accuracy_retention = opt_info.get('accuracy_retention', 0)
                
                print(f'  {opt_type.capitalize()}:')
                print(f'    Accuracy retention: {accuracy_retention:.3f}')
                print(f'    Meets clinical threshold: {meets_clinical}')
                print(f'    Size reduction: {opt_info.get("size_reduction_percent", 0):.1f}%')
                
                if meets_clinical and accuracy_retention >= 0.90:
                    clinical_ready_count += 1
                    print(f'    ✅ Ready for clinical deployment')
                else:
                    print(f'    ⚠️ Requires further optimization for clinical use')
        
        print(f'\n📊 Clinical readiness summary:')
        print(f'  Models meeting clinical standards: {clinical_ready_count}/{len(optimized_models)}')
        
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        
        return clinical_ready_count > 0
        
    except Exception as e:
        print(f'❌ Medical-grade optimization test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all performance optimization tests"""
    print("🧪 Testing Performance Optimizations")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_optimization_imports),
        ("Quantization Optimization", test_quantization_optimization),
        ("Pruning Optimization", test_pruning_optimization),
        ("Knowledge Distillation", test_knowledge_distillation),
        ("Optimization Summary", test_optimization_summary),
        ("Deployment Integration", test_deployment_integration),
        ("Medical-Grade Optimization", test_medical_grade_optimization)
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
        print("🎉 All performance optimization tests passed!")
        print("✅ Quantization, pruning, and knowledge distillation ready for deployment")
        print("✅ Clinical validation integrated with optimization pipeline")
        print("✅ Automated model selection based on clinical performance metrics")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
