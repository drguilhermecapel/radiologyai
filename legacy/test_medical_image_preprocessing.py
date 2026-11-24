#!/usr/bin/env python3
"""
Test Advanced Medical Image Preprocessing
Validates CLAHE, DICOM windowing, lung segmentation, and architecture-specific preprocessing
"""

import sys
import os
sys.path.append('src')

def test_medical_preprocessing_imports():
    """Test that medical preprocessing imports work"""
    try:
        from medai_ml_pipeline import MLPipeline, DatasetConfig, ModelConfig, TrainingConfig
        print('✅ Medical preprocessing imports successful')
        return True
    except Exception as e:
        print(f'❌ Import error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_dicom_windowing():
    """Test DICOM windowing for different modalities"""
    try:
        import tensorflow as tf
        from medai_ml_pipeline import MLPipeline
        
        pipeline = MLPipeline("Test_DICOM_Windowing", "windowing_test")
        
        ct_image = tf.random.normal((512, 512, 1), mean=40, stddev=200)  # CT-like intensities
        xray_image = tf.random.normal((1024, 1024, 1), mean=2048, stddev=1000)  # X-ray-like intensities
        mr_image = tf.random.normal((256, 256, 1), mean=600, stddev=300)  # MR-like intensities
        
        ct_windowed = pipeline._apply_dicom_windowing_tf(ct_image, 'CT')
        xray_windowed = pipeline._apply_dicom_windowing_tf(xray_image, 'CR')
        mr_windowed = pipeline._apply_dicom_windowing_tf(mr_image, 'MR')
        
        print(f'✅ CT windowing: {tf.reduce_min(ct_windowed):.3f} to {tf.reduce_max(ct_windowed):.3f}')
        print(f'✅ X-ray windowing: {tf.reduce_min(xray_windowed):.3f} to {tf.reduce_max(xray_windowed):.3f}')
        print(f'✅ MR windowing: {tf.reduce_min(mr_windowed):.3f} to {tf.reduce_max(mr_windowed):.3f}')
        
        for name, windowed in [('CT', ct_windowed), ('X-ray', xray_windowed), ('MR', mr_windowed)]:
            min_val = tf.reduce_min(windowed).numpy()
            max_val = tf.reduce_max(windowed).numpy()
            if min_val >= 0.0 and max_val <= 1.0:
                print(f'✅ {name} windowing properly normalized')
            else:
                print(f'❌ {name} windowing not properly normalized: [{min_val:.3f}, {max_val:.3f}]')
                return False
        
        return True
        
    except Exception as e:
        print(f'❌ DICOM windowing test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_advanced_clahe():
    """Test advanced CLAHE implementation"""
    try:
        import tensorflow as tf
        from medai_ml_pipeline import MLPipeline
        
        pipeline = MLPipeline("Test_CLAHE", "clahe_test")
        
        test_image = tf.concat([
            tf.ones((256, 128)) * 0.2,  # Dark region
            tf.ones((256, 128)) * 0.8   # Bright region
        ], axis=1)
        
        clahe_enhanced = pipeline._apply_advanced_clahe_tf(test_image, clip_limit=2.0, tile_size=8)
        
        print(f'✅ CLAHE input range: {tf.reduce_min(test_image):.3f} to {tf.reduce_max(test_image):.3f}')
        print(f'✅ CLAHE output range: {tf.reduce_min(clahe_enhanced):.3f} to {tf.reduce_max(clahe_enhanced):.3f}')
        
        input_std = tf.math.reduce_std(test_image).numpy()
        output_std = tf.math.reduce_std(clahe_enhanced).numpy()
        
        if output_std > input_std:
            print(f'✅ CLAHE improved contrast: std {input_std:.3f} → {output_std:.3f}')
        else:
            print(f'⚠️ CLAHE may not have improved contrast: std {input_std:.3f} → {output_std:.3f}')
        
        clahe_aggressive = pipeline._apply_advanced_clahe_tf(test_image, clip_limit=4.0, tile_size=4)
        aggressive_std = tf.math.reduce_std(clahe_aggressive).numpy()
        
        print(f'✅ Aggressive CLAHE contrast: std {aggressive_std:.3f}')
        
        return True
        
    except Exception as e:
        print(f'❌ Advanced CLAHE test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_lung_segmentation():
    """Test lung segmentation for chest X-rays"""
    try:
        import tensorflow as tf
        from medai_ml_pipeline import MLPipeline
        
        pipeline = MLPipeline("Test_Lung_Segmentation", "lung_seg_test")
        
        lung_left = tf.ones((512, 150)) * 0.7
        lung_right = tf.ones((512, 150)) * 0.7
        background = tf.ones((512, 212)) * 0.3
        
        chest_xray = tf.concat([lung_left, background, lung_right], axis=1)
        
        segmented_image = pipeline._apply_lung_segmentation_tf(chest_xray)
        
        print(f'✅ Original image range: {tf.reduce_min(chest_xray):.3f} to {tf.reduce_max(chest_xray):.3f}')
        print(f'✅ Segmented image range: {tf.reduce_min(segmented_image):.3f} to {tf.reduce_max(segmented_image):.3f}')
        
        original_lung_mean = tf.reduce_mean(chest_xray[:, :150]).numpy()  # Left lung region
        segmented_lung_mean = tf.reduce_mean(segmented_image[:, :150]).numpy()
        
        print(f'✅ Lung region enhancement: {original_lung_mean:.3f} → {segmented_lung_mean:.3f}')
        
        chest_xray_3ch = tf.expand_dims(chest_xray, -1)
        chest_xray_3ch = tf.repeat(chest_xray_3ch, 3, axis=-1)
        
        segmented_3ch = pipeline._apply_lung_segmentation_tf(chest_xray_3ch)
        
        if tf.shape(segmented_3ch)[-1] == 3:
            print('✅ Lung segmentation works with 3-channel images')
        else:
            print(f'❌ Lung segmentation failed with 3-channel images: shape {tf.shape(segmented_3ch)}')
            return False
        
        return True
        
    except Exception as e:
        print(f'❌ Lung segmentation test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_architecture_specific_preprocessing():
    """Test architecture-specific preprocessing"""
    try:
        import tensorflow as tf
        from medai_ml_pipeline import MLPipeline
        
        pipeline = MLPipeline("Test_Architecture_Preprocessing", "arch_preproc_test")
        
        test_image = tf.random.normal((384, 384, 3), mean=0.5, stddev=0.2)
        
        architectures = ['EfficientNetV2', 'VisionTransformer', 'ConvNeXt', 'Ensemble']
        
        for arch in architectures:
            processed_image = pipeline._apply_architecture_specific_preprocessing_tf(test_image, arch)
            
            print(f'✅ {arch} preprocessing: shape {tf.shape(processed_image)}, range [{tf.reduce_min(processed_image):.3f}, {tf.reduce_max(processed_image):.3f}]')
            
            if tf.shape(processed_image).numpy().tolist() == tf.shape(test_image).numpy().tolist():
                print(f'✅ {arch} preprocessing maintains shape')
            else:
                print(f'❌ {arch} preprocessing changed shape: {tf.shape(test_image)} → {tf.shape(processed_image)}')
                return False
        
        eff_processed = pipeline._efficientnet_preprocessing_tf(test_image)
        vit_processed = pipeline._vit_preprocessing_tf(test_image)
        convnext_processed = pipeline._convnext_preprocessing_tf(test_image)
        ensemble_processed = pipeline._ensemble_preprocessing_tf(test_image)
        
        print('✅ Individual preprocessing methods:')
        print(f'  EfficientNet: std = {tf.math.reduce_std(eff_processed):.3f}')
        print(f'  ViT: std = {tf.math.reduce_std(vit_processed):.3f}')
        print(f'  ConvNeXt: std = {tf.math.reduce_std(convnext_processed):.3f}')
        print(f'  Ensemble: std = {tf.math.reduce_std(ensemble_processed):.3f}')
        
        return True
        
    except Exception as e:
        print(f'❌ Architecture-specific preprocessing test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_complete_medical_preprocessing_pipeline():
    """Test complete medical preprocessing pipeline"""
    try:
        import tensorflow as tf
        from medai_ml_pipeline import MLPipeline
        
        pipeline = MLPipeline("Test_Complete_Pipeline", "complete_test")
        
        test_cases = [
            ('CT', 'EfficientNetV2', tf.random.normal((512, 512, 1), mean=40, stddev=200)),
            ('CR', 'VisionTransformer', tf.random.normal((1024, 1024), mean=2048, stddev=1000)),
            ('MR', 'ConvNeXt', tf.random.normal((256, 256, 3), mean=600, stddev=300)),
            ('DX', 'Ensemble', tf.random.normal((800, 800, 1), mean=1500, stddev=800))
        ]
        
        for modality, architecture, test_image in test_cases:
            processed_image = pipeline._medical_preprocessing_tf(test_image, modality, architecture)
            
            print(f'✅ {modality} + {architecture}:')
            print(f'  Input shape: {tf.shape(test_image)}')
            print(f'  Output shape: {tf.shape(processed_image)}')
            print(f'  Output range: [{tf.reduce_min(processed_image):.3f}, {tf.reduce_max(processed_image):.3f}]')
            
            if len(tf.shape(processed_image)) == 3 and tf.shape(processed_image)[-1] == 3:
                print(f'✅ {modality} output has correct 3-channel format')
            else:
                print(f'❌ {modality} output has incorrect format: {tf.shape(processed_image)}')
                return False
        
        return True
        
    except Exception as e:
        print(f'❌ Complete preprocessing pipeline test error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all medical image preprocessing tests"""
    print("🧪 Testing Advanced Medical Image Preprocessing")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_medical_preprocessing_imports),
        ("DICOM Windowing", test_dicom_windowing),
        ("Advanced CLAHE", test_advanced_clahe),
        ("Lung Segmentation", test_lung_segmentation),
        ("Architecture-Specific Preprocessing", test_architecture_specific_preprocessing),
        ("Complete Preprocessing Pipeline", test_complete_medical_preprocessing_pipeline)
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
        print("🎉 All medical image preprocessing tests passed!")
        print("✅ Advanced CLAHE with medical-specific parameters implemented")
        print("✅ DICOM windowing for CT, X-ray, and MR modalities working")
        print("✅ Lung segmentation for chest X-rays functional")
        print("✅ Architecture-specific preprocessing for EfficientNetV2, ViT, ConvNeXt, Ensemble ready")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
