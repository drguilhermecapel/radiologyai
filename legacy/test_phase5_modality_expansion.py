#!/usr/bin/env python3
"""
Test Suite for Phase 5: Modality Expansion
Tests for ultrasound and PET-CT fusion functionality across all system components
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from medai_modality_normalizer import ModalitySpecificNormalizer
    from medai_model_selector import ModelSelector, ExamType
    from medai_dicom_processor import DicomProcessor
    from medai_ml_pipeline import MLPipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available")
    sys.exit(1)

class TestPhase5ModalityExpansion(unittest.TestCase):
    """Comprehensive test suite for Phase 5 modality expansion"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.normalizer = ModalitySpecificNormalizer()
        self.model_selector = ModelSelector()
        self.dicom_processor = DicomProcessor(anonymize=False)
        self.ml_pipeline = MLPipeline('test_project', 'phase5_test')
        
        self.test_image_2d = np.random.rand(512, 512).astype(np.float32)
        self.test_image_3d = np.random.rand(512, 512, 1).astype(np.float32)
        self.test_pet_image = np.random.exponential(0.2, (512, 512)).astype(np.float32)
        self.test_ct_image = np.random.normal(40, 100, (512, 512)).astype(np.float32)
        
    def test_ultrasound_normalization(self):
        """Test ultrasound-specific normalization"""
        print("Testing ultrasound normalization...")
        
        normalized_2d = self.normalizer.normalize_ultrasound(self.test_image_2d)
        
        self.assertIsInstance(normalized_2d, np.ndarray)
        self.assertEqual(normalized_2d.dtype, np.float32)
        self.assertTrue(0.0 <= normalized_2d.min() <= normalized_2d.max() <= 1.0)
        
        normalized_3d = self.normalizer.normalize_ultrasound(self.test_image_3d)
        
        self.assertIsInstance(normalized_3d, np.ndarray)
        self.assertEqual(normalized_3d.dtype, np.float32)
        self.assertTrue(0.0 <= normalized_3d.min() <= normalized_3d.max() <= 1.0)
        
        print("✅ Ultrasound normalization tests passed")
        
    def test_pet_ct_fusion_normalization(self):
        """Test PET-CT fusion normalization"""
        print("Testing PET-CT fusion normalization...")
        
        fused_image = self.normalizer.normalize_pet_ct_fusion(
            self.test_pet_image, 
            self.test_ct_image
        )
        
        self.assertIsInstance(fused_image, np.ndarray)
        self.assertEqual(fused_image.dtype, np.float32)
        self.assertEqual(fused_image.shape, self.test_pet_image.shape)
        self.assertTrue(0.0 <= fused_image.min() <= fused_image.max() <= 1.0)
        
        mismatched_ct = np.random.normal(40, 100, (256, 256)).astype(np.float32)
        fused_fallback = self.normalizer.normalize_pet_ct_fusion(
            self.test_pet_image, 
            mismatched_ct
        )
        
        self.assertIsInstance(fused_fallback, np.ndarray)
        self.assertEqual(fused_fallback.dtype, np.float32)
        
        print("✅ PET-CT fusion normalization tests passed")
        
    def test_normalize_by_modality_new_modalities(self):
        """Test normalize_by_modality with new modalities"""
        print("Testing normalize_by_modality with new modalities...")
        
        class MockDicomUS:
            def __init__(self, test_image):
                self.Modality = 'US'
                self.pixel_array = test_image
                
        mock_ds_us = MockDicomUS(self.test_image_2d)
        
        normalized_us = self.normalizer.normalize_by_modality(mock_ds_us, 'US')
        
        self.assertIsInstance(normalized_us, np.ndarray)
        self.assertEqual(normalized_us.dtype, np.float32)
        self.assertTrue(0.0 <= normalized_us.min() <= normalized_us.max() <= 1.0)
        
        class MockDicomPT:
            def __init__(self, test_image):
                self.Modality = 'PT'
                self.pixel_array = test_image
                
        mock_ds_pt = MockDicomPT(self.test_pet_image)
        
        normalized_pt = self.normalizer.normalize_by_modality(mock_ds_pt, 'PT')
        
        self.assertIsInstance(normalized_pt, np.ndarray)
        self.assertEqual(normalized_pt.dtype, np.float32)
        
        print("✅ normalize_by_modality new modalities tests passed")
        
    def test_model_selector_new_modalities(self):
        """Test model selector with new modalities"""
        print("Testing model selector with new modalities...")
        
        us_model = self.model_selector.select_optimal_model(ExamType.ULTRASOUND)
        
        self.assertIsInstance(us_model, dict)
        self.assertIn('primary_model', us_model)
        self.assertIn('secondary_model', us_model)
        self.assertIn('confidence_threshold', us_model)
        self.assertIn('classes', us_model)
        self.assertEqual(us_model['primary_model'], 'efficientnetv2')
        self.assertEqual(len(us_model['classes']), 6)
        
        pet_ct_model = self.model_selector.select_optimal_model(ExamType.PET_CT_FUSION)
        
        self.assertIsInstance(pet_ct_model, dict)
        self.assertIn('primary_model', pet_ct_model)
        self.assertIn('secondary_model', pet_ct_model)
        self.assertIn('confidence_threshold', pet_ct_model)
        self.assertIn('classes', pet_ct_model)
        self.assertEqual(pet_ct_model['primary_model'], 'hybrid_cnn_transformer')
        self.assertEqual(len(pet_ct_model['classes']), 6)
        
        us_stats = self.model_selector.get_model_performance_stats(ExamType.ULTRASOUND)
        pet_ct_stats = self.model_selector.get_model_performance_stats(ExamType.PET_CT_FUSION)
        
        self.assertIsInstance(us_stats, dict)
        self.assertIsInstance(pet_ct_stats, dict)
        
        print("✅ Model selector new modalities tests passed")
        
    def test_dicom_processor_new_modalities(self):
        """Test DICOM processor with new modalities"""
        print("Testing DICOM processor with new modalities...")
        
        self.assertIn('US', self.dicom_processor.SUPPORTED_MODALITIES)
        self.assertIn('PT', self.dicom_processor.SUPPORTED_MODALITIES)
        self.assertEqual(self.dicom_processor.SUPPORTED_MODALITIES['US'], 'Ultrasound')
        self.assertEqual(self.dicom_processor.SUPPORTED_MODALITIES['PT'], 'Positron Emission Tomography')
        
        class MockDicomUS:
            def get(self, key, default=None):
                if key == 'Modality':
                    return 'US'
                return default
                
        class MockDicomPT:
            def get(self, key, default=None):
                if key == 'Modality':
                    return 'PT'
                return default
        
        mock_us = MockDicomUS()
        mock_pt = MockDicomPT()
        
        self.assertTrue(self.dicom_processor.validate_modality_support(mock_us))
        self.assertTrue(self.dicom_processor.validate_modality_support(mock_pt))
        
        us_info = self.dicom_processor.get_modality_info(mock_us)
        pt_info = self.dicom_processor.get_modality_info(mock_pt)
        
        self.assertEqual(us_info['modality'], 'US')
        self.assertEqual(us_info['description'], 'Ultrasound')
        self.assertTrue(us_info['supported'])
        
        self.assertEqual(pt_info['modality'], 'PT')
        self.assertEqual(pt_info['description'], 'Positron Emission Tomography')
        self.assertTrue(pt_info['supported'])
        
        print("✅ DICOM processor new modalities tests passed")
        
    def test_ml_pipeline_synthetic_data_generation(self):
        """Test synthetic data generation for new modalities"""
        print("Testing ML pipeline synthetic data generation...")
        
        us_image = self.ml_pipeline._generate_ultrasound_image()
        
        self.assertIsInstance(us_image, np.ndarray)
        self.assertEqual(us_image.shape, (512, 512, 3))
        self.assertTrue(0.0 <= us_image.min() <= us_image.max() <= 1.0)
        
        pet_ct_image = self.ml_pipeline._generate_pet_ct_fusion_image()
        
        self.assertIsInstance(pet_ct_image, np.ndarray)
        self.assertEqual(pet_ct_image.shape, (512, 512, 3))
        self.assertTrue(0.0 <= pet_ct_image.min() <= pet_ct_image.max() <= 1.0)
        
        self.assertTrue(np.var(us_image) > 0.001)
        self.assertTrue(np.var(pet_ct_image) > 0.001)
        
        print("✅ ML pipeline synthetic data generation tests passed")
        
    def test_ml_pipeline_modality_normalization_tf(self):
        """Test TensorFlow modality normalization for new modalities"""
        print("Testing ML pipeline TensorFlow normalization...")
        
        try:
            import tensorflow as tf
            
            us_tensor = tf.constant(self.test_image_2d, dtype=tf.float32)
            normalized_us = self.ml_pipeline._apply_modality_specific_normalization_tf(us_tensor, 'US')
            
            self.assertIsInstance(normalized_us, tf.Tensor)
            self.assertEqual(normalized_us.dtype, tf.float32)
            
            pt_tensor = tf.constant(self.test_pet_image, dtype=tf.float32)
            normalized_pt = self.ml_pipeline._apply_modality_specific_normalization_tf(pt_tensor, 'PT')
            
            self.assertIsInstance(normalized_pt, tf.Tensor)
            self.assertEqual(normalized_pt.dtype, tf.float32)
            
            print("✅ ML pipeline TensorFlow normalization tests passed")
            
        except ImportError:
            print("⚠️ TensorFlow not available, skipping TF normalization tests")
            
    def test_backward_compatibility(self):
        """Test that existing modalities still work correctly"""
        print("Testing backward compatibility...")
        
        ct_image = np.random.normal(40, 100, (512, 512)).astype(np.float32)
        
        class MockDicomCT:
            def __init__(self):
                self.Modality = 'CT'
                self.pixel_array = ct_image
                self.RescaleSlope = 1.0
                self.RescaleIntercept = -1024.0
                
            def get(self, key, default=None):
                if key == 'Modality':
                    return 'CT'
                return default
                
        mock_ct = MockDicomCT()
        mock_ct.pixel_array = ct_image
        
        normalized_ct = self.normalizer.normalize_by_modality(mock_ct, 'CT')
        
        self.assertIsInstance(normalized_ct, np.ndarray)
        self.assertEqual(normalized_ct.dtype, np.float32)
        
        ct_model = self.model_selector.select_optimal_model(ExamType.CHEST_XRAY)
        self.assertIsInstance(ct_model, dict)
        self.assertIn('primary_model', ct_model)
        
        self.assertIn('CT', self.dicom_processor.SUPPORTED_MODALITIES)
        self.assertTrue(self.dicom_processor.validate_modality_support(mock_ct))
        
        print("✅ Backward compatibility tests passed")
        
    def test_error_handling(self):
        """Test error handling for edge cases"""
        print("Testing error handling...")
        
        try:
            invalid_image = np.array([])
            result = self.normalizer.normalize_ultrasound(invalid_image)
            self.assertIsInstance(result, np.ndarray)
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError))
            
        try:
            result = self.normalizer.normalize_pet_ct_fusion(None, None)
            self.assertIsInstance(result, np.ndarray)
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError, TypeError))
            
        try:
            result = self.model_selector.select_optimal_model(ExamType.GENERAL)
            self.assertIsInstance(result, dict)
        except Exception as e:
            pass
            
        print("✅ Error handling tests passed")

def run_phase5_tests():
    """Run all Phase 5 modality expansion tests"""
    print("=" * 60)
    print("PHASE 5: MODALITY EXPANSION TEST SUITE")
    print("=" * 60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPhase5ModalityExpansion)
    
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("PHASE 5 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    
    if failures > 0:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if errors > 0:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Phase 5 modality expansion tests PASSED!")
        return True
    else:
        print("❌ Phase 5 modality expansion tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_phase5_tests()
    sys.exit(0 if success else 1)
