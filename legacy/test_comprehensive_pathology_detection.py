#!/usr/bin/env python3
"""
Comprehensive Pathology Detection Test Suite
Tests all pathology detection capabilities to ensure no bias and accurate diagnosis
Based on scientific guide validation requirements
"""

import os
import sys
import numpy as np
import logging
from pathlib import Path
import json
from datetime import datetime

sys.path.append('src')

try:
    from medai_integration_manager import MedAIIntegrationManager
    from medai_clinical_evaluation import ClinicalPerformanceEvaluator
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePathologyTester:
    """Comprehensive testing suite for pathology detection"""
    
    def __init__(self):
        self.results = {
            'test_timestamp': datetime.now().isoformat(),
            'pathology_tests': {},
            'clinical_metrics': {},
            'bias_analysis': {},
            'overall_assessment': {}
        }
        
        try:
            self.integration_manager = MedAIIntegrationManager()
            self.clinical_evaluator = ClinicalPerformanceEvaluator()
            logger.info("✅ AI components initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing AI components: {e}")
            raise
    
    def create_synthetic_test_images(self):
        """Create synthetic test images for each pathology class"""
        logger.info("Creating synthetic test images for pathology testing...")
        
        test_images = {}
        pathologies = ['normal', 'pneumonia', 'pleural_effusion', 'fracture', 'tumor']
        
        for pathology in pathologies:
            if pathology == 'normal':
                image = np.random.normal(0.3, 0.1, (224, 224, 3))
                image = np.clip(image, 0, 1)
            elif pathology == 'pneumonia':
                image = np.random.normal(0.4, 0.15, (224, 224, 3))
                image[80:140, 60:160] += 0.3
                image = np.clip(image, 0, 1)
            elif pathology == 'pleural_effusion':
                image = np.random.normal(0.35, 0.12, (224, 224, 3))
                image[150:220, :] += 0.25
                image = np.clip(image, 0, 1)
            elif pathology == 'fracture':
                image = np.random.normal(0.6, 0.1, (224, 224, 3))
                image[100:120, 80:140] -= 0.4
                image = np.clip(image, 0, 1)
            elif pathology == 'tumor':
                image = np.random.normal(0.4, 0.1, (224, 224, 3))
                y, x = np.ogrid[:224, :224]
                mask = (x - 112)**2 + (y - 112)**2 < 30**2
                image[mask] += 0.3
                image = np.clip(image, 0, 1)
            
            test_images[pathology] = image.astype(np.float32)
        
        logger.info(f"✅ Created {len(test_images)} synthetic test images")
        return test_images
    
    def test_pathology_detection(self, test_images):
        """Test pathology detection for each image type"""
        logger.info("Testing pathology detection capabilities...")
        
        pathology_results = {}
        
        for expected_pathology, image in test_images.items():
            logger.info(f"Testing {expected_pathology} detection...")
            
            try:
                result = self.integration_manager.analyze_image(
                    image, 
                    'chest_xray', 
                    generate_attention_map=False
                )
                
                predicted_class = result.get('predicted_class', 'Normal').lower()
                confidence = result.get('confidence', 0.0)
                all_scores = result.get('all_scores', {})
                
                is_correct = predicted_class == expected_pathology.lower()
                
                pathology_results[expected_pathology] = {
                    'predicted_class': predicted_class,
                    'expected_class': expected_pathology,
                    'confidence': float(confidence),
                    'is_correct': is_correct,
                    'all_scores': {k: float(v) for k, v in all_scores.items()},
                    'processing_time': result.get('processing_time', 0.0)
                }
                
                logger.info(f"  Expected: {expected_pathology}")
                logger.info(f"  Predicted: {predicted_class}")
                logger.info(f"  Confidence: {confidence:.3f}")
                logger.info(f"  Correct: {'✅' if is_correct else '❌'}")
                
            except Exception as e:
                logger.error(f"❌ Error testing {expected_pathology}: {e}")
                pathology_results[expected_pathology] = {
                    'error': str(e),
                    'is_correct': False
                }
        
        self.results['pathology_tests'] = pathology_results
        return pathology_results
    
    def analyze_bias(self, pathology_results):
        """Analyze for pneumonia bias and other biases"""
        logger.info("Analyzing model bias...")
        
        bias_analysis = {
            'pneumonia_bias': False,
            'bias_score': 0.0,
            'prediction_distribution': {},
            'confidence_analysis': {},
            'bias_warnings': []
        }
        
        predictions = [result.get('predicted_class', 'unknown') for result in pathology_results.values()]
        unique_predictions = list(set(predictions))
        
        for pred_class in unique_predictions:
            count = predictions.count(pred_class)
            bias_analysis['prediction_distribution'][pred_class] = count
        
        pneumonia_predictions = predictions.count('pneumonia')
        total_predictions = len(predictions)
        pneumonia_ratio = pneumonia_predictions / total_predictions if total_predictions > 0 else 0
        
        if pneumonia_ratio > 0.6:  # More than 60% pneumonia predictions
            bias_analysis['pneumonia_bias'] = True
            bias_analysis['bias_score'] = pneumonia_ratio
            bias_analysis['bias_warnings'].append(f"High pneumonia bias detected: {pneumonia_ratio:.1%} of predictions")
        
        for pathology, result in pathology_results.items():
            if 'confidence' in result:
                bias_analysis['confidence_analysis'][pathology] = result['confidence']
        
        non_pneumonia_confidences = [
            result['confidence'] for pathology, result in pathology_results.items()
            if pathology != 'pneumonia' and 'confidence' in result
        ]
        
        if non_pneumonia_confidences:
            avg_non_pneumonia_confidence = np.mean(non_pneumonia_confidences)
            if avg_non_pneumonia_confidence < 0.7:
                bias_analysis['bias_warnings'].append(
                    f"Low confidence in non-pneumonia cases: {avg_non_pneumonia_confidence:.3f}"
                )
        
        self.results['bias_analysis'] = bias_analysis
        logger.info(f"Bias analysis complete. Pneumonia bias: {'❌ DETECTED' if bias_analysis['pneumonia_bias'] else '✅ NOT DETECTED'}")
        
        return bias_analysis
    
    def test_clinical_metrics(self, pathology_results):
        """Test clinical metrics calculation"""
        logger.info("Testing clinical metrics calculation...")
        
        try:
            y_true = []
            y_pred = []
            
            class_mapping = {'normal': 0, 'pneumonia': 1, 'pleural_effusion': 2, 'fracture': 3, 'tumor': 4}
            
            for expected_pathology, result in pathology_results.items():
                if 'predicted_class' in result:
                    y_true.append(class_mapping.get(expected_pathology, 0))
                    y_pred.append(class_mapping.get(result['predicted_class'], 0))
            
            if len(y_true) > 0:
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                
                accuracy = np.mean(y_true == y_pred)
                
                clinical_metrics = {
                    'accuracy': float(accuracy),
                    'total_tests': len(y_true),
                    'correct_predictions': int(np.sum(y_true == y_pred)),
                    'meets_clinical_threshold': accuracy >= 0.85
                }
                
                try:
                    advanced_metrics = self.clinical_evaluator.calculate_metrics(y_true, y_pred)
                    clinical_metrics.update(advanced_metrics)
                except Exception as e:
                    logger.warning(f"Could not calculate advanced clinical metrics: {e}")
                
                self.results['clinical_metrics'] = clinical_metrics
                logger.info(f"Clinical metrics calculated. Accuracy: {accuracy:.3f}")
                
                return clinical_metrics
            else:
                logger.warning("No valid predictions for clinical metrics calculation")
                return {}
                
        except Exception as e:
            logger.error(f"Error calculating clinical metrics: {e}")
            return {'error': str(e)}
    
    def generate_assessment_report(self):
        """Generate overall assessment report"""
        logger.info("Generating comprehensive assessment report...")
        
        pathology_results = self.results.get('pathology_tests', {})
        bias_analysis = self.results.get('bias_analysis', {})
        clinical_metrics = self.results.get('clinical_metrics', {})
        
        correct_predictions = sum(1 for result in pathology_results.values() if result.get('is_correct', False))
        total_tests = len(pathology_results)
        overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0
        
        clinical_ready = (
            overall_accuracy >= 0.85 and
            not bias_analysis.get('pneumonia_bias', False) and
            clinical_metrics.get('meets_clinical_threshold', False)
        )
        
        assessment = {
            'overall_accuracy': overall_accuracy,
            'total_tests': total_tests,
            'correct_predictions': correct_predictions,
            'clinical_ready': clinical_ready,
            'pneumonia_bias_detected': bias_analysis.get('pneumonia_bias', False),
            'bias_warnings': bias_analysis.get('bias_warnings', []),
            'recommendations': []
        }
        
        if not clinical_ready:
            if overall_accuracy < 0.85:
                assessment['recommendations'].append("Improve model accuracy through additional training")
            if bias_analysis.get('pneumonia_bias', False):
                assessment['recommendations'].append("Address pneumonia bias through balanced training data")
            if not clinical_metrics.get('meets_clinical_threshold', False):
                assessment['recommendations'].append("Enhance clinical validation and metrics")
        else:
            assessment['recommendations'].append("System meets clinical standards for deployment")
        
        self.results['overall_assessment'] = assessment
        
        logger.info("="*60)
        logger.info("COMPREHENSIVE PATHOLOGY DETECTION TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"Correct Predictions: {correct_predictions}/{total_tests}")
        logger.info(f"Pneumonia Bias: {'❌ DETECTED' if bias_analysis.get('pneumonia_bias', False) else '✅ NOT DETECTED'}")
        logger.info(f"Clinical Ready: {'✅ YES' if clinical_ready else '❌ NO'}")
        
        if assessment['recommendations']:
            logger.info("\nRecommendations:")
            for rec in assessment['recommendations']:
                logger.info(f"  • {rec}")
        
        return assessment
    
    def save_results(self, filename="comprehensive_pathology_test_results.json"):
        """Save test results to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"✅ Test results saved to {filename}")
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
    
    def run_comprehensive_test(self):
        """Run the complete test suite"""
        logger.info("Starting comprehensive pathology detection test suite...")
        
        try:
            test_images = self.create_synthetic_test_images()
            
            pathology_results = self.test_pathology_detection(test_images)
            
            bias_analysis = self.analyze_bias(pathology_results)
            
            clinical_metrics = self.test_clinical_metrics(pathology_results)
            
            assessment = self.generate_assessment_report()
            
            self.save_results()
            
            logger.info("✅ Comprehensive pathology detection test completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive test: {e}")
            raise

def main():
    """Main function to run comprehensive pathology tests"""
    try:
        tester = ComprehensivePathologyTester()
        results = tester.run_comprehensive_test()
        
        assessment = results.get('overall_assessment', {})
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)
        print(f"✅ Test completed successfully")
        print(f"📊 Overall accuracy: {assessment.get('overall_accuracy', 0):.1%}")
        print(f"🏥 Clinical ready: {'YES' if assessment.get('clinical_ready', False) else 'NO'}")
        print(f"⚖️ Pneumonia bias: {'DETECTED' if assessment.get('pneumonia_bias_detected', False) else 'NOT DETECTED'}")
        
        return 0 if assessment.get('clinical_ready', False) else 1
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
