#!/usr/bin/env python3
"""
Test Clinical Validation Framework
Validates enhanced clinical evaluation with condition-specific thresholds
"""

import sys
import os
sys.path.append('src')

def test_clinical_validation_imports():
    """Test that clinical validation framework imports work"""
    try:
        from medai_clinical_evaluation import (
            ClinicalPerformanceEvaluator, 
            ClinicalValidationFramework,
            RadiologyBenchmark
        )
        print('✅ Clinical validation framework imports successful')
        return True
    except Exception as e:
        print(f'❌ Import error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_condition_specific_validation():
    """Test condition-specific validation with different thresholds"""
    try:
        from medai_clinical_evaluation import ClinicalValidationFramework
        
        framework = ClinicalValidationFramework()
        
        condition_metrics = {
            'pneumothorax': {  # Critical condition
                'sensitivity': 0.96,
                'specificity': 0.91
            },
            'tumor': {  # Critical condition
                'sensitivity': 0.94,  # Below threshold
                'specificity': 0.92
            },
            'pneumonia': {  # Moderate condition
                'sensitivity': 0.91,
                'specificity': 0.89  # Below threshold
            },
            'pleural_effusion': {  # Moderate condition
                'sensitivity': 0.92,
                'specificity': 0.91
            },
            'normal': {  # Standard condition
                'sensitivity': 0.85,
                'specificity': 0.88
            }
        }
        
        general_metrics = {
            'accuracy': 0.87,
            'recall': 0.89,
            'precision': 0.86
        }
        
        validation_result = framework.validate_for_clinical_use(
            general_metrics, 
            condition_metrics
        )
        
        print(f'✅ Validation completed')
        print(f'✅ Clinical approval: {validation_result["approved_for_clinical_use"]}')
        print(f'✅ Clinical readiness: {validation_result["overall_clinical_readiness"]}')
        
        condition_validation = validation_result['condition_specific_validation']
        
        pneumothorax_validation = condition_validation['pneumothorax']
        assert pneumothorax_validation['condition_type'] == 'CRITICAL'
        assert pneumothorax_validation['sensitivity']['threshold'] == 0.95
        assert pneumothorax_validation['clinical_ready'] == True
        print('✅ Pneumothorax (critical) validation correct')
        
        tumor_validation = condition_validation['tumor']
        assert tumor_validation['condition_type'] == 'CRITICAL'
        assert tumor_validation['sensitivity']['passed'] == False  # 0.94 < 0.95
        print('✅ Tumor (critical) validation correct - failed as expected')
        
        pneumonia_validation = condition_validation['pneumonia']
        assert pneumonia_validation['condition_type'] == 'MODERATE'
        assert pneumonia_validation['sensitivity']['threshold'] == 0.90
        assert pneumonia_validation['specificity']['passed'] == False  # 0.89 < 0.90
        print('✅ Pneumonia (moderate) validation correct')
        
        risk_assessment = validation_result['clinical_risk_assessment']
        tumor_risk = risk_assessment['tumor']
        assert tumor_risk['risk_level'] == 'HIGH_RISK'
        print('✅ Risk assessment correct - tumor marked as HIGH_RISK')
        
        return True
        
    except Exception as e:
        print(f'❌ Condition-specific validation error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_compliance_reporting():
    """Test clinical compliance reporting"""
    try:
        from medai_clinical_evaluation import ClinicalValidationFramework
        
        framework = ClinicalValidationFramework()
        
        perfect_metrics = {
            'pneumothorax': {'sensitivity': 0.96, 'specificity': 0.92},
            'tumor': {'sensitivity': 0.97, 'specificity': 0.93},
            'pneumonia': {'sensitivity': 0.92, 'specificity': 0.91},
            'pleural_effusion': {'sensitivity': 0.93, 'specificity': 0.92},
            'normal': {'sensitivity': 0.85, 'specificity': 0.88}
        }
        
        general_metrics = {'accuracy': 0.90, 'recall': 0.91, 'precision': 0.89}
        
        validation_result = framework.validate_for_clinical_use(
            general_metrics, 
            perfect_metrics
        )
        
        compliance_report = validation_result['compliance_report']
        
        assert compliance_report['regulatory_readiness'] == True
        assert compliance_report['clinical_deployment_readiness'] == True
        assert compliance_report['overall_compliance'] == True
        
        print('✅ Compliance reporting works - perfect metrics approved')
        
        failing_metrics = {
            'pneumothorax': {'sensitivity': 0.93, 'specificity': 0.92},  # Critical fails
            'pneumonia': {'sensitivity': 0.88, 'specificity': 0.91}     # Moderate fails
        }
        
        failing_validation = framework.validate_for_clinical_use(
            general_metrics, 
            failing_metrics
        )
        
        failing_compliance = failing_validation['compliance_report']
        assert failing_compliance['regulatory_readiness'] == False
        print('✅ Compliance reporting correctly identifies failures')
        
        return True
        
    except Exception as e:
        print(f'❌ Compliance reporting error: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_clinical_impact_assessment():
    """Test clinical impact assessment"""
    try:
        from medai_clinical_evaluation import ClinicalValidationFramework
        
        framework = ClinicalValidationFramework()
        
        test_cases = [
            ('pneumothorax', 0.96, 0.92, 'READY_FOR_CLINICAL_USE'),
            ('pneumothorax', 0.92, 0.91, 'REQUIRES_RADIOLOGIST_CONFIRMATION'),
            ('pneumothorax', 0.88, 0.90, 'NOT_RECOMMENDED_FOR_CLINICAL_USE'),
            ('pneumonia', 0.92, 0.91, 'READY_FOR_SCREENING'),
            ('pneumonia', 0.87, 0.89, 'SUITABLE_FOR_TRIAGE'),
            ('pneumonia', 0.82, 0.85, 'REQUIRES_IMPROVEMENT'),
            ('normal', 0.85, 0.88, 'SUITABLE_FOR_SUPPORT_TOOL'),
            ('normal', 0.75, 0.80, 'REQUIRES_IMPROVEMENT')
        ]
        
        for condition, sensitivity, specificity, expected_impact in test_cases:
            impact = framework._assess_clinical_impact(condition, sensitivity, specificity)
            assert impact == expected_impact, f"Expected {expected_impact}, got {impact} for {condition}"
            print(f'✅ Clinical impact correct for {condition}: {impact}')
        
        return True
        
    except Exception as e:
        print(f'❌ Clinical impact assessment error: {e}')
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all clinical validation framework tests"""
    print("🧪 Testing Enhanced Clinical Validation Framework")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_clinical_validation_imports),
        ("Condition-Specific Validation", test_condition_specific_validation),
        ("Compliance Reporting", test_compliance_reporting),
        ("Clinical Impact Assessment", test_clinical_impact_assessment)
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
        print("🎉 All clinical validation framework tests passed!")
        print("✅ Enhanced clinical validation ready for medical deployment")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
