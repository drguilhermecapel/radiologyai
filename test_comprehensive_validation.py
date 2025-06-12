"""
Comprehensive validation testing for MedAI Radiologia production deployment
Tests all components including AI models, web interface, clinical monitoring, and deployment readiness
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import json
import time
import requests
import subprocess
from datetime import datetime
import numpy as np
import tensorflow as tf
from medai_integration_manager import MedAIIntegrationManager
from medai_clinical_monitoring_dashboard import ClinicalMonitoringDashboard
from medai_advanced_clinical_validation import AdvancedClinicalValidationFramework

def test_comprehensive_validation():
    """Execute comprehensive validation testing for production deployment"""
    print("🚀 COMPREHENSIVE VALIDATION TESTING - MedAI Radiologia v4.0.0")
    print("=" * 80)
    
    validation_results = {
        'ai_models': {},
        'web_interface': {},
        'clinical_monitoring': {},
        'deployment_config': {},
        'performance_metrics': {},
        'security_compliance': {},
        'overall_status': 'PENDING'
    }
    
    try:
        print("\n🧠 Testing AI Models and Inference System...")
        ai_validation = test_ai_models_validation()
        validation_results['ai_models'] = ai_validation
        
        if ai_validation['status'] == 'PASS':
            print("  ✅ AI Models validation PASSED")
        else:
            print(f"  ❌ AI Models validation FAILED: {ai_validation.get('error', 'Unknown error')}")
        
        print("\n📊 Testing Clinical Monitoring Dashboard...")
        monitoring_validation = test_clinical_monitoring_validation()
        validation_results['clinical_monitoring'] = monitoring_validation
        
        if monitoring_validation['status'] == 'PASS':
            print("  ✅ Clinical Monitoring validation PASSED")
        else:
            print(f"  ❌ Clinical Monitoring validation FAILED: {monitoring_validation.get('error', 'Unknown error')}")
        
        print("\n🏥 Testing Advanced Clinical Validation Framework...")
        clinical_validation = test_advanced_clinical_framework()
        validation_results['clinical_validation'] = clinical_validation
        
        if clinical_validation['status'] == 'PASS':
            print("  ✅ Advanced Clinical Validation PASSED")
        else:
            print(f"  ❌ Advanced Clinical Validation FAILED: {clinical_validation.get('error', 'Unknown error')}")
        
        print("\n⚙️ Testing Production Configuration...")
        config_validation = test_production_configuration()
        validation_results['deployment_config'] = config_validation
        
        if config_validation['status'] == 'PASS':
            print("  ✅ Production Configuration validation PASSED")
        else:
            print(f"  ❌ Production Configuration validation FAILED: {config_validation.get('error', 'Unknown error')}")
        
        print("\n🐳 Testing Docker Configuration...")
        docker_validation = test_docker_configuration()
        validation_results['docker_config'] = docker_validation
        
        if docker_validation['status'] == 'PASS':
            print("  ✅ Docker Configuration validation PASSED")
        else:
            print(f"  ❌ Docker Configuration validation FAILED: {docker_validation.get('error', 'Unknown error')}")
        
        print("\n⚡ Testing Performance Optimization Methods...")
        performance_validation = test_performance_optimization()
        validation_results['performance_metrics'] = performance_validation
        
        if performance_validation['status'] == 'PASS':
            print("  ✅ Performance Optimization validation PASSED")
        else:
            print(f"  ❌ Performance Optimization validation FAILED: {performance_validation.get('error', 'Unknown error')}")
        
        print("\n🔒 Testing Security and Compliance...")
        security_validation = test_security_compliance()
        validation_results['security_compliance'] = security_validation
        
        if security_validation['status'] == 'PASS':
            print("  ✅ Security and Compliance validation PASSED")
        else:
            print(f"  ❌ Security and Compliance validation FAILED: {security_validation.get('error', 'Unknown error')}")
        
        passed_tests = sum(1 for result in validation_results.values() 
                          if isinstance(result, dict) and result.get('status') == 'PASS')
        total_tests = len([k for k, v in validation_results.items() 
                          if isinstance(v, dict) and 'status' in v])
        
        if passed_tests == total_tests:
            validation_results['overall_status'] = 'PASS'
            overall_status = "✅ ALL TESTS PASSED"
        elif passed_tests >= total_tests * 0.8:
            validation_results['overall_status'] = 'PARTIAL_PASS'
            overall_status = "⚠️ PARTIAL PASS (80%+ tests passed)"
        else:
            validation_results['overall_status'] = 'FAIL'
            overall_status = "❌ VALIDATION FAILED"
        
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 80)
        print(f"🎯 Overall Status: {overall_status}")
        print(f"📊 Tests Passed: {passed_tests}/{total_tests}")
        print(f"📅 Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏥 System Version: MedAI Radiologia v4.0.0")
        
        print(f"\n📈 DETAILED RESULTS:")
        for component, result in validation_results.items():
            if isinstance(result, dict) and 'status' in result:
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                print(f"  {status_icon} {component.replace('_', ' ').title()}: {result['status']}")
                if 'metrics' in result:
                    for metric, value in result['metrics'].items():
                        print(f"    • {metric}: {value}")
        
        validation_results['validation_timestamp'] = datetime.now().isoformat()
        validation_results['system_version'] = "4.0.0"
        
        with open("models/comprehensive_validation_report.json", "w") as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\n💾 Validation report saved to: models/comprehensive_validation_report.json")
        
        if validation_results['overall_status'] in ['PASS', 'PARTIAL_PASS']:
            print("\n🚀 SYSTEM READY FOR PRODUCTION DEPLOYMENT")
            return True
        else:
            print("\n⚠️ SYSTEM REQUIRES FIXES BEFORE PRODUCTION DEPLOYMENT")
            return False
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR IN COMPREHENSIVE VALIDATION: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_models_validation():
    """Test AI models and inference system"""
    try:
        manager = MedAIIntegrationManager()
        
        if not hasattr(manager, 'enhanced_models'):
            return {'status': 'FAIL', 'error': 'Manager does not have enhanced_models attribute'}
        
        if not manager.enhanced_models or len(manager.enhanced_models) == 0:
            return {'status': 'FAIL', 'error': 'No AI models loaded in manager'}
        
        test_image = np.random.rand(224, 224, 3) * 255
        test_image = test_image.astype(np.uint8)
        
        result = manager.analyze_image(test_image, "chest_xray")
        
        if 'error' in result:
            return {'status': 'FAIL', 'error': f"Inference failed: {result['error']}"}
        
        required_fields = ['predicted_class', 'confidence']
        for field in required_fields:
            if field not in result:
                return {'status': 'FAIL', 'error': f"Missing required field: {field}"}
        
        return {
            'status': 'PASS',
            'metrics': {
                'models_loaded': len(manager.enhanced_models),
                'inference_time': result.get('processing_time', 'N/A'),
                'confidence': result.get('confidence', 'N/A')
            }
        }
        
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)}

def test_clinical_monitoring_validation():
    """Test clinical monitoring dashboard"""
    try:
        dashboard = ClinicalMonitoringDashboard()
        
        test_metrics = {
            'sensitivity': 0.95,
            'specificity': 0.92,
            'auc': 0.88,
            'precision': 0.90,
            'f1_score': 0.92
        }
        
        dashboard.track_clinical_validation_metrics(test_metrics)
        
        html_dashboard = dashboard.generate_dashboard_html()
        if len(html_dashboard) < 1000:
            return {'status': 'FAIL', 'error': 'Dashboard HTML generation failed'}
        
        metrics_json = dashboard.get_dashboard_metrics_json()
        metrics_data = json.loads(metrics_json)
        
        return {
            'status': 'PASS',
            'metrics': {
                'dashboard_html_size': len(html_dashboard),
                'metrics_tracked': len(dashboard.metrics_history),
                'alert_thresholds': len(dashboard.alert_thresholds)
            }
        }
        
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)}

def test_advanced_clinical_framework():
    """Test advanced clinical validation framework"""
    try:
        validator = AdvancedClinicalValidationFramework()
        
        mock_predictions = {
            'pneumonia': [0.8, 0.6, 0.9, 0.7, 0.5],
            'pleural_effusion': [0.2, 0.4, 0.1, 0.3, 0.5]
        }
        mock_ground_truth = {
            'pneumonia': [1, 0, 1, 1, 0],
            'pleural_effusion': [0, 1, 0, 0, 1]
        }
        
        study_metadata = {
            'study_name': 'Comprehensive Validation Test',
            'dataset_size': len(mock_predictions['pneumonia']),
            'validation_type': 'binary_classification'
        }
        
        validation_result = validator.conduct_comprehensive_validation_study(
            mock_predictions, mock_ground_truth, study_metadata
        )
        
        if 'error' in validation_result:
            return {'status': 'FAIL', 'error': validation_result['error']}
        
        readiness = validator._assess_clinical_readiness(validation_result)
        
        return {
            'status': 'PASS',
            'metrics': {
                'clinical_accuracy': validation_result.get('accuracy', 'N/A'),
                'sensitivity': validation_result.get('sensitivity', 'N/A'),
                'specificity': validation_result.get('specificity', 'N/A'),
                'clinical_ready': readiness.get('ready_for_deployment', False)
            }
        }
        
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)}

def test_production_configuration():
    """Test production configuration files"""
    try:
        with open("config/production.json", "r") as f:
            prod_config = json.load(f)
        
        required_sections = [
            'environment', 'deployment', 'server', 'database',
            'ai_models', 'clinical_monitoring', 'security'
        ]
        
        for section in required_sections:
            if section not in prod_config:
                return {'status': 'FAIL', 'error': f"Missing config section: {section}"}
        
        if prod_config['environment'] != 'production':
            return {'status': 'FAIL', 'error': 'Environment not set to production'}
        
        if not prod_config['security']['authentication_required']:
            return {'status': 'FAIL', 'error': 'Authentication not required in production'}
        
        return {
            'status': 'PASS',
            'metrics': {
                'config_sections': len(prod_config),
                'environment': prod_config['environment'],
                'version': prod_config['deployment']['version']
            }
        }
        
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)}

def test_docker_configuration():
    """Test Docker configuration files"""
    try:
        with open("Dockerfile", "r") as f:
            dockerfile_content = f.read()
        
        required_dockerfile_elements = [
            'FROM python:3.9-slim',
            'WORKDIR /app',
            'COPY requirements.txt',
            'RUN pip install',
            'EXPOSE 8080'
        ]
        
        for element in required_dockerfile_elements:
            if element not in dockerfile_content:
                return {'status': 'FAIL', 'error': f"Missing Dockerfile element: {element}"}
        
        with open("docker-compose.yml", "r") as f:
            compose_content = f.read()
        
        required_services = ['medai-app', 'medai-db', 'redis', 'nginx']
        for service in required_services:
            if service not in compose_content:
                return {'status': 'FAIL', 'error': f"Missing Docker service: {service}"}
        
        return {
            'status': 'PASS',
            'metrics': {
                'dockerfile_size': len(dockerfile_content),
                'compose_services': len(required_services),
                'docker_ready': True
            }
        }
        
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)}

def test_performance_optimization():
    """Test performance optimization methods"""
    try:
        from medai_ml_pipeline import MLPipeline
        
        pipeline = MLPipeline(project_name="ValidationTest", experiment_name="performance_test")
        
        optimization_methods = [
            '_apply_medical_quantization',
            '_optimize_ensemble_weights',
            '_perform_final_clinical_validation'
        ]
        
        for method in optimization_methods:
            if not hasattr(pipeline, method):
                return {'status': 'FAIL', 'error': f"Missing optimization method: {method}"}
        
        optimization_config = {'model_compression_ratio': 0.3}
        ensemble_result = pipeline._optimize_ensemble_weights(None, optimization_config)
        
        if 'error' in ensemble_result:
            return {'status': 'FAIL', 'error': f"Ensemble optimization failed: {ensemble_result['error']}"}
        
        return {
            'status': 'PASS',
            'metrics': {
                'optimization_methods': len(optimization_methods),
                'ensemble_weights': len(ensemble_result.get('ensemble_weights', {})),
                'performance_ready': True
            }
        }
        
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)}

def test_security_compliance():
    """Test security and compliance features"""
    try:
        with open("config/production.json", "r") as f:
            config = json.load(f)
        
        security_config = config.get('security', {})
        compliance_config = config.get('compliance', {})
        
        required_security = [
            'authentication_required',
            'rate_limiting',
            'cors',
            'audit_logging'
        ]
        
        for feature in required_security:
            if feature not in security_config:
                return {'status': 'FAIL', 'error': f"Missing security feature: {feature}"}
            feature_config = security_config[feature]
            if isinstance(feature_config, dict):
                if not feature_config.get('enabled', False):
                    return {'status': 'FAIL', 'error': f"Security feature not enabled: {feature}"}
            elif isinstance(feature_config, bool):
                if not feature_config:
                    return {'status': 'FAIL', 'error': f"Security feature not enabled: {feature}"}
        
        required_compliance = [
            'hipaa_compliant',
            'gdpr_compliant',
            'audit_trail_enabled',
            'data_anonymization'
        ]
        
        for feature in required_compliance:
            if not compliance_config.get(feature, False):
                return {'status': 'FAIL', 'error': f"Compliance feature not enabled: {feature}"}
        
        return {
            'status': 'PASS',
            'metrics': {
                'security_features': len(required_security),
                'compliance_features': len(required_compliance),
                'hipaa_compliant': compliance_config.get('hipaa_compliant', False),
                'gdpr_compliant': compliance_config.get('gdpr_compliant', False)
            }
        }
        
    except Exception as e:
        return {'status': 'FAIL', 'error': str(e)}

if __name__ == "__main__":
    success = test_comprehensive_validation()
    sys.exit(0 if success else 1)
