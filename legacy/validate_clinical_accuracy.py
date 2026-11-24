#!/usr/bin/env python3
"""
Clinical Accuracy Validation Script
Validates trained models against clinical accuracy requirements (>85%)
"""

import os
import sys
import json
import logging
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from medai_clinical_continuous_validator import ClinicalContinuousValidator, ValidationResult
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ClinicalValidation')

def load_training_results():
    """Load training results from the clinical training directory"""
    training_dir = Path("models/clinical_training")
    
    results = {}
    
    report_file = training_dir / "comprehensive_training_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            results['comprehensive_report'] = json.load(f)
    
    history_file = training_dir / "efficientnetv2_s_progressive_history.json"
    if history_file.exists():
        with open(history_file, 'r') as f:
            results['training_history'] = json.load(f)
    
    return results

def validate_clinical_accuracy_targets(training_results):
    """Validate if models meet clinical accuracy targets"""
    logger.info("=== CLINICAL ACCURACY VALIDATION ===")
    
    if 'comprehensive_report' not in training_results:
        logger.error("No comprehensive training report found")
        return False
    
    report = training_results['comprehensive_report']
    
    targets = {
        'sensitivity': 0.85,
        'specificity': 0.85, 
        'accuracy': 0.85,
        'auc': 0.85
    }
    
    logger.info(f"Clinical accuracy targets: {targets}")
    
    model_performance = report.get('model_performance', {})
    
    validation_results = []
    
    for model_name, performance in model_performance.items():
        logger.info(f"\n--- Validating {model_name} ---")
        
        metrics = performance.get('metrics', {})
        
        actual_metrics = {
            'sensitivity': metrics.get('sensitivity', 0.0),
            'specificity': metrics.get('specificity', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'auc': metrics.get('auc', 0.0)
        }
        
        logger.info(f"Actual metrics: {actual_metrics}")
        
        meets_targets = {}
        overall_pass = True
        
        for metric, target in targets.items():
            actual = actual_metrics[metric]
            meets = actual >= target
            meets_targets[metric] = meets
            
            if not meets:
                overall_pass = False
            
            status = "✅ PASS" if meets else "❌ FAIL"
            logger.info(f"{metric.capitalize()}: {actual:.3f} (target: {target:.3f}) {status}")
        
        clinical_ready = performance.get('clinical_ready', False)
        logger.info(f"Clinical ready: {'✅ YES' if clinical_ready else '❌ NO'}")
        
        validation_results.append({
            'model': model_name,
            'metrics': actual_metrics,
            'meets_targets': meets_targets,
            'overall_pass': overall_pass,
            'clinical_ready': clinical_ready
        })
    
    return validation_results

def generate_validation_report(validation_results, training_results):
    """Generate comprehensive clinical validation report"""
    logger.info("\n=== GENERATING VALIDATION REPORT ===")
    
    report = {
        'validation_timestamp': '2025-06-13T03:04:00Z',
        'clinical_accuracy_validation': {
            'overall_summary': {
                'total_models_validated': len(validation_results),
                'models_meeting_targets': sum(1 for r in validation_results if r['overall_pass']),
                'clinical_ready_models': sum(1 for r in validation_results if r['clinical_ready']),
                'validation_success_rate': 0.0
            },
            'model_results': validation_results
        },
        'clinical_recommendations': [],
        'training_data_summary': training_results.get('comprehensive_report', {}).get('training_summary', {})
    }
    
    if len(validation_results) > 0:
        report['clinical_accuracy_validation']['overall_summary']['validation_success_rate'] = \
            report['clinical_accuracy_validation']['overall_summary']['models_meeting_targets'] / len(validation_results)
    
    recommendations = []
    
    if report['clinical_accuracy_validation']['overall_summary']['models_meeting_targets'] == 0:
        recommendations.append("CRÍTICO: Nenhum modelo atende aos critérios de precisão clínica (>85%)")
        recommendations.append("AÇÃO REQUERIDA: Revisar arquitetura do modelo e pipeline de treinamento")
        recommendations.append("SUGESTÃO: Implementar modelos pré-treinados específicos para imagens médicas")
    
    recommendations.append("RECOMENDAÇÃO: Implementar validação cruzada multi-institucional")
    recommendations.append("RECOMENDAÇÃO: Adicionar métricas de incerteza e calibração")
    
    report['clinical_recommendations'] = recommendations
    
    report_path = Path("models/clinical_training/clinical_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Validation report saved to: {report_path}")
    
    return report

def main():
    """Main validation function"""
    logger.info("Starting clinical accuracy validation...")
    
    training_results = load_training_results()
    
    if not training_results:
        logger.error("No training results found. Please run training first.")
        return False
    
    validation_results = validate_clinical_accuracy_targets(training_results)
    
    if not validation_results:
        logger.error("Clinical accuracy validation failed")
        return False
    
    validation_report = generate_validation_report(validation_results, training_results)
    
    logger.info("\n=== VALIDATION SUMMARY ===")
    summary = validation_report['clinical_accuracy_validation']['overall_summary']
    
    logger.info(f"Total models validated: {summary['total_models_validated']}")
    logger.info(f"Models meeting >85% targets: {summary['models_meeting_targets']}")
    logger.info(f"Clinical ready models: {summary['clinical_ready_models']}")
    logger.info(f"Validation success rate: {summary['validation_success_rate']:.1%}")
    
    logger.info("\n=== CLINICAL RECOMMENDATIONS ===")
    for rec in validation_report['clinical_recommendations']:
        logger.info(f"• {rec}")
    
    if summary['models_meeting_targets'] > 0:
        logger.info("\n🎉 VALIDATION PASSED: At least one model meets clinical accuracy targets")
        return True
    else:
        logger.error("\n❌ VALIDATION FAILED: No models meet clinical accuracy targets")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
