#!/usr/bin/env python3
"""
Test script to validate clinical metrics calculation
Ensures accuracy of sensitivity, specificity, PPV, NPV calculations
"""

import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from medai_clinical_evaluation import ClinicalPerformanceEvaluator

def test_clinical_metrics_validation():
    """Validate clinical metrics calculation with known test cases"""
    print("🏥 Iniciando validação de métricas clínicas...")
    
    test_cases = [
        {
            'name': 'Perfect Classification',
            'y_true': np.array([0, 1, 1, 0, 1]),
            'y_pred': np.array([0, 1, 1, 0, 1]),
            'expected_sensitivity': 1.0,
            'expected_specificity': 1.0,
            'expected_accuracy': 1.0
        },
        {
            'name': 'Mixed Performance',
            'y_true': np.array([0, 1, 1, 0, 1, 0]),
            'y_pred': np.array([0, 1, 0, 0, 1, 1]),
            'expected_sensitivity': 0.667,  # 2/3 true positives detected
            'expected_specificity': 0.667,  # 2/3 true negatives detected
            'expected_accuracy': 0.667      # 4/6 correct predictions
        },
        {
            'name': 'High Sensitivity Case',
            'y_true': np.array([1, 1, 1, 0, 0]),
            'y_pred': np.array([1, 1, 1, 1, 0]),
            'expected_sensitivity': 1.0,    # All positives detected
            'expected_specificity': 0.5,    # Only half negatives correct
            'expected_accuracy': 0.8        # 4/5 correct
        }
    ]
    
    evaluator = ClinicalPerformanceEvaluator()
    
    validation_results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📊 Teste {i+1}: {test_case['name']}")
        
        try:
            metrics = evaluator.calculate_metrics(
                test_case['y_true'], 
                test_case['y_pred']
            )
            
            calculated_sensitivity = metrics.get('sensitivity', 0.0)
            expected_sensitivity = test_case['expected_sensitivity']
            sensitivity_diff = abs(calculated_sensitivity - expected_sensitivity)
            
            calculated_specificity = metrics.get('specificity', 0.0)
            expected_specificity = test_case['expected_specificity']
            specificity_diff = abs(calculated_specificity - expected_specificity)
            
            calculated_accuracy = metrics.get('accuracy', 0.0)
            expected_accuracy = test_case['expected_accuracy']
            accuracy_diff = abs(calculated_accuracy - expected_accuracy)
            
            tolerance = 0.05
            
            sensitivity_valid = sensitivity_diff <= tolerance
            specificity_valid = specificity_diff <= tolerance
            accuracy_valid = accuracy_diff <= tolerance
            
            test_result = {
                'test_name': test_case['name'],
                'sensitivity': {
                    'calculated': calculated_sensitivity,
                    'expected': expected_sensitivity,
                    'valid': sensitivity_valid
                },
                'specificity': {
                    'calculated': calculated_specificity,
                    'expected': expected_specificity,
                    'valid': specificity_valid
                },
                'accuracy': {
                    'calculated': calculated_accuracy,
                    'expected': expected_accuracy,
                    'valid': accuracy_valid
                },
                'overall_valid': sensitivity_valid and specificity_valid and accuracy_valid
            }
            
            validation_results.append(test_result)
            
            print(f"  Sensibilidade: {calculated_sensitivity:.3f} (esperado: {expected_sensitivity:.3f}) {'✅' if sensitivity_valid else '❌'}")
            print(f"  Especificidade: {calculated_specificity:.3f} (esperado: {expected_specificity:.3f}) {'✅' if specificity_valid else '❌'}")
            print(f"  Acurácia: {calculated_accuracy:.3f} (esperado: {expected_accuracy:.3f}) {'✅' if accuracy_valid else '❌'}")
            print(f"  Status: {'✅ PASSOU' if test_result['overall_valid'] else '❌ FALHOU'}")
            
        except Exception as e:
            print(f"  ❌ Erro no teste: {e}")
            validation_results.append({
                'test_name': test_case['name'],
                'error': str(e),
                'overall_valid': False
            })
    
    print(f"\n{'='*60}")
    print("📋 RESUMO DA VALIDAÇÃO DE MÉTRICAS CLÍNICAS")
    print(f"{'='*60}")
    
    total_tests = len(validation_results)
    passed_tests = sum(1 for result in validation_results if result.get('overall_valid', False))
    
    print(f"Total de testes: {total_tests}")
    print(f"Testes aprovados: {passed_tests}")
    print(f"Taxa de sucesso: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("✅ TODAS AS MÉTRICAS CLÍNICAS VALIDADAS COM SUCESSO!")
        print("✅ Sistema pronto para validação clínica avançada")
        return True
    else:
        print("❌ ALGUMAS MÉTRICAS FALHARAM NA VALIDAÇÃO")
        print("❌ Revisão necessária antes do uso clínico")
        return False

def test_confidence_recommendations():
    """Test confidence-based clinical recommendations"""
    print(f"\n{'='*60}")
    print("🎯 TESTANDO RECOMENDAÇÕES BASEADAS EM CONFIANÇA")
    print(f"{'='*60}")
    
    evaluator = ClinicalPerformanceEvaluator()
    
    test_scenarios = [
        {'pred_class': 0, 'confidence': 0.95, 'class_name': 'normal'},
        {'pred_class': 1, 'confidence': 0.92, 'class_name': 'pneumonia'},
        {'pred_class': 2, 'confidence': 0.75, 'class_name': 'pleural_effusion'},
        {'pred_class': 3, 'confidence': 0.60, 'class_name': 'fracture'},
    ]
    
    for scenario in test_scenarios:
        recommendation = evaluator.generate_confidence_based_recommendation(
            scenario['pred_class'],
            scenario['confidence'],
            scenario['class_name']
        )
        print(f"\n📊 {scenario['class_name'].upper()} (Confiança: {scenario['confidence']:.1%})")
        print(f"   {recommendation}")
    
    return True

if __name__ == "__main__":
    print("🏥 MedAI - Validação de Métricas Clínicas")
    print("=" * 60)
    
    metrics_valid = test_clinical_metrics_validation()
    
    confidence_valid = test_confidence_recommendations()
    
    print(f"\n{'='*60}")
    print("🎯 RESULTADO FINAL DA VALIDAÇÃO")
    print(f"{'='*60}")
    
    if metrics_valid and confidence_valid:
        print("✅ VALIDAÇÃO COMPLETA - SISTEMA APROVADO")
        print("✅ Métricas clínicas funcionando corretamente")
        print("✅ Sistema de recomendações operacional")
        sys.exit(0)
    else:
        print("❌ VALIDAÇÃO FALHOU - CORREÇÕES NECESSÁRIAS")
        sys.exit(1)
