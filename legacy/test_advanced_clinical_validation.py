"""
Teste do Framework de Validação Clínica Avançada
Valida a implementação dos estudos de validação clínica
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from medai_advanced_clinical_validation import AdvancedClinicalValidationFramework
from medai_clinical_evaluation import ClinicalPerformanceEvaluator
import json

def test_advanced_clinical_validation():
    """Testa o framework de validação clínica avançada"""
    print("🧪 Testando Framework de Validação Clínica Avançada...")
    
    validation_framework = AdvancedClinicalValidationFramework()
    
    np.random.seed(42)
    n_samples = 100
    
    model_predictions = {
        'pneumonia': np.random.rand(n_samples),
        'pleural_effusion': np.random.rand(n_samples),
        'fracture': np.random.rand(n_samples),
        'tumor': np.random.rand(n_samples),
        'normal': np.random.rand(n_samples)
    }
    
    ground_truth = {
        'pneumonia': np.random.randint(0, 2, n_samples),
        'pleural_effusion': np.random.randint(0, 2, n_samples),
        'fracture': np.random.randint(0, 2, n_samples),
        'tumor': np.random.randint(0, 2, n_samples),
        'normal': np.random.randint(0, 2, n_samples)
    }
    
    study_metadata = {
        'study_name': 'MedAI Clinical Validation Study',
        'dataset_size': n_samples,
        'validation_date': '2025-06-11',
        'model_version': '4.0.0',
        'validation_protocol': 'FDA Guidelines for AI/ML Medical Devices'
    }
    
    print("📊 Conduzindo estudo de validação abrangente...")
    
    validation_study = validation_framework.conduct_comprehensive_validation_study(
        model_predictions=model_predictions,
        ground_truth=ground_truth,
        study_metadata=study_metadata
    )
    
    assert 'study_id' in validation_study
    assert 'results' in validation_study
    assert 'clinical_readiness' in validation_study
    assert 'benchmark_comparison' in validation_study
    
    print("✅ Estudo de validação conduzido com sucesso")
    
    print("🔍 Testando análise por condição...")
    for condition in model_predictions.keys():
        if condition in validation_study['results']:
            result = validation_study['results'][condition]
            assert 'basic_metrics' in result
            assert 'clinical_criteria' in result
            assert 'validation_status' in result
            print(f"  ✅ {condition}: Validação completa")
    
    clinical_readiness = validation_study['clinical_readiness']
    readiness_level = clinical_readiness.get('readiness_level', 'UNKNOWN')
    print(f"🏥 Nível de Prontidão Clínica: {readiness_level}")
    
    benchmark_comparison = validation_study['benchmark_comparison']
    print(f"📈 Comparações com Benchmarks: {len(benchmark_comparison)} condições")
    
    print("📋 Gerando relatório de validação...")
    report = validation_framework.generate_validation_report(validation_study)
    assert len(report) > 0
    print("✅ Relatório gerado com sucesso")
    
    output_path = "models/advanced_clinical_validation_study.json"
    validation_framework.save_validation_study(validation_study, output_path)
    print(f"💾 Estudo salvo em: {output_path}")
    
    return validation_study

def test_enhanced_clinical_evaluator():
    """Testa o avaliador clínico aprimorado"""
    print("\n🧪 Testando ClinicalPerformanceEvaluator Aprimorado...")
    
    evaluator = ClinicalPerformanceEvaluator()
    
    test_conditions = [
        ('pneumonia', 'MODERATE'),
        ('tumor', 'CRITICAL'),
        ('fracture', 'CRITICAL'),
        ('pleural_effusion', 'MODERATE'),
        ('normal', 'STANDARD')
    ]
    
    for condition, expected_category in test_conditions:
        category = evaluator._get_clinical_category(condition)
        assert category == expected_category, f"Categoria incorreta para {condition}: {category} != {expected_category}"
        print(f"  ✅ {condition}: {category}")
    
    print("🔍 Testando validação de padrões clínicos...")
    
    validation_result = evaluator._validate_clinical_standards('pneumonia', 0.92, 0.88)
    assert isinstance(validation_result, dict)
    assert 'meets_standards' in validation_result
    assert 'benchmark_comparison' in validation_result
    print("  ✅ Validação de padrões clínicos funcionando")
    
    assert 'pneumonia' in evaluator.literature_benchmarks
    assert 'fracture' in evaluator.literature_benchmarks
    print("  ✅ Benchmarks da literatura carregados")
    
    print("✅ ClinicalPerformanceEvaluator aprimorado validado")

def test_clinical_thresholds():
    """Testa os thresholds clínicos configurados"""
    print("\n🧪 Testando Thresholds Clínicos...")
    
    evaluator = ClinicalPerformanceEvaluator()
    
    critical_thresholds = evaluator.clinical_thresholds['critical']
    assert critical_thresholds['min_sensitivity'] == 0.95
    assert critical_thresholds['min_specificity'] == 0.90
    print("  ✅ Thresholds críticos: Sensibilidade ≥95%, Especificidade ≥90%")
    
    moderate_thresholds = evaluator.clinical_thresholds['moderate']
    assert moderate_thresholds['min_sensitivity'] == 0.90
    assert moderate_thresholds['min_specificity'] == 0.85
    print("  ✅ Thresholds moderados: Sensibilidade ≥90%, Especificidade ≥85%")
    
    standard_thresholds = evaluator.clinical_thresholds['standard']
    assert standard_thresholds['min_sensitivity'] == 0.85
    assert standard_thresholds['min_specificity'] == 0.92
    print("  ✅ Thresholds padrão: Sensibilidade ≥85%, Especificidade ≥92%")
    
    print("✅ Todos os thresholds clínicos validados")

def main():
    """Executa todos os testes de validação clínica avançada"""
    print("🚀 Iniciando Testes de Validação Clínica Avançada")
    print("=" * 60)
    
    try:
        validation_study = test_advanced_clinical_validation()
        
        test_enhanced_clinical_evaluator()
        
        test_clinical_thresholds()
        
        print("\n" + "=" * 60)
        print("✅ TODOS OS TESTES DE VALIDAÇÃO CLÍNICA PASSARAM")
        print("🏥 Framework de validação clínica avançada implementado com sucesso")
        
        clinical_readiness = validation_study.get('clinical_readiness', {})
        print(f"\n📊 RESUMO DA VALIDAÇÃO:")
        print(f"   • Nível de Prontidão: {clinical_readiness.get('readiness_level', 'N/A')}")
        print(f"   • Condições Avaliadas: {len(validation_study.get('results', {}))}")
        print(f"   • Benchmarks Comparados: {len(validation_study.get('benchmark_comparison', {}))}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO NOS TESTES: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
