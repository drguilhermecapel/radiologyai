#!/usr/bin/env python3
"""
MedAI Radiologia - Sistema de Validação Completa
Testa todas as funcionalidades implementadas do sistema de modelos pré-treinados
Versão: 1.0.0
"""

import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Testa importação de todos os módulos criados"""
    print("🧪 TESTE 1: Importação de Módulos")
    print("=" * 50)
    
    modules_to_test = [
        ('medai_pretrained_loader', 'PreTrainedModelLoader'),
        ('medai_model_downloader', 'ModelDownloader'),
        ('medai_smart_model_manager', 'SmartModelManager'),
        ('medai_model_validator', 'ModelValidator'),
        ('medai_sota_models_real', 'SOTAModelManager'),
        ('medai_inference_system', 'MedAIInferenceSystem'),
        ('medai_integration_manager', 'MedAIIntegrationManager')
    ]
    
    results = {}
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name)
            if hasattr(module, class_name):
                results[module_name] = "✅ SUCESSO"
                print(f"✅ {module_name}.{class_name} - Importado com sucesso")
            else:
                results[module_name] = f"❌ ERRO - Classe {class_name} não encontrada"
                print(f"❌ {module_name}.{class_name} - Classe não encontrada")
        except Exception as e:
            results[module_name] = f"❌ ERRO - {str(e)}"
            print(f"❌ {module_name}.{class_name} - Erro: {e}")
    
    success_count = sum(1 for result in results.values() if result.startswith("✅"))
    total_count = len(results)
    
    print(f"\n📊 Resultado: {success_count}/{total_count} módulos importados com sucesso")
    return results

def test_directory_structure():
    """Testa estrutura de diretórios de modelos"""
    print("\n🧪 TESTE 2: Estrutura de Diretórios")
    print("=" * 50)
    
    required_paths = [
        "models",
        "models/pre_trained",
        "models/model_registry.json",
        "models/pre_trained/README.md",
        "models/pre_trained/.gitkeep"
    ]
    
    results = {}
    
    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            results[path_str] = "✅ EXISTE"
            print(f"✅ {path_str} - Existe")
        else:
            results[path_str] = "❌ NÃO EXISTE"
            print(f"❌ {path_str} - Não existe")
    
    registry_path = Path("models/model_registry.json")
    if registry_path.exists():
        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            models_count = len(registry.get("models", {}))
            print(f"✅ Registry carregado: {models_count} modelos registrados")
            results["registry_content"] = f"✅ {models_count} modelos"
            
            for model_id in registry.get("models", {}):
                print(f"   - {model_id}")
                
        except Exception as e:
            print(f"❌ Erro ao carregar registry: {e}")
            results["registry_content"] = f"❌ ERRO - {e}"
    
    success_count = sum(1 for result in results.values() if result.startswith("✅"))
    total_count = len(results)
    
    print(f"\n📊 Resultado: {success_count}/{total_count} estruturas verificadas com sucesso")
    return results

def test_pretrained_loader():
    """Testa funcionalidades do PreTrainedModelLoader"""
    print("\n🧪 TESTE 3: PreTrainedModelLoader")
    print("=" * 50)
    
    results = {}
    
    try:
        from medai_pretrained_loader import PreTrainedModelLoader
        
        loader = PreTrainedModelLoader()
        results["initialization"] = "✅ SUCESSO"
        print("✅ PreTrainedModelLoader inicializado")
        
        try:
            available_models = loader.get_available_models()
            results["get_available_models"] = f"✅ {len(available_models)} modelos"
            print(f"✅ Modelos disponíveis: {len(available_models)}")
            
            for model_id in available_models:
                print(f"   - {model_id}")
                
        except Exception as e:
            results["get_available_models"] = f"❌ ERRO - {e}"
            print(f"❌ Erro ao obter modelos disponíveis: {e}")
        
        try:
            local_models = loader.check_local_models()
            results["check_local_models"] = f"✅ {len(local_models)} locais"
            print(f"✅ Modelos locais verificados: {len(local_models)}")
            
        except Exception as e:
            results["check_local_models"] = f"❌ ERRO - {e}"
            print(f"❌ Erro ao verificar modelos locais: {e}")
        
        try:
            if hasattr(loader, 'get_smart_management_recommendations'):
                recommendations = loader.get_smart_management_recommendations()
                results["smart_recommendations"] = "✅ SUCESSO"
                print("✅ Recomendações de gerenciamento obtidas")
            else:
                results["smart_recommendations"] = "⚠️ MÉTODO NÃO ENCONTRADO"
                print("⚠️ Método get_smart_management_recommendations não encontrado")
                
        except Exception as e:
            results["smart_recommendations"] = f"❌ ERRO - {e}"
            print(f"❌ Erro ao obter recomendações: {e}")
            
    except Exception as e:
        results["initialization"] = f"❌ ERRO - {e}"
        print(f"❌ Erro ao inicializar PreTrainedModelLoader: {e}")
    
    success_count = sum(1 for result in results.values() if result.startswith("✅"))
    total_count = len(results)
    
    print(f"\n📊 Resultado: {success_count}/{total_count} funcionalidades testadas com sucesso")
    return results

def test_integration_manager():
    """Testa integração com MedAIIntegrationManager"""
    print("\n🧪 TESTE 4: MedAIIntegrationManager")
    print("=" * 50)
    
    results = {}
    
    try:
        from medai_integration_manager import MedAIIntegrationManager
        
        print("✅ MedAIIntegrationManager importado")
        results["import"] = "✅ SUCESSO"
        
        manager_class = MedAIIntegrationManager
        
        required_methods = [
            '_initialize_pretrained_system',
            '_initialize_smart_model_management',
            '_initialize_components'
        ]
        
        for method_name in required_methods:
            if hasattr(manager_class, method_name):
                results[f"method_{method_name}"] = "✅ EXISTE"
                print(f"✅ Método {method_name} existe")
            else:
                results[f"method_{method_name}"] = "❌ NÃO EXISTE"
                print(f"❌ Método {method_name} não existe")
        
    except Exception as e:
        results["import"] = f"❌ ERRO - {e}"
        print(f"❌ Erro ao importar MedAIIntegrationManager: {e}")
    
    success_count = sum(1 for result in results.values() if result.startswith("✅"))
    total_count = len(results)
    
    print(f"\n📊 Resultado: {success_count}/{total_count} verificações bem-sucedidas")
    return results

def test_installers():
    """Testa arquivos de instalação"""
    print("\n🧪 TESTE 5: Sistemas de Instalação")
    print("=" * 50)
    
    results = {}
    
    installer_files = [
        "MedAI_Radiologia_Installer.py",
        "MedAI_Model_Installer.py",
        "build_with_models.py"
    ]
    
    for installer_file in installer_files:
        path = Path(installer_file)
        if path.exists():
            results[installer_file] = "✅ EXISTE"
            print(f"✅ {installer_file} - Existe")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.startswith('#!/usr/bin/env python3'):
                    print(f"   ✅ Shebang correto")
                
                if 'class' in content and 'def' in content:
                    print(f"   ✅ Estrutura de classe encontrada")
                    
            except Exception as e:
                print(f"   ❌ Erro ao verificar conteúdo: {e}")
        else:
            results[installer_file] = "❌ NÃO EXISTE"
            print(f"❌ {installer_file} - Não existe")
    
    success_count = sum(1 for result in results.values() if result.startswith("✅"))
    total_count = len(results)
    
    print(f"\n📊 Resultado: {success_count}/{total_count} instaladores verificados")
    return results

def test_documentation():
    """Testa documentação criada"""
    print("\n🧪 TESTE 6: Documentação")
    print("=" * 50)
    
    results = {}
    
    doc_files = [
        "MODELS_LICENSE.md",
        "models/pre_trained/README.md"
    ]
    
    for doc_file in doc_files:
        path = Path(doc_file)
        if path.exists():
            results[doc_file] = "✅ EXISTE"
            print(f"✅ {doc_file} - Existe")
            
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                word_count = len(content.split())
                print(f"   📄 {word_count} palavras")
                
                if word_count > 100:
                    print(f"   ✅ Documentação substancial")
                else:
                    print(f"   ⚠️ Documentação curta")
                    
            except Exception as e:
                print(f"   ❌ Erro ao ler arquivo: {e}")
        else:
            results[doc_file] = "❌ NÃO EXISTE"
            print(f"❌ {doc_file} - Não existe")
    
    success_count = sum(1 for result in results.values() if result.startswith("✅"))
    total_count = len(results)
    
    print(f"\n📊 Resultado: {success_count}/{total_count} documentos verificados")
    return results

def generate_final_report(all_results):
    """Gera relatório final de todos os testes"""
    print("\n" + "=" * 60)
    print("RELATÓRIO FINAL DE VALIDAÇÃO")
    print("=" * 60)
    
    total_tests = 0
    total_successes = 0
    
    for test_name, results in all_results.items():
        successes = sum(1 for result in results.values() if result.startswith("✅"))
        total = len(results)
        
        total_tests += total
        total_successes += successes
        
        percentage = (successes / total * 100) if total > 0 else 0
        
        print(f"📊 {test_name}: {successes}/{total} ({percentage:.1f}%)")
    
    overall_percentage = (total_successes / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 RESULTADO GERAL: {total_successes}/{total_tests} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 90:
        print("🎉 EXCELENTE - Sistema validado com sucesso!")
        status = "EXCELENTE"
    elif overall_percentage >= 75:
        print("✅ BOM - Sistema funcional com pequenos problemas")
        status = "BOM"
    elif overall_percentage >= 50:
        print("⚠️ REGULAR - Sistema precisa de correções")
        status = "REGULAR"
    else:
        print("❌ CRÍTICO - Sistema precisa de revisão completa")
        status = "CRÍTICO"
    
    report = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "overall_percentage": overall_percentage,
        "status": status,
        "total_tests": total_tests,
        "total_successes": total_successes,
        "detailed_results": all_results
    }
    
    report_path = Path("test_validation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Relatório salvo em: {report_path}")
    
    return report

def main():
    """Função principal de validação"""
    print("🚀 INICIANDO VALIDAÇÃO COMPLETA DO SISTEMA MEDAI RADIOLOGIA")
    print("=" * 60)
    print(f"⏰ Início: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    all_results = {}
    
    try:
        all_results["Importação de Módulos"] = test_imports()
        all_results["Estrutura de Diretórios"] = test_directory_structure()
        all_results["PreTrainedModelLoader"] = test_pretrained_loader()
        all_results["MedAIIntegrationManager"] = test_integration_manager()
        all_results["Sistemas de Instalação"] = test_installers()
        all_results["Documentação"] = test_documentation()
        
        final_report = generate_final_report(all_results)
        
        print(f"\n⏰ Fim: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("🏁 VALIDAÇÃO CONCLUÍDA!")
        
        return final_report["overall_percentage"] >= 75
        
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO NA VALIDAÇÃO: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
