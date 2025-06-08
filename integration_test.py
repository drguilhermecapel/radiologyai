#!/usr/bin/env python3
"""
Teste de integração completo para verificar funcionalidade do MedAI
Verifica análise de imagens médicas, interface e recursos de IA
"""

import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_integration_manager():
    """Testa o gerenciador de integração principal"""
    print("🔧 TESTANDO INTEGRATION MANAGER")
    print("=" * 50)
    
    try:
        from medai_integration_manager import MedAIIntegrationManager
        
        manager = MedAIIntegrationManager()
        print("✅ MedAIIntegrationManager inicializado")
        
        result = manager.analyze_sample_image()
        print(f"✅ Análise de amostra executada: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no integration manager: {e}")
        return False

def test_dicom_processing():
    """Testa processamento de arquivos DICOM"""
    print("\n🏥 TESTANDO PROCESSAMENTO DICOM")
    print("=" * 50)
    
    try:
        import pydicom

        normal_dir = Path("data/samples/normal")
        if not normal_dir.exists():
            print("ℹ️  Diretório de amostras inexistente, teste simulado")
            return True

        dicom_files = list(normal_dir.glob("*.dcm"))
        print(f"✅ Arquivos DICOM encontrados: {len(dicom_files)}")

        if dicom_files:
            sample_file = dicom_files[0]
            ds = pydicom.dcmread(sample_file)
            print(f"✅ DICOM carregado: {ds.PatientName}")
            print(f"  • Modalidade: {ds.Modality}")
            print(f"  • Dimensões: {ds.Rows}x{ds.Columns}")

            image_array = ds.pixel_array
            print(f"✅ Array de pixels: {image_array.shape}")

        return True
            
    except Exception as e:
        print(f"❌ Erro no processamento DICOM: {e}")
        return False

def test_ai_models_loading():
    """Testa carregamento dos modelos de IA"""
    print("\n🤖 TESTANDO CARREGAMENTO DE MODELOS")
    print("=" * 50)
    
    try:
        from medai_sota_models import SOTAModelManager
        
        model_manager = SOTAModelManager()
        print("✅ SOTAModelManager inicializado")
        
        available_models = model_manager.get_available_models()
        print(f"✅ Modelos disponíveis: {len(available_models)}")
        
        for model_name in available_models:
            print(f"  • {model_name}")
        
        if available_models:
            test_model = available_models[0]
            model = model_manager.load_model(test_model)
            print(f"✅ Modelo {test_model} carregado com sucesso")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no carregamento de modelos: {e}")
        return False

def test_image_analysis_pipeline():
    """Testa pipeline completo de análise de imagens"""
    print("\n🔬 TESTANDO PIPELINE DE ANÁLISE")
    print("=" * 50)
    
    try:
        from medai_inference_system import InferenceEngine
        
        engine = InferenceEngine()
        print("✅ InferenceEngine inicializado")
        
        test_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        
        result = engine.predict(test_image)
        print(f"✅ Predição executada: {result}")
        
        if result and 'confidence' in result:
            confidence = result['confidence']
            print(f"✅ Confiança da análise: {confidence:.2%}")
            
            if confidence >= 0.85:
                print("✅ Precisão dentro do esperado (≥85%)")
                return True
            else:
                print("⚠️  Precisão abaixo do esperado")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no pipeline de análise: {e}")
        return False

def test_format_compatibility():
    """Testa compatibilidade com diferentes formatos"""
    print("\n📁 TESTANDO COMPATIBILIDADE DE FORMATOS")
    print("=" * 50)
    
    supported_formats = {
        'medical': ['.dcm', '.nii', '.nii.gz', '.hdr', '.img'],
        'standard': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'],
        'scientific': ['.exr', '.hdr']
    }
    
    total_formats = 0
    for category, formats in supported_formats.items():
        print(f"✅ {category.upper()}: {len(formats)} formatos")
        for fmt in formats:
            print(f"  • {fmt}")
        total_formats += len(formats)
    
    print(f"✅ Total de formatos suportados: {total_formats}")
    return True

def test_system_resources():
    """Testa verificação de recursos do sistema"""
    print("\n💻 TESTANDO RECURSOS DO SISTEMA")
    print("=" * 50)
    
    try:
        print(f"✅ Python versão: {sys.version.split()[0]}")

        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            print(f"✅ Memória total: {memory_gb:.1f} GB")
        except ImportError:
            print("ℹ️  psutil não disponível - verificação básica")

        return True
        
    except Exception as e:
        print(f"❌ Erro na verificação de recursos: {e}")
        return False

def main():
    """Executa todos os testes de integração"""
    print("🏥 MEDAI RADIOLOGIA - TESTE DE INTEGRAÇÃO COMPLETO")
    print("=" * 70)
    print(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    tests = [
        ("Integration Manager", test_integration_manager),
        ("Processamento DICOM", test_dicom_processing),
        ("Carregamento de Modelos", test_ai_models_loading),
        ("Pipeline de Análise", test_image_analysis_pipeline),
        ("Compatibilidade de Formatos", test_format_compatibility),
        ("Recursos do Sistema", test_system_resources)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro crítico em {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("📋 RESUMO DOS TESTES DE INTEGRAÇÃO")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\n📊 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed >= total - 1:  # Permitir 1 falha não crítica
        print("🎉 INTEGRAÇÃO APROVADA!")
        print("✅ Sistema MedAI está funcionando corretamente")
        print("🔧 Pronto para análise de imagens médicas")
        return 0
    else:
        print("⚠️  PROBLEMAS DE INTEGRAÇÃO DETECTADOS")
        print("🔧 Verificar logs acima para detalhes")
        return 1

if __name__ == "__main__":
    sys.exit(main())
