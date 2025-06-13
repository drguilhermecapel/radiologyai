#!/usr/bin/env python3
"""
MedAI Radiologia - Teste das Correções Implementadas
Script para verificar se todas as correções estão funcionando
"""

import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def setup_logging():
    """Configura logging para testes"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger('MedAI-Test')

def test_imports():
    """Testa se todos os imports estão funcionando"""
    logger = setup_logging()
    logger.info("🧪 Testando imports...")
    
    tests = []
    
    try:
        from src.web_server import app, initialize_medai_system, convert_numpy_to_json
        tests.append(("Flask Server", True, "OK"))
    except Exception as e:
        tests.append(("Flask Server", False, str(e)))
    
    try:
        from src.medai_fastapi_server import app as fastapi_app, convert_numpy_to_json as fastapi_convert
        tests.append(("FastAPI Server", True, "OK"))
    except Exception as e:
        tests.append(("FastAPI Server", False, str(e)))
    
    try:
        from main_unified import main, check_dependencies
        tests.append(("Main Unificado", True, "OK"))
    except Exception as e:
        tests.append(("Main Unificado", False, str(e)))
    
    try:
        import numpy as np
        from src.web_server import convert_numpy_to_json
        
        test_data = {
            'array': np.array([1, 2, 3]),
            'float': np.float32(3.14),
            'int': np.int64(42),
            'bool': np.bool_(True)
        }
        
        converted = convert_numpy_to_json(test_data)
        assert isinstance(converted['array'], list)
        assert isinstance(converted['float'], float)
        assert isinstance(converted['int'], int)
        assert isinstance(converted['bool'], bool)
        
        tests.append(("Conversão NumPy", True, "OK"))
    except Exception as e:
        tests.append(("Conversão NumPy", False, str(e)))
    
    logger.info("\n" + "="*60)
    logger.info("RELATÓRIO DE TESTES DAS CORREÇÕES")
    logger.info("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, success, message in tests:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        logger.info(f"{test_name:20} | {status} | {message}")
        if success:
            passed += 1
        else:
            failed += 1
    
    logger.info("="*60)
    logger.info(f"RESUMO: {passed} passaram, {failed} falharam")
    logger.info("="*60)
    
    return failed == 0

def test_flask_endpoints():
    """Testa os endpoints do Flask"""
    logger = setup_logging()
    logger.info("🧪 Testando endpoints Flask...")
    
    try:
        from src.web_server import app
        
        with app.test_client() as client:
            response = client.get('/api/status')
            assert response.status_code == 200
            
            data = response.get_json()
            assert 'status' in data
            assert 'app_name' in data
            
            logger.info("✅ Endpoint /api/status funcionando")
            
            response = client.get('/')
            assert response.status_code == 200
            
            logger.info("✅ Página principal funcionando")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro nos testes Flask: {e}")
        return False

def test_fastapi_endpoints():
    """Testa os endpoints do FastAPI"""
    logger = setup_logging()
    logger.info("🧪 Testando endpoints FastAPI...")
    
    try:
        from fastapi.testclient import TestClient
        from src.medai_fastapi_server import app
        
        client = TestClient(app)
        
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'app_name' in data
        
        logger.info("✅ Endpoint /api/v1/health funcionando")
        
        response = client.get("/api/v1/models")
        assert response.status_code in [200, 503]  # 503 se sistema não inicializado
        
        logger.info("✅ Endpoint /api/v1/models funcionando")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro nos testes FastAPI: {e}")
        return False

def main():
    """Executa todos os testes"""
    logger = setup_logging()
    logger.info("🏥 Iniciando testes das correções MedAI Radiologia")
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_flask_endpoints():
        all_passed = False
    
    if not test_fastapi_endpoints():
        all_passed = False
    
    if all_passed:
        logger.info("\n🎉 TODOS OS TESTES PASSARAM! Sistema corrigido com sucesso.")
        return 0
    else:
        logger.error("\n❌ ALGUNS TESTES FALHARAM. Verifique os erros acima.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
