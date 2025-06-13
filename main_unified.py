#!/usr/bin/env python3
"""
MedAI Radiologia - Sistema Unificado de Inicialização
Script principal corrigido para execução do sistema
"""

import sys
import os
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def setup_logging():
    """Configura o sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('MedAI')

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    logger = setup_logging()
    missing_deps = []
    
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow: {tf.__version__}")
    except ImportError:
        missing_deps.append("tensorflow")
    
    try:
        import numpy as np
        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import PIL
        logger.info(f"Pillow: {PIL.__version__}")
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import flask
        logger.info(f"Flask: {flask.__version__}")
    except ImportError:
        missing_deps.append("flask")
    
    try:
        import fastapi
        logger.info(f"FastAPI: {fastapi.__version__}")
    except ImportError:
        missing_deps.append("fastapi")
    
    if missing_deps:
        logger.error(f"Dependências faltando: {', '.join(missing_deps)}")
        logger.info("Execute: pip install -r requirements.txt")
        return False
    
    logger.info("Todas as dependências estão instaladas")
    return True

def run_web_server(port=5000):
    """Executa o servidor Flask"""
    logger = setup_logging()
    logger.info("Iniciando servidor Flask...")
    
    try:
        from src.web_server import app, initialize_medai_system
        
        if initialize_medai_system():
            logger.info(f"Servidor Flask disponível em http://localhost:{port}")
            app.run(host='0.0.0.0', port=port, debug=False)
        else:
            logger.error("Falha na inicialização do sistema MedAI")
            return False
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor Flask: {e}")
        return False
    
    return True

def run_fastapi_server(port=8000):
    """Executa o servidor FastAPI"""
    logger = setup_logging()
    logger.info("Iniciando servidor FastAPI...")
    
    try:
        from src.medai_fastapi_server import main
        main()
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor FastAPI: {e}")
        return False
    
    return True

def run_gui():
    """Executa a interface gráfica"""
    logger = setup_logging()
    logger.info("Iniciando interface gráfica...")
    
    try:
        from PyQt5.QtWidgets import QApplication
        from src.medai_gui_main import MedAIMainWindow
        
        app = QApplication(sys.argv)
        app.setApplicationName("MedAI Radiologia")
        app.setOrganizationName("Dr. Guilherme Capel")
        
        window = MedAIMainWindow()
        window.show()
        
        return app.exec_()
    except ImportError:
        logger.warning("PyQt5 não disponível, tentando modo web...")
        return run_web_server()
    except Exception as e:
        logger.error(f"Erro ao iniciar GUI: {e}")
        return False

def main():
    """Função principal com argumentos de linha de comando"""
    parser = argparse.ArgumentParser(description='MedAI Radiologia - Sistema de Análise de Imagens Médicas')
    parser.add_argument('--mode', choices=['web', 'api', 'gui', 'all'], default='gui',
                       help='Modo de execução (padrão: gui)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Porta para servidor web (padrão: 5000)')
    parser.add_argument('--api-port', type=int, default=8000,
                       help='Porta para servidor API (padrão: 8000)')
    parser.add_argument('--check-deps', action='store_true',
                       help='Verificar dependências e sair')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info("🏥 Iniciando MedAI Radiologia v3.0.0")
    
    if args.check_deps:
        return 0 if check_dependencies() else 1
    
    if not check_dependencies():
        logger.error("Dependências faltando. Execute com --check-deps para detalhes.")
        return 1
    
    if args.mode == 'web':
        return 0 if run_web_server(args.port) else 1
    elif args.mode == 'api':
        return 0 if run_fastapi_server(args.api_port) else 1
    elif args.mode == 'gui':
        return run_gui()
    elif args.mode == 'all':
        import threading
        import time
        
        api_thread = threading.Thread(target=run_fastapi_server, args=(args.api_port,))
        api_thread.daemon = True
        api_thread.start()
        
        time.sleep(2)
        
        return 0 if run_web_server(args.port) else 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
