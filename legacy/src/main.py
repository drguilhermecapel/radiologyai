#!/usr/bin/env python3
"""
MedAI Radiologia - Sistema Principal
Versão 3.0.0 - Branch 36
"""

import sys
import os
import logging
import warnings
from pathlib import Path

# Configurar encoding UTF-8
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suprimir warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('medai.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('MedAI')

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    try:
        import tensorflow
        import numpy
        import PyQt5
        import flask
        import cv2
        logger.info("✅ Dependências principais verificadas")
        return True
    except ImportError as e:
        logger.error(f"❌ Dependência faltando: {e}")
        return False

def main():
    """Função principal"""
    logger.info("MedAI Radiologia v3.0.0 - Iniciando...")
    
    # Verificar dependências
    if not check_dependencies():
        logger.error("Por favor, instale as dependências: pip install -r requirements.txt")
        return 1
    
    # Tentar diferentes modos de execução
    mode = "gui"  # Modo padrão
    
    # Verificar argumentos de linha de comando
    if len(sys.argv) > 1:
        if "--web" in sys.argv or "-w" in sys.argv:
            mode = "web"
        elif "--cli" in sys.argv or "-c" in sys.argv:
            mode = "cli"
        elif "--help" in sys.argv or "-h" in sys.argv:
            print("MedAI Radiologia - Opções:")
            print("  --gui    : Interface gráfica (padrão)")
            print("  --web    : Interface web")
            print("  --cli    : Interface linha de comando")
            print("  --help   : Mostra esta ajuda")
            return 0
    
    # Executar modo selecionado
    if mode == "gui":
        try:
            logger.info("Iniciando interface gráfica...")
            from medai_gui_main import MedAIMainWindow
            from PyQt5.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            
            # Tentar criar janela principal
            try:
                window = MedAIMainWindow()
                window.show()
                return app.exec_()
            except AttributeError as e:
                logger.warning(f"Problema na GUI: {e}")
                logger.info("Tentando modo web como alternativa...")
                mode = "web"
                
        except ImportError as e:
            logger.warning(f"PyQt5 não disponível: {e}")
            logger.info("Mudando para modo web...")
            mode = "web"
    
    if mode == "web":
        try:
            logger.info("Iniciando servidor web...")
            import web_server
            web_server.main()
            return 0
        except Exception as e:
            logger.error(f"Erro no modo web: {e}")
            mode = "cli"
    
    if mode == "cli":
        try:
            logger.info("Iniciando modo CLI...")
            import medai_cli
            medai_cli.cli()
            return 0
        except Exception as e:
            logger.error(f"Erro no modo CLI: {e}")
    
    logger.error("Nenhum modo de execução disponível!")
    print("\nSugestões:")
    print("1. Verifique se os arquivos necessários estão presentes")
    print("2. Execute: pip install -r requirements.txt")
    print("3. Tente executar diretamente:")
    print("   - python web_server.py")
    print("   - python medai_cli.py")
    
    return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erro não tratado: {e}")
        sys.exit(1)
