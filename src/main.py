#!/usr/bin/env python3
"""
MedAI Radiologia - Sistema de Análise de Imagens Médicas por IA
Arquivo principal para execução da aplicação
"""

import sys
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

sys.path.insert(0, str(Path(__file__).parent))

try:
    from medai_main_structure import Config, logger
except ImportError:
    class Config:
        APP_NAME = "MedAI Radiologia"
        APP_VERSION = "3.0.0"
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('MedAI')

try:
    from medai_gui_main import MedAIMainWindow
except ImportError:
    logger.warning("GUI não disponível")
    MedAIMainWindow = None

try:
    from medai_setup_initialize import SystemInitializer
except ImportError:
    logger.warning("System initializer não disponível")
    # Criar mock do SystemInitializer
    class SystemInitializer:
        def initialize_system(self):
            return True

def check_environment():
    """Verifica se o ambiente está configurado corretamente"""
    try:
        import tensorflow as tf
        import numpy as np
        import PIL
        import cv2
        logger.info("Dependências principais verificadas com sucesso")
        return True
    except ImportError as e:
        logger.error(f"Dependência faltando: {e}")
        return False

def main():
    """Função principal"""
    logger.info("🏥 Iniciando MedAI Radiologia v3.0.0")
    
    if not check_environment():
        logger.warning("Reinicie o programa após a instalação das dependências")
        input("Pressione Enter para sair...")
        sys.exit(1)
    
    if '--help' in sys.argv or '-h' in sys.argv:
        print("""
MedAI Radiologia - Sistema de Análise de Imagens Médicas

Uso:
  python main.py [opções]

Opções:
  --web       Iniciar servidor web
  --gui       Iniciar interface gráfica (padrão)
  --port N    Porta do servidor web (padrão: 5000)
  --help      Mostrar esta mensagem
        """)
        sys.exit(0)
    
    if '--web' in sys.argv:
        logger.info("Iniciando em modo servidor web...")
        try:
            from web_server import app, initialize_medai_system
            
            initialize_medai_system()
            
            port = 5000
            if '--port' in sys.argv:
                idx = sys.argv.index('--port')
                if idx + 1 < len(sys.argv):
                    port = int(sys.argv[idx + 1])
            
            logger.info(f"Servidor disponível em http://localhost:{port}")
            app.run(host='0.0.0.0', port=port, debug=False)
            
        except Exception as e:
            logger.error(f"Erro ao iniciar servidor web: {e}")
            sys.exit(1)
    else:
        logger.info("Iniciando interface gráfica...")
        try:
            from PyQt5.QtWidgets import QApplication
            try:
                from medai_gui_main import MedAIMainWindow
            except ImportError:
                logger.warning("GUI main não disponível")
                MedAIMainWindow = None
            
            if MedAIMainWindow:
                app = QApplication(sys.argv)
                app.setApplicationName("MedAI Radiologia")
                app.setOrganizationName("Dr. Guilherme Capel")
                
                window = MedAIMainWindow()
                window.show()
                
                sys.exit(app.exec_())
            else:
                raise ImportError("GUI não disponível")
            
        except ImportError:
            logger.warning("PyQt5 não disponível, iniciando em modo web...")
            os.system(f'"{sys.executable}" "{__file__}" --web')
        except Exception as e:
            logger.error(f"Erro ao iniciar GUI: {e}")
            logger.info("Tentando modo web como alternativa...")
            os.system(f'"{sys.executable}" "{__file__}" --web')

if __name__ == "__main__":
    sys.exit(main())
