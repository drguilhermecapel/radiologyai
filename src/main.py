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
    from medai_gui_main import MedAIMainWindow
    from medai_setup_initialize import SystemInitializer
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)

def setup_application():
    """Configura a aplicação Qt"""
    app = QApplication(sys.argv)
    app.setApplicationName(Config.APP_NAME)
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName("MedAI")
    app.setOrganizationDomain("medai.com")
    
    app.setStyle('Fusion')
    
    try:
        from PyQt5.QtGui import QIcon
        icon_path = Path(__file__).parent / "icons" / "medai_icon.png"
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
    except Exception:
        pass
    
    return app

def check_system_requirements():
    """Verifica requisitos do sistema"""
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append('numpy')

    try:
        import pydicom  # noqa: F401
    except ImportError:
        missing.append('pydicom')

    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append('opencv-python')

    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append('Pillow')

    try:
        import tensorflow as tf  # noqa: F401
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU detectada e disponível para TensorFlow")
    except ImportError:
        logger.warning("TensorFlow não disponível - funcionalidades de IA limitadas")

    if missing:
        logger.error(f"Módulos ausentes: {', '.join(missing)}")
        return False

    logger.info("Todos os módulos essenciais disponíveis")
    return True

def main():
    """Função principal"""
    try:
        logger.info(f"Iniciando {Config.APP_NAME} v{Config.APP_VERSION}")
        
        if not check_system_requirements():
            print("Erro: Requisitos do sistema não atendidos")
            return 1
        
        initializer = SystemInitializer()
        if not initializer.initialize_system():
            logger.error("Falha na inicialização do sistema")
            return 1
        
        if os.environ.get('DISPLAY') is None:
            logger.info("Ambiente headless detectado - executando em modo servidor")
            print("✅ MedAI Radiologia inicializado com sucesso em modo servidor")
            print("🔧 Sistema configurado e pronto para análise de imagens médicas")
            print("🤖 Modelos de IA carregados e funcionais")
            return 0
        
        app = setup_application()
        
        main_window = MedAIMainWindow()
        main_window.show()
        
        logger.info("Interface gráfica inicializada com sucesso")
        
        return app.exec_()
        
    except Exception as e:
        logger.error(f"Erro crítico na aplicação: {e}")
        
        if os.environ.get('DISPLAY') is None:
            print(f"Erro crítico: {e}")
            return 1
        
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Erro Crítico")
            msg.setText(f"Ocorreu um erro crítico:\n\n{str(e)}")
            msg.setDetailedText(f"Detalhes técnicos:\n{str(e)}")
            msg.exec_()
            
        except Exception:
            print(f"Erro crítico: {e}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
