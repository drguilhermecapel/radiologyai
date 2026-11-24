# simple_demo.py
"""
Demonstração simples de como usar o sistema de treinamento
Este script mostra o fluxo básico de trabalho
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("DEMONSTRACAO - Sistema NIH ChestX-ray14")
print("=" * 60)

def show_workflow():
    """Mostra o fluxo de trabalho completo"""
    
    print("\nFLUXO DE TRABALHO COMPLETO:\n")
    
    steps = [
        ("1. VERIFICAR AMBIENTE", "python verify_environment.py"),
        ("2. CONFIGURACAO RAPIDA", "python quick_start_nih_training.py"),
        ("3. ANALISAR DATASET", "python analyze_dataset.py"),
        ("4. TREINAR MODELO", "python train_simple.py"),
        ("5. TESTAR MODELO", "python test_trained_model.py")
    ]
    
    for step, command in steps:
        print(f"{step}")
        print(f"   Comando: {command}")
        print(f"   {'=' * 50}")
        print()

def show_file_structure():
    """Mostra estrutura de arquivos necessária"""
    
    print("\nESTRUTURA DE ARQUIVOS NECESSARIA:\n")
    
    structure = """
    Pasta do Projeto/
    |-- config_training.py
    |-- quick_start_nih_training.py
    |-- verify_environment.py
    |-- train_chest_xray.py
    |-- analyze_dataset.py
    |-- test_trained_model.py
    |-- simple_demo.py (este arquivo)
    
    D:/NIH_CHEST_XRAY/
    |-- images/
    |   |-- 00000001_000.png
    |   |-- 00000001_001.png
    |   |-- ... (112,120 imagens)
    |-- Data_Entry_2017_v2020.csv
    |-- models_trained/ (criado automaticamente)
    """
    
    print(structure)

def show_quick_commands():
    """Mostra comandos rápidos"""
    
    print("\nCOMANDOS RAPIDOS:\n")
    
    print("Instalação completa e treinamento:")
    print("   python quick_start_nih_training.py")
    print()
    
    print("Apenas verificar ambiente:")
    print("   python verify_environment.py")
    print()
    
    print("Treinar com configuração customizada:")
    print("   1. Edite config_training.py")
    print("   2. Execute: python train_chest_xray.py")
    print()
    
    print("Testar uma imagem após treinamento:")
    print("   python test_trained_model.py")

def show_common_issues():
    """Mostra problemas comuns e soluções"""
    
    print("\nPROBLEMAS COMUNS E SOLUCOES:\n")
    
    issues = [
        ("Erro de memória", 
         "Reduza batch_size para 8 ou 4 em config_training.py"),
        
        ("Dataset não encontrado", 
         "Verifique se existe D:/NIH_CHEST_XRAY/images/"),
        
        ("Treinamento muito lento", 
         "Use GPU ou reduza epochs e image_size"),
        
        ("ImportError tensorflow", 
         "Execute: pip install tensorflow==2.15.0"),
        
        ("Modelo não converge", 
         "Ajuste learning_rate ou use menos classes")
    ]
    
    for problem, solution in issues:
        print(f"Problema: {problem}")
        print(f"Solução: {solution}")
        print()

def show_expected_results():
    """Mostra resultados esperados"""
    
    print("\nRESULTADOS ESPERADOS APOS TREINAMENTO:\n")
    
    print("Métricas de performance:")
    print("- AUC médio: 0.75-0.85")
    print("- Acurácia: 80-90%")
    print("- Sensibilidade (Pneumonia): > 80%")
    print("- Especificidade: > 85%")
    print()
    
    print("Arquivos gerados em D:/NIH_CHEST_XRAY/models_trained/:")
    print("- best_model_*.h5 (modelo com melhor AUC)")
    print("- final_model_*.h5 (modelo final)")
    print("- training_log_*.csv (histórico de treino)")
    print("- evaluation_report_*.json (métricas de teste)")
    print("- training_history_*.png (gráficos)")

def main():
    """Função principal"""
    
    while True:
        print("\n" + "=" * 60)
        print("MENU PRINCIPAL")
        print("=" * 60)
        print("\n1. Ver fluxo de trabalho")
        print("2. Ver estrutura de arquivos")
        print("3. Ver comandos rápidos")
        print("4. Ver problemas comuns")
        print("5. Ver resultados esperados")
        print("6. Iniciar verificação de ambiente")
        print("7. Sair")
        
        choice = input("\nEscolha uma opção (1-7): ")
        
        if choice == "1":
            show_workflow()
        elif choice == "2":
            show_file_structure()
        elif choice == "3":
            show_quick_commands()
        elif choice == "4":
            show_common_issues()
        elif choice == "5":
            show_expected_results()
        elif choice == "6":
            print("\nIniciando verificação...")
            os.system("python verify_environment.py")
        elif choice == "7":
            print("\nEncerrando...")
            break
        else:
            print("\nOpção inválida!")
        
        input("\nPressione ENTER para continuar...")

if __name__ == "__main__":
    print("\nBem-vindo ao sistema de treinamento NIH ChestX-ray14!")
    print("Este é um guia interativo para ajudá-lo a começar.")
    main()
