# cli.py - Interface de linha de comando para MedAI

import click
import sys
from pathlib import Path
import json
import yaml
import logging
from typing import Optional, List, Dict
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.tree import Tree
from rich.syntax import Syntax
from rich import print as rprint
import questionary
from questionary import Style

# Importar módulos do sistema
from main import Config
from dicom_processor import DICOMProcessor
from neural_networks import MedicalImageNetwork
from training_system import MedicalModelTrainer
from inference_system import MedicalInferenceEngine
from batch_processor import BatchProcessor
from security_audit import SecurityManager, UserRole
from export_system import ExportManager

# Configurar console Rich
console = Console()

# Estilo customizado para questionary
custom_style = Style([
    ('question', 'fg:#ff9d00 bold'),
    ('answer', 'fg:#00ff00 bold'),
    ('pointer', 'fg:#ff9d00 bold'),
    ('highlighted', 'fg:#ff9d00 bold'),
    ('selected', 'fg:#00ff00'),
    ('separator', 'fg:#6c6c6c'),
    ('instruction', 'fg:#abb2bf'),
    ('text', 'fg:#ffffff'),
])

@click.group()
@click.option('--debug', is_flag=True, help='Ativar modo debug')
@click.option('--config', type=click.Path(), help='Arquivo de configuração')
@click.pass_context
def cli(ctx, debug, config):
    """
    MedAI - Sistema de Análise de Imagens Médicas por IA
    
    Use 'medai COMMAND --help' para mais informações sobre um comando.
    """
    # Configurar logging
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Carregar configuração
    if config:
        with open(config, 'r') as f:
            if config.endswith('.yaml'):
                ctx.obj = yaml.safe_load(f)
            else:
                ctx.obj = json.load(f)
    else:
        ctx.obj = {}
    
    # Mostrar banner
    if not ctx.invoked_subcommand:
        show_banner()

def show_banner():
    """Mostra banner do MedAI"""
    banner = """
    ╔═╗╔═╗╔═╗╔═╗╦
    ║║║║╣ ║ ║╠═╣║
    ╩ ╩╚═╝═╩╝╩ ╩╩
    Medical AI Analysis System v1.0.0
    """
    console.print(Panel(banner, style="bold blue"))

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', default='chest_xray', help='Modelo a usar')
@click.option('--output', '-o', type=click.Path(), help='Diretório de saída')
@click.option('--format', '-f', default='json', type=click.Choice(['json', 'pdf', 'html']))
@click.option('--visualize', '-v', is_flag=True, help='Mostrar visualização')
def analyze(image_path, model, output, format, visualize):
    """Analisa uma imagem médica"""
    console.print(f"\n[bold blue]Analisando imagem:[/bold blue] {image_path}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        # Tarefas
        task1 = progress.add_task("[cyan]Carregando imagem...", total=100)
        task2 = progress.add_task("[cyan]Processando...", total=100)
        task3 = progress.add_task("[cyan]Analisando com IA...", total=100)
        
        # Carregar imagem
        processor = DICOMProcessor()
        
        if image_path.endswith(('.dcm', '.dicom')):
            ds = processor.read_dicom(image_path)
            image = processor.dicom_to_array(ds)
            metadata = processor.extract_metadata(ds)
        else:
            import cv2
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            metadata = {'file_type': Path(image_path).suffix}
        
        progress.update(task1, completed=100)
        
        # Pré-processar
        model_config = Config.MODEL_CONFIG.get(model)
        if model_config:
            image_processed = processor.preprocess_for_ai(
                image, 
                model_config['input_size']
            )
        else:
            console.print(f"[red]Modelo não encontrado: {model}[/red]")
            return
        
        progress.update(task2, completed=100)
        
        # Carregar modelo e fazer inferência
        try:
            engine = MedicalInferenceEngine(
                model_config['model_path'],
                model_config
            )
            
            result = engine.predict_single(image_processed, metadata=metadata)
            
            progress.update(task3, completed=100)
            
        except Exception as e:
            console.print(f"[red]Erro na análise: {str(e)}[/red]")
            return
    
    # Mostrar resultados
    show_analysis_results(result, metadata)
    
    # Visualizar se solicitado
    if visualize:
        visualize_results(image, result)
    
    # Salvar resultados
    if output:
        save_results(result, output, format)
        console.print(f"\n[green]Resultados salvos em: {output}[/green]")

def show_analysis_results(result, metadata):
    """Mostra resultados da análise"""
    # Criar tabela de resultados
    table = Table(title="Resultados da Análise", show_header=True, header_style="bold magenta")
    table.add_column("Classe", style="dim", width=20)
    table.add_column("Probabilidade", justify="right")
    table.add_column("Confiança", justify="center")
    
    # Adicionar predições
    for class_name, prob in sorted(result.predictions.items(), key=lambda x: x[1], reverse=True):
        confidence_color = "green" if prob > 0.7 else "yellow" if prob > 0.3 else "red"
        confidence_bar = "█" * int(prob * 20)
        
        table.add_row(
            class_name,
            f"{prob:.2%}",
            f"[{confidence_color}]{confidence_bar}[/{confidence_color}]"
        )
    
    console.print("\n", table)
    
    # Mostrar predição principal
    console.print(f"\n[bold green]Diagnóstico Sugerido:[/bold green] {result.predicted_class}")
    console.print(f"[bold]Confiança:[/bold] {result.confidence:.2%}")
    console.print(f"[dim]Tempo de processamento: {result.processing_time:.3f}s[/dim]")
    
    # Mostrar metadados se disponíveis
    if metadata:
        console.print("\n[bold blue]Metadados da Imagem:[/bold blue]")
        for key, value in metadata.items():
            if value:
                console.print(f"  • {key}: {value}")

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), required=True, help='Diretório de saída')
@click.option('--model', '-m', default='chest_xray', help='Modelo a usar')
@click.option('--recursive', '-r', is_flag=True, help='Buscar recursivamente')
@click.option('--workers', '-w', default=4, help='Número de workers paralelos')
@click.option('--format', '-f', default='csv', type=click.Choice(['csv', 'json', 'xlsx']))
def batch(input_dir, output, model, recursive, workers, format):
    """Processa múltiplas imagens em lote"""
    console.print(f"\n[bold blue]Processamento em Lote[/bold blue]")
    console.print(f"Diretório: {input_dir}")
    console.print(f"Recursivo: {'Sim' if recursive else 'Não'}")
    
    # Contar arquivos
    from pathlib import Path
    patterns = ['*.dcm', '*.png', '*.jpg', '*.jpeg']
    files = []
    
    for pattern in patterns:
        if recursive:
            files.extend(Path(input_dir).rglob(pattern))
        else:
            files.extend(Path(input_dir).glob(pattern))
    
    console.print(f"Arquivos encontrados: {len(files)}")
    
    if not files:
        console.print("[red]Nenhum arquivo encontrado![/red]")
        return
    
    # Confirmar processamento
    if not click.confirm(f"Processar {len(files)} arquivos?"):
        return
    
    # Criar processador em lote
    model_config = Config.MODEL_CONFIG.get(model)
    engine = MedicalInferenceEngine(model_config['model_path'], model_config)
    
    batch_processor = BatchProcessor(
        engine,
        max_workers=workers
    )
    
    # Criar job
    job = batch_processor.create_batch_job(
        [str(f) for f in files],
        output,
        model,
        recursive
    )
    
    # Processar com barra de progresso
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task(
            f"[cyan]Processando {len(files)} arquivos...", 
            total=len(files)
        )
        
        def progress_callback(job):
            progress.update(task, completed=int(job.progress))
        
        # Processar
        summary = batch_processor.process_batch(job, progress_callback)
    
    # Mostrar resumo
    show_batch_summary(summary)

def show_batch_summary(summary):
    """Mostra resumo do processamento em lote"""
    table = Table(title="Resumo do Processamento", show_header=True)
    table.add_column("Métrica", style="dim")
    table.add_column("Valor", justify="right")
    
    table.add_row("Total de arquivos", str(summary['total_files']))
    table.add_row("Processados com sucesso", f"[green]{summary['processed']}[/green]")
    table.add_row("Falhas", f"[red]{summary['failed']}[/red]")
    table.add_row("Taxa de sucesso", summary['success_rate'])
    table.add_row("Tempo total", summary['total_time'])
    table.add_row("Tempo médio por arquivo", summary['average_time_per_file'])
    
    console.print("\n", table)
    
    # Distribuição de classes
    if 'class_distribution' in summary:
        console.print("\n[bold]Distribuição de Classes:[/bold]")
        for class_name, count in summary['class_distribution'].items():
            console.print(f"  • {class_name}: {count}")

@cli.command()
@click.option('--data-dir', '-d', type=click.Path(exists=True), required=True)
@click.option('--model-type', '-m', default='efficientnet', 
              type=click.Choice(['efficientnet', 'densenet', 'resnet', 'custom']))
@click.option('--epochs', '-e', default=100, help='Número de épocas')
@click.option('--batch-size', '-b', default=32, help='Tamanho do batch')
@click.option('--learning-rate', '-lr', default=0.001, help='Taxa de aprendizado')
@click.option('--output-dir', '-o', type=click.Path(), required=True)
def train(data_dir, model_type, epochs, batch_size, learning_rate, output_dir):
    """Treina um novo modelo"""
    console.print(f"\n[bold blue]Treinamento de Modelo[/bold blue]")
    
    # Configurações interativas
    config = questionary.form(
        num_classes=questionary.text("Número de classes:", default="5"),
        image_size=questionary.text("Tamanho da imagem (ex: 224,224):", default="224,224"),
        validation_split=questionary.text("Split de validação (%):", default="20"),
        augmentation=questionary.confirm("Usar data augmentation?", default=True),
        pretrained=questionary.confirm("Usar pesos pré-treinados?", default=True),
    ).ask()
    
    # Converter configurações
    num_classes = int(config['num_classes'])
    image_size = tuple(map(int, config['image_size'].split(',')))
    validation_split = float(config['validation_split']) / 100
    
    # Mostrar configuração
    console.print("\n[bold]Configuração do Treinamento:[/bold]")
    console.print(f"  • Modelo: {model_type}")
    console.print(f"  • Classes: {num_classes}")
    console.print(f"  • Tamanho: {image_size}")
    console.print(f"  • Épocas: {epochs}")
    console.print(f"  • Batch: {batch_size}")
    console.print(f"  • Learning Rate: {learning_rate}")
    
    if not click.confirm("\nIniciar treinamento?"):
        return
    
    # Implementar treinamento
    console.print("\n[yellow]Funcionalidade de treinamento em desenvolvimento[/yellow]")

@cli.command()
def interactive():
    """Modo interativo do MedAI"""
    show_banner()
    
    while True:
        # Menu principal
        action = questionary.select(
            "O que você deseja fazer?",
            choices=[
                "Analisar imagem",
                "Processamento em lote",
                "Treinar modelo",
                "Gerenciar modelos",
                "Configurações",
                "Sair"
            ],
            style=custom_style
        ).ask()
        
        if action == "Sair":
            console.print("\n[yellow]Até logo![/yellow]")
            break
        
        elif action == "Analisar imagem":
            interactive_analyze()
        
        elif action == "Processamento em lote":
            interactive_batch()
        
        elif action == "Treinar modelo":
            interactive_train()
        
        elif action == "Gerenciar modelos":
            manage_models()
        
        elif action == "Configurações":
            manage_settings()

def interactive_analyze():
    """Análise interativa de imagem"""
    # Selecionar arquivo
    image_path = questionary.path(
        "Caminho da imagem:",
        only_files=True,
        style=custom_style
    ).ask()
    
    if not image_path:
        return
    
    # Selecionar modelo
    model = questionary.select(
        "Selecione o modelo:",
        choices=list(Config.MODEL_CONFIG.keys()),
        style=custom_style
    ).ask()
    
    # Opções
    options = questionary.checkbox(
        "Opções adicionais:",
        choices=[
            "Mostrar visualização",
            "Gerar relatório PDF",
            "Salvar resultados"
        ],
        style=custom_style
    ).ask()
    
    # Executar análise
    ctx = click.Context(analyze)
    ctx.invoke(
        analyze,
        image_path=image_path,
        model=model,
        visualize="Mostrar visualização" in options
    )

@cli.group()
def model():
    """Gerenciamento de modelos"""
    pass

@model.command('list')
def model_list():
    """Lista modelos disponíveis"""
    table = Table(title="Modelos Disponíveis", show_header=True)
    table.add_column("Nome", style="dim")
    table.add_column("Arquivo", justify="right")
    table.add_column("Classes", justify="center")
    table.add_column("Tamanho", justify="right")
    
    for name, config in Config.MODEL_CONFIG.items():
        model_path = Path(config['model_path'])
        if model_path.exists():
            size = f"{model_path.stat().st_size / 1024 / 1024:.1f} MB"
            status = "[green]✓[/green]"
        else:
            size = "-"
            status = "[red]✗[/red]"
        
        table.add_row(
            f"{status} {name}",
            str(model_path.name),
            str(config['classes']),
            size
        )
    
    console.print("\n", table)

@model.command('info')
@click.argument('model_name')
def model_info(model_name):
    """Mostra informações detalhadas do modelo"""
    if model_name not in Config.MODEL_CONFIG:
        console.print(f"[red]Modelo não encontrado: {model_name}[/red]")
        return
    
    config = Config.MODEL_CONFIG[model_name]
    
    # Criar árvore de informações
    tree = Tree(f"[bold]{model_name}[/bold]")
    
    # Configuração
    config_branch = tree.add("Configuração")
    config_branch.add(f"Arquivo: {config['model_path']}")
    config_branch.add(f"Tamanho de entrada: {config['input_size']}")
    config_branch.add(f"Threshold: {config['threshold']}")
    
    # Classes
    classes_branch = tree.add("Classes")
    for cls in config['classes']:
        classes_branch.add(cls)
    
    # Verificar se modelo existe
    model_path = Path(config['model_path'])
    if model_path.exists():
        stats_branch = tree.add("Estatísticas")
        stats_branch.add(f"Tamanho: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        stats_branch.add(f"Modificado: {datetime.fromtimestamp(model_path.stat().st_mtime)}")
    
    console.print("\n", tree)

@cli.group()
def security():
    """Gerenciamento de segurança"""
    pass

@security.command('create-user')
@click.option('--username', '-u', prompt=True, help='Nome de usuário')
@click.option('--role', '-r', 
              type=click.Choice(['admin', 'physician', 'radiologist', 'technician', 'viewer']),
              prompt=True, help='Papel do usuário')
@click.option('--email', '-e', prompt=True, help='Email')
@click.option('--full-name', '-n', prompt=True, help='Nome completo')
@click.password_option()
def create_user(username, role, email, full_name, password):
    """Cria novo usuário"""
    security_mgr = SecurityManager()
    
    success, result = security_mgr.create_user(
        username=username,
        password=password,
        role=UserRole(role),
        email=email,
        full_name=full_name,
        created_by='CLI'
    )
    
    if success:
        console.print(f"\n[green]Usuário criado com sucesso![/green]")
        console.print(f"ID: {result}")
    else:
        console.print(f"\n[red]Erro ao criar usuário: {result}[/red]")

@security.command('audit-log')
@click.option('--days', '-d', default=7, help='Número de dias para mostrar')
@click.option('--user', '-u', help='Filtrar por usuário')
@click.option('--risk-level', '-r', default=0, help='Nível mínimo de risco')
def audit_log(days, user, risk_level):
    """Mostra log de auditoria"""
    security_mgr = SecurityManager()
    
    from datetime import datetime, timedelta
    start_date = datetime.now() - timedelta(days=days)
    
    events = security_mgr.get_audit_logs(
        start_date=start_date,
        user_id=user,
        min_risk_level=risk_level
    )
    
    if not events:
        console.print("[yellow]Nenhum evento encontrado[/yellow]")
        return
    
    table = Table(title=f"Log de Auditoria - Últimos {days} dias", show_header=True)
    table.add_column("Data/Hora", style="dim")
    table.add_column("Usuário")
    table.add_column("Evento")
    table.add_column("IP")
    table.add_column("Risco", justify="center")
    table.add_column("Status", justify="center")
    
    for event in events[:50]:  # Limitar a 50 eventos
        # Cor baseada no risco
        if event['risk_level'] >= 8:
            risk_color = "red"
        elif event['risk_level'] >= 5:
            risk_color = "yellow"
        else:
            risk_color = "green"
        
        # Status
        status = "[green]✓[/green]" if event['success'] else "[red]✗[/red]"
        
        table.add_row(
            event['timestamp'][:19],
            event['user_id'] or '-',
            event['event_type'],
            event['ip_address'],
            f"[{risk_color}]{event['risk_level']}[/{risk_color}]",
            status
        )
    
    console.print("\n", table)

@cli.group()
def export():
    """Exportação de dados"""
    pass

@export.command('results')
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--format', '-f', 
              type=click.Choice(['pdf', 'html', 'docx', 'json', 'csv']),
              default='pdf', help='Formato de exportação')
@click.option('--output', '-o', type=click.Path(), required=True)
@click.option('--template', '-t', type=click.Path(exists=True), help='Template customizado')
def export_results(input_file, format, output, template):
    """Exporta resultados de análise"""
    export_mgr = ExportManager()
    
    # Carregar resultados
    with open(input_file, 'r') as f:
        if input_file.endswith('.json'):
            data = json.load(f)
        else:
            console.print("[red]Formato de entrada não suportado[/red]")
            return
    
    # Preparar conteúdo para relatório
    content = {
        'title': 'Relatório de Análise Médica - MedAI',
        'metadata': {
            'Data': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'Sistema': 'MedAI v1.0.0',
            'Arquivo': Path(input_file).name
        },
        'sections': [
            {
                'title': 'Resultados da Análise',
                'text': f"Diagnóstico: {data.get('predicted_class', 'N/A')}",
                'table': [
                    ['Classe', 'Probabilidade'],
                    *[[k, f"{v:.2%}"] for k, v in data.get('predictions', {}).items()]
                ]
            }
        ]
    }
    
    # Exportar
    with console.status(f"Exportando para {format.upper()}..."):
        try:
            export_info = export_mgr.export_report(
                content,
                output,
                format=format,
                template=template
            )
            
            console.print(f"\n[green]Exportação concluída![/green]")
            console.print(f"Arquivo: {export_info['path']}")
            console.print(f"Tamanho: {export_info['size_bytes'] / 1024:.1f} KB")
            
        except Exception as e:
            console.print(f"\n[red]Erro na exportação: {str(e)}[/red]")

@cli.command()
@click.option('--port', '-p', default=8080, help='Porta do servidor')
@click.option('--host', '-h', default='localhost', help='Host do servidor')
@click.option('--debug', is_flag=True, help='Modo debug')
def server(port, host, debug):
    """Inicia servidor web do MedAI"""
    console.print(f"\n[bold blue]Iniciando servidor MedAI[/bold blue]")
    console.print(f"Host: {host}")
    console.print(f"Porta: {port}")
    console.print(f"Debug: {'Sim' if debug else 'Não'}")
    
    console.print("\n[yellow]Servidor web em desenvolvimento[/yellow]")
    console.print("Use 'medai analyze' para análise via CLI")

@cli.command()
def config():
    """Mostra configuração atual"""
    tree = Tree("[bold]Configuração MedAI[/bold]")
    
    # Diretórios
    dirs_branch = tree.add("Diretórios")
    dirs_branch.add(f"Base: {Config.BASE_DIR}")
    dirs_branch.add(f"Dados: {Config.DATA_DIR}")
    dirs_branch.add(f"Modelos: {Config.MODELS_DIR}")
    dirs_branch.add(f"Logs: {Config.LOGS_DIR}")
    dirs_branch.add(f"Relatórios: {Config.REPORTS_DIR}")
    
    # Modelos
    models_branch = tree.add("Modelos Configurados")
    for name in Config.MODEL_CONFIG.keys():
        models_branch.add(name)
    
    # Sistema
    system_branch = tree.add("Sistema")
    system_branch.add(f"Versão: {Config.APP_VERSION}")
    system_branch.add(f"GPU: {'Habilitada' if Config.GPU_ENABLED else 'Desabilitada'}")
    system_branch.add(f"Cache: {Config.CACHE_SIZE} imagens")
    
    console.print("\n", tree)

@cli.command()
@click.option('--all', '-a', is_flag=True, help='Executar todos os testes')
@click.option('--unit', '-u', is_flag=True, help='Testes unitários')
@click.option('--integration', '-i', is_flag=True, help='Testes de integração')
@click.option('--performance', '-p', is_flag=True, help='Testes de performance')
def test(all, unit, integration, performance):
    """Executa testes do sistema"""
    console.print("\n[bold blue]Executando Testes[/bold blue]")
    
    tests_to_run = []
    if all:
        tests_to_run = ['unit', 'integration', 'performance']
    else:
        if unit:
            tests_to_run.append('unit')
        if integration:
            tests_to_run.append('integration')
        if performance:
            tests_to_run.append('performance')
    
    if not tests_to_run:
        tests_to_run = ['unit']  # Padrão
    
    # Executar testes
    import subprocess
    
    for test_type in tests_to_run:
        console.print(f"\n[cyan]Executando testes {test_type}...[/cyan]")
        
        cmd = ['python', '-m', 'pytest', f'tests/{test_type}', '-v']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green]✓ Testes {test_type} passaram![/green]")
            else:
                console.print(f"[red]✗ Testes {test_type} falharam[/red]")
                console.print(result.stdout)
                
        except Exception as e:
            console.print(f"[red]Erro ao executar testes: {str(e)}[/red]")

def save_results(result, output_dir, format):
    """Salva resultados da análise"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'json':
        output_file = output_path / f'resultado_{timestamp}.json'
        data = {
            'timestamp': datetime.now().isoformat(),
            'predicted_class': result.predicted_class,
            'confidence': result.confidence,
            'predictions': result.predictions,
            'processing_time': result.processing_time
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format == 'pdf':
        # Implementar geração de PDF
        output_file = output_path / f'relatorio_{timestamp}.pdf'
        console.print("[yellow]Geração de PDF em desenvolvimento[/yellow]")
    
    elif format == 'html':
        # Implementar geração de HTML
        output_file = output_path / f'relatorio_{timestamp}.html'
        console.print("[yellow]Geração de HTML em desenvolvimento[/yellow]")

def visualize_results(image, result):
    """Visualiza resultados (placeholder)"""
    console.print("\n[yellow]Visualização em desenvolvimento[/yellow]")
    console.print("Use a interface gráfica para visualização completa")

if __name__ == '__main__':
    cli()
