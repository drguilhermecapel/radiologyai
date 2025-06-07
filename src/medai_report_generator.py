# report_generator.py - Sistema de geração de relatórios médicos

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, 
    Table, TableStyle, PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tempfile
from datetime import datetime
import io
from typing import Dict, List, Optional, Any
import logging
import base64

logger = logging.getLogger('MedAI.Reports')

class MedicalReportGenerator:
    """
    Gerador de relatórios médicos profissionais
    Cria documentos PDF estruturados com resultados de análise
    """
    
    def __init__(self, institution_name: str = "MedAI Radiologia",
                 institution_logo: Optional[str] = None):
        self.institution_name = institution_name
        self.institution_logo = institution_logo
        self.temp_dir = tempfile.mkdtemp()
        
        # Estilos
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Cria estilos customizados para o relatório"""
        # Título principal
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#003366'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtítulo
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#0066cc'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Texto normal
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
        
        # Texto de destaque
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['BodyText'],
            fontSize=12,
            textColor=colors.HexColor('#cc0000'),
            alignment=TA_LEFT
        ))
        
        # Rodapé
        self.styles.add(ParagraphStyle(
            name='Footer',
            parent=self.styles['Normal'],
            fontSize=9,
            textColor=colors.gray,
            alignment=TA_CENTER
        ))
    
    def generate_report(self,
                       output_path: str,
                       patient_info: Dict[str, Any],
                       analysis_results: Dict[str, Any],
                       image_path: Optional[str] = None,
                       physician_name: str = "Sistema Automatizado",
                       additional_notes: str = "") -> bool:
        """
        Gera relatório médico completo
        
        Args:
            output_path: Caminho para salvar o PDF
            patient_info: Informações do paciente
            analysis_results: Resultados da análise de IA
            image_path: Caminho da imagem analisada
            physician_name: Nome do médico responsável
            additional_notes: Observações adicionais
            
        Returns:
            True se gerado com sucesso
        """
        try:
            # Criar documento
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # Construir conteúdo
            story = []
            
            # Cabeçalho
            story.extend(self._create_header())
            
            # Informações do paciente
            story.extend(self._create_patient_section(patient_info))
            
            # Informações do exame
            story.extend(self._create_exam_section(patient_info))
            
            # Resultados da análise
            story.extend(self._create_analysis_section(analysis_results))
            
            # Imagens
            if image_path:
                story.extend(self._create_images_section(
                    image_path, 
                    analysis_results.get('heatmap_path')
                ))
            
            # Interpretação e conclusões
            story.extend(self._create_interpretation_section(
                analysis_results, 
                additional_notes
            ))
            
            # Assinatura
            story.extend(self._create_signature_section(physician_name))
            
            # Gerar PDF
            doc.build(story, onFirstPage=self._add_page_number, 
                     onLaterPages=self._add_page_number)
            
            logger.info(f"Relatório gerado: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")
            return False
    
    def _create_header(self) -> List:
        """Cria cabeçalho do relatório"""
        elements = []
        
        # Logo e título
        if self.institution_logo and Path(self.institution_logo).exists():
            logo = Image(self.institution_logo, width=2*inch, height=1*inch)
            logo.hAlign = 'CENTER'
            elements.append(logo)
            elements.append(Spacer(1, 12))
        
        # Título da instituição
        elements.append(Paragraph(
            self.institution_name,
            self.styles['CustomTitle']
        ))
        
        # Subtítulo
        elements.append(Paragraph(
            "RELATÓRIO DE ANÁLISE RADIOLÓGICA POR INTELIGÊNCIA ARTIFICIAL",
            self.styles['Heading2']
        ))
        
        elements.append(Spacer(1, 20))
        
        # Linha divisória
        elements.append(Table(
            [['']],
            colWidths=[doc.width],
            style=TableStyle([
                ('LINEBELOW', (0, 0), (-1, -1), 2, colors.HexColor('#003366'))
            ])
        ))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_patient_section(self, patient_info: Dict) -> List:
        """Cria seção com informações do paciente"""
        elements = []
        
        elements.append(Paragraph(
            "DADOS DO PACIENTE",
            self.styles['CustomHeading']
        ))
        
        # Tabela com dados do paciente
        patient_data = [
            ['ID do Paciente:', patient_info.get('PatientID', 'Não informado')],
            ['Nome:', patient_info.get('PatientName', 'Anonimizado')],
            ['Idade:', patient_info.get('PatientAge', 'Não informada')],
            ['Sexo:', patient_info.get('PatientSex', 'Não informado')],
            ['Data de Nascimento:', patient_info.get('PatientBirthDate', 'Não informada')]
        ]
        
        table = Table(patient_data, colWidths=[4*cm, 10*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_exam_section(self, exam_info: Dict) -> List:
        """Cria seção com informações do exame"""
        elements = []
        
        elements.append(Paragraph(
            "DADOS DO EXAME",
            self.styles['CustomHeading']
        ))
        
        exam_data = [
            ['Modalidade:', exam_info.get('Modality', 'Não especificada')],
            ['Data do Exame:', self._format_date(exam_info.get('StudyDate', ''))],
            ['Descrição:', exam_info.get('StudyDescription', 'Não disponível')],
            ['Parte do Corpo:', exam_info.get('BodyPartExamined', 'Não especificada')],
            ['Equipamento:', exam_info.get('Manufacturer', 'Não informado')]
        ]
        
        # Adicionar parâmetros técnicos se disponíveis
        if exam_info.get('KVP'):
            exam_data.append(['KVP:', exam_info['KVP']])
        if exam_info.get('ExposureTime'):
            exam_data.append(['Tempo de Exposição:', f"{exam_info['ExposureTime']} ms"])
        
        table = Table(exam_data, colWidths=[4*cm, 10*cm])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_analysis_section(self, results: Dict) -> List:
        """Cria seção com resultados da análise"""
        elements = []
        
        elements.append(Paragraph(
            "RESULTADOS DA ANÁLISE",
            self.styles['CustomHeading']
        ))
        
        # Resultado principal
        predicted_class = results.get('predicted_class', 'Indeterminado')
        confidence = results.get('confidence', 0)
        
        # Destaque do resultado principal
        result_text = f"<b>Classificação:</b> {predicted_class}<br/>"
        result_text += f"<b>Confiança:</b> {confidence:.1%}"
        
        # Cor baseada na confiança
        if confidence > 0.8:
            style = self.styles['Highlight']
        else:
            style = self.styles['CustomBody']
        
        elements.append(Paragraph(result_text, style))
        elements.append(Spacer(1, 12))
        
        # Tabela de probabilidades
        elements.append(Paragraph(
            "Distribuição de Probabilidades:",
            self.styles['BodyText']
        ))
        
        prob_data = [['Classe', 'Probabilidade']]
        predictions = results.get('predictions', {})
        
        for class_name, prob in sorted(predictions.items(), 
                                      key=lambda x: x[1], reverse=True):
            prob_data.append([class_name, f"{prob:.1%}"])
        
        prob_table = Table(prob_data, colWidths=[8*cm, 4*cm])
        prob_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        elements.append(prob_table)
        elements.append(Spacer(1, 20))
        
        # Métricas de confiança
        if 'uncertainty_metrics' in results:
            elements.append(Paragraph(
                "Métricas de Incerteza:",
                self.styles['BodyText']
            ))
            
            metrics = results['uncertainty_metrics']
            metrics_text = f"""
            • Entropia Normalizada: {metrics.get('normalized_entropy', 0):.2%}<br/>
            • Margem de Decisão: {metrics.get('margin', 0):.2f}<br/>
            • Variância: {metrics.get('variance', 0):.4f}
            """
            
            elements.append(Paragraph(metrics_text, self.styles['CustomBody']))
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_images_section(self, 
                              original_path: str,
                              heatmap_path: Optional[str] = None) -> List:
        """Cria seção com imagens"""
        elements = []
        
        elements.append(Paragraph(
            "IMAGENS",
            self.styles['CustomHeading']
        ))
        
        # Preparar imagens lado a lado
        images_data = []
        
        # Imagem original
        if Path(original_path).exists():
            img_original = Image(original_path, width=6*cm, height=6*cm)
            img_original.hAlign = 'CENTER'
            images_data.append([img_original, 'Imagem Original'])
        
        # Heatmap se disponível
        if heatmap_path and Path(heatmap_path).exists():
            img_heatmap = Image(heatmap_path, width=6*cm, height=6*cm)
            img_heatmap.hAlign = 'CENTER'
            images_data.append([img_heatmap, 'Mapa de Atenção'])
        
        # Criar tabela com imagens
        if images_data:
            # Transpor para layout horizontal
            if len(images_data) == 2:
                img_table = Table(
                    [[images_data[0][0], images_data[1][0]],
                     [Paragraph(images_data[0][1], self.styles['BodyText']),
                      Paragraph(images_data[1][1], self.styles['BodyText'])]],
                    colWidths=[7*cm, 7*cm]
                )
            else:
                img_table = Table(
                    [[images_data[0][0]],
                     [Paragraph(images_data[0][1], self.styles['BodyText'])]],
                    colWidths=[14*cm]
                )
            
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12)
            ]))
            
            elements.append(KeepTogether(img_table))
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_interpretation_section(self, 
                                     results: Dict,
                                     additional_notes: str) -> List:
        """Cria seção de interpretação e conclusões"""
        elements = []
        
        elements.append(Paragraph(
            "INTERPRETAÇÃO E CONCLUSÕES",
            self.styles['CustomHeading']
        ))
        
        # Interpretação automatizada
        interpretation = self._generate_interpretation(results)
        elements.append(Paragraph(interpretation, self.styles['CustomBody']))
        
        # Notas adicionais
        if additional_notes:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(
                "Observações Adicionais:",
                self.styles['BodyText']
            ))
            elements.append(Paragraph(additional_notes, self.styles['CustomBody']))
        
        # Avisos e disclaimers
        elements.append(Spacer(1, 20))
        disclaimer = """
        <b>IMPORTANTE:</b> Esta análise foi realizada por um sistema de 
        inteligência artificial e deve ser utilizada como ferramenta auxiliar 
        de diagnóstico. O resultado deve ser sempre interpretado por um 
        profissional médico qualificado em conjunto com o histórico clínico 
        do paciente e outros exames complementares.
        """
        
        elements.append(Paragraph(
            disclaimer,
            ParagraphStyle(
                name='Disclaimer',
                parent=self.styles['BodyText'],
                fontSize=9,
                textColor=colors.HexColor('#666666'),
                borderWidth=1,
                borderColor=colors.HexColor('#cccccc'),
                borderPadding=10,
                backColor=colors.HexColor('#f5f5f5')
            )
        ))
        
        return elements
    
    def _create_signature_section(self, physician_name: str) -> List:
        """Cria seção de assinatura"""
        elements = []
        
        elements.append(Spacer(1, 40))
        
        # Data e hora
        current_time = datetime.now()
        date_text = current_time.strftime("%d de %B de %Y às %H:%M")
        
        # Tabela de assinatura
        signature_data = [
            ['_' * 40],
            [physician_name],
            ['Médico Responsável'],
            [f'Data: {date_text}']
        ]
        
        sig_table = Table(signature_data, colWidths=[8*cm])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 1), (0, 1), 12),
            ('FONTNAME', (0, 1), (0, 1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 2), (0, 3), 9),
            ('TEXTCOLOR', (0, 2), (0, 3), colors.grey)
        ]))
        
        # Alinhar à direita
        sig_table.hAlign = 'RIGHT'
        elements.append(sig_table)
        
        return elements
    
    def _generate_interpretation(self, results: Dict) -> str:
        """Gera interpretação automatizada dos resultados"""
        predicted_class = results.get('predicted_class', 'Indeterminado')
        confidence = results.get('confidence', 0)
        
        # Templates de interpretação baseados na classe e confiança
        interpretations = {
            'Normal': {
                'high': "A análise por inteligência artificial não identificou alterações significativas na imagem radiológica. O padrão observado é compatível com achados dentro dos limites da normalidade.",
                'medium': "A análise sugere achados dentro dos limites da normalidade, porém recomenda-se correlação clínica devido ao grau moderado de confiança.",
                'low': "Os achados sugerem padrão normal, mas devido à baixa confiança da análise, recomenda-se revisão cuidadosa e correlação com dados clínicos."
            },
            'Pneumonia': {
                'high': "A análise identificou padrões radiológicos sugestivos de processo pneumônico. Observam-se alterações compatíveis com consolidação pulmonar.",
                'medium': "Há indícios de alterações sugestivas de pneumonia. Recomenda-se correlação clínica e laboratorial para confirmação diagnóstica.",
                'low': "Possíveis alterações sugestivas de processo infeccioso pulmonar. Devido à incerteza da análise, é imprescindível avaliação médica detalhada."
            },
            'COVID-19': {
                'high': "Os padrões identificados são altamente sugestivos de pneumonia por COVID-19, com achados típicos de opacidades em vidro fosco.",
                'medium': "Observam-se alterações que podem ser compatíveis com COVID-19. Correlação com teste RT-PCR e quadro clínico é essencial.",
                'low': "Alterações inespecíficas que podem estar relacionadas a COVID-19. Investigação adicional é necessária."
            },
            'Fratura': {
                'high': "A análise detectou descontinuidade cortical compatível com fratura. Recomenda-se avaliação ortopédica imediata.",
                'medium': "Possível linha de fratura identificada. Sugere-se projeções adicionais para melhor caracterização.",
                'low': "Alterações suspeitas que podem representar fratura. Avaliação clínica e radiográfica adicional é recomendada."
            }
        }
        
        # Determinar nível de confiança
        if confidence > 0.8:
            conf_level = 'high'
        elif confidence > 0.6:
            conf_level = 'medium'
        else:
            conf_level = 'low'
        
        # Obter interpretação apropriada
        if predicted_class in interpretations:
            interpretation = interpretations[predicted_class][conf_level]
        else:
            interpretation = f"A análise identificou '{predicted_class}' com {confidence:.1%} de confiança. Correlação clínica é recomendada."
        
        # Adicionar informações sobre o nível de confiança
        interpretation += f"\n\nNível de confiança da análise: {confidence:.1%}"
        
        # Adicionar recomendações baseadas na confiança
        if confidence < 0.7:
            interpretation += "\n\n<b>Nota:</b> Devido ao grau de incerteza, recomenda-se fortemente uma segunda opinião médica e/ou exames complementares."
        
        return interpretation
    
    def _format_date(self, date_str: str) -> str:
        """Formata data DICOM para formato legível"""
        if not date_str or len(date_str) < 8:
            return "Não informada"
        
        try:
            # Formato DICOM: YYYYMMDD
            year = date_str[:4]
            month = date_str[4:6]
            day = date_str[6:8]
            
            # Converter para objeto datetime
            date_obj = datetime(int(year), int(month), int(day))
            
            # Formatar em português
            months_pt = {
                1: 'janeiro', 2: 'fevereiro', 3: 'março', 4: 'abril',
                5: 'maio', 6: 'junho', 7: 'julho', 8: 'agosto',
                9: 'setembro', 10: 'outubro', 11: 'novembro', 12: 'dezembro'
            }
            
            return f"{day} de {months_pt[date_obj.month]} de {year}"
            
        except:
            return date_str
    
    def _add_page_number(self, canvas, doc):
        """Adiciona número de página"""
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.grey)
        
        # Número da página
        page_num = canvas.getPageNumber()
        text = f"Página {page_num}"
        canvas.drawCentredString(doc.pagesize[0]/2, 1*cm, text)
        
        # Rodapé
        footer_text = f"{self.institution_name} - Relatório gerado por MedAI"
        canvas.drawCentredString(doc.pagesize[0]/2, 0.5*cm, footer_text)
        
        canvas.restoreState()
    
    def generate_html_report(self, 
                           output_path: str,
                           patient_info: Dict,
                           analysis_results: Dict,
                           image_path: Optional[str] = None) -> bool:
        """
        Gera relatório em formato HTML
        
        Args:
            output_path: Caminho para salvar o HTML
            patient_info: Informações do paciente
            analysis_results: Resultados da análise
            image_path: Caminho da imagem
            
        Returns:
            True se gerado com sucesso
        """
        try:
            # Template HTML
            html_template = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório MedAI - {patient_id}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #003366;
            text-align: center;
            border-bottom: 3px solid #0066cc;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #0066cc;
            margin-top: 30px;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .info-table td {{
            padding: 10px;
            border: 1px solid #ddd;
        }}
        .info-table td:first-child {{
            background-color: #f0f0f0;
            font-weight: bold;
            width: 30%;
        }}
        .result-box {{
            background-color: #e8f4f8;
            border: 2px solid #0066cc;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }}
        .confidence-high {{
            color: #008000;
            font-weight: bold;
        }}
        .confidence-medium {{
            color: #ff8c00;
            font-weight: bold;
        }}
        .confidence-low {{
            color: #dc143c;
            font-weight: bold;
        }}
        .disclaimer {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin-top: 30px;
            font-size: 0.9em;
        }}
        .images {{
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }}
        .image-container {{
            text-align: center;
        }}
        .image-container img {{
            max-width: 300px;
            max-height: 300px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .probability-bar {{
            background-color: #f0f0f0;
            border-radius: 5px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .probability-fill {{
            background-color: #0066cc;
            color: white;
            padding: 5px;
            text-align: right;
        }}
        @media print {{
            body {{
                background-color: white;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{institution_name}</h1>
        <h2 style="text-align: center;">Relatório de Análise Radiológica por IA</h2>
        
        <h2>Dados do Paciente</h2>
        <table class="info-table">
            <tr>
                <td>ID do Paciente</td>
                <td>{patient_id}</td>
            </tr>
            <tr>
                <td>Nome</td>
                <td>{patient_name}</td>
            </tr>
            <tr>
                <td>Idade</td>
                <td>{patient_age}</td>
            </tr>
            <tr>
                <td>Sexo</td>
                <td>{patient_sex}</td>
            </tr>
        </table>
        
        <h2>Dados do Exame</h2>
        <table class="info-table">
            <tr>
                <td>Modalidade</td>
                <td>{modality}</td>
            </tr>
            <tr>
                <td>Data do Exame</td>
                <td>{study_date}</td>
            </tr>
            <tr>
                <td>Descrição</td>
                <td>{study_description}</td>
            </tr>
        </table>
        
        <h2>Resultados da Análise</h2>
        <div class="result-box">
            <h3>Classificação: {predicted_class}</h3>
            <p>Confiança: <span class="{confidence_class}">{confidence:.1%}</span></p>
            
            <h4>Distribuição de Probabilidades:</h4>
            {probability_bars}
        </div>
        
        {images_section}
        
        <h2>Interpretação</h2>
        <p>{interpretation}</p>
        
        <div class="disclaimer">
            <strong>IMPORTANTE:</strong> Esta análise foi realizada por um sistema de 
            inteligência artificial e deve ser utilizada como ferramenta auxiliar 
            de diagnóstico. O resultado deve ser sempre interpretado por um 
            profissional médico qualificado.
        </div>
        
        <div style="margin-top: 50px; text-align: right;">
            <p>_______________________________________<br>
            Médico Responsável<br>
            <small>Data: {current_date}</small></p>
        </div>
    </div>
</body>
</html>
            """
            
            # Preparar dados
            confidence = analysis_results.get('confidence', 0)
            confidence_class = 'confidence-high' if confidence > 0.8 else \
                             'confidence-medium' if confidence > 0.6 else \
                             'confidence-low'
            
            # Criar barras de probabilidade
            probability_bars = ""
            for class_name, prob in analysis_results.get('predictions', {}).items():
                probability_bars += f"""
                <div class="probability-bar">
                    <div class="probability-fill" style="width: {prob*100}%">
                        {class_name}: {prob:.1%}
                    </div>
                </div>
                """
            
            # Seção de imagens
            images_section = ""
            if image_path and Path(image_path).exists():
                # Converter imagem para base64
                with open(image_path, 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                
                images_section = f"""
                <h2>Imagens</h2>
                <div class="images">
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_base64}" alt="Imagem Original">
                        <p>Imagem Original</p>
                    </div>
                </div>
                """
            
            # Preencher template
            html_content = html_template.format(
                institution_name=self.institution_name,
                patient_id=patient_info.get('PatientID', 'N/A'),
                patient_name=patient_info.get('PatientName', 'Anonimizado'),
                patient_age=patient_info.get('PatientAge', 'N/A'),
                patient_sex=patient_info.get('PatientSex', 'N/A'),
                modality=patient_info.get('Modality', 'N/A'),
                study_date=self._format_date(patient_info.get('StudyDate', '')),
                study_description=patient_info.get('StudyDescription', 'N/A'),
                predicted_class=analysis_results.get('predicted_class', 'Indeterminado'),
                confidence=confidence,
                confidence_class=confidence_class,
                probability_bars=probability_bars,
                images_section=images_section,
                interpretation=self._generate_interpretation(analysis_results),
                current_date=datetime.now().strftime('%d/%m/%Y %H:%M')
            )
            
            # Salvar arquivo
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Relatório HTML gerado: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório HTML: {str(e)}")
            return False
