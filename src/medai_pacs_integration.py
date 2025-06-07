# pacs_integration.py - Sistema de integração com PACS e conformidade com padrões médicos

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pynetdicom import AE, evt, StoragePresentationContexts, debug_logger
from pynetdicom.sop_class import (
    CTImageStorage, MRImageStorage, 
    XRayRadiographicImageStorage,
    SecondaryCaptureImageStorage,
    StructuredReportStorage
)
import hl7
import fhirclient.models.diagnosticreport as dr
import fhirclient.models.observation as obs
import fhirclient.models.patient as patient
import fhirclient.models.imagingstudy as imaging
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import base64

logger = logging.getLogger('MedAI.PACS')

@dataclass
class PACSConfig:
    """Configuração para conexão PACS"""
    ae_title: str = "MEDAI_SCU"
    pacs_ae_title: str = "PACS_SCP"
    pacs_host: str = "localhost"
    pacs_port: int = 11112
    timeout: int = 30
    max_pdu: int = 16384
    
@dataclass
class HL7Config:
    """Configuração para integração HL7"""
    sending_application: str = "MEDAI"
    sending_facility: str = "AI_LAB"
    receiving_application: str = "HIS"
    receiving_facility: str = "HOSPITAL"
    version: str = "2.5"

class PACSIntegration:
    """
    Sistema de integração com PACS (Picture Archiving and Communication System)
    Suporta DICOM, HL7 e FHIR para interoperabilidade médica
    """
    
    def __init__(self, pacs_config: PACSConfig, hl7_config: HL7Config):
        self.pacs_config = pacs_config
        self.hl7_config = hl7_config
        
        # Configurar Application Entity
        self.ae = AE(ae_title=self.pacs_config.ae_title)
        self.ae.network_timeout = self.pacs_config.timeout
        self.ae.maximum_pdu_size = self.pacs_config.max_pdu
        
        # Adicionar contextos de apresentação
        self._setup_presentation_contexts()
        
        # Cliente FHIR (configurar servidor se disponível)
        self.fhir_client = None
        
        logger.info(f"PACSIntegration inicializado - AE Title: {self.pacs_config.ae_title}")
    
    def _setup_presentation_contexts(self):
        """Configura contextos de apresentação DICOM"""
        # Adicionar contextos de storage
        self.ae.requested_contexts = StoragePresentationContexts
        
        # Adicionar contextos adicionais
        additional_contexts = [
            StructuredReportStorage,
            SecondaryCaptureImageStorage
        ]
        
        for context in additional_contexts:
            self.ae.add_requested_context(context)
    
    def query_pacs(self, 
                   query_params: Dict[str, str],
                   query_level: str = "STUDY") -> List[Dict]:
        """
        Consulta PACS usando C-FIND
        
        Args:
            query_params: Parâmetros de consulta (PatientID, StudyDate, etc)
            query_level: Nível da consulta (PATIENT, STUDY, SERIES, IMAGE)
            
        Returns:
            Lista de resultados encontrados
        """
        from pynetdicom import AE
        from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelFind
        
        # Criar dataset de consulta
        ds = Dataset()
        ds.QueryRetrieveLevel = query_level
        
        # Adicionar parâmetros de consulta
        for tag, value in query_params.items():
            setattr(ds, tag, value)
        
        # Campos a retornar
        ds.PatientName = ''
        ds.PatientID = ''
        ds.StudyDate = ''
        ds.StudyTime = ''
        ds.StudyDescription = ''
        ds.StudyInstanceUID = ''
        ds.Modality = ''
        
        results = []
        
        try:
            # Estabelecer associação
            assoc = self.ae.associate(
                self.pacs_config.pacs_host,
                self.pacs_config.pacs_port,
                contexts=[StudyRootQueryRetrieveInformationModelFind]
            )
            
            if assoc.is_established:
                # Enviar C-FIND
                responses = assoc.send_c_find(
                    ds, 
                    StudyRootQueryRetrieveInformationModelFind
                )
                
                for (status, identifier) in responses:
                    if status and identifier:
                        # Converter para dicionário
                        result = {
                            'PatientName': str(identifier.PatientName),
                            'PatientID': str(identifier.PatientID),
                            'StudyDate': str(identifier.StudyDate),
                            'StudyTime': str(identifier.StudyTime),
                            'StudyDescription': str(identifier.StudyDescription),
                            'StudyInstanceUID': str(identifier.StudyInstanceUID),
                            'Modality': str(identifier.Modality)
                        }
                        results.append(result)
                
                # Liberar associação
                assoc.release()
                
                logger.info(f"C-FIND concluído: {len(results)} resultados")
            else:
                logger.error("Falha ao estabelecer associação com PACS")
                
        except Exception as e:
            logger.error(f"Erro na consulta PACS: {str(e)}")
            
        return results
    
    def retrieve_study(self,
                      study_uid: str,
                      output_dir: str) -> List[str]:
        """
        Recupera estudo do PACS usando C-MOVE
        
        Args:
            study_uid: UID do estudo
            output_dir: Diretório de saída
            
        Returns:
            Lista de arquivos recuperados
        """
        from pynetdicom import AE, evt
        from pynetdicom.sop_class import StudyRootQueryRetrieveInformationModelMove
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        retrieved_files = []
        
        # Handler para C-STORE (receber imagens)
        def handle_store(event):
            """Handle C-STORE request"""
            ds = event.dataset
            ds.file_meta = event.file_meta
            
            # Gerar nome de arquivo
            filename = f"{ds.SOPInstanceUID}.dcm"
            filepath = output_path / filename
            
            # Salvar arquivo
            ds.save_as(str(filepath), write_like_original=False)
            retrieved_files.append(str(filepath))
            
            return 0x0000  # Success
        
        # Configurar handlers
        handlers = [(evt.EVT_C_STORE, handle_store)]
        
        # Criar AE para receber imagens
        storage_ae = AE(ae_title=self.pacs_config.ae_title)
        storage_ae.supported_contexts = StoragePresentationContexts
        
        # Iniciar servidor SCP em thread separada
        import threading
        scp_thread = threading.Thread(
            target=storage_ae.start_server,
            args=(('', 11113),),
            kwargs={'evt_handlers': handlers}
        )
        scp_thread.daemon = True
        scp_thread.start()
        
        try:
            # Criar dataset de consulta
            ds = Dataset()
            ds.QueryRetrieveLevel = 'STUDY'
            ds.StudyInstanceUID = study_uid
            
            # Estabelecer associação para C-MOVE
            assoc = self.ae.associate(
                self.pacs_config.pacs_host,
                self.pacs_config.pacs_port,
                contexts=[StudyRootQueryRetrieveInformationModelMove]
            )
            
            if assoc.is_established:
                # Enviar C-MOVE
                responses = assoc.send_c_move(
                    ds,
                    self.pacs_config.ae_title,  # Destination AE
                    StudyRootQueryRetrieveInformationModelMove
                )
                
                for (status, identifier) in responses:
                    if status:
                        logger.info(f"C-MOVE status: 0x{status.Status:04X}")
                
                assoc.release()
                
                logger.info(f"C-MOVE concluído: {len(retrieved_files)} arquivos")
            else:
                logger.error("Falha ao estabelecer associação para C-MOVE")
                
        except Exception as e:
            logger.error(f"Erro no C-MOVE: {str(e)}")
        finally:
            # Parar servidor SCP
            storage_ae.shutdown()
            
        return retrieved_files
    
    def send_to_pacs(self, dicom_file: str) -> bool:
        """
        Envia arquivo DICOM para PACS usando C-STORE
        
        Args:
            dicom_file: Caminho do arquivo DICOM
            
        Returns:
            True se enviado com sucesso
        """
        try:
            # Ler arquivo DICOM
            ds = pydicom.dcmread(dicom_file)
            
            # Determinar SOP Class baseado na modalidade
            if ds.Modality == 'CT':
                context = CTImageStorage
            elif ds.Modality == 'MR':
                context = MRImageStorage
            elif ds.Modality == 'CR' or ds.Modality == 'DX':
                context = XRayRadiographicImageStorage
            else:
                context = SecondaryCaptureImageStorage
            
            # Estabelecer associação
            assoc = self.ae.associate(
                self.pacs_config.pacs_host,
                self.pacs_config.pacs_port,
                contexts=[context]
            )
            
            if assoc.is_established:
                # Enviar C-STORE
                status = assoc.send_c_store(ds)
                
                # Verificar status
                if status:
                    logger.info(f"C-STORE status: 0x{status.Status:04X}")
                    success = status.Status == 0x0000
                else:
                    success = False
                
                # Liberar associação
                assoc.release()
                
                return success
            else:
                logger.error("Falha ao estabelecer associação para C-STORE")
                return False
                
        except Exception as e:
            logger.error(f"Erro no C-STORE: {str(e)}")
            return False
    
    def create_structured_report(self,
                               analysis_results: Dict,
                               original_study_uid: str,
                               patient_info: Dict) -> FileDataset:
        """
        Cria Structured Report DICOM com resultados da análise
        
        Args:
            analysis_results: Resultados da análise de IA
            original_study_uid: UID do estudo original
            patient_info: Informações do paciente
            
        Returns:
            Dataset do Structured Report
        """
        # Criar FileDataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = StructuredReportStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        
        # Criar dataset principal
        ds = FileDataset(
            None, {},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )
        
        # Informações do paciente
        ds.PatientName = patient_info.get('PatientName', '^')
        ds.PatientID = patient_info.get('PatientID', '')
        ds.PatientBirthDate = patient_info.get('PatientBirthDate', '')
        ds.PatientSex = patient_info.get('PatientSex', '')
        
        # Informações do estudo
        ds.StudyDate = datetime.now().strftime('%Y%m%d')
        ds.StudyTime = datetime.now().strftime('%H%M%S')
        ds.StudyInstanceUID = generate_uid()
        ds.StudyID = 'AI_ANALYSIS'
        
        # Informações da série
        ds.SeriesDate = ds.StudyDate
        ds.SeriesTime = ds.StudyTime
        ds.SeriesInstanceUID = generate_uid()
        ds.SeriesNumber = '999'
        ds.SeriesDescription = 'AI Analysis Report'
        
        # Informações do SR
        ds.SOPClassUID = StructuredReportStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = 'SR'
        ds.ContentDate = ds.StudyDate
        ds.ContentTime = ds.StudyTime
        
        # Referência ao estudo original
        ds.ReferencedStudySequence = [Dataset()]
        ds.ReferencedStudySequence[0].ReferencedSOPClassUID = '1.2.840.10008.3.1.2.3.1'
        ds.ReferencedStudySequence[0].ReferencedSOPInstanceUID = original_study_uid
        
        # Conteúdo do SR
        ds.ValueType = 'CONTAINER'
        ds.ConceptNameCodeSequence = [Dataset()]
        ds.ConceptNameCodeSequence[0].CodeValue = '126000'
        ds.ConceptNameCodeSequence[0].CodingSchemeDesignator = 'DCM'
        ds.ConceptNameCodeSequence[0].CodeMeaning = 'Imaging Report'
        
        # Adicionar resultados da análise
        content_sequence = []
        
        # Achados principais
        finding = Dataset()
        finding.RelationshipType = 'CONTAINS'
        finding.ValueType = 'TEXT'
        finding.ConceptNameCodeSequence = [Dataset()]
        finding.ConceptNameCodeSequence[0].CodeValue = '121071'
        finding.ConceptNameCodeSequence[0].CodingSchemeDesignator = 'DCM'
        finding.ConceptNameCodeSequence[0].CodeMeaning = 'Finding'
        finding.TextValue = f"AI Analysis: {analysis_results['predicted_class']} " \
                           f"(Confidence: {analysis_results['confidence']:.2%})"
        content_sequence.append(finding)
        
        # Adicionar probabilidades de cada classe
        for class_name, probability in analysis_results['predictions'].items():
            prob_item = Dataset()
            prob_item.RelationshipType = 'CONTAINS'
            prob_item.ValueType = 'NUM'
            prob_item.ConceptNameCodeSequence = [Dataset()]
            prob_item.ConceptNameCodeSequence[0].CodeValue = '121072'
            prob_item.ConceptNameCodeSequence[0].CodingSchemeDesignator = 'DCM'
            prob_item.ConceptNameCodeSequence[0].CodeMeaning = 'Probability'
            
            prob_item.MeasuredValueSequence = [Dataset()]
            prob_item.MeasuredValueSequence[0].NumericValue = str(probability)
            prob_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence = [Dataset()]
            prob_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeValue = '%'
            prob_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodingSchemeDesignator = 'UCUM'
            prob_item.MeasuredValueSequence[0].MeasurementUnitsCodeSequence[0].CodeMeaning = 'percent'
            
            content_sequence.append(prob_item)
        
        ds.ContentSequence = content_sequence
        
        return ds
    
    def create_hl7_message(self,
                          patient_data: Dict,
                          analysis_results: Dict,
                          message_type: str = "ORU") -> str:
        """
        Cria mensagem HL7 com resultados da análise
        
        Args:
            patient_data: Dados do paciente
            analysis_results: Resultados da análise
            message_type: Tipo de mensagem HL7 (ORU para resultados)
            
        Returns:
            Mensagem HL7 formatada
        """
        # Criar mensagem HL7
        message = hl7.Message([
            [
                'MSH',
                '|',
                '^~\\&',
                self.hl7_config.sending_application,
                self.hl7_config.sending_facility,
                self.hl7_config.receiving_application,
                self.hl7_config.receiving_facility,
                datetime.now().strftime('%Y%m%d%H%M%S'),
                '',
                f'{message_type}^R01',
                str(int(datetime.now().timestamp())),
                'P',
                self.hl7_config.version
            ]
        ])
        
        # Segmento PID (Patient Identification)
        pid_segment = [
            'PID',
            '1',
            '',
            patient_data.get('PatientID', ''),
            '',
            patient_data.get('PatientName', '').replace(' ', '^'),
            '',
            patient_data.get('PatientBirthDate', ''),
            patient_data.get('PatientSex', ''),
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            ''
        ]
        message.append(pid_segment)
        
        # Segmento OBR (Observation Request)
        obr_segment = [
            'OBR',
            '1',
            '',
            generate_uid()[:20],
            'AI_ANALYSIS',
            '',
            '',
            datetime.now().strftime('%Y%m%d%H%M%S'),
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            'F',  # Final result
            '',
            '',
            '',
            '',
            '',
            ''
        ]
        message.append(obr_segment)
        
        # Segmentos OBX (Observation Result)
        # Resultado principal
        obx_main = [
            'OBX',
            '1',
            'ST',  # String
            'AI_PREDICTION',
            '',
            analysis_results['predicted_class'],
            '',
            '',
            'A',  # Abnormal
            '',
            '',
            'F',
            datetime.now().strftime('%Y%m%d%H%M%S')
        ]
        message.append(obx_main)
        
        # Confiança
        obx_confidence = [
            'OBX',
            '2',
            'NM',  # Numeric
            'AI_CONFIDENCE',
            '',
            str(analysis_results['confidence']),
            '%',
            '',
            'N',  # Normal
            '',
            '',
            'F',
            datetime.now().strftime('%Y%m%d%H%M%S')
        ]
        message.append(obx_confidence)
        
        # Probabilidades por classe
        for i, (class_name, prob) in enumerate(analysis_results['predictions'].items(), 3):
            obx_prob = [
                'OBX',
                str(i),
                'NM',
                f'PROB_{class_name.upper()}',
                '',
                str(prob),
                '%',
                '',
                'N',
                '',
                '',
                'F',
                datetime.now().strftime('%Y%m%d%H%M%S')
            ]
            message.append(obx_prob)
        
        return str(message)
    
    def create_fhir_diagnostic_report(self,
                                    patient_data: Dict,
                                    analysis_results: Dict,
                                    study_data: Dict) -> Dict:
        """
        Cria DiagnosticReport FHIR com resultados da análise
        
        Args:
            patient_data: Dados do paciente
            analysis_results: Resultados da análise
            study_data: Dados do estudo de imagem
            
        Returns:
            Recurso FHIR DiagnosticReport em formato dict
        """
        # Criar DiagnosticReport
        report = {
            "resourceType": "DiagnosticReport",
            "id": generate_uid(),
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                    "code": "RAD",
                    "display": "Radiology"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "82692-5",
                    "display": "AI-Assisted Radiology Report"
                }],
                "text": "Análise de Imagem por Inteligência Artificial"
            },
            "subject": {
                "reference": f"Patient/{patient_data.get('PatientID', 'unknown')}",
                "display": patient_data.get('PatientName', 'Unknown Patient')
            },
            "effectiveDateTime": datetime.now().isoformat(),
            "issued": datetime.now().isoformat(),
            "performer": [{
                "display": "MedAI System"
            }],
            "resultsInterpreter": [{
                "display": "MedAI Neural Network Model"
            }],
            "imagingStudy": [{
                "reference": f"ImagingStudy/{study_data.get('StudyInstanceUID', '')}"
            }],
            "conclusion": f"Análise por IA detectou: {analysis_results['predicted_class']} "
                         f"com confiança de {analysis_results['confidence']:.1%}",
            "conclusionCode": [{
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": self._map_to_snomed(analysis_results['predicted_class']),
                    "display": analysis_results['predicted_class']
                }]
            }]
        }
        
        # Adicionar observações
        observations = []
        
        # Observação principal
        main_obs = {
            "resourceType": "Observation",
            "id": generate_uid(),
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "imaging",
                    "display": "Imaging"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "59776-5",
                    "display": "Procedure findings"
                }]
            },
            "subject": {
                "reference": f"Patient/{patient_data.get('PatientID', 'unknown')}"
            },
            "effectiveDateTime": datetime.now().isoformat(),
            "valueString": analysis_results['predicted_class'],
            "interpretation": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    "code": "A" if analysis_results['confidence'] > 0.8 else "I",
                    "display": "Abnormal" if analysis_results['confidence'] > 0.8 else "Intermediate"
                }]
            }]
        }
        observations.append(main_obs)
        
        # Observações de probabilidade
        for class_name, prob in analysis_results['predictions'].items():
            prob_obs = {
                "resourceType": "Observation",
                "id": generate_uid(),
                "status": "final",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "59777-3",
                        "display": "Probability"
                    }],
                    "text": f"Probabilidade de {class_name}"
                },
                "subject": {
                    "reference": f"Patient/{patient_data.get('PatientID', 'unknown')}"
                },
                "effectiveDateTime": datetime.now().isoformat(),
                "valueQuantity": {
                    "value": prob * 100,
                    "unit": "%",
                    "system": "http://unitsofmeasure.org",
                    "code": "%"
                }
            }
            observations.append(prob_obs)
        
        # Adicionar referências às observações no report
        report["result"] = [{"reference": f"Observation/{obs['id']}"} 
                           for obs in observations]
        
        # Criar bundle com report e observações
        bundle = {
            "resourceType": "Bundle",
            "id": generate_uid(),
            "type": "collection",
            "entry": [
                {"resource": report},
                *[{"resource": obs} for obs in observations]
            ]
        }
        
        return bundle
    
    def _map_to_snomed(self, diagnosis: str) -> str:
        """Mapeia diagnóstico para código SNOMED CT"""
        # Mapeamento simplificado - em produção usar terminologia completa
        snomed_map = {
            "Normal": "17621005",
            "Pneumonia": "233604007",
            "COVID-19": "840544004",
            "Tuberculose": "56717001",
            "Cardiomegalia": "8186001",
            "Fratura": "125605004",
            "Tumor": "108369006",
            "Hemorragia": "50960005"
        }
        
        return snomed_map.get(diagnosis, "260413007")  # Unknown
    
    def export_to_cda(self,
                     patient_data: Dict,
                     analysis_results: Dict,
                     output_path: str):
        """
        Exporta resultados para CDA (Clinical Document Architecture)
        
        Args:
            patient_data: Dados do paciente
            analysis_results: Resultados da análise
            output_path: Caminho para salvar o CDA
        """
        # Criar documento CDA
        root = ET.Element("ClinicalDocument", 
                         xmlns="urn:hl7-org:v3",
                         {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance"})
        
        # Header
        type_id = ET.SubElement(root, "typeId")
        type_id.set("root", "2.16.840.1.113883.1.3")
        type_id.set("extension", "POCD_HD000040")
        
        # ID do documento
        doc_id = ET.SubElement(root, "id")
        doc_id.set("root", generate_uid())
        
        # Código do documento
        code = ET.SubElement(root, "code")
        code.set("code", "18748-4")
        code.set("codeSystem", "2.16.840.1.113883.6.1")
        code.set("displayName", "Diagnostic Imaging Report")
        
        # Título
        title = ET.SubElement(root, "title")
        title.text = "Relatório de Análise de Imagem por IA"
        
        # Data efetiva
        effective_time = ET.SubElement(root, "effectiveTime")
        effective_time.set("value", datetime.now().strftime("%Y%m%d%H%M%S"))
        
        # Confidencialidade
        confidentiality = ET.SubElement(root, "confidentialityCode")
        confidentiality.set("code", "N")
        confidentiality.set("codeSystem", "2.16.840.1.113883.5.25")
        
        # Paciente
        record_target = ET.SubElement(root, "recordTarget")
        patient_role = ET.SubElement(record_target, "patientRole")
        
        patient_id = ET.SubElement(patient_role, "id")
        patient_id.set("root", "2.16.840.1.113883.19.5")
        patient_id.set("extension", patient_data.get("PatientID", ""))
        
        patient = ET.SubElement(patient_role, "patient")
        name = ET.SubElement(patient, "name")
        given = ET.SubElement(name, "given")
        given.text = patient_data.get("PatientName", "").split()[0] if patient_data.get("PatientName") else ""
        family = ET.SubElement(name, "family")
        family.text = patient_data.get("PatientName", "").split()[-1] if patient_data.get("PatientName") else ""
        
        # Autor (sistema)
        author = ET.SubElement(root, "author")
        author_time = ET.SubElement(author, "time")
        author_time.set("value", datetime.now().strftime("%Y%m%d%H%M%S"))
        
        assigned_author = ET.SubElement(author, "assignedAuthor")
        author_id = ET.SubElement(assigned_author, "id")
        author_id.set("root", "2.16.840.1.113883.19.5")
        author_id.set("extension", "MEDAI_SYSTEM")
        
        # Custodiante
        custodian = ET.SubElement(root, "custodian")
        assigned_custodian = ET.SubElement(custodian, "assignedCustodian")
        represented_org = ET.SubElement(assigned_custodian, "representedCustodianOrganization")
        org_id = ET.SubElement(represented_org, "id")
        org_id.set("root", "2.16.840.1.113883.19.5")
        org_name = ET.SubElement(represented_org, "name")
        org_name.text = "MedAI Institution"
        
        # Body estruturado
        component = ET.SubElement(root, "component")
        structured_body = ET.SubElement(component, "structuredBody")
        
        # Seção de resultados
        results_component = ET.SubElement(structured_body, "component")
        section = ET.SubElement(results_component, "section")
        
        # Template ID para resultados de imagem
        template_id = ET.SubElement(section, "templateId")
        template_id.set("root", "2.16.840.1.113883.10.20.6.1.1")
        
        # Código da seção
        section_code = ET.SubElement(section, "code")
        section_code.set("code", "59776-5")
        section_code.set("codeSystem", "2.16.840.1.113883.6.1")
        section_code.set("displayName", "Findings")
        
        # Título da seção
        section_title = ET.SubElement(section, "title")
        section_title.text = "Achados da Análise por IA"
        
        # Texto narrativo
        text = ET.SubElement(section, "text")
        paragraph = ET.SubElement(text, "paragraph")
        paragraph.text = f"A análise por inteligência artificial identificou: {analysis_results['predicted_class']} " \
                        f"com confiança de {analysis_results['confidence']:.1%}"
        
        # Entries estruturadas
        for class_name, prob in analysis_results['predictions'].items():
            entry = ET.SubElement(section, "entry")
            observation = ET.SubElement(entry, "observation", classCode="OBS", moodCode="EVN")
            
            obs_template = ET.SubElement(observation, "templateId")
            obs_template.set("root", "2.16.840.1.113883.10.20.6.2.8")
            
            obs_id = ET.SubElement(observation, "id")
            obs_id.set("root", generate_uid())
            
            obs_code = ET.SubElement(observation, "code")
            obs_code.set("code", "59776-5")
            obs_code.set("codeSystem", "2.16.840.1.113883.6.1")
            
            obs_status = ET.SubElement(observation, "statusCode")
            obs_status.set("code", "completed")
            
            obs_value = ET.SubElement(observation, "value")
            obs_value.set("{http://www.w3.org/2001/XMLSchema-instance}type", "ST")
            obs_value.text = f"{class_name}: {prob:.1%}"
        
        # Salvar documento
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding="UTF-8", xml_declaration=True)
        
        logger.info(f"Documento CDA salvo em: {output_path}")
    
    def send_to_integration_server(self,
                                 message: str,
                                 server_url: str,
                                 message_type: str = "HL7") -> bool:
        """
        Envia mensagem para servidor de integração
        
        Args:
            message: Mensagem a enviar (HL7, FHIR, etc)
            server_url: URL do servidor
            message_type: Tipo de mensagem
            
        Returns:
            True se enviado com sucesso
        """
        try:
            headers = {
                'Content-Type': 'application/json' if message_type == 'FHIR' else 'text/plain',
                'Accept': 'application/json'
            }
            
            if message_type == 'HL7':
                # HL7 via MLLP seria ideal, mas usando HTTP para simplicidade
                data = message.encode('utf-8')
            else:
                data = message if isinstance(message, str) else json.dumps(message)
            
            response = requests.post(
                server_url,
                data=data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Mensagem {message_type} enviada com sucesso")
                return True
            else:
                logger.error(f"Erro ao enviar mensagem: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao enviar para servidor: {str(e)}")
            return False
