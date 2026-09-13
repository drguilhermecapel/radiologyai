"""Hierarquia de exceções do RadiologyAI.

Princípio de projeto (ROADMAP §4.2): *nenhum caminho de degradação silenciosa*.
O sistema legado devolvia um modelo dummy quando não havia pesos e caía numa
heurística de OpenCV. Aqui, cada uma dessas situações levanta uma exceção
específica e o processamento para.
"""

from __future__ import annotations


class RadiologyAIError(Exception):
    """Base de todas as exceções do pacote."""


# --- ingestão -------------------------------------------------------------


class IngestionError(RadiologyAIError):
    """Falha ao ler ou interpretar um arquivo de entrada."""


class InvalidDICOMError(IngestionError):
    """Arquivo não é DICOM válido ou está corrompido."""


class MissingTagError(IngestionError):
    """Tag DICOM obrigatória ausente para a operação pedida."""


# --- escopo / uso pretendido ----------------------------------------------


class OutOfScopeError(RadiologyAIError):
    """Entrada fora do uso pretendido declarado (REG-01 §2, perigo H-03).

    Levantada quando modalidade, incidência, idade ou qualidade da imagem caem
    fora do que a versão em uso foi validada para processar. É um controle de
    risco, não uma conveniência: o sistema recusa, nunca degrada.
    """


# --- modelos --------------------------------------------------------------


class ModelError(RadiologyAIError):
    """Base para problemas de modelo."""


class ModelNotFoundError(ModelError):
    """Model card ou arquivo de pesos inexistente."""


class WeightsIntegrityError(ModelError):
    """sha256 dos pesos não confere com o declarado no model card (perigo H-04)."""


class ModelCardError(ModelError):
    """Model card malformado ou inconsistente."""


# --- inferência -----------------------------------------------------------


class InferenceError(RadiologyAIError):
    """Falha durante a inferência."""


class BackendUnavailableError(InferenceError):
    """Backend de inferência exigido não está instalado.

    Levantada em vez de recorrer a qualquer substituto. Instale o extra
    correspondente (por exemplo ``pip install 'radiologyai[ml]'``).
    """


# --- avaliação ------------------------------------------------------------


class EvaluationError(RadiologyAIError):
    """Falha na avaliação de desempenho."""


class PatientLeakageError(EvaluationError):
    """Interseção de patient_id entre splits (ROADMAP §7, governança de dados).

    Vazamento por paciente é a forma mais comum de um número publicado de IA
    radiológica estar errado. É erro fatal, nunca aviso.
    """
