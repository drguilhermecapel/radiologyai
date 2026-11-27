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


class UnsupportedInputError(IngestionError):
    """Entrada decodificável, mas fora do que o sistema sabe tratar sem adivinhar.

    Exemplos: série com espaçamento entre cortes irregular, diretório com mais
    de uma série sem escolha explícita, multiframe enhanced sem geometria.
    """


# --- escopo / uso pretendido ----------------------------------------------


class OutOfScopeError(RadiologyAIError):
    """Entrada fora do uso pretendido declarado (REG-01 §2, perigo H-03).

    Levantada quando modalidade, incidência, idade ou qualidade da imagem caem
    fora do que a versão em uso foi validada para processar. É um controle de
    risco, não uma conveniência: o sistema recusa, nunca degrada.
    """


class ExamTypeUnknownError(OutOfScopeError):
    """Tipo de exame não pôde ser estabelecido pelos metadados; um humano deve escolher."""


# --- configuração ---------------------------------------------------------


class ConfigurationError(RadiologyAIError):
    """Configuração, roteamento ou armazenamento de artefatos inválido."""


class ArtifactStoreError(ConfigurationError):
    """Artefato ausente, corrompido ou referência que escapa do repositório."""


# --- modelos --------------------------------------------------------------


class ModelError(RadiologyAIError):
    """Base para problemas de modelo."""


class ModelNotFoundError(ModelError):
    """Model card ou arquivo de pesos inexistente."""


class WeightsIntegrityError(ModelError):
    """sha256 dos pesos não confere com o declarado no model card (perigo H-04)."""


class ModelCardError(ModelError):
    """Model card malformado ou inconsistente."""


class ManifestError(ModelCardError):
    """Manifesto de modelo (``radiologyai.analysis``) malformado ou que viola regra do registro."""


class ModelUnavailableError(ModelError):
    """Modelo não pode ser usado: planejado, pesos ausentes ou integridade reprovada.

    Nunca deve ser engolida e substituída por um resultado. A API a mapeia para
    HTTP 503; o pipeline a registra em ``ModelRun.status == "unavailable"``.
    """

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(f"modelo {model_id!r} indisponível: {reason}")
        self.model_id = model_id
        self.reason = reason


class IntegrityCheckError(ModelUnavailableError, WeightsIntegrityError):
    """Arquivo de pesos (ou membro de arquivo compactado) não confere com o sha256 fixado."""


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


# --- catálogo e evidência -------------------------------------------------


class CatalogError(RadiologyAIError):
    """Catálogo de condições, protocolo ou níveis ausente, malformado ou inconsistente."""


class EvidenceError(RadiologyAIError):
    """Registros de evidência inconsistentes para o cálculo de nível.

    O nível de evidência é calculado, nunca digitado. Quando os registros não
    permitem um cálculo bem definido, o sistema para em vez de escolher um nível.
    """


class RegressionGateError(RadiologyAIError, ValueError):
    """Entrada inválida para o portão de regressão de métricas (arquivo ou parâmetro).

    Herda de ``ValueError`` para manter compatíveis os chamadores que já a capturam.
    """
