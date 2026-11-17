"""
Configuración centralizada del motor de IA y Geoprocesamiento de FitoVigIA Enterprise.

Aplica la Regla de Oro:
- Todo lo interno y todo lo visible al usuario final está en español.
- Todo contrato externo, protocolo o interfaz con librerías externas se conserva en inglés.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del Componente Backend y del Monorepo
# ---------------------------------------------------------------------------
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]  # backend/
RAIZ_MONOREPO = RAIZ_PROYECTO.parent  # FitoVigIA/
PROJECT_ROOT = RAIZ_PROYECTO

RAIZ_DATASET = Path(os.environ.get("FITOVIGIA_DATASET_ROOT", str(RAIZ_PROYECTO / "dataset")))
DATASET_ROOT = RAIZ_DATASET

RAIZ_DATASET_CURADO = RAIZ_DATASET
CURATED_DATASET_ROOT = RAIZ_DATASET_CURADO

DIR_CURADO_DRON = RAIZ_DATASET / "01_dron_dosel_cls"
CURATED_DRONE_DIR = DIR_CURADO_DRON
DIR_DATASET_DRON = DIR_CURADO_DRON
DRONE_DATASET_DIR = DIR_CURADO_DRON

DIR_CURADO_CAMPO_HOJA = RAIZ_DATASET / "02_campo_hoja_cls"
CURATED_FIELD_LEAF_DIR = DIR_CURADO_CAMPO_HOJA
DIR_PADDY_DOCTOR = DIR_CURADO_CAMPO_HOJA
PADDY_DOCTOR_DIR = DIR_CURADO_CAMPO_HOJA
DIR_HOJA_ARROZ = DIR_CURADO_CAMPO_HOJA
RICE_LEAF_DIR = DIR_CURADO_CAMPO_HOJA

DIR_CURADO_PLAGAS_YOLO = RAIZ_DATASET / "03_deteccion_plagas_yolo"
CURATED_PEST_YOLO_DIR = DIR_CURADO_PLAGAS_YOLO
DIR_DATASET_PLAGAS = DIR_CURADO_PLAGAS_YOLO / "clasificacion_9_plagas"
PEST_DATASET_DIR = DIR_DATASET_PLAGAS

# Dominio 4 (Nuevo): Repositorio para Capturas Reales de Campo en el Tolima (Semillero ISIDEAD)
DIR_TOLIMA_REAL_ISIDEAD = RAIZ_DATASET / "04_tolima_real_isidead"
TOLIMA_REAL_DATASET_DIR = DIR_TOLIMA_REAL_ISIDEAD
FRENTES_ACTA1_ISIDEAD = [
    "01_plagas_y_sintomas",
    "02_estado_planta_dosel",
    "03_estado_suelo",
]

# Gobernanza de Origen de Datos (Distinción estricta entre Pre-entrenamiento Público vs. Campo Real Tolima)
ORIGEN_PREENTRENAMIENTO_PUBLICO = "PREENTRENAMIENTO_REPOSITORIO_ABIERTO_INTERNET"
ORIGEN_CAMPO_TOLIMA_REAL = "RECOLECCION_LOCAL_TOLIMA_SEMILLERO_ISIDEAD"

ORIGENES_DATOS_FITOVIGIA: dict[str, dict] = {
    ORIGEN_PREENTRENAMIENTO_PUBLICO: {
        "codigo": ORIGEN_PREENTRENAMIENTO_PUBLICO,
        "etiqueta_ui": "Pre-entrenamiento Base (Repositorios Libres de Internet)",
        "descripcion": (
            "Imágenes descargadas de repositorios abiertos internacionales (Kaggle Rice Leaf, "
            "Paddy Doctor, IP102 / RP11). Utilizadas exclusivamente para Transfer Learning "
            "inicial y pruebas de arquitectura; no representan mediciones reales del Tolima."
        ),
        "validado_en_campo_tolima": False,
    },
    ORIGEN_CAMPO_TOLIMA_REAL: {
        "codigo": ORIGEN_CAMPO_TOLIMA_REAL,
        "etiqueta_ui": "Datos Locales Reales – Tolima (Semillero ISIDEAD)",
        "descripcion": (
            "Fotografías y vuelos de drones (UAV) capturados en lotes arroceros de Ibagué "
            "(Meseta), Saldaña y El Espinal bajo el protocolo DEXPLOS y las Actas N.° 1 y N.° 3 "
            "de 2026 del Semillero ISIDEAD."
        ),
        "validado_en_campo_tolima": True,
    },
}

# Parámetros Fotogramétricos UAV y Espectrales fundamentados en Literatura Científica
# (Rojas et al., 2018; Lin et al., 2025; Qiang et al., 2025; Ding et al., 2025; Li et al., 2025; Gong et al., 2021; RAC 100)
PARAMETROS_UAV_LITERATURA: dict[str, float | tuple[float, float]] = {
    "altura_vuelo_min_m": 15.0,
    "altura_vuelo_max_m": 40.0,
    "altura_vuelo_optima_m": 25.0,
    "gsd_min_cm_px": 0.35,
    "gsd_max_cm_px": 1.80,
    "traslape_longitudinal_min_pct": 70.0,
    "traslape_longitudinal_max_pct": 85.0,
    "traslape_transversal_min_pct": 70.0,
    "traslape_transversal_max_pct": 85.0,
    "umbral_exg_vegetacion_arroz": 0.08,
    "umbral_vari_estres_clorotico": -0.02,
}

# Separación Taxonómica RP11 (Ding et al., 2025) vs. Síntomas Fitopatológicos (Lin et al., 2025; Guzmán Brand y Gélvez, 2025)
TAXONOMIA_RP11_TOLIMA: dict[str, dict] = {
    "insectos_adultos_y_larvas_yolo": [
        {"codigo": "sogatodes_orizicola", "nombre": "Sogata del arroz (Sogatodes orizicola)", "estadio": "Adulto / Ninfa", "frente_acta1": "01_plagas_y_sintomas"},
        {"codigo": "diatraea_saccharalis", "nombre": "Barrenador del tallo (Diatraea saccharalis)", "estadio": "Adulto / Larva", "frente_acta1": "01_plagas_y_sintomas"},
        {"codigo": "rupela_albinella", "nombre": "Novia del arroz (Rupela albinella)", "estadio": "Adulto", "frente_acta1": "01_plagas_y_sintomas"},
        {"codigo": "spodoptera_frugiperda", "nombre": "Gusano cogollero (Spodoptera frugiperda)", "estadio": "Larva / Adulto", "frente_acta1": "01_plagas_y_sintomas"},
        {"codigo": "cnaphalocrocis_medinalis", "nombre": "Enrollador de la hoja (Cnaphalocrocis medinalis)", "estadio": "Adulto / Larva", "frente_acta1": "01_plagas_y_sintomas"},
    ],
    "sintomas_planta_y_dosel_cnn": [
        {"codigo": "piricularia", "nombre": "Piricularia foliar y de panícula (Pyricularia oryzae)", "frente_acta1": "02_estado_planta_dosel"},
        {"codigo": "mancha_parda", "nombre": "Mancha parda / Helmintosporiosis (Bipolaris oryzae)", "frente_acta1": "02_estado_planta_dosel"},
        {"codigo": "anublo_bacteriano_panicula", "nombre": "Añublo bacteriano de la panícula (Burkholderia glumae)", "frente_acta1": "02_estado_planta_dosel"},
        {"codigo": "virosis_tungro", "nombre": "Complejo viral Hoja Blanca / Tungro", "frente_acta1": "02_estado_planta_dosel"},
    ],
    "cobertura_y_suelo": [
        {"codigo": "sano", "nombre": "Dosel de arroz con vigor fotosintético normal (ExG/VARI óptimo)", "frente_acta1": "02_estado_planta_dosel"},
        {"codigo": "suelo_desnudo", "nombre": "Suelo desnudo / Lámina de agua / Pérdida de densidad de siembra", "frente_acta1": "03_estado_suelo"},
    ],
}

RUTA_DATA_YAML_YOLO = DIR_CURADO_PLAGAS_YOLO / "data.yaml"
YOLO_DATA_YAML_PATH = RUTA_DATA_YAML_YOLO

RUTA_REPORTE_CURADURIA = RAIZ_DATASET / "reporte_curaduria.json"
CURATION_REPORT_PATH = RUTA_REPORTE_CURADURIA

DIR_MODELOS = Path(os.environ.get("FITOVIGIA_MODELS_DIR", str(RAIZ_PROYECTO / "models")))
MODELS_DIR = DIR_MODELOS
DIR_EJECUCION = RAIZ_PROYECTO / "runtime"
DIR_ARTEFACTOS = Path(
    os.environ.get("FITOVIGIA_ARTIFACTS_DIR", str(DIR_EJECUCION / "artifacts"))
)
ARTIFACTS_DIR = DIR_ARTEFACTOS
DIR_ALMACENAMIENTO_LOCAL = Path(
    os.environ.get("FITOVIGIA_STORAGE_LOCAL_DIR", str(DIR_EJECUCION / "storage_local"))
)
STORAGE_LOCAL_DIR = DIR_ALMACENAMIENTO_LOCAL
DIR_ASSETS_MOVIL = RAIZ_MONOREPO / "mobile" / "assets" / "models"
MOBILE_ASSETS_DIR = DIR_ASSETS_MOVIL

DIR_MANIFIESTOS = RAIZ_PROYECTO / "engine" / "data"
MANIFEST_DIR = DIR_MANIFIESTOS
RUTA_MANIFIESTO_MULTINIVEL = DIR_MANIFIESTOS / "manifiesto_multinivel.csv"
FULL_MANIFEST_PATH = RUTA_MANIFIESTO_MULTINIVEL
RUTA_MANIFIESTO_PARTICIONADO = DIR_MANIFIESTOS / "manifiesto_particionado.csv"
SPLIT_MANIFEST_PATH = RUTA_MANIFIESTO_PARTICIONADO

# Artefactos de Modelos Exportados (Dominio 1: Dron 4 clases y Dominio 2: Campo Hoja 10 clases)
RUTA_MODELO_PYTORCH = DIR_MODELOS / "fitovigia_mejor_modelo.pth"
PYTORCH_MODEL_PATH = RUTA_MODELO_PYTORCH
RUTA_MODELO_ONNX_SERVIDOR = DIR_MODELOS / "fitovigia_server_fp32.onnx"
ONNX_SERVER_MODEL_PATH = RUTA_MODELO_ONNX_SERVIDOR
RUTA_MODELO_ONNX_MOVIL = DIR_MODELOS / "fitovigia_mobile_int8.onnx"
ONNX_MOBILE_MODEL_PATH = RUTA_MODELO_ONNX_MOVIL
RUTA_METADATOS_MODELO = DIR_MODELOS / "model_metadata.json"
MODEL_METADATA_PATH = RUTA_METADATOS_MODELO

RUTA_MODELO_CAMPO_PYTORCH = DIR_MODELOS / "fitovigia_campo_hoja_mejor.pth"
RUTA_MODELO_CAMPO_ONNX_SERVIDOR = DIR_MODELOS / "fitovigia_campo_hoja_fp32.onnx"
RUTA_MODELO_CAMPO_ONNX_MOVIL = DIR_MODELOS / "fitovigia_campo_hoja_int8.onnx"
RUTA_METADATOS_CAMPO_MODELO = DIR_MODELOS / "model_field_leaf_metadata.json"

# ---------------------------------------------------------------------------
# Configuración de MLflow Tracking & Model Registry
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.environ.get(
    "FITOVIGIA_MLFLOW_URI", f"sqlite:///{(DIR_EJECUCION / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = os.environ.get("FITOVIGIA_MLFLOW_EXPERIMENT", "FitoVigIA_Tolima_Fitosanitario")
MLFLOW_MODEL_NAME = os.environ.get("FITOVIGIA_MLFLOW_MODEL_NAME", "fitovigia_rice_classifier")
MLFLOW_DEFAULT_STAGE = os.environ.get("FITOVIGIA_MLFLOW_STAGE", "Staging")

# ---------------------------------------------------------------------------
# Parámetros de Hardware (Optimizados para RTX 3050 4 GB VRAM en WSL 2)
# ---------------------------------------------------------------------------
USAR_CUDA = os.environ.get("FITOVIGIA_USE_CUDA", "true").lower() == "true"
USE_CUDA = USAR_CUDA
AMP_FP16 = os.environ.get("FITOVIGIA_AMP_FP16", "true").lower() == "true"
TAMANO_IMAGEN = (224, 224)
IMG_SIZE = TAMANO_IMAGEN
CANALES_IMAGEN = 3
IMG_CHANNELS = CANALES_IMAGEN
TAMANO_LOTE = int(os.environ.get("FITOVIGIA_BATCH_SIZE", "16"))
BATCH_SIZE = TAMANO_LOTE
PASOS_ACUMULACION_GRADIENTE = int(os.environ.get("FITOVIGIA_GRAD_ACCUM_STEPS", "2"))
GRAD_ACCUM_STEPS = PASOS_ACUMULACION_GRADIENTE
MUESTRAS_POR_CLASE = int(os.environ.get("FITOVIGIA_SUBSET_PER_CLASS", "300"))
SUBSET_PER_CLASS = MUESTRAS_POR_CLASE
PROPORCIONES_PARTICION = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_RATIOS = PROPORCIONES_PARTICION
SEMILLA = 42
SEED = SEMILLA

# Estadísticas estándar de ImageNet para Transfer Learning en PyTorch y ONNX
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Taxonomía Fitosanitaria en Español (Regla de Oro)
# ---------------------------------------------------------------------------
CLASES_DRON = [
    "sano",
    "roya_parda",
    "roya_amarilla",
    "suelo_desnudo",
]
DRONE_CLASSES = CLASES_DRON

CLASES_CAMPO_HOJA = [
    "sano",
    "piricularia",
    "anublo_bacteriano_hoja",
    "raya_bacteriana_hoja",
    "anublo_bacteriano_panicula",
    "mancha_parda",
    "virosis_tungro",
    "escaldado_carbon_hoja",
    "dano_barrenador_tallo",
    "dano_insecto_foliar",
]
FIELD_LEAF_CLASSES = CLASES_CAMPO_HOJA

CLASES_MULTINIVEL = [
    "sano",
    "roya_parda",
    "roya_amarilla",
    "anublo_bacteriano_hoja",
    "piricularia",
    "mancha_parda",
    "virosis_tungro",
    "barrenador_tallo",
]
MULTILEVEL_CLASSES = CLASES_MULTINIVEL

# Mapeo de compatibilidad desde etiquetas crudas en inglés hacia etiquetas internas en español
MAPEO_ETIQUETAS_A_ESPANOL: dict[str, str] = {
    "healthy": "sano",
    "normal": "sano",
    "brown_rust": "roya_parda",
    "yellow_rust": "roya_amarilla",
    "ground": "suelo_desnudo",
    "blast": "piricularia",
    "bacterial_leaf_blight": "anublo_bacteriano_hoja",
    "bacterial_leaf_streak": "raya_bacteriana_hoja",
    "bacterial_panicle_blight": "anublo_bacteriano_panicula",
    "brown_spot": "mancha_parda",
    "tungro": "virosis_tungro",
    "leaf_scald_smut": "escaldado_carbon_hoja",
    "stem_borer_damage": "dano_barrenador_tallo",
    "stem_borer": "barrenador_tallo",
    "foliar_insect_damage": "dano_insecto_foliar",
    "unknown": "desconocido_no_arroz",
    "out_of_domain": "desconocido_no_arroz",
}


def normalizar_etiqueta_espanol(etiqueta: str) -> str:
    """Garantiza que cualquier etiqueta fitosanitaria se resuelva en español."""
    limpia = etiqueta.strip().lower()
    return MAPEO_ETIQUETAS_A_ESPANOL.get(limpia, limpia)


CATALOGO_AGRONOMICO: dict[str, dict] = {
    "sano": {
        "nombre_comun": "Cultivo Saludable",
        "agente_causal": "N/A (Vigor fotosintético normal)",
        "nivel_riesgo": "BAJO",
        "severidad_base": 0.05,
        "recomendacion_mip": "Mantener monitoreo periódico de lámina de agua y nutrición balanceada (N-P-K).",
    },
    "roya_parda": {
        "nombre_comun": "Roya Parda Foliar",
        "agente_causal": "Puccinia spp. / Complejo fúngico foliar",
        "nivel_riesgo": "ALTO",
        "severidad_base": 0.78,
        "recomendacion_mip": "Inspeccionar foco georreferenciado; evaluar aplicación localizada de triazoles/estrobilurinas bajo aval agronómico.",
    },
    "roya_amarilla": {
        "nombre_comun": "Roya Amarilla / Clorosis Infecciosa",
        "agente_causal": "Puccinia striiformis / Patógeno foliar",
        "nivel_riesgo": "ALTO",
        "severidad_base": 0.82,
        "recomendacion_mip": "Verificar drenaje del lote, descartar deficiencia de nitrógeno y aplicar control fúngico focalizado en parche.",
    },
    "suelo_desnudo": {
        "nombre_comun": "Suelo Desnudo / Pérdida de Densidad",
        "agente_causal": "Falla de germinación, daño por aves o vuelco",
        "nivel_riesgo": "MEDIO",
        "severidad_base": 0.35,
        "recomendacion_mip": "Revisar nivelación láser del lote, lámina de riego y densidad de siembra en la zona descubierta.",
    },
    "anublo_bacteriano_hoja": {
        "nombre_comun": "Añublo Bacteriano del Arroz",
        "agente_causal": "Xanthomonas oryzae pv. oryzae",
        "nivel_riesgo": "CRITICO",
        "severidad_base": 0.90,
        "recomendacion_mip": "Evitar exceso de fertilización nitrogenada, desinfectar herramientas y aislar el foco de riego para frenar dispersión.",
    },
    "raya_bacteriana_hoja": {
        "nombre_comun": "Raya Bacteriana de la Hoja",
        "agente_causal": "Xanthomonas oryzae pv. oryzicola",
        "nivel_riesgo": "ALTO",
        "severidad_base": 0.76,
        "recomendacion_mip": "Reducir humedad excesiva en el dosel, emplear semilla certificada y monitorear lotes vecinos.",
    },
    "anublo_bacteriano_panicula": {
        "nombre_comun": "Añublo Bacteriano de la Panícula",
        "agente_causal": "Burkholderia glumae",
        "nivel_riesgo": "CRITICO",
        "severidad_base": 0.91,
        "recomendacion_mip": "Vigilancia prioritaria en floración ante altas temperaturas nocturnas; manejo preventivo según guía Fedearroz/ICA.",
    },
    "piricularia": {
        "nombre_comun": "Piricularia / Añublo del Arroz",
        "agente_causal": "Pyricularia oryzae (Magnaporthe oryzae)",
        "nivel_riesgo": "CRITICO",
        "severidad_base": 0.95,
        "recomendacion_mip": "Alerta prioritaria: mantener lámina de agua estable y aplicar fungicida específico (ej. Triciclazol/Isoprotiolano) según umbral ICA.",
    },
    "mancha_parda": {
        "nombre_comun": "Mancha Parda (Helmintosporiosis)",
        "agente_causal": "Bipolaris oryzae",
        "nivel_riesgo": "ALTO",
        "severidad_base": 0.75,
        "recomendacion_mip": "Asociado a suelos con deficiencia de potasio o silicio; corregir nutrición edáfica y tratar focos severos.",
    },
    "virosis_tungro": {
        "nombre_comun": "Virosis Complejo Tungro / Hoja Blanca",
        "agente_causal": "Complejo viral transmitido por cicadélidos/delfácidos",
        "nivel_riesgo": "CRITICO",
        "severidad_base": 0.92,
        "recomendacion_mip": "Monitorear población de vectores con red entomológica (10 pases dobles) y eliminar plantas reservorio.",
    },
    "escaldado_carbon_hoja": {
        "nombre_comun": "Escaldado Foliar / Carbón de la Hoja",
        "agente_causal": "Microdochium oryzae / Entyloma oryzae",
        "nivel_riesgo": "MEDIO",
        "severidad_base": 0.65,
        "recomendacion_mip": "Controlar malezas hospederas en caballones y evitar densidades de siembra excesivas.",
    },
    "dano_barrenador_tallo": {
        "nombre_comun": "Daño por Barrenador (Corazón Muerto / Panícula Blanca)",
        "agente_causal": "Diatraea saccharalis / Rupela albinella",
        "nivel_riesgo": "ALTO",
        "severidad_base": 0.85,
        "recomendacion_mip": "Liberación biológica de Trichogramma exiguum o Telenomus rowani y destrucción de socas post-cosecha.",
    },
    "barrenador_tallo": {
        "nombre_comun": "Barrenador del Tallo (Corazón Muerto / Panícula Blanca)",
        "agente_causal": "Diatraea saccharalis / Rupela albinella",
        "nivel_riesgo": "ALTO",
        "severidad_base": 0.85,
        "recomendacion_mip": "Liberación biológica de Trichogramma exiguum o Telenomus rowani y destrucción de socas post-cosecha.",
    },
    "dano_insecto_foliar": {
        "nombre_comun": "Daño por Insectos Defoliadores / Enrolladores",
        "agente_causal": "Cnaphalocrocis medinalis / Dicladispa armigera / Spodoptera",
        "nivel_riesgo": "ALTO",
        "severidad_base": 0.74,
        "recomendacion_mip": "Evaluar porcentaje de área foliar afectada; priorizar control biológico (Bacillus thuringiensis) sobre umbral económico.",
    },
    "desconocido_no_arroz": {
        "nombre_comun": "Desconocido / Fuera de Dominio (No-Arroz)",
        "agente_causal": "Imagen fuera de dominio o baja confianza espectral",
        "nivel_riesgo": "NO_APLICA",
        "severidad_base": 0.0,
        "recomendacion_mip": "La fotografía no corresponde a tejido foliar ni dosel de arroz con nitidez suficiente. Encuadre la hoja dentro de las guías visuales a 15-20 cm con luz natural.",
    },
}
AGRONOMIC_KNOWLEDGE_BASE = CATALOGO_AGRONOMICO

# ---------------------------------------------------------------------------
# Zonas y Distritos Arroceros de Referencia en el Tolima (EPSG:4326)
# ---------------------------------------------------------------------------
ZONAS_ARROCERAS_TOLIMA: dict[str, dict] = {
    "EL_ESPINAL": {
        "municipio": "El Espinal",
        "distrito_riego": "Usocoello",
        "center_lat": 4.1486,
        "center_lon": -74.8830,
        "bbox": (-74.9100, 4.1300, -74.8600, 4.1700),
    },
    "SALDANA": {
        "municipio": "Saldaña",
        "distrito_riego": "Usosaldaña",
        "center_lat": 3.9317,
        "center_lon": -75.0153,
        "bbox": (-75.0400, 3.9100, -74.9900, 3.9550),
    },
    "GUAMO": {
        "municipio": "Guamo",
        "distrito_riego": "Usocoello - Sector Guamo",
        "center_lat": 4.0308,
        "center_lon": -74.9700,
        "bbox": (-74.9950, 4.0100, -74.9450, 4.0550),
    },
    "PURIFICACION": {
        "municipio": "Purificación",
        "distrito_riego": "Prado - Purificación",
        "center_lat": 3.8586,
        "center_lon": -74.9314,
        "bbox": (-74.9550, 3.8350, -74.9050, 3.8800),
    },
    "IBAGUE_MESETA": {
        "municipio": "Ibagué (Meseta)",
        "distrito_riego": "Meseta de Ibagué",
        "center_lat": 4.4125,
        "center_lon": -75.1542,
        "bbox": (-75.1800, 4.3900, -75.1300, 4.4350),
    },
    "LERIDA": {
        "municipio": "Lérida",
        "distrito_riego": "Asorrecio",
        "center_lat": 4.8622,
        "center_lon": -74.9108,
        "bbox": (-74.9350, 4.8400, -74.8850, 4.8850),
    },
}
TOLIMA_RICE_ZONES = ZONAS_ARROCERAS_TOLIMA

AVISO_AGRONOMICO = (
    "AVISO AGRONOMICO: FitoVigIA es una plataforma empresarial de soporte a la decisión para "
    "Manejo Integrado de Plagas (MIP) en arroz. Ninguna alerta automatizada sustituye el "
    "concepto técnico ni la prescripción de un Ingeniero Agrónomo con matrícula profesional "
    "vigente conforme a las regulaciones del ICA."
)
AGRONOMIC_DISCLAIMER = AVISO_AGRONOMICO
