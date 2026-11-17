"""
Configuración tipada del Backend Empresarial de FitoVigIA (Pydantic Settings).

Aplica la Regla de Oro:
- Clase e instancias internas en español (`ConfiguracionEmpresarial`, `configuracion`).
- Interfaz exigida por `pydantic_settings` (`BaseSettings`, `SettingsConfigDict`, `model_config`)
  y nombres de variables de entorno mapeadas se conservan según contrato externo.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from engine.config import RAIZ_PROYECTO


class ConfiguracionEmpresarial(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FITOVIGIA_", extra="ignore")

    env: str = "produccion"
    app_name: str = "FitoVigIA Empresarial - Vigilancia Fitosanitaria del Tolima"
    version: str = "1.0.0"

    secret_key: str = "fitovigia-tolima-clave-secreta-empresarial-2026-owasp-asvs-256bit"
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "fitovigia-enterprise"
    jwt_audience: str = "fitovigia-clients"
    access_token_expire_minutes: int = 480

    # Límites de seguridad OWASP ASVS / PortSwigger
    cors_allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://10.0.2.2:8000",
    ]
    max_upload_image_mb: int = 15
    max_upload_orthomosaic_mb: int = 100
    auth_rate_limit_per_minute: int = 15

    # Base de datos principal: PostgreSQL + PostGIS
    database_url: str = os.environ.get(
        "FITOVIGIA_DATABASE_URL",
        "postgresql+psycopg2://fitovigia:fitovigia_secret@localhost:5432/fitovigia_geo",
    )
    # Base de datos conveniente de respaldo local: SQLite
    sqlite_fallback_url: str = os.environ.get(
        "FITOVIGIA_SQLITE_FALLBACK_URL",
        f"sqlite:///{(RAIZ_PROYECTO / 'runtime' / 'fitovigia_enterprise.db').as_posix()}",
    )

    redis_url: str = os.environ.get("FITOVIGIA_REDIS_URL", "redis://localhost:6379/0")
    celery_eager_fallback: bool = (
        os.environ.get("FITOVIGIA_CELERY_EAGER_FALLBACK", "true").lower() == "true"
    )

    minio_endpoint: str = os.environ.get("FITOVIGIA_MINIO_ENDPOINT", "localhost:9000")
    minio_access_key: str = os.environ.get("FITOVIGIA_MINIO_ACCESS_KEY", "fitovigia_admin")
    minio_secret_key: str = os.environ.get("FITOVIGIA_MINIO_SECRET_KEY", "fitovigia_minio_secret")
    minio_bucket: str = os.environ.get("FITOVIGIA_MINIO_BUCKET", "fitovigia-tolima-orthomosaics")
    minio_secure: bool = False

    storage_local_dir: Path = RAIZ_PROYECTO / "runtime" / "storage_local"


configuracion = ConfiguracionEmpresarial()
EnterpriseSettings = ConfiguracionEmpresarial
settings = configuracion
