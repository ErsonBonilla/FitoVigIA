"""
Servicio de Almacenamiento de Objetos Empresarial (MinIO S3 + Respaldo Local)
con Protección Anti-Path Traversal (OWASP Top 10 A01 / ASVS V12.3 / PortSwigger).

Aplica la Regla de Oro:
- Clase, métodos y variables internas en español (`ServicioAlmacenamientoObjetos`,
  `intentar_conectar_minio`, `guardar_bytes`, `validar_y_resolver_ruta_segura`).
- Interfaz del SDK `minio` (`Minio`, `bucket_exists`, `make_bucket`, `put_object`) en inglés.
"""

from __future__ import annotations

import io
import re
import socket
from pathlib import Path

from backend.core.config import configuracion

PATRON_SEGMENTO_RUTA = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def sanitizar_identificador_ruta(valor: str) -> str:
    """Valida que un identificador (ej. `lot_id`) no contenga secuencias de escape de directorio."""
    limpio = valor.strip()
    if (
        not limpio
        or ".." in limpio
        or "/" in limpio
        or "\\" in limpio
        or "\x00" in limpio
        or not PATRON_SEGMENTO_RUTA.match(limpio)
    ):
        raise ValueError(
            f"Identificador de ruta inválido o intento de Path Traversal bloqueado: '{valor}'"
        )
    return limpio


class ServicioAlmacenamientoObjetos:
    def __init__(self) -> None:
        self.raiz_local: Path = configuracion.storage_local_dir.resolve()
        self.raiz_local.mkdir(parents=True, exist_ok=True)
        self.deposito = configuracion.minio_bucket
        self.cliente_minio = None
        self.minio_activo = False

    def validar_y_resolver_ruta_segura(self, clave_objeto: str) -> tuple[str, Path]:
        """Garantiza que `clave_objeto` permanezca estrictamente confinada dentro de `self.raiz_local`."""
        if not clave_objeto or "\\" in clave_objeto or "\x00" in clave_objeto or ".." in clave_objeto:
            raise ValueError("Clave de objeto rechazada por política Anti-Path Traversal (ASVS V12.3).")

        partes = [p for p in clave_objeto.strip("/").split("/") if p]
        if not partes or len(partes) > 6:
            raise ValueError("Estructura de clave de objeto inválida.")

        for segmento in partes:
            if segmento in {".", ".."} or not PATRON_SEGMENTO_RUTA.match(segmento):
                raise ValueError(
                    f"Segmento de ruta no permitido ('{segmento}') por política Anti-Path Traversal."
                )

        clave_normalizada = "/".join(partes)
        ruta_destino = (self.raiz_local / clave_normalizada).resolve()
        if not ruta_destino.is_relative_to(self.raiz_local):
            raise ValueError("Escape de directorio detectado fuera de storage_local.")

        return clave_normalizada, ruta_destino

    def intentar_conectar_minio(self) -> bool:
        try:
            partes = configuracion.minio_endpoint.split(":")
            host = partes[0]
            puerto = int(partes[1]) if len(partes) > 1 else 9000
            with socket.create_connection((host, puerto), timeout=0.4):
                pass

            from minio import Minio

            cliente = Minio(
                configuracion.minio_endpoint,
                access_key=configuracion.minio_access_key,
                secret_key=configuracion.minio_secret_key,
                secure=configuracion.minio_secure,
            )
            if not cliente.bucket_exists(self.deposito):
                cliente.make_bucket(self.deposito)
            self.cliente_minio = cliente
            self.minio_activo = True
            return True
        except Exception:
            self.minio_activo = False
            return False

    def guardar_bytes(
        self,
        clave_objeto: str,
        datos: bytes,
        tipo_contenido: str = "application/octet-stream",
    ) -> dict:
        """Guarda un objeto en MinIO (si está activo) y mantiene réplica confinada en `storage_local/`."""
        clave_segura, ruta_local = self.validar_y_resolver_ruta_segura(clave_objeto)
        ruta_local.parent.mkdir(parents=True, exist_ok=True)
        ruta_local.write_bytes(datos)

        motor_almacenamiento = "sistema_archivos_local"
        if self.minio_activo and self.cliente_minio is not None:
            try:
                self.cliente_minio.put_object(
                    self.deposito,
                    clave_segura,
                    io.BytesIO(datos),
                    length=len(datos),
                    content_type=tipo_contenido,
                )
                motor_almacenamiento = f"minio://{self.deposito}/{clave_segura}"
            except Exception:
                motor_almacenamiento = "sistema_archivos_local"

        return {
            "object_key": clave_segura,
            "local_path": str(ruta_local),
            "storage_backend": motor_almacenamiento,
            "size_bytes": len(datos),
        }

    # Propiedades y alias de compatibilidad
    @property
    def minio_active(self) -> bool:
        return self.minio_activo

    try_connect_minio = intentar_conectar_minio
    save_bytes = guardar_bytes


servicio_almacenamiento = ServicioAlmacenamientoObjetos()
ObjectStorageService = ServicioAlmacenamientoObjetos
storage_service = servicio_almacenamiento
