"""
Endpoints de Inferencia Visual Multinivel: Fotografía Móvil en Terreno y Ortomosaicos de Dron (/api/v1/inference).

Aplica la Regla de Oro:
- Lógica interna, modelos ORM (`DeteccionFitosanitariaORM`, `TrabajoOrtomosaicoORM`),
  estados (`PENDIENTE`, `PROCESANDO`, `COMPLETADO`) y mensajes al usuario en español.
- Rutas y esquemas del contrato HTTP/JSON en inglés.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.core.config import configuracion
from backend.core.database import (
    DeteccionFitosanitariaORM,
    TrabajoOrtomosaicoORM,
    obtener_sesion_bd,
)
from backend.core.storage import sanitizar_identificador_ruta, servicio_almacenamiento
from backend.schemas.api_schemas import (
    RespuestaPrediccionCampo,
    RespuestaTrabajoOrtomosaico,
)
from backend.services.inference_service import inference_service
from backend.workers.celery_app import (
    ejecutar_analisis_ortomosaico,
    procesar_ortomosaico_tarea,
)

router = APIRouter(prefix="/inference", tags=["Inferencia IA Multinivel (Móvil & Dron)"])

TIPOS_IMAGEN_PERMITIDOS = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
    "application/octet-stream",
}


def validar_firma_magic_bytes_imagen(datos: bytes, permitir_tiff: bool = False) -> tuple[str, str]:
    """Verifica las firmas binarias reales (Magic Bytes) para impedir subida de archivos falsificados (PortSwigger File Upload)."""
    if len(datos) < 12:
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido está vacío o truncado.",
        )
    if datos.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if datos.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if permitir_tiff and datos.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif", "image/tiff"
    raise HTTPException(
        status_code=400,
        detail="Firma binaria de archivo no válida. Solo se permiten imágenes genuinas JPEG, PNG o GeoTIFF.",
    )


def _validar_parametros_entrada_inferencia(lot_id: str, latitude: float, longitude: float) -> str:
    try:
        lote_seguro = sanitizar_identificador_ruta(lot_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not (-90.0 <= latitude <= 90.0) or not (-180.0 <= longitude <= 180.0):
        raise HTTPException(
            status_code=400,
            detail="Coordenadas geográficas fuera de rango válido EPSG:4326.",
        )
    return lote_seguro


@router.post("/online", response_model=RespuestaPrediccionCampo)
@router.post("/mobile", response_model=RespuestaPrediccionCampo)
async def predecir_imagen_movil_campo(
    file: UploadFile = File(
        ..., description="Fotografía foliar o de plaga capturada en terreno (JPEG/PNG)"
    ),
    lot_id: str = Form(default="LOTE-ESPINAL-01"),
    latitude: float = Form(default=4.1486),
    longitude: float = Form(default=-74.8830),
    domain: str = Form(default="dron"),
    sesion: Session = Depends(obtener_sesion_bd),
):
    lote_seguro = _validar_parametros_entrada_inferencia(lot_id, latitude, longitude)

    if file.content_type and file.content_type.lower() not in TIPOS_IMAGEN_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail="Formato MIME no soportado. Envíe una imagen JPEG o PNG válida.",
        )

    limite_bytes = configuracion.max_upload_image_mb * 1024 * 1024
    bytes_crudos = await file.read(limite_bytes + 1)
    if len(bytes_crudos) > limite_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"La imagen supera el tamaño máximo permitido ({configuracion.max_upload_image_mb} MB).",
        )

    ext_segura, mime_detectado = validar_firma_magic_bytes_imagen(bytes_crudos, permitir_tiff=False)

    try:
        resultado = inference_service.predict(
            bytes_crudos,
            lot_id=lote_seguro,
            latitude=latitude,
            longitude=longitude,
            source_type="MOVIL_EN_LINEA",
            domain=domain,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="No fue posible decodificar ni procesar la imagen suministrada.",
        ) from exc

    clave_objeto = f"movil/{lote_seguro}/{uuid.uuid4().hex[:12]}_captura{ext_segura}"
    servicio_almacenamiento.guardar_bytes(
        clave_objeto, bytes_crudos, tipo_contenido=mime_detectado
    )

    deteccion = DeteccionFitosanitariaORM(
        id_registro_cliente=f"EN-LINEA-{uuid.uuid4().hex[:10]}",
        lote_id=lote_seguro,
        tipo_fuente="MOVIL_EN_LINEA",
        etiqueta_enfermedad=resultado["disease_label"],
        confianza=resultado["confidence"],
        indice_severidad=resultado["severity_index"],
        latitud=latitude,
        longitud=longitude,
        punto_wkt=f"POINT({longitude} {latitude})",
        version_modelo=resultado["model_version"],
    )
    sesion.add(deteccion)
    sesion.commit()

    return RespuestaPrediccionCampo(**resultado)


@router.post("/orthomosaic", response_model=RespuestaTrabajoOrtomosaico)
async def enviar_ortomosaico_dron(
    file: UploadFile = File(..., description="Ortomosaico GeoTIFF o captura aérea de dron (VANT)"),
    lot_id: str = Form(default="LOTE-ESPINAL-01"),
    sesion: Session = Depends(obtener_sesion_bd),
):
    lote_seguro = _validar_parametros_entrada_inferencia(lot_id, 4.1486, -74.8830)

    limite_bytes = configuracion.max_upload_orthomosaic_mb * 1024 * 1024
    bytes_crudos = await file.read(limite_bytes + 1)
    if not bytes_crudos:
        raise HTTPException(status_code=400, detail="El archivo de ortomosaico está vacío.")
    if len(bytes_crudos) > limite_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"El ortomosaico supera el límite máximo permitido ({configuracion.max_upload_orthomosaic_mb} MB).",
        )

    ext_segura, mime_detectado = validar_firma_magic_bytes_imagen(bytes_crudos, permitir_tiff=True)

    id_tarea = f"ORTO-{uuid.uuid4().hex[:12].upper()}"
    clave_objeto = f"ortomosaicos/{lote_seguro}/{id_tarea}{ext_segura}"
    objeto_guardado = servicio_almacenamiento.guardar_bytes(
        clave_objeto, bytes_crudos, tipo_contenido=mime_detectado
    )

    trabajo = TrabajoOrtomosaicoORM(
        id_tarea=id_tarea,
        lote_id=lote_seguro,
        estado="PENDIENTE",
        clave_objeto_almacenamiento=clave_objeto,
    )
    sesion.add(trabajo)
    sesion.commit()

    if configuracion.celery_eager_fallback:
        ejecutar_analisis_ortomosaico(id_tarea, lote_seguro, objeto_guardado["local_path"])
    else:
        try:
            procesar_ortomosaico_tarea.delay(id_tarea, lote_seguro, objeto_guardado["local_path"])
        except Exception:
            ejecutar_analisis_ortomosaico(id_tarea, lote_seguro, objeto_guardado["local_path"])

    sesion.refresh(trabajo)
    return RespuestaTrabajoOrtomosaico(
        task_id=trabajo.id_tarea,
        lot_id=trabajo.lote_id,
        status=trabajo.estado,
        total_patches=trabajo.total_parches,
        infested_patches=trabajo.parches_infestados,
        infection_rate=trabajo.tasa_infestacion,
        heatmap_geojson_url=f"/api/v1/geospatial/heatmap/{lote_seguro}",
        message="Ortomosaico procesado y capa de calor actualizada exitosamente.",
    )


@router.get("/tasks/{task_id}", response_model=RespuestaTrabajoOrtomosaico)
def obtener_estado_trabajo_ortomosaico(
    task_id: str, sesion: Session = Depends(obtener_sesion_bd)
):
    trabajo = (
        sesion.query(TrabajoOrtomosaicoORM)
        .filter(TrabajoOrtomosaicoORM.id_tarea == task_id)
        .first()
    )
    if trabajo is None:
        raise HTTPException(status_code=404, detail=f"No existe la tarea '{task_id}'")
    return RespuestaTrabajoOrtomosaico(
        task_id=trabajo.id_tarea,
        lot_id=trabajo.lote_id,
        status=trabajo.estado,
        total_patches=trabajo.total_parches,
        infested_patches=trabajo.parches_infestados,
        infection_rate=trabajo.tasa_infestacion,
        heatmap_geojson_url=f"/api/v1/geospatial/heatmap/{trabajo.lote_id}",
        message=f"Estado de procesamiento: {trabajo.estado}",
    )
