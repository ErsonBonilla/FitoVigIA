"""
Suite de Pruebas Automatizadas E2E de FitoVigIA Enterprise bajo la Regla de Oro (pytest).

Verifica:
1. Regla de Oro en `dataset/`: dominios `01_dron_dosel_cls`, `02_campo_hoja_cls`
   y `03_deteccion_plagas_yolo` con etiquetas y particiones internas en español y protocolo
   Ultralytics YOLO (`images/train`, `labels/train`, `data.yaml`) en inglés.
2. Preprocesamiento radiométrico CLAHE (OpenCV) y tensores Albumentations.
3. Arquitecturas PyTorch (`ClasificadorCNNFitoVigIA`), `DetectorFitosanitarioYOLO` y exportación
   ONNX con etiquetas en español (`sano`, `roya_parda`, `roya_amarilla`, `suelo_desnudo`).
4. Geoprocesamiento de ortomosaicos GeoTIFF (Rasterio) y Mapas de Calor GeoJSON (GeoPandas).
5. Plataforma Backend FastAPI: Salud GPU/ONNX, Auth JWT, Lotes del Tolima, Inferencia Móvil,
   Ortomosaicos de Dron (Celery) y Sincronización Offline-First de la App Flutter.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from engine.architectures import (
    DetectorFitosanitarioYOLO,
    auditar_salud_neuronas,
    construir_clasificador_pytorch,
)
from engine.config import (
    CLASES_CAMPO_HOJA,
    CLASES_DRON,
    DIR_ASSETS_MOVIL,
    DIR_DATASET_DRON,
    DIR_TOLIMA_REAL_ISIDEAD,
    ORIGEN_PREENTRENAMIENTO_PUBLICO,
    RUTA_DATA_YAML_YOLO,
    RUTA_MODELO_CAMPO_ONNX_MOVIL,
    RUTA_MODELO_CAMPO_ONNX_SERVIDOR,
    RUTA_MODELO_CAMPO_PYTORCH,
    RUTA_MODELO_ONNX_MOVIL,
    RUTA_MODELO_ONNX_SERVIDOR,
    RUTA_MODELO_PYTORCH,
)
from engine.data_governance import (
    auditar_fuga_datos,
    cargar_manifiesto_dominio_curado,
    construir_manifiesto_dron,
    crear_subconjunto_balanceado,
    obtener_resumen_curaduria,
    particionar_manifiesto_agrupado,
)
from engine.exporter import asegurar_artefactos_por_defecto
from engine.geospatial import (
    construir_geodataframe_mapa_calor,
    crear_ortomosaico_geotiff_muestra,
    exportar_artefactos_mapa_calor,
    seccionar_ortomosaico_en_parches,
    validar_parametros_vuelo_uav_rac100,
)
from engine.preprocessing import (
    aplicar_correccion_radiometrica_clahe,
    calcular_indices_vegetativos_rgb_uav,
    preprocesar_para_inferencia,
)


def _crear_bytes_jpeg_prueba() -> bytes:
    generador = np.random.RandomState(42)
    img = generador.randint(40, 210, size=(224, 224, 3), dtype=np.uint8)
    ok, codificado = cv2.imencode(".jpg", img)
    assert ok
    return codificado.tobytes()


def test_gobernanza_datos_y_particion_sin_fuga():
    auditoria = auditar_fuga_datos(DIR_DATASET_DRON)
    assert auditoria["total_raw_files"] > 3000
    assert auditoria["duplicate_copy_files_count"] >= 1

    manifiesto = construir_manifiesto_dron(DIR_DATASET_DRON, excluir_copias=True)
    assert not manifiesto["is_duplicate_copy"].any()
    assert set(manifiesto["label_name"].unique()) == set(CLASES_DRON)

    subconjunto = crear_subconjunto_balanceado(manifiesto, por_clase=40)
    df_particionado = particionar_manifiesto_agrupado(subconjunto)
    fuga = (df_particionado.groupby("group_id")["particion"].nunique() > 1).sum()
    assert int(fuga) == 0

    # Validar los 3 dominios especializados bajo la Regla de Oro en dataset/
    curaduria = obtener_resumen_curaduria()
    assert curaduria["dominio_01_dron_dosel"]["grupos_con_fuga"] == 0
    assert curaduria["dominio_01_dron_dosel"]["clases"] == [
        "sano",
        "roya_parda",
        "roya_amarilla",
        "suelo_desnudo",
    ]
    assert curaduria["dominio_02_campo_hoja"]["total_imagenes_curadas"] > 15000
    assert "piricularia" in curaduria["dominio_02_campo_hoja"]["clases"]
    assert curaduria["dominio_03_plagas_yolo"]["total_cajas_delimitadoras_yolo"] == 62227
    assert RUTA_DATA_YAML_YOLO.exists()

    df_hoja = cargar_manifiesto_dominio_curado("campo_hoja")
    assert set(df_hoja["particion"].unique()) == {"entrenamiento", "validacion", "prueba"}
    assert set(df_hoja["split"].unique()) == {"train", "val", "test"}


def test_preprocesamiento_clahe_y_detector_yolo():
    generador = np.random.RandomState(7)
    muestra_rgb = generador.randint(30, 220, size=(224, 224, 3), dtype=np.uint8)
    mejorada = aplicar_correccion_radiometrica_clahe(muestra_rgb)
    assert mejorada.shape == (224, 224, 3)

    tensor_nchw = preprocesar_para_inferencia(mejorada)
    assert tensor_nchw.shape == (1, 3, 224, 224)
    assert tensor_nchw.dtype == np.float32

    detector = DetectorFitosanitarioYOLO()
    cajas = detector.localizar_lesiones(mejorada, predicted_class="roya_parda")
    assert len(cajas) >= 1
    assert cajas[0]["label"] == "roya_parda"
    assert len(cajas[0]["bbox_xyxy"]) == 4

    # Verificar índices vegetativos UAV RGB (ExG y VARI) y validador RAC 100 / GSD
    indices = calcular_indices_vegetativos_rgb_uav(mejorada)
    assert "exg_mean" in indices and "vari_mean" in indices and "cobertura_dosel_pct" in indices
    vuelo = validar_parametros_vuelo_uav_rac100(altitude_m=25.0, overlap_front_pct=80.0, overlap_side_pct=75.0)
    assert vuelo["rac100_vlos_compliant"] is True
    assert vuelo["photogrammetry_valid"] is True
    assert DIR_TOLIMA_REAL_ISIDEAD.exists()


def test_clasificador_pytorch_y_exportacion_onnx_movil():
    modelo_congelado = construir_clasificador_pytorch(
        backbone_name="efficientnet_b0", num_classes=4, fine_tune=False, pretrained=False
    )
    entrenables_congelado = sum(
        p.numel() for p in modelo_congelado.parameters() if p.requires_grad
    )

    modelo_afinado = construir_clasificador_pytorch(
        backbone_name="efficientnet_b0", num_classes=4, fine_tune=True, pretrained=False
    )
    entrenables_afinado = sum(p.numel() for p in modelo_afinado.parameters() if p.requires_grad)
    assert entrenables_afinado > entrenables_congelado

    # Verificar soporte de MobileNetV2 para el borde móvil
    modelo_mobilenet = construir_clasificador_pytorch(
        backbone_name="mobilenet_v2", num_classes=4, fine_tune=True, pretrained=False
    )
    assert sum(p.numel() for p in modelo_mobilenet.parameters() if p.requires_grad) > 0

    meta = asegurar_artefactos_por_defecto()
    assert RUTA_MODELO_ONNX_SERVIDOR.exists()
    assert RUTA_MODELO_ONNX_MOVIL.exists()
    assert (DIR_ASSETS_MOVIL / "labels_tolima.json").exists()
    assert meta["classes"] == ["sano", "roya_parda", "roya_amarilla", "suelo_desnudo"]
    # Verificar compresión cuantizada INT8 real (~4.5 MB vs ~16.5 MB FP32)
    assert meta["mobile_size_mb"] < meta["server_size_mb"] * 0.45
    assert meta["compression_ratio"] >= 3.0


def test_salud_neuronal_y_pesos_entrenados_gpu():
    import torch

    # 1. Verificar modelo entrenado de Dron VANT (4 clases)
    assert RUTA_MODELO_PYTORCH.exists()
    ckpt_dron = torch.load(RUTA_MODELO_PYTORCH, map_location="cpu", weights_only=False)
    assert "run_id" in ckpt_dron
    assert ckpt_dron["metrics"]["f1_macro"] >= 0.80
    assert ckpt_dron["metrics"]["roc_auc_ovr"] >= 0.94

    modelo_dron = construir_clasificador_pytorch(
        backbone_name=ckpt_dron["backbone_name"],
        num_classes=len(CLASES_DRON),
        fine_tune=True,
        pretrained=False,
    )
    modelo_dron.load_state_dict(ckpt_dron["state_dict"])

    df_dron = cargar_manifiesto_dominio_curado("dron")
    muestra_prueba = df_dron[df_dron["particion"] == "prueba"].groupby("label").head(4)
    tensores_x = [
        torch.from_numpy(preprocesar_para_inferencia(p)[0])
        for p in muestra_prueba["path"].tolist()
    ]
    lote_x = torch.stack(tensores_x, dim=0)
    lote_y = torch.tensor(muestra_prueba["label"].tolist(), dtype=torch.long)

    salud_dron = auditar_salud_neuronas(modelo_dron, lote_x, lote_y)
    assert salud_dron["neuronas_saludables"] is True
    assert salud_dron["neuronas_muertas"] == 0
    assert salud_dron["ratio_neuronas_activas"] == 1.0
    assert salud_dron["clases_predichas_unicas"] >= 3
    assert salud_dron["exactitud_lote_evaluado"] >= 0.75

    # 2. Verificar modelo entrenado de Campo Hoja (10 clases)
    assert RUTA_MODELO_CAMPO_PYTORCH.exists()
    assert RUTA_MODELO_CAMPO_ONNX_SERVIDOR.exists()
    assert RUTA_MODELO_CAMPO_ONNX_MOVIL.exists()
    ckpt_hoja = torch.load(RUTA_MODELO_CAMPO_PYTORCH, map_location="cpu", weights_only=False)
    assert ckpt_hoja["metrics"]["roc_auc_ovr"] >= 0.90


def test_ortomosaico_rasterio_y_mapa_calor_geopandas(tmp_path: Path):
    ruta_tif = crear_ortomosaico_geotiff_muestra(
        tmp_path / "espinal_prueba.tif", zone_key="EL_ESPINAL", width=448, height=448
    )
    assert ruta_tif.exists()

    parches = seccionar_ortomosaico_en_parches(ruta_tif, patch_size=224)
    assert len(parches) == 4
    for p in parches:
        p["disease_label"] = "roya_parda"
        p["confidence"] = 0.91

    gdf = construir_geodataframe_mapa_calor(parches, lot_id="LOTE-PRUEBA-01")
    salida = exportar_artefactos_mapa_calor(gdf, lot_id="LOTE-PRUEBA-01", out_dir=tmp_path)
    assert Path(salida["geojson_path"]).exists()
    assert Path(salida["png_path"]).exists()
    assert salida["infection_rate"] == 1.0


def test_endpoints_fastapi_empresarial():
    with TestClient(app) as cliente:
        # 1. Monitoreo de salud
        r_salud = cliente.get("/api/v1/health")
        assert r_salud.status_code == 200
        datos_salud = r_salud.json()
        assert datos_salud["status"] == "ok"
        assert datos_salud["model_loaded"] is True

        # 2. Autenticación JWT (/api/v1/auth/login y /api/v1/auth/token)
        r_auth = cliente.post(
            "/api/v1/auth/login",
            json={"username": "agronomo@fitovigia.co", "password": "FitoVigIA2026*"},
        )
        assert r_auth.status_code == 200
        token = r_auth.json()["access_token"]
        assert token

        # 3. Fincas y Lotes del Tolima
        r_lotes = cliente.get("/api/v1/farms/lots")
        assert r_lotes.status_code == 200
        lotes = r_lotes.json()
        assert len(lotes) >= 2
        assert lotes[0]["id"] == "LOTE-ESPINAL-01"

        # 4. Inferencia en línea (/api/v1/inference/online) y control fuera de dominio
        bytes_jpg = _crear_bytes_jpeg_prueba()
        r_pred = cliente.post(
            "/api/v1/inference/online",
            files={"file": ("hoja_arroz.jpg", io.BytesIO(bytes_jpg), "image/jpeg")},
            data={"lot_id": "LOTE-ESPINAL-01", "latitude": "4.1486", "longitude": "-74.8830"},
        )
        assert r_pred.status_code == 200
        datos_pred = r_pred.json()
        assert datos_pred["disease_label"] in {
            "sano",
            "roya_parda",
            "roya_amarilla",
            "suelo_desnudo",
        }
        assert datos_pred["dataset_origin"] == ORIGEN_PREENTRENAMIENTO_PUBLICO
        assert datos_pred["acta1_study_front"] in {
            "01_plagas_y_sintomas",
            "02_estado_planta_dosel",
            "03_estado_suelo",
        }
        assert "exg_mean" in datos_pred["vegetative_indices"]
        assert "AVISO AGRONOMICO" in datos_pred["disclaimer"]

        # Imagen negra plana -> debe ser rechazada como 'desconocido_no_arroz'
        img_plana = np.zeros((224, 224, 3), dtype=np.uint8)
        _, jpg_plano = cv2.imencode(".jpg", img_plana)
        r_fuera = cliente.post(
            "/api/v1/inference/online",
            files={"file": ("no_arroz.jpg", io.BytesIO(jpg_plano.tobytes()), "image/jpeg")},
            data={"lot_id": "LOTE-ESPINAL-01"},
        )
        assert r_fuera.status_code == 200
        assert r_fuera.json()["disease_label"] == "desconocido_no_arroz"

        # 5. Sincronización asíncrona (/api/v1/sync)
        r_sync = cliente.post(
            "/api/v1/sync",
            json={
                "device_id": "ANDROID-TOLIMA-CAMPO-01",
                "records": [
                    {
                        "client_record_id": "PRUEBA-OFFLINE-001",
                        "lot_id": "LOTE-ESPINAL-01",
                        "disease_label": "roya_parda",
                        "confidence": 0.94,
                        "severity_index": 0.73,
                        "latitude": 4.1490,
                        "longitude": -74.8825,
                    }
                ],
            },
        )
        assert r_sync.status_code == 200
        datos_sync = r_sync.json()
        assert datos_sync["received_count"] == 1
        assert "PRUEBA-OFFLINE-001" in datos_sync["synced_ids"]

        # 6. Carga de Ortomosaico de Dron (/api/v1/drones/upload)
        r_orto = cliente.post(
            "/api/v1/drones/upload",
            files={"file": ("vuelo_dron.jpg", io.BytesIO(bytes_jpg), "image/jpeg")},
            data={"lot_id": "LOTE-ESPINAL-01"},
        )
        assert r_orto.status_code == 200
        datos_orto = r_orto.json()
        assert datos_orto["status"] == "COMPLETADO"
        assert datos_orto["total_patches"] >= 4

        # 7. Consulta del Mapa de Calor GeoJSON (/api/v1/maps/heatmap/{farm_id})
        r_calor = cliente.get("/api/v1/maps/heatmap/FINCA-TOL-01")
        assert r_calor.status_code == 200
        datos_calor = r_calor.json()
        assert datos_calor["geojson"]["type"] == "FeatureCollection"
        assert len(datos_calor["geojson"]["features"]) >= 1


def test_seguridad_owasp_asvs_portswigger():
    """Audita controles defensivos contra vectores OWASP Top 10, ASVS v4.0.3 y PortSwigger."""
    import pytest
    from backend.core.security import _cifrar_contrasena, verificar_contrasena
    from backend.core.storage import servicio_almacenamiento

    # 1. Criptografía PBKDF2-HMAC-SHA256 con sal y comparación en tiempo constante (OWASP A02 / ASVS V2.4)
    hash_1 = _cifrar_contrasena("ClaveSeguraTolima2026*")
    hash_2 = _cifrar_contrasena("ClaveSeguraTolima2026*")
    assert hash_1.startswith("pbkdf2_sha256$")
    assert hash_1 != hash_2  # Sales aleatorias únicas por cada derivación
    assert verificar_contrasena("ClaveSeguraTolima2026*", hash_1) is True
    assert verificar_contrasena("ClaveIncorrecta2026*", hash_1) is False

    # 2. Prevención de Path Traversal en Almacenamiento de Objetos (OWASP A01 / ASVS V12.3 / PortSwigger)
    with pytest.raises(ValueError):
        servicio_almacenamiento.guardar_bytes("../../etc/passwd", b"malicioso")
    with pytest.raises(ValueError):
        servicio_almacenamiento.guardar_bytes("movil/..\\..\\secret.txt", b"malicioso")

    with TestClient(app) as cliente:
        # 3. Cabeceras HTTP de Seguridad (OWASP A05 / ASVS V14.4)
        r_health = cliente.get("/api/v1/health")
        assert r_health.headers.get("X-Content-Type-Options") == "nosniff"
        assert r_health.headers.get("X-Frame-Options") == "DENY"
        assert r_health.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
        assert "max-age=31536000" in r_health.headers.get("Strict-Transport-Security", "")
        assert "default-src 'none'" in r_health.headers.get("Content-Security-Policy", "")
        assert r_health.headers.get("X-Request-ID")

        # 4. Prevención de Mass Assignment / Parameter Pollution (OWASP A04 / ASVS V5.1)
        r_mass = cliente.post(
            "/api/v1/auth/login",
            json={
                "username": "agronomo@fitovigia.co",
                "password": "FitoVigIA2026*",
                "role": "SUPERADMIN_INYECTADO",
            },
        )
        assert r_mass.status_code == 422

        # 5. Rechazo de Archivos Falsificados por Magic Bytes (OWASP A08 / PortSwigger File Upload)
        falso_jpeg = b"<html><script>alert('xss')</script></html>"
        r_spoof = cliente.post(
            "/api/v1/inference/online",
            files={"file": ("exploit.jpg", io.BytesIO(falso_jpeg), "image/jpeg")},
            data={"lot_id": "LOTE-ESPINAL-01"},
        )
        assert r_spoof.status_code == 400
        assert "Firma binaria" in r_spoof.json()["detail"]

        # 6. Bloqueo de Path Traversal en parámetros de formulario (`lot_id`)
        bytes_jpg = _crear_bytes_jpeg_prueba()
        r_traversal = cliente.post(
            "/api/v1/inference/online",
            files={"file": ("hoja.jpg", io.BytesIO(bytes_jpg), "image/jpeg")},
            data={"lot_id": "../../escape_dir"},
        )
        assert r_traversal.status_code == 400

        # 7. Seguridad JWT: revocación en `/auth/logout` y rechazo de tokens manipulados (PortSwigger JWT)
        r_login = cliente.post(
            "/api/v1/auth/login",
            json={"username": "agronomo@fitovigia.co", "password": "FitoVigIA2026*"},
        )
        assert r_login.status_code == 200
        jwt_valido = r_login.json()["access_token"]

        r_me = cliente.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {jwt_valido}"})
        assert r_me.status_code == 200
        assert r_me.json()["username"] == "agronomo@fitovigia.co"

        r_logout = cliente.post(
            "/api/v1/auth/logout", headers={"Authorization": f"Bearer {jwt_valido}"}
        )
        assert r_logout.status_code == 200

        # Intentar reutilizar el JWT revocado -> debe devolver 401 Unauthorized
        r_revocado = cliente.get(
            "/api/v1/auth/me", headers={"Authorization": f"Bearer {jwt_valido}"}
        )
        assert r_revocado.status_code == 401

