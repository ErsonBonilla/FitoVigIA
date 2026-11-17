"""
Motor Geoespacial de FitoVigIA Enterprise (Rasterio + GeoPandas + Shapely).

Aplica la Regla de Oro:
- Lógica interna, etiquetas de patología (`sano`, `roya_parda`, `roya_amarilla`, etc.),
  zonas de severidad y leyendas de mapas en español.
- Estándar RFC 7946 GeoJSON (`FeatureCollection`, `features`, `geometry`, `properties`)
  y CRS `EPSG:4326` conservados en inglés por protocolo internacional.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.geometry import Point, box

from engine.config import (
    CATALOGO_AGRONOMICO,
    DIR_ARTEFACTOS,
    ORIGEN_PREENTRENAMIENTO_PUBLICO,
    PARAMETROS_UAV_LITERATURA,
    ZONAS_ARROCERAS_TOLIMA,
    normalizar_etiqueta_espanol,
)


def validar_parametros_vuelo_uav_rac100(
    altitude_m: float = 25.0,
    overlap_front_pct: float = 80.0,
    overlap_side_pct: float = 75.0,
    sensor_width_mm: float = 13.2,
    focal_length_mm: float = 8.8,
    image_width_px: int = 4000,
) -> dict:
    """Valida los parámetros de captura aérea UAV (Variable Independiente X del diseño DEXPLOS).

    Fundamentado en:
    - Petro et al. (2018) y Rojas et al. (2018) [Mapeo aéreo de arroz con UAV en Colombia].
    - Lin et al. (2025) y Qiang et al. (2025) [Detección de piricularia y plagas en arroz con UAV y YOLO].
    - García Díaz y Quimbayo Castiblanco (2016) [Norma RAC 100 de la Aeronáutica Civil].
    """
    gsd_cm_px = round((altitude_m * sensor_width_mm) / ( max(focal_length_mm, 0.1) * max(image_width_px, 1) ) * 100.0, 3)
    alt_ok = float(PARAMETROS_UAV_LITERATURA["altura_vuelo_min_m"]) <= altitude_m <= float(PARAMETROS_UAV_LITERATURA["altura_vuelo_max_m"])
    ov_front_ok = float(PARAMETROS_UAV_LITERATURA["traslape_longitudinal_min_pct"]) <= overlap_front_pct <= float(PARAMETROS_UAV_LITERATURA["traslape_longitudinal_max_pct"])
    ov_side_ok = float(PARAMETROS_UAV_LITERATURA["traslape_transversal_min_pct"]) <= overlap_side_pct <= float(PARAMETROS_UAV_LITERATURA["traslape_transversal_max_pct"])
    gsd_ok = float(PARAMETROS_UAV_LITERATURA["gsd_min_cm_px"]) <= gsd_cm_px <= float(PARAMETROS_UAV_LITERATURA["gsd_max_cm_px"])

    return {
        "altitude_m": altitude_m,
        "gsd_cm_px": gsd_cm_px,
        "overlap_front_pct": overlap_front_pct,
        "overlap_side_pct": overlap_side_pct,
        "rac100_vlos_compliant": alt_ok,
        "photogrammetry_valid": bool(alt_ok and ov_front_ok and ov_side_ok and gsd_ok),
    }


def crear_ortomosaico_geotiff_muestra(
    out_tif_path: Path,
    zone_key: str = "EL_ESPINAL",
    width: int = 672,
    height: int = 672,
    source_images: list[Path] | None = None,
) -> Path:
    """Construye un ortomosaico GeoTIFF georreferenciado (`EPSG:4326`) sobre un lote del Tolima."""
    zona = ZONAS_ARROCERAS_TOLIMA.get(zone_key, ZONAS_ARROCERAS_TOLIMA["EL_ESPINAL"])
    min_lon, min_lat, max_lon, max_lat = zona["bbox"]
    transformacion = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    lienzo_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tam_parche = 224
    idx = 0

    for y0 in range(0, height, tam_parche):
        for x0 in range(0, width, tam_parche):
            alto_celda = min(tam_parche, height - y0)
            ancho_celda = min(tam_parche, width - x0)
            if source_images and idx < len(source_images) and source_images[idx].exists():
                bytes_crudos = source_images[idx].read_bytes()
                bgr = cv2.imdecode(np.frombuffer(bytes_crudos, np.uint8), cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    lienzo_rgb[y0 : y0 + alto_celda, x0 : x0 + ancho_celda] = cv2.resize(
                        rgb, (ancho_celda, alto_celda)
                    )
                    idx += 1
                    continue

            generador = np.random.RandomState(42 + idx)
            celda = np.zeros((alto_celda, ancho_celda, 3), dtype=np.uint8)
            if (x0 // tam_parche == 1) and (y0 // tam_parche == 1):
                celda[..., 0] = generador.randint(145, 195, size=(alto_celda, ancho_celda))
                celda[..., 1] = generador.randint(120, 165, size=(alto_celda, ancho_celda))
                celda[..., 2] = generador.randint(35, 75, size=(alto_celda, ancho_celda))
            else:
                celda[..., 0] = generador.randint(40, 85, size=(alto_celda, ancho_celda))
                celda[..., 1] = generador.randint(135, 195, size=(alto_celda, ancho_celda))
                celda[..., 2] = generador.randint(35, 75, size=(alto_celda, ancho_celda))
            lienzo_rgb[y0 : y0 + alto_celda, x0 : x0 + ancho_celda] = celda
            idx += 1

    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    chw = np.transpose(lienzo_rgb, (2, 0, 1))
    with rasterio.open(
        out_tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=chw.dtype,
        crs="EPSG:4326",
        transform=transformacion,
    ) as dst:
        dst.write(chw)

    return out_tif_path


def seccionar_ortomosaico_en_parches(
    geotiff_path: Path,
    patch_size: int = 224,
) -> list[dict]:
    """Lee un ortomosaico GeoTIFF con Rasterio y extrae parches RGB con su polígono GPS real."""
    parches: list[dict] = []
    with rasterio.open(geotiff_path) as src:
        alto, ancho = src.height, src.width
        id_parche = 0
        for fila_off in range(0, alto, patch_size):
            for col_off in range(0, ancho, patch_size):
                alto_v = min(patch_size, alto - fila_off)
                ancho_v = min(patch_size, ancho - col_off)
                ventana = Window(col_off=col_off, row_off=fila_off, width=ancho_v, height=alto_v)
                transform_ventana = src.window_transform(ventana)

                datos = src.read(window=ventana)
                if datos.shape[0] >= 3:
                    rgb = np.transpose(datos[:3], (1, 2, 0)).astype(np.uint8)
                else:
                    gris = datos[0].astype(np.uint8)
                    rgb = np.stack([gris, gris, gris], axis=-1)

                izq, arr = transform_ventana @ (0, 0)
                der, aba = transform_ventana @ (ancho_v, alto_v)
                min_lon, max_lon = min(izq, der), max(izq, der)
                min_lat, max_lat = min(aba, arr), max(aba, arr)
                lon_centro = round((min_lon + max_lon) / 2.0, 6)
                lat_centro = round((min_lat + max_lat) / 2.0, 6)

                parches.append(
                    {
                        "patch_id": f"parche_{id_parche:04d}",
                        "row_off": fila_off,
                        "col_off": col_off,
                        "image_rgb": rgb,
                        "bbox_gps": (min_lon, min_lat, max_lon, max_lat),
                        "center_lat": lat_centro,
                        "center_lon": lon_centro,
                    }
                )
                id_parche += 1
    return parches


def construir_geodataframe_mapa_calor(
    patch_predictions: list[dict],
    lot_id: str = "LOTE-ESPINAL-01",
    municipio: str = "El Espinal",
) -> gpd.GeoDataFrame:
    """Construye un GeoDataFrame (EPSG:4326) con el nivel de severidad fitosanitaria por celda."""
    registros: list[dict] = []
    geometrias = []

    for item in patch_predictions:
        enfermedad = normalizar_etiqueta_espanol(item.get("disease_label", "sano"))
        confianza = float(item.get("confidence", 0.90))
        info_kb = CATALOGO_AGRONOMICO.get(enfermedad, CATALOGO_AGRONOMICO["sano"])
        sev_base = float(info_kb["severidad_base"])
        indice_severidad = round(
            sev_base * confianza if enfermedad != "sano" else 0.05 * (1.0 - confianza), 4
        )

        if enfermedad == "desconocido_no_arroz":
            zona_calor = "FUERA_DOMINIO_GRIS"
            color_gradcam = "#6C757D"
            opacidad_gradcam = 0.4
            estado_badge = "Objeto No Identificado"
        elif indice_severidad >= 0.65:
            zona_calor = "NIVEL_2_FOCO_INFECCION"
            color_gradcam = "#EF476F"
            opacidad_gradcam = 0.9
            estado_badge = "Plaga Detectada"
        elif indice_severidad >= 0.30:
            zona_calor = "NIVEL_1_ACTIVACION_MEDIA"
            color_gradcam = "#FFD166"
            opacidad_gradcam = 0.7
            estado_badge = "Alerta Temprana"
        else:
            zona_calor = "NIVEL_0_BAJA_ACTIVACION"
            color_gradcam = "#00B4D8"
            opacidad_gradcam = 0.3
            estado_badge = "Cultivo Sano"

        if "bbox_gps" in item and item["bbox_gps"]:
            min_lon, min_lat, max_lon, max_lat = item["bbox_gps"]
            geom = box(min_lon, min_lat, max_lon, max_lat)
            c_lat = round((min_lat + max_lat) / 2.0, 6)
            c_lon = round((min_lon + max_lon) / 2.0, 6)
        else:
            c_lat = float(item.get("latitude", 4.1486))
            c_lon = float(item.get("longitude", -74.8830))
            geom = Point(c_lon, c_lat).buffer(0.0008, cap_style=3)

        geometrias.append(geom)
        registros.append(
            {
                "lot_id": lot_id,
                "municipio": municipio,
                "patch_id": item.get("patch_id", f"celda_{len(registros):03d}"),
                "disease_label": enfermedad,
                "nombre_comun": info_kb["nombre_comun"],
                "estado_badge": estado_badge,
                "confidence": round(confianza, 4),
                "severity_index": indice_severidad,
                "heat_zone": zona_calor,
                "gradcam_color": color_gradcam,
                "gradcam_opacity": opacidad_gradcam,
                "nivel_riesgo": info_kb["nivel_riesgo"],
                "latitude": c_lat,
                "longitude": c_lon,
                "dataset_origin": item.get("dataset_origin", ORIGEN_PREENTRENAMIENTO_PUBLICO),
                "recomendacion_mip": info_kb["recomendacion_mip"],
            }
        )

    return gpd.GeoDataFrame(registros, geometry=geometrias, crs="EPSG:4326")


def exportar_artefactos_mapa_calor(
    gdf: gpd.GeoDataFrame,
    lot_id: str = "LOTE-ESPINAL-01",
    out_dir: Path = DIR_ARTEFACTOS / "mapas_calor",
) -> dict:
    """Exporta el Mapa de Calor Fitosanitario como capa GeoJSON estándar y visualización PNG (espectro Magma/Grad-CAM)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ruta_geojson = out_dir / f"mapa_calor_{lot_id}.geojson"
    ruta_png = out_dir / f"mapa_calor_{lot_id}.png"

    cadena_geojson = gdf.to_json()
    ruta_geojson.write_text(cadena_geojson, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9.6, 7.2), dpi=300)
    gdf.plot(
        column="severity_index",
        ax=ax,
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
        edgecolor="#1C2833",
        linewidth=1.2,
        legend=True,
        legend_kwds={
            "label": "Índice de Severidad Fitosanitaria (0,00 = Vigor Sano | 1,00 = Foco Crítico)",
            "shrink": 0.88,
        },
    )

    for _, fila in gdf.iterrows():
        centroide = fila.geometry.centroid
        etiqueta_corta = str(fila["disease_label"]).replace("_", " ").upper()
        conf_pct = float(fila.get("confidence", 0.95)) * 100.0
        ax.annotate(
            f"{fila['patch_id'].upper()}\n{etiqueta_corta}\nSev: {fila['severity_index']:.2f} ({conf_pct:.1f}%)",
            xy=(centroide.x, centroide.y),
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#1C2833",
            bbox={"boxstyle": "round,pad=0.32", "fc": "white", "alpha": 0.88, "ec": "#1B4F72", "lw": 0.9},
        )

    ax.set_title(
        f"Mapa de Calor Fitosanitario Georreferenciado – {lot_id} ({gdf['municipio'].iloc[0]}, Tolima)",
        fontsize=11.0,
        fontweight="bold",
        color="#1B4F72",
        pad=12,
    )
    ax.set_xlabel("Longitud Geográfica WGS84 (EPSG:4326)", fontsize=9.5, fontweight="bold", color="#1C2833")
    ax.set_ylabel("Latitud Geográfica WGS84 (EPSG:4326)", fontsize=9.5, fontweight="bold", color="#1C2833")
    ax.grid(True, linestyle="--", alpha=0.35, color="#566573")
    fig.tight_layout()
    fig.savefig(ruta_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    total_celdas = len(gdf)
    celdas_infestadas = int((gdf["disease_label"] != "sano").sum())
    tasa_infestacion = round(celdas_infestadas / max(1, total_celdas), 4)

    return {
        "lot_id": lot_id,
        "total_cells": total_celdas,
        "infested_cells": celdas_infestadas,
        "infection_rate": tasa_infestacion,
        "mean_severity_index": round(float(gdf["severity_index"].mean()), 4),
        "geojson_path": str(ruta_geojson),
        "png_path": str(ruta_png),
        "geojson": json.loads(cadena_geojson),
    }


# Alias de compatibilidad
create_sample_orthomosaic_geotiff = crear_ortomosaico_geotiff_muestra
slice_orthomosaic_into_patches = seccionar_ortomosaico_en_parches
build_heatmap_geodataframe = construir_geodataframe_mapa_calor
export_heatmap_artifacts = exportar_artefactos_mapa_calor
