# -*- coding: utf-8 -*-
"""
Módulo de Explicabilidad Neuronal e Interpretabilidad Visual en PyTorch para FitoVigIA.

Implementa sobre el modelo entrenado `ClasificadorCNNFitoVigIA` (`fitovigia_mejor_modelo.pth`):
1. Diagrama de la arquitectura exacta en PyTorch con conteo real de parámetros.
2. Visualización de filtros convolucionales (`Kernels 3x3`) y distribución de pesos (`Weights`).
3. Mapas de características (`Feature Maps` / Capas de activación superficial, intermedia y profunda).
4. Mapas de calor `Grad-CAM` (Gradient-weighted Class Activation Mapping) sobre imágenes reales de dron.
5. Reducción de dimensionalidad (`t-SNE` y `PCA`) de los embeddings latentes de 256 dimensiones (`head[3]`).
6. Matriz de Confusión y Curvas ROC Multiclase (One-vs-Rest) con datos reales de validación.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from engine.architectures import ClasificadorCNNFitoVigIA, construir_clasificador_pytorch, fijar_semillas_reproducibles
from engine.config import (
    CLASES_DRON,
    DIR_ARTEFACTOS,
    RUTA_MANIFIESTO_PARTICIONADO,
    RUTA_MODELO_PYTORCH,
    SEMILLA,
    TAMANO_LOTE,
)
from engine.preprocessing import (
    aplicar_correccion_radiometrica_clahe,
    cargar_y_preparar_imagen,
    construir_cargadores_datos,
    preprocesar_para_inferencia,
)


def cargar_modelo_dron_entrenado(ruta_ckpt: Path = RUTA_MODELO_PYTORCH) -> tuple[ClasificadorCNNFitoVigIA, dict]:
    """Carga el checkpoint real entrenado en GPU CUDA (`fitovigia_mejor_modelo.pth`)."""
    ckpt = torch.load(ruta_ckpt, map_location="cpu", weights_only=False)
    modelo = construir_clasificador_pytorch(
        backbone_name=ckpt.get("backbone_name", "efficientnet_b0"),
        num_classes=len(ckpt.get("class_names", CLASES_DRON)),
        dropout=ckpt.get("dropout", 0.25),
        fine_tune=ckpt.get("fine_tune", True),
        pretrained=False,
    )
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()
    return modelo, ckpt


def generar_diagrama_arquitectura_pytorch(
    modelo: ClasificadorCNNFitoVigIA,
    ruta_salida: Path,
) -> Path:
    """Genera el diagrama técnico de la arquitectura PyTorch `ClasificadorCNNFitoVigIA` con conteo real de parámetros."""
    total_params = sum(p.numel() for p in modelo.parameters())
    trainable_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    head_params = sum(p.numel() for p in modelo.head.parameters())

    fig, ax = plt.subplots(figsize=(11.2, 5.4), dpi=220)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        5.6,
        5.08,
        "Arquitectura de la Red Neuronal Convolucional en PyTorch: ClasificadorCNNFitoVigIA (EfficientNet-B0 + Cabezal SiLU)",
        fontsize=10.2,
        fontweight="bold",
        color="#1B4F72",
        ha="center",
        va="center",
    )
    ax.text(
        5.6,
        4.76,
        f"Parámetros Totales: {total_params:,} | Entrenables (Fine-Tuning Fase B): {trainable_params:,} | "
        f"Congelados (Base ImageNet): {frozen_params:,} | Cabezal Clasificador: {head_params:,}".replace(",", "."),
        fontsize=8.2,
        color="#2C3E50",
        ha="center",
        va="center",
    )

    bloques = [
        (0.25, 2.55, 1.45, 1.85, "#1B4F72", "ENTRADA +\nPREPROCESO",
         ["Imagen Dron RGB", "CLAHE (Canal L)", "Normalización", "Tensor NCHW:", "(B, 3, 224, 224)"]),
        (2.00, 2.55, 1.65, 1.85, "#2874A6", "STEM + MBConv\nCONGELADOS",
         ["features[0]: Conv3x3", "features[1..4]:", "MBConv1 / MBConv6", "BatchNorm2d .eval()", "Salida: (B, 80, 28, 28)"]),
        (3.95, 2.55, 1.75, 1.85, "#117864", "MBConv PROFUNDO\n(FINE-TUNING)",
         ["features[5..7]: MBConv6", "features[8]: Conv1x1", "Squeeze-and-Excitation", "Gradientes Activos", "Salida: (B, 1280, 7, 7)"]),
        (6.00, 2.55, 1.35, 1.85, "#6C3483", "AGRUPACIÓN\nESPACIAL",
         ["AdaptiveAvgPool2d", "Ventana 7x7 -> 1x1", "Flatten Tensor", "Vector Latente:", "(B, 1280)"]),
        (7.65, 2.55, 1.65, 1.85, "#B9770E", "CABEZAL DENSO\n(self.head[0..4])",
         ["LayerNorm(1280)", "Dropout(p=0.25)", "Linear(1280 -> 256)", "Init: Kaiming Normal", "SiLU + Dropout(0.125)"]),
        (9.60, 2.55, 1.35, 1.85, "#78281F", "CAPA SALIDA\n(self.head[5])",
         ["Linear(256 -> 4)", "Init: Xavier (g=0.5)", "Softmax / Logits:", "• sano • roya_parda", "• roya_am. • suelo"]),
    ]

    for idx, (x, y, w, h, col, tit, lineas) in enumerate(bloques):
        caja = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05", facecolor="#F8F9F9", edgecolor=col, linewidth=1.8
        )
        ax.add_patch(caja)
        cab = patches.FancyBboxPatch(
            (x, y + h - 0.48), w, 0.48, boxstyle="round,pad=0.03", facecolor=col, edgecolor="none"
        )
        ax.add_patch(cab)
        ax.text(x + w / 2, y + h - 0.24, tit, color="white", fontsize=7.6, fontweight="bold", ha="center", va="center")
        for j, lin in enumerate(lineas):
            ax.text(
                x + w / 2,
                y + h - 0.72 - j * 0.24,
                lin,
                color="#1C2833",
                fontsize=7.3,
                fontweight="bold" if "Tensor" in lin or "Salida" in lin or "Linear" in lin else "normal",
                ha="center",
                va="center",
            )
        if idx < len(bloques) - 1:
            nx = bloques[idx + 1][0]
            ax.annotate(
                "",
                xy=(nx - 0.03, y + h / 2),
                xytext=(x + w + 0.03, y + h / 2),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#2C3E50"),
            )

    # Panel inferior: Detalle matemático de ingeniería neuronal implementada en architectures.py
    panel_inf = patches.FancyBboxPatch(
        (0.25, 0.25), 10.7, 1.95, boxstyle="round,pad=0.05", facecolor="#EBF5FB", edgecolor="#2874A6", linewidth=1.4
    )
    ax.add_patch(panel_inf)
    ax.text(
        5.6,
        1.96,
        "Mecanismos de Estabilidad Matemática y Auditoría Neuronal Implementados en backend/engine/architectures.py",
        fontsize=8.6,
        fontweight="bold",
        color="#1B4F72",
        ha="center",
        va="center",
    )

    detalles = [
        ("1. Protección de BatchNorm2d en Transfer Learning",
         "Sobrescritura de modelo.train(mode) para forzar\nmodulo.eval() en todas las capas BatchNorm2d del\nbackbone, preservando media y varianza de ImageNet."),
        ("2. Activación SiLU (Swish) + Kaiming Normal",
         "Sustitución de ReLU por SiLU(x) = x · σ(x) e\ninicialización Kaiming fan_in en Linear(1280, 256),\ngarantizando 256/256 neuronas activas (0 muertas)."),
        ("3. Gancho de Auditoría y Exportación ONNX",
         "Forward hook en head[3] (256-D) para auditar normas\nde gradiente (0,776 / 0,844) y compilación a ONNX\nFP32 (16,55 MB) e INT8 cuantizado (4,51 MB)."),
    ]
    for k, (dtit, dbody) in enumerate(detalles):
        dx = 0.45 + k * 3.52
        sub = patches.FancyBboxPatch(
            (dx, 0.40), 3.30, 1.35, boxstyle="round,pad=0.04", facecolor="white", edgecolor="#2874A6", linewidth=1.0
        )
        ax.add_patch(sub)
        ax.text(dx + 1.65, 1.52, dtit, fontsize=7.6, fontweight="bold", color="#1B4F72", ha="center", va="center")
        ax.text(dx + 1.65, 0.90, dbody, fontsize=7.2, color="#1C2833", ha="center", va="center", linespacing=1.25)

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def visualizar_filtros_y_pesos_reales(
    modelo: ClasificadorCNNFitoVigIA,
    ruta_salida: Path,
) -> Path:
    """Extrae y grafica los 32 filtros convolucionales 3x3 reales del Stem y las distribuciones de pesos del cabezal."""
    pesos_conv0 = modelo.backbone.features[0][0].weight.detach().cpu().numpy()  # (32, 3, 3, 3)
    pesos_fc1 = modelo.head[2].weight.detach().cpu().numpy().ravel()  # (256 * 1280,)
    pesos_fc2 = modelo.head[5].weight.detach().cpu().numpy().ravel()  # (4 * 256,)

    fig = plt.figure(figsize=(11.0, 4.6), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0], wspace=0.28)

    # Panel izquierdo: Cuadrícula 4x8 con los 32 kernels 3x3 RGB reales de features[0][0]
    gs_left = gs[0].subgridspec(5, 8, hspace=0.28, wspace=0.12, height_ratios=[0.28, 1, 1, 1, 1])
    ax_tit = fig.add_subplot(gs_left[0, :])
    ax_tit.axis("off")
    ax_tit.text(
        0.5,
        0.5,
        "a) 32 Filtros Convolucionales (Kernels 3×3 RGB)\nCapa Inicial features[0][0] (Tensor: 32×3×3×3)",
        fontsize=8.8,
        fontweight="bold",
        color="#1B4F72",
        ha="center",
        va="center",
    )

    for idx in range(32):
        r = (idx // 8) + 1
        c = idx % 8
        ax_k = fig.add_subplot(gs_left[r, c])
        kernel = np.transpose(pesos_conv0[idx], (1, 2, 0))  # (3, 3, 3)
        k_min, k_max = kernel.min(), kernel.max()
        kernel_norm = (kernel - k_min) / max(k_max - k_min, 1e-6)
        ax_k.imshow(kernel_norm, interpolation="nearest")
        ax_k.set_title(f"F{idx+1}", fontsize=6.5, pad=1.5, color="#1C2833")
        ax_k.axis("off")

    # Panel derecho: Distribución empírica de los pesos sinápticos reales del cabezal
    gs_right = gs[1].subgridspec(2, 1, hspace=0.42)
    ax_w1 = fig.add_subplot(gs_right[0])
    ax_w1.hist(pesos_fc1, bins=60, color="#1B4F72", edgecolor="none", alpha=0.88, density=True)
    ax_w1.axvline(float(np.mean(pesos_fc1)), color="#D35400", linestyle="--", lw=1.4, label=f"Media: {np.mean(pesos_fc1):.4f}")
    ax_w1.set_title(
        f"b) Distribución de Pesos Reales head[2] Linear(1280→256)\n(n = {len(pesos_fc1):,} pesos | σ = {np.std(pesos_fc1):.4f})".replace(",", "."),
        fontsize=8.6,
        fontweight="bold",
        color="#1B4F72",
    )
    ax_w1.set_ylabel("Densidad", fontsize=8)
    ax_w1.legend(fontsize=7.5, loc="upper right")
    ax_w1.grid(alpha=0.3, linestyle="--")

    ax_w2 = fig.add_subplot(gs_right[1])
    ax_w2.hist(pesos_fc2, bins=35, color="#117864", edgecolor="#1C2833", linewidth=0.4, alpha=0.88, density=True)
    ax_w2.axvline(float(np.mean(pesos_fc2)), color="#C0392B", linestyle="--", lw=1.4, label=f"Media: {np.mean(pesos_fc2):.4f}")
    ax_w2.set_title(
        f"c) Distribución de Pesos Reales head[5] Linear(256→4)\n(n = {len(pesos_fc2):,} pesos | σ = {np.std(pesos_fc2):.4f})".replace(",", "."),
        fontsize=8.6,
        fontweight="bold",
        color="#1B4F72",
    )
    ax_w2.set_xlabel("Valor del Peso Sináptico (w)", fontsize=8)
    ax_w2.set_ylabel("Densidad", fontsize=8)
    ax_w2.legend(fontsize=7.5, loc="upper right")
    ax_w2.grid(alpha=0.3, linestyle="--")

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def visualizar_mapas_caracteristicas_reales(
    modelo: ClasificadorCNNFitoVigIA,
    ruta_imagen: Path,
    ruta_salida: Path,
) -> Path:
    """Extrae mediante forward hooks los mapas de activación reales en 3 profundidades del backbone."""
    img_rgb_clahe = cargar_y_preparar_imagen(ruta_imagen, aplicar_clahe=True)
    img_rgb_res = cv2.resize(img_rgb_clahe, (224, 224))
    tensor_entrada = torch.from_numpy(preprocesar_para_inferencia(ruta_imagen, aplicar_clahe=True))

    capturas: dict[str, np.ndarray] = {}
    capas_objetivo = {
        "superficial": modelo.backbone.features[1],   # (1, 16, 112, 112)
        "intermedia": modelo.backbone.features[4],    # (1, 80, 28, 28)
        "profunda": modelo.backbone.features[8],      # (1, 1280, 7, 7)
    }

    ganchos = []
    for nombre, capa in capas_objetivo.items():
        def _crear_hook(clave: str):
            def _hook(_mod, _inp, out):
                capturas[clave] = out.detach().cpu().numpy()[0]
            return _hook
        ganchos.append(capa.register_forward_hook(_crear_hook(nombre)))

    with torch.no_grad():
        logits = modelo(tensor_entrada)
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))

    for g in ganchos:
        g.remove()

    fig, axes = plt.subplots(3, 5, figsize=(11.0, 6.4), dpi=220)
    niveles = [
        ("superficial", "Nivel 1: Capa Superficial\nfeatures[1] (112×112×16)"),
        ("intermedia", "Nivel 2: Capa Intermedia\nfeatures[4] (28×28×80)"),
        ("profunda", "Nivel 3: Capa Profunda\nfeatures[8] (7×7×1280)"),
    ]

    for fila_idx, (clave, etiqueta_nivel) in enumerate(niveles):
        fmap = capturas[clave]  # (C, H, W)
        energia_canal = np.mean(np.abs(fmap), axis=(1, 2))
        top_canales = np.argsort(energia_canal)[::-1][:3]

        # Columna 0: Imagen de referencia + título del nivel
        axes[fila_idx, 0].imshow(img_rgb_res)
        axes[fila_idx, 0].set_title(etiqueta_nivel, fontsize=8.0, fontweight="bold", color="#1B4F72")
        axes[fila_idx, 0].axis("off")

        # Columnas 1..3: Los 3 canales de mayor energía de activación en esa capa
        for j, ch_idx in enumerate(top_canales):
            mapa_ch = fmap[ch_idx]
            axes[fila_idx, j + 1].imshow(mapa_ch, cmap="viridis", interpolation="bilinear")
            axes[fila_idx, j + 1].set_title(
                f"Canal #{int(ch_idx)}\n(Act. máx: {mapa_ch.max():.2f})", fontsize=7.6, color="#1C2833"
            )
            axes[fila_idx, j + 1].axis("off")

        # Columna 4: Promedio espacial de activación a través de todos los canales de la capa
        mapa_medio = np.mean(np.maximum(fmap, 0), axis=0)
        axes[fila_idx, 4].imshow(mapa_medio, cmap="magma", interpolation="bilinear")
        axes[fila_idx, 4].set_title(" Energía Media\nMulticanal (ReLU)", fontsize=7.8, fontweight="bold", color="#78281F")
        axes[fila_idx, 4].axis("off")

    fig.suptitle(
        f"Mapas de Características Reales (Feature Maps) en PyTorch – Muestra: {CLASES_DRON[pred_idx]} (Confianza: {probs[pred_idx]*100:.1f}%)",
        fontsize=10.0,
        fontweight="bold",
        color="#1B4F72",
        y=1.01,
    )
    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def calcular_gradcam_imagen(
    modelo: ClasificadorCNNFitoVigIA,
    ruta_imagen: Path,
    clase_objetivo: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Calcula matemáticamente el mapa Grad-CAM sobre la última capa convolucional `backbone.features[8]`."""
    bytes_img = Path(ruta_imagen).read_bytes()
    bgr = cv2.imdecode(np.frombuffer(bytes_img, dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb_orig = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (224, 224))
    rgb_clahe = cv2.resize(aplicar_correccion_radiometrica_clahe(rgb_orig), (224, 224))

    tensor_x = torch.from_numpy(preprocesar_para_inferencia(ruta_imagen, aplicar_clahe=True))
    tensor_x.requires_grad_(True)

    activaciones: list[torch.Tensor] = []
    gradientes: list[torch.Tensor] = []

    capa_conv = modelo.backbone.features[8]
    h_fwd = capa_conv.register_forward_hook(lambda _m, _i, out: activaciones.append(out))
    h_bwd = capa_conv.register_full_backward_hook(lambda _m, _gi, go: gradientes.append(go[0]))

    try:
        modelo.eval()
        modelo.zero_grad(set_to_none=True)
        logits = modelo(tensor_x)
        probs = torch.softmax(logits, dim=1)
        idx_cls = int(torch.argmax(probs, dim=1).item()) if clase_objetivo is None else int(clase_objetivo)
        confianza = float(probs[0, idx_cls].item())
        logits[0, idx_cls].backward()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    act = activaciones[0].detach()[0]  # (1280, 7, 7)
    grad = gradientes[0].detach()[0]   # (1280, 7, 7)
    pesos_alfa = torch.mean(grad, dim=(1, 2), keepdim=True)  # (1280, 1, 1)
    cam = torch.sum(pesos_alfa * act, dim=0)  # (7, 7)
    cam = F.relu(cam).cpu().numpy()

    cam_min, cam_max = float(cam.min()), float(cam.max())
    cam_norm = (cam - cam_min) / max(cam_max - cam_min, 1e-6)
    cam_224 = cv2.resize(cam_norm, (224, 224), interpolation=cv2.INTER_CUBIC)
    cam_224 = np.clip(cam_224, 0.0, 1.0)

    mapa_color_bgr = cv2.applyColorMap(np.uint8(255 * cam_224), cv2.COLORMAP_JET)
    mapa_color_rgb = cv2.cvtColor(mapa_color_bgr, cv2.COLOR_BGR2RGB)
    superposicion = np.clip(0.55 * rgb_clahe.astype(np.float32) + 0.45 * mapa_color_rgb.astype(np.float32), 0, 255).astype(np.uint8)

    return rgb_orig, rgb_clahe, cam_224, superposicion, idx_cls, confianza


def _seleccionar_muestra_representativa(
    modelo: ClasificadorCNNFitoVigIA,
    df_val: pd.DataFrame,
    idx_etiqueta: int,
) -> Path:
    """Selecciona la muestra de validación de la clase `idx_etiqueta` mejor clasificada por el modelo."""
    candidatos = df_val[df_val["label"] == idx_etiqueta]
    mejor_ruta = Path(candidatos.iloc[0]["path"])
    mejor_conf = -1.0
    modelo.eval()
    with torch.no_grad():
        for _, fila in candidatos.head(15).iterrows():
            ruta_c = Path(fila["path"])
            tx = torch.from_numpy(preprocesar_para_inferencia(ruta_c, aplicar_clahe=True))
            probs = torch.softmax(modelo(tx), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[idx_etiqueta].item())
            if pred == idx_etiqueta and conf > mejor_conf:
                mejor_conf = conf
                mejor_ruta = ruta_c
    return mejor_ruta


def visualizar_gradcam_multiclase_real(
    modelo: ClasificadorCNNFitoVigIA,
    df_particionado: pd.DataFrame,
    ruta_salida: Path,
) -> Path:
    """Genera un panel comparativo Grad-CAM sobre muestras reales de validación para las 4 clases del modelo."""
    df_val = df_particionado[df_particionado["split"].isin(["val", "test"])].reset_index(drop=True)
    clases_mostrar = ["roya_amarilla", "roya_parda", "sano", "suelo_desnudo"]

    fig, axes = plt.subplots(len(clases_mostrar), 4, figsize=(10.6, 8.6), dpi=220)
    columnas_tit = [
        "1. Captura Aérea Original\n(Dron VANT 224×224)",
        "2. Corrección Radiométrica\nCLAHE (Espacio CIELAB)",
        "3. Mapa de Activación\nGrad-CAM Puro (features[8])",
        "4. Superposición Térmica\nGrad-CAM + Inferencia",
    ]

    for fila, nombre_cls in enumerate(clases_mostrar):
        idx_etiqueta = CLASES_DRON.index(nombre_cls)
        ruta_img = _seleccionar_muestra_representativa(modelo, df_val, idx_etiqueta)

        orig, clahe_img, cam_224, overlay, pred_idx, conf = calcular_gradcam_imagen(
            modelo, ruta_img, clase_objetivo=idx_etiqueta
        )

        axes[fila, 0].imshow(orig)
        axes[fila, 0].set_ylabel(nombre_cls.upper(), fontsize=8.4, fontweight="bold", color="#1B4F72")
        axes[fila, 1].imshow(clahe_img)
        axes[fila, 2].imshow(cam_224, cmap="jet", vmin=0.0, vmax=1.0)
        axes[fila, 3].imshow(overlay)
        axes[fila, 3].set_xlabel(
            f"Pred: {CLASES_DRON[pred_idx]} ({conf*100:.1f}%)",
            fontsize=8.2,
            fontweight="bold",
            color="#117864" if pred_idx == idx_etiqueta else "#C0392B",
        )

        for col in range(4):
            if fila == 0:
                axes[fila, col].set_title(columnas_tit[col], fontsize=8.4, fontweight="bold", color="#1B4F72")
            axes[fila, col].set_xticks([])
            axes[fila, col].set_yticks([])

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def visualizar_reduccion_dimensionalidad_tsne(
    modelo: ClasificadorCNNFitoVigIA,
    df_particionado: pd.DataFrame,
    ruta_salida: Path,
) -> Path:
    """Extrae los embeddings reales de 256-D (`head[3]`) en validación/prueba y grafica t-SNE y PCA 2D."""
    fijar_semillas_reproducibles(SEMILLA)
    df_eval = df_particionado[df_particionado["split"].isin(["val", "test"])].reset_index(drop=True)
    cargadores = construir_cargadores_datos(
        df_eval.assign(split="val"), batch_size=TAMANO_LOTE, apply_clahe=True
    )
    loader = cargadores["val"]

    embeddings_lista: list[np.ndarray] = []
    etiquetas_lista: list[np.ndarray] = []

    gancho = modelo.head[3].register_forward_hook(
        lambda _m, _i, out: embeddings_lista.append(out.detach().cpu().numpy())
    )
    try:
        modelo.eval()
        with torch.no_grad():
            for lx, ly in loader:
                _ = modelo(lx)
                etiquetas_lista.append(ly.numpy())
    finally:
        gancho.remove()

    X_emb = np.concatenate(embeddings_lista, axis=0)  # (N, 256)
    y_true = np.concatenate(etiquetas_lista, axis=0)  # (N,)

    tsne = TSNE(n_components=2, perplexity=28, random_state=SEMILLA, init="pca", learning_rate="auto")
    Z_tsne = tsne.fit_transform(X_emb)

    pca = PCA(n_components=2, random_state=SEMILLA)
    Z_pca = pca.fit_transform(X_emb)
    var_explicada = pca.explained_variance_ratio_ * 100.0

    colores_clase = {
        0: ("#117864", "o", "sano (Cultivo saludable)"),
        1: ("#B9770E", "s", "roya_parda (Puccinia spp.)"),
        2: ("#C0392B", "^", "roya_amarilla (P. striiformis)"),
        3: ("#5D6D7E", "D", "suelo_desnudo (Pérdida densidad)"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=220)

    for idx_cls, (col, marcador, etiqueta) in colores_clase.items():
        mask = y_true == idx_cls
        ax1.scatter(
            Z_tsne[mask, 0], Z_tsne[mask, 1], c=col, marker=marcador, s=36, alpha=0.85, edgecolors="white", linewidth=0.5, label=etiqueta
        )
        ax2.scatter(
            Z_pca[mask, 0], Z_pca[mask, 1], c=col, marker=marcador, s=36, alpha=0.85, edgecolors="white", linewidth=0.5, label=etiqueta
        )

    ax1.set_title(f"a) Proyección No Lineal t-SNE (256-D → 2-D, n = {len(y_true)})", fontsize=9.2, fontweight="bold", color="#1B4F72")
    ax1.set_xlabel("Dimensión Latente t-SNE 1", fontsize=8.2)
    ax1.set_ylabel("Dimensión Latente t-SNE 2", fontsize=8.2)
    ax1.legend(fontsize=7.6, loc="best", framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle="--")

    ax2.set_title(
        f"b) Componentes Principales PCA (Var. explicada: {var_explicada[0]+var_explicada[1]:.1f}%)",
        fontsize=9.2,
        fontweight="bold",
        color="#1B4F72",
    )
    ax2.set_xlabel(f"PC1 ({var_explicada[0]:.1f}% varianza)", fontsize=8.2)
    ax2.set_ylabel(f"PC2 ({var_explicada[1]:.1f}% varianza)", fontsize=8.2)
    ax2.legend(fontsize=7.6, loc="best", framealpha=0.9)
    ax2.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def generar_matriz_confusion_y_roc_reales(
    modelo: ClasificadorCNNFitoVigIA,
    df_particionado: pd.DataFrame,
    ruta_salida: Path,
) -> Path:
    """Evalúa `fitovigia_mejor_modelo.pth` en el conjunto de validación y grafica Matriz de Confusión y Curvas ROC."""
    cargadores = construir_cargadores_datos(df_particionado, batch_size=TAMANO_LOTE, apply_clahe=True)
    val_loader = cargadores["val"]

    y_real_l: list[np.ndarray] = []
    y_prob_l: list[np.ndarray] = []

    modelo.eval()
    with torch.no_grad():
        for lx, ly in val_loader:
            logits = modelo(lx)
            probs = torch.softmax(logits, dim=1)
            y_real_l.append(ly.numpy())
            y_prob_l.append(probs.numpy())

    y_real = np.concatenate(y_real_l, axis=0)
    y_prob = np.concatenate(y_prob_l, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    mc = confusion_matrix(y_real, y_pred, labels=list(range(len(CLASES_DRON))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=220)
    disp = ConfusionMatrixDisplay(confusion_matrix=mc, display_labels=CLASES_DRON)
    disp.plot(ax=ax1, cmap="Blues", colorbar=False, xticks_rotation=20)
    ax1.set_title(f"a) Matriz de Confusión en Validación (n = {len(y_real)})", fontsize=9.2, fontweight="bold", color="#1B4F72")
    ax1.set_xlabel("Clase Predicha por ClasificadorCNNFitoVigIA", fontsize=8.2)
    ax1.set_ylabel("Clase Fitosanitaria Real (Ground Truth)", fontsize=8.2)

    y_bin = label_binarize(y_real, classes=list(range(len(CLASES_DRON))))
    colores_roc = ["#117864", "#B9770E", "#C0392B", "#2874A6"]
    for i, (cls_name, col) in enumerate(zip(CLASES_DRON, colores_roc)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_cls = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax2.plot(fpr, tpr, color=col, lw=2.0, label=f"{cls_name} (AUC = {auc_cls:.3f})")

    auc_macro = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2, label="Clasificador Aleatorio (0.500)")
    ax2.set_title(f"b) Curvas ROC Multiclase OvR (Macro AUC = {auc_macro:.4f})", fontsize=9.2, fontweight="bold", color="#1B4F72")
    ax2.set_xlabel("Tasa de Falsos Positivos (1 - Especificidad)", fontsize=8.2)
    ax2.set_ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)", fontsize=8.2)
    ax2.legend(loc="lower right", fontsize=7.8)
    ax2.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def generar_todas_visualizaciones_explicabilidad(dir_salida: Path | None = None) -> dict[str, str]:
    """Ejecuta y exporta las 6 visualizaciones reales de explicabilidad neuronal de FitoVigIA."""
    if dir_salida is None:
        dir_salida = DIR_ARTEFACTOS / "explicabilidad"
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)

    modelo, _ = cargar_modelo_dron_entrenado(RUTA_MODELO_PYTORCH)
    df_particionado = pd.read_csv(RUTA_MANIFIESTO_PARTICIONADO)
    df_val = df_particionado[df_particionado["split"].isin(["val", "test"])].reset_index(drop=True)

    # Seleccionar una muestra real bien clasificada de roya_amarilla para el análisis de Feature Maps
    muestra_roya = _seleccionar_muestra_representativa(
        modelo, df_val, CLASES_DRON.index("roya_amarilla")
    )

    rutas = {
        "arquitectura_pytorch": str(generar_diagrama_arquitectura_pytorch(modelo, dir_salida / "01_arquitectura_pytorch_cnn.png")),
        "filtros_y_pesos": str(visualizar_filtros_y_pesos_reales(modelo, dir_salida / "02_filtros_kernels_y_pesos.png")),
        "mapas_caracteristicas": str(visualizar_mapas_caracteristicas_reales(modelo, muestra_roya, dir_salida / "03_mapas_caracteristicas_capas.png")),
        "gradcam_multiclase": str(visualizar_gradcam_multiclase_real(modelo, df_particionado, dir_salida / "04_gradcam_multiclase_dron.png")),
        "tsne_embeddings": str(visualizar_reduccion_dimensionalidad_tsne(modelo, df_particionado, dir_salida / "05_tsne_pca_embeddings_256d.png")),
        "matriz_confusion_y_roc": str(generar_matriz_confusion_y_roc_reales(modelo, df_particionado, dir_salida / "06_matriz_confusion_y_roc_val.png")),
    }
    return rutas


if __name__ == "__main__":
    resultado = generar_todas_visualizaciones_explicabilidad()
    for k, v in resultado.items():
        print(f"[OK] {k}: {v}")
