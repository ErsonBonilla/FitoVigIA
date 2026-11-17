"""
Endpoints de Autenticación Empresarial JWT (/api/v1/auth) con Protección Anti-Fuerza Bruta
y Revocación de Sesión (OWASP ASVS V2 / V3).

Aplica la Regla de Oro:
- Funciones, variables y mensajes de error internos en español.
- Rutas REST y esquemas del contrato JSON (`access_token`, `token_type`) en inglés.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from backend.core.security import (
    autenticar_usuario,
    crear_token_acceso,
    extraer_clave_cliente_auth,
    limitador_auth,
    obtener_usuario_actual_opcional,
    revocar_token_jti,
)
from backend.schemas.api_schemas import RespuestaToken, SolicitudToken

router = APIRouter(prefix="/auth", tags=["Seguridad y Roles"])


@router.post("/login", response_model=RespuestaToken)
@router.post("/token", response_model=RespuestaToken)
def iniciar_sesion_para_token(carga: SolicitudToken, request: Request):
    clave_cliente = extraer_clave_cliente_auth(request, carga.username)
    limitador_auth.verificar_permitido(clave_cliente)

    usuario = autenticar_usuario(carga.username, carga.password)
    if usuario is None:
        limitador_auth.registrar_fallo(clave_cliente)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales inválidas. Verifique su usuario y contraseña corporativa.",
        )

    limitador_auth.limpiar_intentos(clave_cliente)
    token = crear_token_acceso(
        sujeto=usuario["usuario"],
        rol=usuario["rol"],
        distrito=usuario["distrito"],
    )
    return RespuestaToken(
        access_token=token,
        role=usuario["rol"],
        district=usuario["distrito"],
        full_name=usuario["nombre_completo"],
    )


@router.post("/logout")
def cerrar_sesion_y_revocar_token(usuario: dict = Depends(obtener_usuario_actual_opcional)):
    jti = usuario.get("jti")
    if jti:
        revocar_token_jti(str(jti))
    return {"status": "revoked", "message": "Sesión cerrada y token JWT invalidado exitosamente."}


@router.get("/me")
def obtener_perfil_operador_actual(usuario: dict = Depends(obtener_usuario_actual_opcional)):
    return {
        "username": usuario["username"],
        "role": usuario["role"],
        "district": usuario["district"],
    }
