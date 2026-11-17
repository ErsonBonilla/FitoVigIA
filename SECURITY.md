# Política de Seguridad de FitoVigIA Enterprise

**FitoVigIA** aplica un modelo de defensa en profundidad basado en **OWASP Top 10**, **OWASP ASVS v4.0.3 (Nivel 2)** y las guías de mitigación de **PortSwigger Web Security Academy**.

---

## 1. Versiones Soportadas

| Versión | Estado de Soporte | Componentes Cubiertos |
| :--- | :--- | :--- |
| `1.0.x` | Soportada activamente | `backend/` (FastAPI + Engine IA), `frontend/` (Deno SSR), `mobile/` (Flutter) |

---

## 2. Arquitectura de Controles de Seguridad Implementados

### A. Prevención en el Ciclo de Desarrollo (DevSecOps Local y CI/CD)
* **Lefthook (`lefthook.yml`):** Orquesta hooks bloqueantes en `pre-commit` y `pre-push`.
* **Gitleaks (`.gitleaks.toml`):** Escanea el árbol de trabajo y el historial de Git para impedir la filtración de credenciales, claves privadas PEM o tokens JWT.
* **Bandit & Ruff Security (`backend/pyproject.toml`):** Análisis estático SAST automatizado sobre el código Python verificando ausencia de inyecciones, criptografía insegura y manejo defectuoso de excepciones (CWE-703).
* **Dependabot (`.github/dependabot.yml`):** Vigilancia semanal de vulnerabilidades CVE en paquetes de Python, Flutter (`pub`) y GitHub Actions.

### B. Controles de Seguridad en Tiempo de Ejecución (OWASP ASVS & PortSwigger)
1. **Autenticación y Criptografía (`OWASP A02 / A07` · `ASVS V2`):**
   * Derivación de contraseñas con sal aleatoria única (`16 bytes`) mediante `PBKDF2-HMAC-SHA256` (`210,000` iteraciones).
   * Comparación en tiempo constante con `hmac.compare_digest` y derivación ficticia para neutralizar *Timing Attacks* de enumeración de usuarios.
   * Limitador de tasa en memoria (`LimitadorFuerzaBrutaAuth`) contra fuerza bruta en `/api/v1/auth/login` y `/api/v1/auth/token` (HTTP `429 Too Many Requests`).
2. **Gestión de Sesiones JWT (`ASVS V3.5` · `PortSwigger JWT Security`):**
   * Validación estricta de algoritmo (`HS256`, bloqueando `alg: none`), emisor (`iss`), audiencia (`aud`), vigencia (`exp`, `nbf`, `iat`) e identificador único (`jti`).
   * Revocación inmediata de tokens mediante `POST /api/v1/auth/logout`.
3. **Almacenamiento Seguro y Subida de Archivos (`OWASP A01 / A08` · `ASVS V12`):**
   * Confinamiento estricto contra *Path Traversal* (`resolve().is_relative_to(...)` y `sanitizar_identificador_ruta`).
   * Verificación de firmas binarias reales (*Magic Bytes* JPEG, PNG y GeoTIFF) y límites de tamaño (`413 Payload Too Large`).
4. **Cabeceras HTTP, Anti-XSS y Seguridad Móvil (`OWASP A03 / A05` · `ASVS V5 / V14`):**
   * Cabeceras `Content-Security-Policy`, `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy` y `X-Request-ID`.
   * Sanitización anti-XSS en servidor (`escaparHtml`) y cliente (`escaparHtmlCliente` en visores GeoJSON/Leaflet).
   * Bloqueo de tráfico en texto claro en Android mediante `network_security_config.xml` y contenedores Docker sin privilegios (`no-new-privileges:true`).

---

## 3. Reporte de Vulnerabilidades

Si identificas una vulnerabilidad de seguridad en cualquiera de los 3 componentes de **FitoVigIA**, por favor repórtala de manera privada a través de la pestaña **Security -> Report a vulnerability** (GitHub Private Vulnerability Reporting) del repositorio antes de divulgarla públicamente.
