import { manejarSolicitudPortal } from "../main.ts";
import { escaparHtml, renderizarDocumentoSSR } from "../routes/_layout.ts";

Deno.test("Portal Web FitoVigIA SSR responde en /dashboard, /lotes, /mapas, /reportes y /health con cabeceras OWASP", async () => {
  const rutas = ["/dashboard", "/lotes", "/mapas", "/reportes"];

  for (const ruta of rutas) {
    const req = new Request(`http://localhost:3000${ruta}`);
    const resp = manejarSolicitudPortal(req);
    if (resp.status !== 200) {
      throw new Error(`La ruta ${ruta} devolvió estado ${resp.status}`);
    }
    if (resp.headers.get("x-content-type-options") !== "nosniff") {
      throw new Error(`Falta cabecera X-Content-Type-Options en ${ruta}`);
    }
    if (resp.headers.get("x-frame-options") !== "DENY") {
      throw new Error(`Falta cabecera X-Frame-Options en ${ruta}`);
    }
    if (!resp.headers.get("content-security-policy")?.includes("frame-ancestors 'none'")) {
      throw new Error(`Falta Content-Security-Policy en ${ruta}`);
    }
    const html = await resp.text();
    if (!html.includes("FitoVigIA Tolima Empresarial")) {
      throw new Error(`El HTML de ${ruta} no contiene el encabezado corporativo en español.`);
    }
  }

  // Verificar sanitización Anti-XSS (OWASP A03 / PortSwigger XSS)
  const payloadXss = `<script>alert("xss")</script>`;
  const escapado = escaparHtml(payloadXss);
  if (escapado.includes("<script>")) {
    throw new Error("escaparHtml no neutralizó la etiqueta <script>.");
  }
  const docSsr = renderizarDocumentoSSR(payloadXss, "/dashboard", "<p>Contenido</p>");
  if (docSsr.includes(`<title><script>`)) {
    throw new Error("renderizarDocumentoSSR permitió inyección XSS en <title>.");
  }

  const reqCss = new Request("http://localhost:3000/static/styles.css");
  const respCss = manejarSolicitudPortal(reqCss);
  const cssText = await respCss.text();
  if (!cssText.includes("--color-verde-arrozal: #1B5E20") || !cssText.includes("--gradcam-high")) {
    throw new Error("El archivo /static/styles.css no contiene los tokens del sistema de diseño.");
  }

  const reqHealth = new Request("http://localhost:3000/health");
  const respHealth = manejarSolicitudPortal(reqHealth);
  const datos = await respHealth.json();
  if (datos.status !== "ok") {
    throw new Error("El endpoint /health del portal Deno no devolvió ok.");
  }
});
