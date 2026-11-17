/**
 * Servidor Principal del Portal Web FitoVigIA (Deno 2.9.5 + Arquitectura de Islas SSR).
 *
 * Rutas en español cumpliendo la Regla de Oro:
 * - `/dashboard` : Tablero fitosanitario institucional (FEDEARROZ / AGROSAVIA)
 * - `/lotes`     : Inventario de fincas y lotes arroceros en el Tolima
 * - `/mapas`     : Visor interactivo Leaflet de ortomosaicos y mapas de calor GeoJSON
 * - `/reportes`  : Reportes MIP y auditoría de modelos de IA
 */

import { renderizarRutaDashboard } from "./routes/dashboard.tsx";
import { renderizarRutaLotes } from "./routes/lotes.tsx";
import { renderizarRutaMapas } from "./routes/mapas.tsx";
import { renderizarRutaReportes } from "./routes/reportes.tsx";

export function manejarSolicitudPortal(req: Request): Response {
  const url = new URL(req.url);
  const ruta = url.pathname;

  const cabecerasSeguridadComunes = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "strict-origin-when-cross-origin",
    "permissions-policy": "camera=(), microphone=(), geolocation=(self)",
    "cross-origin-opener-policy": "same-origin",
    "content-security-policy":
      "default-src 'self'; script-src 'self' 'unsafe-inline' https://unpkg.com; style-src 'self' 'unsafe-inline' https://unpkg.com; img-src 'self' data: https://*.tile.openstreetmap.org https://unpkg.com; connect-src 'self' http://localhost:8000 http://127.0.0.1:8000; frame-ancestors 'none';",
  };

  const cabecerasHtml = {
    "content-type": "text/html; charset=utf-8",
    "cache-control": "no-store",
    ...cabecerasSeguridadComunes,
  };

  if (ruta === "/static/styles.css") {
    const css = Deno.readTextFileSync(new URL("./static/styles.css", import.meta.url));
    return new Response(css, {
      status: 200,
      headers: {
        "content-type": "text/css; charset=utf-8",
        ...cabecerasSeguridadComunes,
      },
    });
  }
  if (ruta === "/" || ruta === "/dashboard") {
    return new Response(renderizarRutaDashboard(), { status: 200, headers: cabecerasHtml });
  }
  if (ruta === "/lotes") {
    return new Response(renderizarRutaLotes(), { status: 200, headers: cabecerasHtml });
  }
  if (ruta === "/mapas") {
    return new Response(renderizarRutaMapas(), { status: 200, headers: cabecerasHtml });
  }
  if (ruta === "/reportes") {
    return new Response(renderizarRutaReportes(), { status: 200, headers: cabecerasHtml });
  }
  if (ruta === "/health") {
    return new Response(
      JSON.stringify({
        status: "ok",
        runtime: `Deno ${Deno.version.deno}`,
        typescript: Deno.version.typescript,
        v8: Deno.version.v8,
        portal: "FitoVigIA Tolima SSR + Islands",
      }),
      {
        status: 200,
        headers: {
          "content-type": "application/json; charset=utf-8",
          ...cabecerasSeguridadComunes,
        },
      },
    );
  }

  return new Response(
    renderizarRutaDashboard(),
    { status: 404, headers: cabecerasHtml },
  );
}

if (import.meta.main) {
  const puerto = Number(Deno.env.get("PORT") ?? "3000");
  console.log(`Portal Web FitoVigIA (Deno ${Deno.version.deno}) activo en http://localhost:${puerto}/dashboard`);
  Deno.serve({ port: puerto }, manejarSolicitudPortal);
}
