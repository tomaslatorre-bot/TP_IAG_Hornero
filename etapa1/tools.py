"""
tools.py: tools de la etapa 1 (Marketing).

- leer_historial: lectura de contenidos_aprobados.csv (qué se le comunicó a cada segmento).
- guardar_pieza:  escritura de una pieza aprobada como fila nueva del mismo CSV.

Además hay funciones comunes (sin @tool) que usa el código del grafo para decidir qué recibe
cada agente y para controlar la pieza: buscar_en_guia y guia_completa (RAG sobre
guia_de_marca.pdf, índice FAISS guardado en disco), ejemplos_de_canal, canales_disponibles,
normalizar_canal, pieza_mas_parecida y dominios_del_historial.

Recordatorios (Clase 4):
- El docstring de cada tool es lo que el LLM lee para decidir cuándo usarla.
- Que cada tool reciba al menos un parámetro.
- Lo que devuelve una tool entra entero en el contexto del LLM: devolvemos resúmenes, no tablas enteras.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import pandas as pd
from langchain_core.tools import tool

from rag import IndiceRAG, _embeber

# Carpeta con los datos del TP. Si mueven los archivos, ajusten esta ruta.
RUTA_DATOS = Path(__file__).parent / "datos"
RUTA_GUIA = RUTA_DATOS / "etapa1" / "guia_de_marca.pdf"
RUTA_CONTENIDOS = RUTA_DATOS / "etapa1" / "contenidos_aprobados.csv"
CARPETA_INDICES = Path(__file__).parent / "indices"

COLUMNAS = ["fecha", "canal", "segmento", "titulo", "texto"]
SEGMENTOS = ["Nuevos", "Recurrentes", "Premium", "Todos"]

# El índice se arma (o se levanta del disco) recién la primera vez que alguien busca
indice_guia = IndiceRAG(RUTA_GUIA, CARPETA_INDICES, "guia_marca")


# --------------------------------------------------------------------------
# Funciones comunes (las usa el código del grafo, no el LLM)
# --------------------------------------------------------------------------

def _leer_contenidos() -> pd.DataFrame:
    return pd.read_csv(RUTA_CONTENIDOS)


def _normalizar(texto: str) -> str:
    return str(texto).strip().lower()


def canales_disponibles() -> list[str]:
    """Canales que ya existen en el historial (para que el estratega use los mismos nombres)."""
    return sorted(_leer_contenidos()["canal"].dropna().unique().tolist())


# Otras formas de nombrar un canal que no contienen su nombre en el historial.
# No son reglas de marca: solo sinónimos para reconocer el canal.
ALIAS_CANAL = {"e-mail": "Email", "mail": "Email", "correo": "Email", "notificación": "Push",
               "wsp": "WhatsApp", "wpp": "WhatsApp"}


def normalizar_canal(texto: str) -> str | None:
    """Lleva un canal escrito libremente ("Instagram (post)", "e-mail", "push notification")
    al nombre exacto que usa el historial. Si hay varios, toma el primero que aparece en el
    texto. Devuelve None si no reconoce ninguno: así no dependemos de que el LLM respete la lista."""
    texto = _normalizar(texto)
    canales = canales_disponibles()
    candidatos = {c.lower(): c for c in canales}
    candidatos.update({a: c for a, c in ALIAS_CANAL.items() if c in canales})
    encontrados = []
    for nombre, canal in candidatos.items():
        m = re.search(rf"(?<!\w){re.escape(nombre)}(?!\w)", texto)
        if m:
            encontrados.append((m.start(), canal))
    return min(encontrados)[1] if encontrados else None


def buscar_en_guia(consultas: list[str], k: int = 1) -> str:
    """Trae de la guía los fragmentos relevantes para varias consultas, sin repetir."""
    chunks = indice_guia.buscar_varias(consultas, k)
    return "\n\n---\n\n".join(chunks)


def guia_completa() -> str:
    """La guía entera, leída del PDF (es corta: unas 3 páginas). La usa el revisor."""
    return indice_guia.texto_completo()


def esqueleto_de_pieza(texto: str) -> str:
    """El formato de una pieza sin su contenido: deja los rótulos (Asunto:, Título:, Texto:), el
    saludo con {nombre}, la línea del botón y la cantidad de hashtags, y reemplaza el resto por
    <...>. Así el redactor ve CÓMO se arma una pieza del canal sin poder copiar QUÉ dice."""
    lineas = []
    for linea in str(texto).splitlines():
        linea = linea.strip()
        if not linea:
            continue
        rotulo = re.match(r"(Asunto|Título|Texto)\s*:", linea, flags=re.IGNORECASE)
        saludo = re.match(r"(Hola,?\s*\{nombre\}\s*[:.,]?)", linea, flags=re.IGNORECASE)
        boton = re.fullmatch(r"\[\s*([^:\]]+):.*\]", linea)
        hashtags = " #<hashtag>" * len(re.findall(r"#\w+", linea))
        if rotulo:
            lineas.append(f"{rotulo.group(1)}: <...>")
        elif boton:
            lineas.append(f"[{boton.group(1)}: <...>]")
        elif saludo:
            resto = " <...>" if linea[saludo.end():].strip() else ""
            lineas.append(saludo.group(1) + resto)
        else:
            lineas.append("<...>" + hashtags)
    return "\n".join(lineas)


def ejemplos_de_canal(canal: str, segmento: str, excluir_textos: list[str] | None = None,
                      cantidad: int = 2, descartar: Callable[[str], bool] | None = None) -> str:
    """Piezas aprobadas del mismo canal, como ejemplo de formato y estilo. Para que el redactor
    no copie algo que este segmento ya recibió, se excluyen las piezas enviadas a ese segmento
    o a 'Todos' (y las que se pasen en excluir_textos). descartar(texto) saca además las piezas
    que no sirven de ejemplo para este pedido (p. ej. promociones que el pedido no trae: el
    redactor terminaba copiando sus cuotas). Si no queda ninguna, se devuelve solo el esqueleto
    del formato de las piezas del canal."""
    df = _leer_contenidos()
    df = df[df["canal"].map(_normalizar) == _normalizar(canal)].sort_values("fecha", ascending=False)
    if df.empty:
        return "(no hay piezas aprobadas de este canal)"
    recibidas = {_normalizar(segmento), "todos"} if _normalizar(segmento) != "todos" else None
    excluir = {_normalizar(t) for t in (excluir_textos or [])}
    if recibidas is None:   # una pieza para 'Todos' le llega a todos: ninguna sirve de ejemplo
        otros = df.iloc[0:0]
    else:
        otros = df[~df["segmento"].map(_normalizar).isin(recibidas) & ~df["texto"].map(_normalizar).isin(excluir)]
        if descartar is not None:
            otros = otros[~otros["texto"].astype(str).map(descartar)]
    if not otros.empty:
        return "\n\n".join(f"TITULO: {f.titulo}\nTEXTO: {f.texto}" for f in otros.head(cantidad).itertuples())
    return ("(Solo el formato: las piezas de este canal ya le llegaron a este segmento, así que no se "
            "muestran. Cada <...> es contenido nuevo que escribís vos.)\n" + esqueleto_de_pieza(df.iloc[0]["texto"]))


def rotulos_del_canal(canal: str) -> list[str]:
    """Rótulos de línea ("Asunto:", "Título:", "Texto:") que llevan TODAS las piezas aprobadas
    del canal. Sale del historial, no de una regla escrita a mano: por ejemplo, los push del
    historial tienen las líneas "Título:" y "Texto:"."""
    df = _leer_contenidos()
    textos = df[df["canal"].map(_normalizar) == _normalizar(canal)]["texto"].astype(str).tolist()
    if not textos:
        return []
    return [r for r in ["Asunto", "Título", "Texto"]
            if all(re.search(rf"^\s*{r}\s*:", t, flags=re.IGNORECASE | re.MULTILINE) for t in textos)]


def dominios_del_historial() -> set[str]:
    """Dominios web que usan las piezas aprobadas (por ejemplo hornero.com.ar). Un link a otro
    dominio en una pieza nueva es inventado."""
    textos = " ".join(_leer_contenidos()["texto"].astype(str)).lower()
    return set(re.findall(r"\b((?:[a-z0-9-]+\.)+(?:com|net|org)(?:\.ar)?)\b", textos))


def pieza_mas_parecida(texto: str) -> dict:
    """Similitud coseno (embeddings de la Clase 1, el mismo modelo del RAG) entre la pieza nueva
    y cada pieza del historial. Devuelve la más parecida: similitud, titulo, fecha, segmento, texto."""
    df = _leer_contenidos()
    if df.empty:
        return {"similitud": 0.0, "titulo": "", "fecha": "", "segmento": "", "texto": ""}
    vectores = _embeber([texto] + df["texto"].astype(str).tolist())
    similitudes = vectores[1:] @ vectores[0]
    i = int(similitudes.argmax())
    fila = df.iloc[i]
    return {"similitud": float(similitudes[i]), "titulo": str(fila["titulo"]), "fecha": str(fila["fecha"]),
            "segmento": str(fila["segmento"]), "texto": str(fila["texto"])}


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@tool
def leer_historial(segmento: Literal["Nuevos", "Recurrentes", "Premium", "Todos"], cantidad: int = 6) -> str:
    """Devuelve las piezas que Marketing ya le mandó a un segmento por CUALQUIER canal (las más
    recientes primero), para no repetir mensajes. Incluye las piezas enviadas a 'Todos', porque
    también le llegaron a ese segmento. No filtra por canal a propósito: un mensaje que el
    segmento ya recibió por email tampoco conviene repetirlo en un push."""
    df = _leer_contenidos()
    if segmento == "Todos":
        filtro = pd.Series(True, index=df.index)
    else:
        filtro = df["segmento"].map(_normalizar).isin([_normalizar(segmento), "todos"])
    df = df[filtro].sort_values("fecha", ascending=False).head(cantidad)

    if df.empty:
        return f"No hay piezas previas para el segmento {segmento}."
    lineas = [f"Últimas {len(df)} piezas para {segmento} (todos los canales):"]
    for f in df.itertuples():
        texto = str(f.texto)
        texto = texto[:250] + ("..." if len(texto) > 250 else "")
        lineas.append(f"- {f.fecha} | {f.canal} | {f.segmento} | {f.titulo} | {texto}")
    return "\n".join(lineas)


@tool
def guardar_pieza(canal: str, segmento: Literal["Nuevos", "Recurrentes", "Premium", "Todos"],
                  titulo: str, texto: str) -> str:
    """Guarda una pieza APROBADA como fila nueva de contenidos_aprobados.csv, con la fecha y hora actual.
    Usar solo con piezas que ya pasaron la revisión de marca."""
    if not (canal.strip() and titulo.strip() and texto.strip()):
        return "No se guardó: canal, título y texto no pueden estar vacíos."
    canal_ok = normalizar_canal(canal)
    if canal_ok is None:
        return f"No se guardó: el canal '{canal}' no es uno de {canales_disponibles()}."
    canal = canal_ok
    if segmento not in SEGMENTOS:
        return f"No se guardó: el segmento debe ser uno de {SEGMENTOS}."

    # Sin espacios al final de cada línea: el modelo deja saltos de Markdown ("  \n")
    texto = "\n".join(linea.rstrip() for linea in texto.strip().splitlines())
    df = _leer_contenidos()
    repetida = (df["titulo"].map(_normalizar) == _normalizar(titulo)) & \
               (df["texto"].map(_normalizar) == _normalizar(texto))
    if repetida.any():
        return "No se guardó: esa pieza ya está en el historial."

    # Mismo formato de fecha que las filas originales: "2026-10-05 10:30"
    fila = pd.DataFrame([{"fecha": datetime.now().strftime("%Y-%m-%d %H:%M"), "canal": canal.strip(),
                          "segmento": segmento, "titulo": titulo.strip(), "texto": texto.strip()}])
    pd.concat([df, fila[COLUMNAS]], ignore_index=True).to_csv(RUTA_CONTENIDOS, index=False)
    return f"Pieza guardada en el historial ({len(df) + 1} piezas en total)."

