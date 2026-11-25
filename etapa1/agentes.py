"""
agentes.py: etapa 1, pipeline secuencial de Marketing con LangGraph.

    pedido ─► ESTRATEGA ─► REDACTOR ─► REVISOR ─► GUARDAR ─► respuesta
                 │            ▲           │
                 │            └─ rechaza ─┘   (como máximo MAX_BORRADORES borradores)
                 └─ fuera de alcance ─► fin

- Estratega (LLM + tool leer_historial): decide si el pedido está en alcance y arma un brief.
  Es el único que ve el historial (del segmento, por todos los canales), para no repetir.
- Redactor (LLM): escribe la pieza. Recibe solo los campos del brief que necesita (sin lo ya
  comunicado), las secciones de la guía que trae el RAG y piezas del mismo canal que el segmento
  NO recibió como ejemplo de formato (si no hay, solo el esqueleto del formato).
- Revisor de marca (LLM): lee la guía de marca COMPLETA y responde punto por punto el checklist
  de la guía más "datos inventados". Recibe solo la pieza, el pedido original y las medidas y
  controles hechos con código.
- Guardar (código): si la pieza se aprobó, la agrega al CSV (siempre, salvo que el pedido diga
  explícitamente que no; ver GUARDAR_SIEMPRE).

Las reglas de marca salen del PDF de la guía, que el revisor lee entero. Qué recibe cada agente
lo decide el código (las funciones nodo_*), no un LLM. Lo que el código puede verificar no se
deja librado al LLM:
- el canal y el segmento se validan contra el historial;
- las medidas (largo, palabras, emojis) se calculan con código;
- los valores (%, $, fechas, cuotas) que no están en el pedido ni en la guía rechazan la pieza;
- una pieza casi igual a una del historial (similitud coseno con embeddings) se rechaza;
- solo se aprueba si el veredicto es exactamente APROBADA y ningún punto del checklist falla.
- los límites de largo, emojis y hashtags se leen de la tabla de reglas por canal de la guía y
  se comparan con código (limites_del_canal).
Además hay listas escritas a mano, para reglas de la guía que el revisor tenía y no aplicaba
(las corridas lo muestran): beneficios (BENEFICIOS), tuteo (TUTEO_*), vigencias sin fecha
(VIGENCIAS), términos internos del brief (TERMINOS_DEL_BRIEF), contactos (CONTACTOS) y
características del producto (CARACTERISTICAS). Son listas cortas, de formas sin ambigüedad,
porque un falso positivo gasta un borrador sin que lo vea el revisor.

main.py solo usa responder(). La etapa 2 reutiliza ejecutar_marketing().
"""

import operator
import os
import re
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from tools import (SEGMENTOS, buscar_en_guia, canales_disponibles, dominios_del_historial, ejemplos_de_canal,
                   guardar_pieza, guia_completa, leer_historial, normalizar_canal,
                   pieza_mas_parecida, rotulos_del_canal)

load_dotenv()

class ContadorUso(BaseCallbackHandler):
    """Cuenta llamadas al LLM y tokens (Clase 4). LangChain llama a on_llm_end cada vez que
    termina una llamada; los tokens son los que informa Groq en usage_metadata.
    Es uno solo para todo el módulo: si dos pedidos corren a la vez, los números se mezclan
    (en la app de Streamlit, con un usuario, no pasa)."""

    def __init__(self):
        self.reiniciar()

    def reiniciar(self):
        self.llamadas, self.entrada, self.salida = 0, 0, 0

    def on_llm_end(self, response, **kwargs):
        self.llamadas += 1
        for generaciones in response.generations:
            for g in generaciones:
                uso = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                self.entrada += uso.get("input_tokens", 0)
                self.salida += uso.get("output_tokens", 0)


contador_uso = ContadorUso()

MODELO = "openai/gpt-oss-20b"
# reasoning_effort="low": con el valor por defecto, gpt-oss-20b a veces razona hasta agotar los
# tokens de salida y devuelve una respuesta vacía. Además gasta muchos menos tokens.
# callbacks: el contador lo comparten los tres agentes, porque usan este mismo llm.
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=MODELO, temperature=0.0, max_retries=5,
               reasoning_effort="low", callbacks=[contador_uso])

# Ciclo de corrección: si el revisor rechaza, el redactor reescribe con las observaciones.
# Es lo que hace un equipo de Marketing real (escribe → revisa → corrige) y evita descartar una
# pieza por un detalle corregible. Con 2 borradores el push (límites de 40/90 caracteres) y
# WhatsApp quedaban rechazados seguido; con 3 alcanzan. Más vueltas suman tiempo y llamadas a Groq.
MAX_BORRADORES = 3        # 1 borrador + 2 correcciones como máximo (cada vuelta suma 2 llamadas al LLM)
# True: guarda toda pieza aprobada, porque la consigna pide entregar el CSV con las piezas de
# TODAS las consultas de prueba. Un pedido que dice explícitamente "no lo guardes" gana igual.
# False: guarda solo si el pedido lo pide ("guardalo").
GUARDAR_SIEMPRE = True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def preguntar(agente, texto: str) -> str:
    resultado = agente.invoke({"messages": [{"role": "user", "content": texto}]})
    return resultado["messages"][-1].content


def tools_llamadas(resultado) -> str:
    return "\n".join(f"→ Tool: {tc['name']}({tc['args']})"
                     for msg in resultado["messages"] for tc in (getattr(msg, "tool_calls", None) or []))


def limpiar(texto: str) -> str:
    # Saca el markdown en negrita que a veces agrega el modelo ("**CANAL:**")
    return re.sub(r"\*\*", "", texto or "").strip()


def extraer_campo(texto: str, campo: str, siguientes: list[str] | None = None) -> str:
    """Extrae el valor de 'CAMPO: valor'. Si se pasan los campos siguientes, toma todo el
    bloque (varias líneas) hasta el próximo campo; si no, solo esa línea."""
    texto = limpiar(texto)
    if siguientes:
        fin = "|".join(re.escape(s) for s in siguientes)
        patron = rf"^{campo}\s*:\s*(.*?)(?=\n\s*(?:{fin})\s*:|\Z)"
    else:
        patron = rf"^{campo}\s*:\s*([^\n]*)"
    m = re.search(patron, texto, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ""


# Incluye los números con emoji (1️⃣ = dígito + U+FE0F + U+20E3), que se usan para listas de tips
EMOJI = re.compile("[0-9#*]\uFE0F?\u20E3|[\U0001F000-\U0001FAFF☀-➿⬀-⯿]")
ROTULOS = ["Asunto", "Título", "Texto"]
ROTULO = re.compile(rf"^\s*(?:{'|'.join(ROTULOS)})\s*:\s*", flags=re.IGNORECASE | re.MULTILINE)


def palabras_de(texto: str) -> int:
    return len(re.findall(r"\w+", texto))


def cuerpo_de_email(texto: str) -> tuple[str | None, str]:
    """Separa el cuerpo de un email: las líneas entre el saludo ("Hola, {nombre}:") y el botón
    ("[Botón: ...]"). Quedan afuera el asunto, el saludo, el botón y los legales del pie.
    Devuelve (cuerpo, "") o (None, motivo) si no encuentra el saludo o el botón."""
    lineas = [l.strip() for l in texto.splitlines() if l.strip()]
    lineas = [l for l in lineas if not re.match(r"asunto\s*:", l, flags=re.IGNORECASE)]
    saludo = next((i for i, l in enumerate(lineas) if re.match(r"hola\b", l, flags=re.IGNORECASE)), None)
    boton = next((i for i, l in enumerate(lineas) if re.fullmatch(r"\[.*\]", l)), None)
    if saludo is None:
        return None, "no se encontró la línea de saludo (\"Hola, ...\")"
    if boton is None or boton <= saludo:
        return None, "no se encontró una línea de botón entre corchetes después del saludo"
    return "\n".join(lineas[saludo + 1:boton]), ""


def medir_pieza(texto: str, canal: str = "") -> str:
    """Medidas objetivas de la pieza, calculadas con código para que el revisor las compare
    con las reglas de la guía. Los LLM cuentan mal caracteres y emojis; acá no hay reglas,
    solo conteos: los límites los pone la guía. Cada medida dice exactamente qué cubre."""
    sin_rotulos = ROTULO.sub("", texto)
    hashtags = len(re.findall(r"#\w+", texto))
    medidas = [f"- Caracteres de toda la pieza (sin rótulos como 'Título:'): {len(sin_rotulos)}",
               f"- Palabras de toda la pieza (sin rótulos): {palabras_de(sin_rotulos)}",
               f"- Hashtags: {hashtags}",
               f"- Emojis en toda la pieza: {len(EMOJI.findall(texto))}"]
    # Si la pieza tiene líneas con rótulo (Asunto:, Título:, Texto:), medimos cada una sin el rótulo
    for rotulo in ROTULOS:
        m = re.search(rf"^\s*{rotulo}\s*:\s*([^\n]*)", texto, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            linea = m.group(1).strip()
            medidas.append(f"- Línea '{rotulo}' (sin el rótulo): {len(linea)} caracteres, "
                           f"{len(EMOJI.findall(linea))} emojis")
    # Email: la guía pone el tope de palabras al cuerpo, no a toda la pieza
    if (normalizar_canal(canal) or "") == "Email":
        cuerpo, motivo = cuerpo_de_email(texto)
        if cuerpo is None:
            medidas.append(f"- Cuerpo del email: no se pudo medir ({motivo}); contalo vos")
        else:
            medidas.append(f"- Cuerpo del email (entre el saludo y el botón; sin asunto, saludo, "
                           f"botón ni legales del pie): {palabras_de(cuerpo)} palabras")
    return "\n".join(medidas)


# --------------------------------------------------------------------------
# Controles hechos con código (antes del revisor)
# --------------------------------------------------------------------------
# El revisor tenía la regla de "no inventar datos" y la de no repetir, y no las aplicaba. Lo que
# se puede verificar con código se verifica acá. No son reglas de marca: se compara la pieza
# contra el pedido, la guía y el historial.

# Valores concretos: porcentajes, montos, fechas DD/MM y cantidad de cuotas
VALORES = re.compile(r"\d+(?:[.,]\d+)?\s*%|\$\s*\d[\d.,]*|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b|\b\d+\s+cuotas\b",
                     flags=re.IGNORECASE)
MARCADOR = re.compile(r"\[[^\]\d]*\]")   # [X%], [DD/MM]: datos pendientes, están permitidos
# (sin dígitos: "[Cupón: 10% del 01/12 al 31/12]" no es un marcador, son datos inventados entre corchetes)
# Beneficios (cuotas, descuento, cupón...). Si el pedido no los trae, solo pueden aparecer con su
# valor como marcador en la misma oración ("un cupón de [X%]"): es el caso del beneficio que la
# guía pide en la recuperación de clientes. Sin marcador ("cuotas sin interés"), son inventados.
BENEFICIOS = ["cuotas", "descuento", "cupón", "sin cargo", "gratis", "garantía", "reintegro", "bonificación",
              "financiación", "facilidades de pago", "opciones de pago", "pago flexible", "pagos flexibles",
              # los que la guía le asigna al segmento Premium: orientan el tono, no son una oferta del pedido
              "acceso anticipado", "atención preferencial", "beneficios exclusivos"]
# Similitud coseno a partir de la cual la pieza se considera una copia de una del historial.
# Calibrado con el historial: entre piezas distintas el máximo es ~0.49; una pieza nueva sobre el
# mismo tema que otra da ~0.68; una copia casi literal da ~1.0.
UMBRAL_COPIA = 0.85


# Tuteo (la guía, sección 2: voseo, no "compra tú"). Solo formas que no se confunden con el
# voseo ni con sustantivos: el presente de tú ("tienes" vs "tenés") en cualquier lugar, y el
# imperativo de tú ("Descubre" vs "Descubrí") solo al empezar una oración, donde le habla al
# cliente (en el medio, "elige" o "encuentra" pueden ser tercera persona). También el pronombre
# "ti" ("para ti", "hablan de ti"): con vos es "para vos". Quedan afuera las palabras que al
# empezar una oración pueden no ser tuteo: "Dale"/"Date" (iguales con vos), "Sigue"/"Cambia"
# (tercera persona: "Sigue el calor"), "Compra"/"Visita"/"Suma"/"Busca"/"Entra"/"Llena"
# (sustantivos o adjetivos: "Compra simple", "Visita guiada", "Suma de ideas").
TUTEO_PRESENTE = re.compile(
    r"\b(?:tú|ti|tienes|puedes|quieres|sabes|eres|haces|vienes|necesitas|buscas|encuentras|disfrutas|"
    r"prefieres|conoces|sientes|piensas|vives|eliges|descubres|mereces|imaginas|recibes|aprovechas)\b",
    flags=re.IGNORECASE)
TUTEO_IMPERATIVO = {"descubre", "elige", "aprovecha", "disfruta", "renueva", "mira", "ven", "haz",
                    "pide", "conoce", "prepara", "transforma", "encuentra", "ingresa", "escoge",
                    "imagina", "equipa", "decora", "atrévete", "anímate", "regálate",
                    # llamados a la acción en tú ("Abre la app": la guía lo escribe así en la tabla
                    # de canales para describir el push, pero en la pieza va "Abrí la app")
                    "abre", "toca", "descarga", "escribe", "pulsa", "accede", "aprende", "únete",
                    "suscríbete", "regístrate", "cuéntanos", "escríbenos", "llámanos", "contáctanos",
                    "déjanos", "mantén", "obtén", "explora", "consigue", "llévate", "olvídate"}
# Vigencias que dan un plazo sin fecha ("esta semana", "por tiempo limitado"): en una promoción,
# si el pedido no las trae, son un dato inventado igual que una fecha y van como marcador
# [DD/MM]. Solo se controlan si la pieza tiene un beneficio o un %: en un post de tips, "este
# fin de semana" no es la vigencia de nada.
VIGENCIAS = re.compile(
    r"\b(?:esta\s+semana|este\s+(?:mes|fin\s+de\s+semana|finde)|solo\s+por\s+hoy|s[oó]lo\s+hoy|"
    r"hasta\s+(?:el\s+)?(?:lunes|martes|mi[eé]rcoles|jueves|viernes|s[aá]bado|domingo)|"
    r"por\s+tiempo\s+limitado|[uú]ltimos\s+d[ií]as|por\s+pocos\s+d[ií]as|hasta\s+fin\s+de\s+mes)\b",
    flags=re.IGNORECASE)


# Palabras del brief y de la guía que no van en una pieza para clientes: si aparecen, el redactor
# copió instrucciones internas ("Voseo", "con voseo rioplatense"). No incluye "tono" ni
# "segmento"... que pueden tener otro sentido ("tonos tierra") o ser válidas.
TERMINOS_DEL_BRIEF = re.compile(r"\b(?:voseo|tuteo|rioplatense|brief|llamado a la acci[oó]n|cta)\b",
                                flags=re.IGNORECASE)


def terminos_del_brief(texto: str) -> list[str]:
    return sorted({m.group(0).lower() for m in TERMINOS_DEL_BRIEF.finditer(texto)})


def formas_de_tuteo(texto: str) -> list[str]:
    encontradas = [m.group(0) for m in TUTEO_PRESENTE.finditer(texto)]
    for oracion in re.split(r"[.!?¡¿\n]", texto):
        # se saltean los signos y los números del principio ("1️⃣ Separa", "2. Elige")
        primera = re.match(r"[\W\d_]*([^\W\d_]+)", oracion.strip())
        if primera and primera.group(1).lower() in TUTEO_IMPERATIVO:
            encontradas.append(primera.group(1))
    return sorted(set(encontradas), key=str.lower)


def vigencias_no_pedidas(texto: str, pedido: str) -> list[str]:
    es_promocion = "%" in texto or any(b in texto.lower() for b in BENEFICIOS)
    if not es_promocion:
        return []
    pedidas = {re.sub(r"\s+", " ", v).lower() for v in VIGENCIAS.findall(pedido)}
    return sorted({re.sub(r"\s+", " ", v).lower() for v in VIGENCIAS.findall(MARCADOR.sub("", texto))} - pedidas)


def _valores(texto: str) -> list[str]:
    return [re.sub(r"\s+", "", v).lower() for v in VALORES.findall(MARCADOR.sub("", texto))]


def beneficios_no_pedidos(texto: str, pedido: str) -> list[str]:
    """Beneficios que el pedido no trae y que el texto menciona sin un marcador [..] en la
    misma oración. Se usa para la pieza y también para el brief del estratega."""
    oraciones = [o.lower() for o in re.split(r"[.!?\n]", texto)]
    return [b for b in BENEFICIOS if b not in pedido.lower()
            and any(b in o and "[" not in o for o in oraciones)]


# Datos de contacto: un email o un teléfono que el pedido no trae es inventado (la guía pide
# "un canal directo de contacto", pero el dato lo pone Marketing: [CONTACTO]).
CONTACTOS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\b0800[\s-]?\d{3}[\s-]?\d{3,4}\b|"
                       r"\b\d{2,4}[\s-]\d{4}[\s-]\d{4}\b")
DOMINIO = re.compile(r"\b((?:[a-z0-9-]+\.)+(?:com|net|org)(?:\.ar)?)\b")
# Características del producto que la regla de NO INVENTAR prohíbe si el pedido no las trae
# (materiales, terminaciones, durabilidad...). El revisor las aprobaba: "materiales resistentes",
# "diseñados para el clima argentino", "personalizá cada pieza".
CARACTERISTICAS = re.compile(
    r"\b(?:resist\w*|impermeable\w*|durader\w*|artesanal\w*|materiales?\b|acabados?\b|"
    r"terminaciones?\b|(?<!seguimiento )(?<!atención )(?<!atencion )personaliz\w*|a\s+medida\b(?!\s+que)|dise[ñn]ad[oa]s?\s+para\s+el\s+clima)",
    flags=re.IGNORECASE)


def contactos_inventados(texto: str, pedido: str) -> list[str]:
    sin_marcadores = MARCADOR.sub("", texto)
    contactos = {c for c in CONTACTOS.findall(sin_marcadores) if c not in pedido}
    sin_emails = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "", sin_marcadores.lower())
    permitidos = dominios_del_historial() | set(DOMINIO.findall(pedido.lower()))
    links = {d for d in DOMINIO.findall(sin_emails) if d not in permitidos}
    return sorted(contactos | links)


def caracteristicas_inventadas(texto: str, pedido: str) -> list[str]:
    return sorted({m.group(0).lower() for m in CARACTERISTICAS.finditer(texto)
                   if m.group(0).lower() not in pedido.lower()})


def fila_del_canal(canal: str) -> str | None:
    """La fila de la tabla "Reglas por canal" de la guía para ese canal, tal como está en el PDF."""
    return next((l.strip() for l in guia_completa().splitlines() if l.strip().startswith("- Canal:")
                 and normalizar_canal(l.split("|")[0].split(":", 1)[1]) == canal), None)


def limites_del_canal(canal: str) -> dict:
    """Límites del canal leídos de la tabla "Reglas por canal" de la guía (no escritos a mano):
    caracteres, título y texto (push), asunto y palabras del cuerpo (email), emojis y hashtags.
    Si la guía no tiene la fila o no se entiende, devuelve {} y no se controla nada."""
    fila = fila_del_canal(canal)
    if fila is None:
        return {}
    celdas = {c.split(":", 1)[0].strip().lower(): c.split(":", 1)[1].strip()
              for c in fila.strip(" -").split("|") if ":" in c}
    numero = lambda patron, texto: (int(m.group(1).replace(".", "")) if (m := re.search(patron, texto, re.I)) else None)
    largo, emojis, hashtags = celdas.get("largo máximo", ""), celdas.get("emojis", ""), celdas.get("hashtags", "")
    limites = {"titulo": numero(r"t[ií]tulo de ([\d.]+)", largo), "texto": numero(r"texto de ([\d.]+)", largo),
               "asunto": numero(r"asunto de ([\d.]+)", largo), "palabras_cuerpo": numero(r"cuerpo de hasta ([\d.]+) palabras", largo)}
    if not any(limites.values()):
        limites["caracteres"] = numero(r"^([\d.]+) caracteres", largo)
    limites["emojis"] = 0 if "ninguno" in emojis.lower() else numero(r"hasta (\d+)", emojis)
    if hashtags.lower().startswith("no"):
        limites["hashtags"] = (0, 0)
    elif (m := re.match(r"(\d+)\s*(?:a|o)\s*(\d+)", hashtags)):
        limites["hashtags"] = (int(m.group(1)), int(m.group(2)))
    limites["hashtag_obligatorio"] = (re.search(r"siempre incluye (#\w+)", hashtags) or [None, None])[1]
    return {k: v for k, v in limites.items() if v is not None}


def fallas_de_limites(texto: str, canal: str) -> list[str]:
    """Compara las medidas de la pieza con los límites del canal y dice cuánto sobra: "recortá al
    menos 9" le sirve al redactor más que "excede el límite" (WhatsApp fallaba 3 borradores así)."""
    lim, fallas = limites_del_canal(canal), []
    def linea(rotulo):
        m = re.search(rf"^\s*{rotulo}\s*:\s*([^\n]*)", texto, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None
    medidas = [("caracteres", "la pieza", len(ROTULO.sub("", texto)), "caracteres"),
               ("titulo", "la línea 'Título:'", len(linea("Título") or ""), "caracteres"),
               ("texto", "la línea 'Texto:'", len(linea("Texto") or ""), "caracteres"),
               ("asunto", "el asunto", len(linea("Asunto") or ""), "caracteres"),
               ("emojis", "la pieza", len(EMOJI.findall(texto)), "emojis")]
    if "palabras_cuerpo" in lim:
        cuerpo, _ = cuerpo_de_email(texto)
        if cuerpo is not None:
            medidas.append(("palabras_cuerpo", "el cuerpo del email", palabras_de(cuerpo), "palabras"))
    for clave, que, valor, unidad in medidas:
        if clave in lim and valor > lim[clave]:
            pista = ""
            if unidad == "caracteres":   # los LLM cuentan mal caracteres; palabras, mejor
                texto_medido = {"titulo": linea("Título"), "texto": linea("Texto"),
                                "asunto": linea("Asunto")}.get(clave) or ROTULO.sub("", texto)
                palabras = palabras_de(texto_medido)
                pista = (f" Tiene {palabras} palabras: dejalo en {max(1, palabras * lim[clave] // valor - 1)} "
                         f"o menos, sacando frases enteras.")
            tipo = "Emojis" if unidad == "emojis" else "Largo"
            recorte = "Sacá" if unidad == "emojis" else "Recortá"
            fallas.append(f"{tipo} (control por código, tabla de la guía para {canal}): {que} admite hasta "
                          f"{lim[clave]} {unidad} y tiene {valor}. {recorte} al menos {valor - lim[clave]}.{pista}")
    if "hashtags" in lim:
        n, (minimo, maximo) = len(re.findall(r"#\w+", texto)), lim["hashtags"]
        if not minimo <= n <= maximo:
            fallas.append(f"Hashtags (control por código, tabla de la guía para {canal}): van de {minimo} a "
                          f"{maximo} y la pieza tiene {n}.")
    obligatorio = lim.get("hashtag_obligatorio")
    if obligatorio and obligatorio.lower() not in texto.lower():
        fallas.append(f"Hashtags (control por código, tabla de la guía para {canal}): falta {obligatorio}.")
    return fallas


# Legales escritos completos ("Válido del DD/MM...", "No acumulable..."). La guía (sección 6) los
# pide así en email e Instagram, pero en WhatsApp van como link a "Bases y condiciones": escritos
# completos, el mensaje no entra en los 300 caracteres (WhatsApp fallaba 3 borradores por eso).
LEGAL_LARGO = re.compile(r"no\s+acumulable|agotar\s+stock|v[aá]lido\s+del\b", flags=re.IGNORECASE)


def frases_compartidas(texto: str, otro: str, n: int = 5) -> list[str]:
    """Oraciones de la pieza que comparten n palabras seguidas con otra pieza: le dicen al
    redactor qué reescribir cuando la pieza es una copia (antes solo sabía "es casi igual")."""
    palabras = lambda t: re.findall(r"\w+", t.lower())
    otras = palabras(otro)
    ngramas = {tuple(otras[i:i + n]) for i in range(len(otras) - n + 1)}
    copiadas = []
    for oracion in re.split(r"(?<=[.!?])\s+|\n", texto):
        w = palabras(oracion)
        if any(tuple(w[i:i + n]) in ngramas for i in range(len(w) - n + 1)):
            copiadas.append(oracion.strip())
    return copiadas


def trae_datos_ajenos(texto: str, pedido: str) -> bool:
    """True si una pieza del historial tiene beneficios o valores (%, $, fechas) que el pedido
    no trae: no sirve de ejemplo, porque el redactor copiaba esas promociones."""
    return bool(beneficios_no_pedidos(texto, pedido)) or any(v not in _valores(pedido) for v in _valores(texto))


def controles_de_codigo(texto: str, pedido: str, canal: str, situacion: str = "",
                        titulo: str = "") -> tuple[str, list[str], dict]:
    """Devuelve (medidas para el revisor, fallas que rechazan la pieza, pieza más parecida).
    - Formato: si todas las piezas del canal en el historial llevan rótulos de línea ("Título:",
      "Texto:"), la pieza nueva también tiene que llevarlos.
    - Valores (%, $, fechas, cuotas) que no están en el pedido ni en la guía (la guía trae los
      legales, como "CFTEA 0%"): son datos inventados y rechazan la pieza.
    - Beneficios (cuotas, descuento, cupón...) que el pedido no trae y aparecen sin un marcador
      [..] en la misma oración: rechazan la pieza.
    - Similitud con la pieza más parecida del historial: si supera UMBRAL_COPIA, es una copia."""
    permitidos = set(_valores(pedido)) | set(_valores(guia_completa()))
    inventados = sorted({v for v in _valores(texto) if v not in permitidos})
    beneficios = beneficios_no_pedidos(texto, pedido)
    tuteo = formas_de_tuteo(texto)
    tuteo_titulo = formas_de_tuteo(titulo)   # el título interno también queda en el CSV
    vigencias = vigencias_no_pedidas(texto, pedido)
    internos = terminos_del_brief(texto)
    contactos = contactos_inventados(texto, pedido)
    caracteristicas = caracteristicas_inventadas(texto, pedido)
    parecida = pieza_mas_parecida(texto)

    medidas = [f"- Valores (%, $, fechas, cuotas) que no están en el pedido ni en la guía: "
               f"{', '.join(inventados) if inventados else 'ninguno'}",
               f"- Beneficios que el pedido no trae, mencionados sin marcador [..]: "
               f"{', '.join(beneficios) if beneficios else 'ninguno'}",
               f"- Formas de tuteo: {', '.join(tuteo) if tuteo else 'ninguna'}",
               f"- Vigencias sin fecha que el pedido no trae (\"esta semana\"...): "
               f"{', '.join(vigencias) if vigencias else 'ninguna'}",
               f"- Términos internos del brief en la pieza: {', '.join(internos) if internos else 'ninguno'}",
               f"- Contactos o links que no están en el pedido ni en el historial: "
               f"{', '.join(contactos) if contactos else 'ninguno'}",
               f"- Características del producto que el pedido no trae: "
               f"{', '.join(caracteristicas) if caracteristicas else 'ninguna'}",
               f"- Pieza más parecida del historial: '{parecida['titulo']}' ({parecida['fecha']}, "
               f"{parecida['segmento']}), similitud {parecida['similitud']:.2f} (copia desde {UMBRAL_COPIA})"]
    fallas = fallas_de_limites(texto, canal)
    faltan = [r for r in rotulos_del_canal(canal)
              if not re.search(rf"^\s*{r}\s*:", texto, flags=re.IGNORECASE | re.MULTILINE)]
    if faltan:
        fallas.append(f"Formato (control por código): las piezas de {canal} llevan las líneas "
                      f"{' y '.join(repr(r + ':') for r in rotulos_del_canal(canal))} dentro del TEXTO, "
                      f"y a esta le faltan {', '.join(repr(r + ':') for r in faltan)}.")
    if inventados:
        fallas.append(f"Datos inventados (control por código): {', '.join(inventados)} no están en el "
                      f"pedido. Sacalos o reemplazalos por marcadores entre corchetes ([X%], [DD/MM]).")
    if beneficios:
        fallas.append(f"Beneficios inventados (control por código): la pieza menciona "
                      f"{', '.join(beneficios)}, que el pedido no trae. Sacalos (si la guía pide un "
                      f"beneficio para esta situación, su valor va como marcador: [X%]).")
    if tuteo:
        fallas.append(f"Tuteo (control por código): {', '.join(tuteo)}. La guía pide voseo rioplatense "
                      f"(\"tenés\", \"podés\", \"descubrí\", \"elegí\", \"abrí la app\", \"para vos\").")
    if tuteo_titulo:
        fallas.append(f"Tuteo en el TITULO interno (control por código): {', '.join(tuteo_titulo)}. "
                      f"También va en voseo, o sin verbo (\"Descuento en Cocina\").")
    if vigencias:
        fallas.append(f"Vigencia inventada (control por código): {', '.join(vigencias)} no está en el "
                      f"pedido. Sacala o usá un marcador ([DD/MM]).")
    if contactos:
        fallas.append(f"Contacto inventado (control por código): {', '.join(contactos)} no está en el pedido "
                      f"ni en el historial. Usá un marcador ([CONTACTO], [LINK]) o un link de "
                      f"{', '.join(sorted(dominios_del_historial())) or 'Hornero'}.")
    if caracteristicas:
        fallas.append(f"Características inventadas (control por código): {', '.join(caracteristicas)}. El pedido "
                      f"no dice cómo está hecho el producto: contá para qué sirve, no cómo está hecho.")
    if "recuperaci" in situacion.lower() and not any(b in texto.lower() for b in BENEFICIOS):
        fallas.append("Recuperación de clientes (control por código, sección 8 de la guía): falta ofrecer un "
                      "beneficio concreto con legales, por ejemplo un cupón de [X%] con su vigencia [DD/MM].")
    if canal == "WhatsApp" and LEGAL_LARGO.search(texto):
        fallas.append("Legales (control por código, sección 6 de la guía): en WhatsApp los legales van como "
                      "link a \"Bases y condiciones\" (por ejemplo, 'Bases y condiciones: hornero.com.ar/legales'), "
                      "no con el texto completo (\"Válido del...\", \"No acumulable...\").")
    if internos:
        fallas.append(f"Brief filtrado (control por código): la pieza dice {', '.join(internos)}, que son "
                      f"instrucciones internas. Escribí el texto para el cliente, sin nombrarlas.")
    if parecida["similitud"] >= UMBRAL_COPIA:
        fallas.append(f"Copia del historial (control por código): la pieza es casi igual a "
                      f"'{parecida['titulo']}' ({parecida['fecha']}), similitud {parecida['similitud']:.2f}. "
                      f"Escribí un mensaje nuevo, con otras frases."
                      + (f" Frases copiadas: {'; '.join(repr(f) for f in frases_compartidas(texto, parecida['texto']))}."
                         if frases_compartidas(texto, parecida["texto"]) else ""))
    return "\n".join(medidas), fallas, parecida


CAMPOS_BRIEF = r"CANAL|SEGMENTO|SITUACI[OÓ]N|OBJETIVO|DATOS DEL PEDIDO|YA COMUNICADO|[AÁ]NGULO|TONO"


def campos_del_brief(brief: str) -> dict[str, str]:
    """{campo: valor} del brief, con los nombres en mayúscula y sin tildes ("ANGULO", "SITUACION")."""
    partes = re.split(rf"^\s*({CAMPOS_BRIEF})\s*:", limpiar(brief), flags=re.IGNORECASE | re.MULTILINE)
    sin_tilde = str.maketrans("ÁÓáó", "AOao")
    return {nombre.upper().translate(sin_tilde): valor.strip() for nombre, valor in zip(partes[1::2], partes[2::2])}


# Campos del brief que describen qué dice la pieza: ahí no puede haber beneficios que el pedido
# no trae. YA COMUNICADO sí los nombra (es lo que el segmento ya recibió) y no llega al redactor.
CAMPOS_CONTENIDO = ["DATOS DEL PEDIDO", "OBJETIVO", "ANGULO"]


def beneficios_del_brief(brief: str, pedido: str) -> list[str]:
    campos = campos_del_brief(brief)
    return beneficios_no_pedidos("\n".join(campos.get(c, "") for c in CAMPOS_CONTENIDO), pedido)


def brief_para_redactor(brief: str, canal: str, segmento: str, pedido: str = "") -> str:
    """Solo los campos del brief que el redactor necesita. Se deja afuera YA COMUNICADO (lo que
    el segmento ya recibió): es para el estratega, y en manos del redactor terminaba copiado.
    Si un campo de contenido todavía trae un beneficio que el pedido no pide (el estratega lo
    sacó del historial), se reemplaza: si no, el redactor recibe dos órdenes contradictorias
    (el brief pide cuotas y la revisión las rechaza) y falla en todos los borradores.
    Si el brief no vino con el formato esperado, se pasa entero."""
    campos = campos_del_brief(brief)
    if len(campos) < 4:
        return brief
    campos["CANAL"], campos["SEGMENTO"] = canal, segmento
    if pedido:
        reemplazos = {"DATOS DEL PEDIDO": pedido, "OBJETIVO": f"comunicar lo que dice el pedido: {pedido}",
                      "ANGULO": "elegilo vos, usando solo los DATOS DEL PEDIDO, sin agregar beneficios"}
        for campo in CAMPOS_CONTENIDO:
            if campo in campos and beneficios_no_pedidos(campos[campo], pedido):
                campos[campo] = reemplazos[campo]
    return "\n".join(f"{nombre}: {valor}" for nombre, valor in campos.items() if nombre != "YA COMUNICADO")


# --------------------------------------------------------------------------
# System prompts (rol, scope y tono de cada agente)
# --------------------------------------------------------------------------

REGLA_NO_INVENTAR = """NO INVENTAR DATOS:
- Nunca inventes datos que no estén en el pedido: precios, montos, porcentajes o códigos de
  descuento, fechas de vigencia, plazos de entrega, stock, cuotas, nombres de productos o colecciones.
- Tampoco inventes características del producto (materiales, terminaciones, medidas, garantías,
  "artesanal", "premium") ni beneficios que el pedido no menciona. Contá para qué sirve, no cómo está hecho.
- Tampoco inventes datos de contacto: emails, teléfonos ni links de otros dominios. Usá
  [CONTACTO] o [LINK], o los links de Hornero que aparecen en los ejemplos.
- Si la pieza necesita un dato que el pedido no da (por ejemplo la vigencia de un descuento),
  dejá un marcador entre corchetes para que Marketing lo complete: [DD/MM], [X%], [CÓDIGO]."""

# Recuperación de clientes: la guía (sección 8) pide contar qué estamos haciendo para mejorar, pero
# el pedido no trae ese dato, y el redactor lo inventaba ("un problema técnico... ya lo
# solucionamos", "un transportista dedicado"). No se resuelve con una lista de palabras: el dato
# va como marcador para que lo complete Marketing, igual que el valor del cupón.
NOTA_RECUPERACION = """RECUPERACIÓN DE CLIENTES (sección 8 de la guía):
- Reconocé el problema que dice el pedido, sin explicar su causa: el pedido no la da, y la guía
  pide no dar excusas ni culpar a terceros.
- Lo que Hornero está haciendo para mejorar va como marcador, [ACCIÓN DE MEJORA], salvo que el
  pedido lo diga. No inventes medidas, servicios ni canales de atención.
- No supongas en qué estado está el pedido (si llegó, si está en camino): el pedido solo dice que
  hubo un problema con la entrega. El asunto reconoce ese problema.
- El beneficio se ofrece en el cuerpo, antes del botón, con su valor como marcador: "te dejamos un
  cupón de [X%] para tu próxima compra". El botón lleva a usarlo y el legal va al pie.
- La acción de mejora va en una oración del cuerpo con el marcador, por ejemplo: "Para que no
  vuelva a pasar, estamos [ACCIÓN DE MEJORA]".
- La guía pide también un canal directo de contacto: va como marcador en una oración del cuerpo,
  por ejemplo: "Si necesitás ayuda, escribinos a [CONTACTO]"."""

# Aclaraciones de la guía que valen para UN canal. Van solo en la entrada del redactor de ese
# canal: en el prompt general, el email copiaba el link de legales de WhatsApp y el WhatsApp
# cerraba con "Abrí la app", que es el llamado del push.
NOTAS_DEL_CANAL = {
    "WhatsApp": ('Los legales van como link a "Bases y condiciones" (sección 6 de la guía), por ejemplo '
                 '"Bases y condiciones: hornero.com.ar/legales", no con el texto completo: así entra en '
                 'los 300 caracteres. El llamado a la acción es un link corto.'),
    "Instagram": 'El llamado a la acción es exactamente "Link en la bio" o "Encontralo en la app", sin mezclarlos.',
    "Email": ('Los legales del beneficio van completos al pie, después del botón (sección 6 de la guía), una '
              'sola vez. El link a "Bases y condiciones" es el formato de WhatsApp, no del email.'),
    "Push": ('La tabla de la guía describe el llamado como "Abre la app": en la pieza va en voseo, '
             '"Abrí la app". Los 90 caracteres de la línea "Texto:" incluyen "Abrí la app." y el emoji: '
             'son unas 14 palabras en total, en oraciones completas.'),
}

SYSTEM_ESTRATEGA = f"""Sos estratega de contenidos del equipo de Marketing de Hornero, un e-commerce
argentino de artículos para el hogar y electro que vende por web, app y marketplaces.

ROL: a partir de un pedido en lenguaje natural, definís QUÉ comunicar y CÓMO. No escribís la pieza.

TONO (de tu respuesta): profesional, concreto y breve. El brief es un documento interno para el
redactor: sin adornos ni frases publicitarias.

SCOPE (en este orden):
1. PRIMERO decidí si el pedido está dentro del alcance: solo piezas de comunicación para clientes
   de Hornero (posts, emails, mensajes de WhatsApp, push, LinkedIn). Si es cualquier otra cosa,
   respondé únicamente FUERA DE ALCANCE, sin usar ninguna tool.
2. Recién si está dentro del alcance, usá la tool leer_historial con el segmento destinatario
   para ver qué se le comunicó hace poco por cualquier canal, y proponé un ángulo que no repita
   esos mensajes.
3. Usá los fragmentos de la guía de marca que te pasamos para elegir el tono según el segmento
   y la situación.

QUÉ PUEDE OFRECER LA PIEZA:
- La descripción de cada segmento en la guía (por ejemplo, que valoran el precio y las cuotas)
  sirve para elegir el TONO y qué destacar de lo que el pedido trae. NO es una oferta: no
  agregues cuotas, descuentos, cupones, envío sin cargo, garantías ni ningún otro beneficio que
  el pedido no mencione.
- El ÁNGULO solo puede usar lo que figura en DATOS DEL PEDIDO. Única excepción: si la guía pide
  un elemento para esa situación (por ejemplo, un beneficio concreto en la recuperación de
  clientes), indicalo, con todos sus valores como marcadores entre corchetes ([X%], [DD/MM]).

{REGLA_NO_INVENTAR}

FORMATO DE RESPUESTA (exactamente estos campos, en este orden, sin markdown):
CANAL: <uno de los canales que te indico en el pedido>
SEGMENTO: <Nuevos, Recurrentes, Premium o Todos>
SITUACIÓN: <promoción, lanzamiento, recuperación de clientes o contenido>
OBJETIVO: <qué queremos lograr con la pieza, en una oración>
DATOS DEL PEDIDO: <solo lo que dice el pedido, sin agregar nada: producto o categoría, descuento, etc.>
YA COMUNICADO: <mensajes recientes a este segmento que hay que evitar repetir>
ÁNGULO: <la idea central de la pieza, distinta de lo ya comunicado y hecha solo con los DATOS DEL PEDIDO>
TONO: <cómo hablarle a este segmento en esta situación, según la guía>

Si el pedido no dice el segmento, elegí el más adecuado; si ninguno aplica, usá Todos.
Respondé en español rioplatense."""

SYSTEM_REDACTOR = f"""Sos redactor/a publicitario/a de Hornero, un e-commerce argentino de artículos para
el hogar y electro.

ROL: escribís UNA pieza de comunicación siguiendo el brief y las reglas de la guía de marca que
te pasamos. Las piezas son solo texto.

SCOPE:
- Solo escribís la pieza que pide el brief, para el canal y el segmento que indica. No cambiás
  el canal, el segmento ni el ángulo, y no agregás piezas extra ni comentarios fuera del formato.
- Tu única fuente de reglas son los fragmentos de la guía; tu única fuente de datos, el brief.
- El brief son instrucciones internas: no se citan ni se copian en la pieza. Escribí con tus
  palabras lo que el brief pide, sin repetir sus frases.

TONO: el que indica la guía para el segmento y la situación del brief, en español rioplatense
con voseo: cálido, claro y práctico, como un vecino que sabe de casas.

REGLAS:
- Seguí todas las reglas de los fragmentos de la guía: voz y tono, reglas del canal (largo,
  hashtags, emojis, llamado a la acción), palabras prohibidas y legales obligatorios.
- Copiá el FORMATO de las piezas de ejemplo del canal (por ejemplo, si los emails empiezan con
  "Asunto:" o los push tienen líneas "Título:" y "Texto:"), pero no copies sus mensajes, sus
  frases ni sus datos: los porcentajes, fechas y beneficios de un ejemplo NO valen para tu pieza.
- Si recibís observaciones de la revisión, corregí TODAS sin cambiar lo que estaba bien.

{REGLA_NO_INVENTAR}

FORMATO DE RESPUESTA (exactamente estos tres campos, sin markdown):
TITULO: <título interno corto y descriptivo, como los del historial ("Recuperación: entrega con
problemas", "Lanzamiento colección Nórdica"): describe la pieza para Marketing, no le habla al cliente>
TEXTO: <la pieza completa tal como se publica, con el mismo formato que las piezas del canal; si el
canal usa líneas como "Asunto:" o "Título:" y "Texto:", van dentro de este campo>
IMAGEN SUGERIDA: <una sola línea describiendo la imagen que podría acompañarla>"""

SYSTEM_REVISOR = """Sos revisor/a de marca de Hornero. Controlás que una pieza cumpla la guía de marca
antes de publicarse.

ROL: revisás, no reescribís. Recibís la guía de marca completa, el pedido original, la pieza y sus
medidas y controles hechos con código (caracteres, palabras, hashtags, emojis, datos que no están
en el pedido), que son exactos: usalos en lugar de contar vos. Fijate qué parte de la pieza cubre
cada medida antes de compararla con la guía.

SCOPE:
- Solo controlás la pieza contra la guía de marca y el pedido original. No evaluás la estrategia
  ni el ángulo, y no proponés una versión reescrita de la pieza.
- Solo marcás incumplimientos que puedas justificar citando la guía o el pedido.

TONO: objetivo, preciso y breve. Cada observación nombra la regla incumplida y qué cambiar.

QUÉ CONTROLÁS, PUNTO POR PUNTO:
- Puntos 1 a 7: los 7 puntos del "Checklist antes de publicar" (sección 10 de la guía), en el
  mismo orden. Para cada uno aplicá las reglas de la guía que le corresponden (por ejemplo, para
  el largo y los hashtags, la tabla de reglas por canal, con sus emojis y llamado a la acción).
- Punto 8, Datos inventados: cualquier precio, porcentaje, fecha, producto, característica del
  producto o beneficio (cuotas, descuento, cupón, envío sin cargo, garantía) que NO esté en el
  pedido es un error, salvo que la guía lo pida para esa situación y sus valores estén como
  marcadores entre corchetes ([DD/MM], [X%]), que son la forma correcta de dejar datos pendientes.

RECUPERACIÓN (sección 8): el pedido no dice qué está haciendo Hornero para mejorar, así que la
pieza lo deja como marcador, [ACCIÓN DE MEJORA], que completa Marketing. Ese marcador cumple el
punto de "contar qué estamos haciendo", igual que [X%] cumple el valor del cupón: no la rechaces
por eso. Sí es un error inventar la causa del problema o una medida concreta que el pedido no trae.

VOSEO (punto 1): las formas de vos llevan la tilde al final ("elegí", "escogé", "tenés", "abrí",
"descubrí") y son voseo correcto. Marcá NO CUMPLE en el punto 1 solo si citás una palabra en tú
("tienes", "elige", "abre", "ti") o en usted.

CANAL: aplicá solo la fila de la tabla de reglas por canal que corresponde al canal de la pieza
(te la pasamos aparte). Las reglas de otros canales no aplican: por ejemplo, #HorneroEnCasa es
de Instagram y un push o un WhatsApp no lleva hashtags. Si la pieza llegó a vos, el código ya
comprobó contra esa fila el largo, los emojis y los hashtags, y cumplen.

LEGALES (sección 6): en WhatsApp los legales van como link a "Bases y condiciones"; un link así
cumple el punto de legales, aunque no esté el texto completo.

Cada punto lleva exactamente una de estas marcas: CUMPLE, NO CUMPLE o NO APLICA (NO APLICA solo
si el punto es condicional y no corresponde, por ejemplo legales cuando no hay promoción).
APROBADA solo si ningún punto es NO CUMPLE. Si la pieza cumple, aprobala aunque se pudiera
escribir de otra manera.

FORMATO DE RESPUESTA (exactamente estos campos, sin markdown):
CHECKLIST:
1. <punto 1 de la guía>: <CUMPLE, NO CUMPLE o NO APLICA> - <justificación breve>
2. ... (un renglón por punto, del 1 al 8)
8. Datos inventados: <CUMPLE, NO CUMPLE o NO APLICA> - <justificación breve>
VEREDICTO: <solo la palabra APROBADA o la palabra RECHAZADA>
OBSERVACIONES: <si está rechazada, lista de incumplimientos concretos, la regla de la guía que
incumplen y cómo corregirlos; si está aprobada, "ninguna">"""

PUNTOS_CHECKLIST = 8   # 7 del checklist de la guía + datos inventados

# Tres agentes con create_agent, cada uno con sus tools (mínimo privilegio):
# solo el estratega lee el historial; nadie escribe en el CSV (lo hace el nodo de código).
agente_estratega = create_agent(model=llm, tools=[leer_historial], system_prompt=SYSTEM_ESTRATEGA)
agente_redactor = create_agent(model=llm, tools=[], system_prompt=SYSTEM_REDACTOR)
agente_revisor = create_agent(model=llm, tools=[], system_prompt=SYSTEM_REVISOR)


# --------------------------------------------------------------------------
# Estado del grafo
# --------------------------------------------------------------------------

class EstadoMarketing(TypedDict, total=False):
    pedido: str
    guardar: bool | None   # True: el pedido pide guardar; False: pide NO guardar; None: no dice
    # Estratega
    brief: str
    canal: str
    segmento: str
    fuera_de_alcance: bool
    sin_canal: bool        # el canal del brief y del pedido no es ninguno de los del historial
    # Redactor
    titulo: str
    texto: str
    imagen: str
    borradores: int
    # Revisor
    aprobada: bool
    observaciones: str
    textos_copiados: list[str]   # piezas del historial que un borrador copió: no se usan de ejemplo
    # Guardado
    resultado_guardado: str
    # Pasos para mostrar en la app (cada nodo agrega los suyos)
    pasos: Annotated[list[dict], operator.add]


# --------------------------------------------------------------------------
# Nodos
# --------------------------------------------------------------------------

def nodo_estratega(estado: EstadoMarketing) -> dict:
    # RAG: secciones de la guía para elegir el tono según segmento y situación
    guia = buscar_en_guia(["segmentos de clientes y cómo hablarles", "voz y tono según la situación"])
    entrada = (f"PEDIDO: {estado['pedido']}\n\n"
               f"CANALES DE HORNERO (usá estos nombres): {', '.join(canales_disponibles())}\n\n"
               f"FRAGMENTOS DE LA GUÍA DE MARCA:\n{guia}")
    resultado = agente_estratega.invoke({"messages": [{"role": "user", "content": entrada}]})
    brief = limpiar(resultado["messages"][-1].content)

    paso = {"titulo": "1. Estratega (brief)", "contenido": f"{tools_llamadas(resultado)}\n\n{brief}".strip()}
    if "FUERA DE ALCANCE" in brief.upper():
        return {"fuera_de_alcance": True, "brief": brief, "pasos": [paso]}

    # Control por código del brief: el estratega ve el historial y a veces arrastra un beneficio
    # de una pieza vieja (p. ej. "cuotas sin interés en Electro" al push de aires). Se le pide
    # rehacerlo una vez; si insiste, brief_para_redactor reemplaza esos campos.
    beneficios = beneficios_del_brief(brief, estado["pedido"])
    if beneficios:
        correccion = (f"Tu brief agrega {', '.join(beneficios)}, que el pedido no trae (lo sacaste del "
                      f"historial o lo supusiste). Rehacé el brief completo, con el mismo formato: "
                      f"DATOS DEL PEDIDO, OBJETIVO y ÁNGULO solo con lo que dice el pedido.")
        resultado = agente_estratega.invoke({"messages": resultado["messages"] +
                                             [{"role": "user", "content": correccion}]})
        brief = limpiar(resultado["messages"][-1].content)
        paso = {"titulo": "1. Estratega (brief corregido por código)",
                "contenido": f"{paso['contenido']}\n\n→ Control por código: {correccion}\n\n{brief}"}

    segmento = (extraer_campo(brief, "SEGMENTO").split() or ["Todos"])[0].strip(".,")
    segmento = next((s for s in SEGMENTOS if s.lower() == segmento.lower()), "Todos")
    # El canal se valida con código, igual que el segmento: se lleva al nombre exacto del
    # historial ("Instagram (post)" → "Instagram"). Si el brief no trae uno válido, se busca
    # en el pedido; si tampoco, se corta el flujo en vez de seguir con un canal desconocido.
    canal = normalizar_canal(extraer_campo(brief, "CANAL")) or normalizar_canal(estado["pedido"])
    if canal is None:
        return {"fuera_de_alcance": False, "sin_canal": True, "brief": brief, "pasos": [paso]}
    return {
        "fuera_de_alcance": False,
        "sin_canal": False,
        "brief": brief,
        "canal": canal,
        "segmento": segmento,
        "borradores": 0,
        "pasos": [paso],
    }


def nodo_redactor(estado: EstadoMarketing) -> dict:
    canal, segmento = estado["canal"], estado["segmento"]
    # RAG: el código decide qué secciones de la guía necesita el redactor
    guia = buscar_en_guia(["voz y tono", f"reglas por canal {canal}", f"cómo hablarle al segmento {segmento}",
                           "palabras y frases prohibidas", "promociones y legales obligatorios",
                           "plazos de entrega"])
    # Ejemplos del canal que este segmento NO recibió (o solo el esqueleto del formato), para
    # que el redactor no copie un mensaje que el segmento ya tiene
    ejemplos = ejemplos_de_canal(canal, segmento, excluir_textos=estado.get("textos_copiados"),
                                 descartar=lambda t: trae_datos_ajenos(t, estado["pedido"]))
    entrada = (f"BRIEF (instrucciones internas: no se citan ni se copian en la pieza):\n"
               f"{brief_para_redactor(estado['brief'], canal, segmento, estado['pedido'])}\n\n"
               f"FRAGMENTOS DE LA GUÍA DE MARCA:\n{guia}\n\n"
               f"PIEZAS DE EJEMPLO DEL CANAL {canal.upper()} (solo formato y estilo; no copies sus "
               f"mensajes ni sus datos):\n{ejemplos}")
    fila = fila_del_canal(canal)
    if fila:   # la regla del canal, completa y desde el primer borrador (los fragmentos del RAG pueden cortarla)
        entrada += f"\n\nREGLAS DEL CANAL {canal.upper()} (fila de la tabla de la guía): {fila}"
    if canal in NOTAS_DEL_CANAL:
        entrada += f"\n{NOTAS_DEL_CANAL[canal]}"
    if "recuperaci" in campos_del_brief(estado["brief"]).get("SITUACION", "").lower():
        entrada += f"\n\n{NOTA_RECUPERACION}"
    rotulos = rotulos_del_canal(canal)
    if rotulos:   # sale del historial: todas las piezas del canal usan estas líneas
        lineas = " y ".join(f"'{r}:'" for r in rotulos)
        entrada += (f"\n\nFORMATO OBLIGATORIO DEL CANAL {canal.upper()}: dentro del campo TEXTO, la pieza "
                    f"lleva las líneas {lineas}, cada una en su renglón.")
    if estado.get("observaciones") and not estado.get("aprobada", True):
        entrada += (f"\n\nTU BORRADOR ANTERIOR:\nTITULO: {estado['titulo']}\nTEXTO: {estado['texto']}\n\n"
                    f"OBSERVACIONES DE LA REVISIÓN (corregí todas):\n{estado['observaciones']}")

    pieza = limpiar(preguntar(agente_redactor, entrada))
    n = estado.get("borradores", 0) + 1
    return {
        "titulo": extraer_campo(pieza, "TITULO", ["TEXTO"]),
        "texto": extraer_campo(pieza, "TEXTO", ["IMAGEN SUGERIDA"]),
        "imagen": extraer_campo(pieza, "IMAGEN SUGERIDA"),
        "borradores": n,
        "pasos": [{"titulo": f"2. Redactor (borrador {n})", "contenido": pieza}],
    }


def nodo_revisor(estado: EstadoMarketing) -> dict:
    # El revisor lee la guía COMPLETA (es corta) para controlar todas las reglas, y recibe
    # solo la pieza y el pedido: no ve el historial ni el brief.
    medidas = medir_pieza(estado["texto"], estado["canal"])
    situacion = campos_del_brief(estado.get("brief", "")).get("SITUACION", "")
    controles, fallas, parecida = controles_de_codigo(estado["texto"], estado["pedido"], estado["canal"], situacion,
                                                      estado.get("titulo", ""))
    entrada = (f"GUÍA DE MARCA COMPLETA:\n{guia_completa()}\n\n"
               f"========\n"
               f"CANAL: {estado['canal']}\nSEGMENTO: {estado['segmento']}\n"
               f"REGLAS DE ESTE CANAL (fila de la tabla de la guía): {fila_del_canal(estado['canal']) or 'no está en la tabla'}\n"
               f"PEDIDO ORIGINAL: {estado['pedido']}\n\n"
               f"PIEZA A REVISAR:\nTITULO: {estado['titulo']}\nTEXTO:\n{estado['texto']}\n\n"
               f"MEDIDAS Y CONTROLES DE LA PIEZA (calculados con código, son exactos):\n{medidas}\n{controles}")
    if fallas:
        # El código ya rechazó la pieza: no se llama al revisor LLM (con hasta MAX_BORRADORES
        # vueltas eran tiempo y tokens de más). El redactor corrige esto primero; el checklist
        # se revisa cuando la pieza pase los controles de código.
        aprobada = False
        observaciones = "\n".join("- " + f for f in fallas)
        revision = "(No se llamó al revisor: la pieza no pasó los controles de código.)"
    else:
        revision = limpiar(preguntar(agente_revisor, entrada))
        aprobada, observaciones = leer_revision(revision)
    copiados = list(estado.get("textos_copiados") or [])
    if parecida["similitud"] >= UMBRAL_COPIA:
        copiados.append(parecida["texto"])
    return {
        "aprobada": aprobada,
        "observaciones": observaciones,
        "textos_copiados": copiados,
        "pasos": [{"titulo": f"3. Revisor de marca (borrador {estado['borradores']})",
                   "contenido": "\n".join([medidas, controles, "", revision]
                                          + (["", "CONTROLES DE CÓDIGO: rechazada", *fallas] if fallas else []))}],
    }


# "Cumple parcialmente" o "cumple a medias" no es CUMPLE: se toma como NO CUMPLE
MARCAS = re.compile(r"\b(NO\s+CUMPLE|NO\s+APLICA|CUMPLE\s+(?:PARCIAL\w*|A\s+MEDIAS|EN\s+PARTE)|CUMPLE)\b",
                    flags=re.IGNORECASE)
# "1.", "1)", "- 1.", "**1.**", "| 1 |", "Punto 1:": el revisor no siempre numera igual
NUMERO_DE_PUNTO = re.compile(r"(?:punto\s*)?(\d+)\s*[.):|\-–]\s*(.*)", flags=re.IGNORECASE)


def leer_checklist(revision: str) -> tuple[dict[int, str], list[str]]:
    """Lee el bloque CHECKLIST: {número de punto: marca} y los renglones marcados NO CUMPLE.
    Solo cuenta la PRIMERA marca de cada renglón (así "NO CUMPLE" no se lee como CUMPLE). Acepta
    "Cumple" en minúscula y renglones con viñetas o negritas: rechazar una pieza válida por cómo
    numeró el revisor era un rechazo espurio."""
    bloque = extraer_campo(revision, "CHECKLIST", ["VEREDICTO"])
    marcas, fallidos = {}, []
    for linea in bloque.splitlines():
        m = NUMERO_DE_PUNTO.match(re.sub(r"^[\s\-*•#>|]+", "", linea).replace("**", ""))
        if not m:
            continue
        marca = MARCAS.search(m.group(2))
        valor = re.sub(r"\s+", " ", marca.group(1)).upper() if marca else ""
        if valor.startswith("CUMPLE "):
            valor = "NO CUMPLE"
        marcas.setdefault(int(m.group(1)), valor)
        if not marca or valor == "NO CUMPLE":
            fallidos.append(linea.strip())
    return marcas, fallidos


def leer_revision(revision: str) -> tuple[bool, str]:
    """Interpreta la respuesta del revisor. Falla cerrado: solo se aprueba si
    - el veredicto es exactamente APROBADA ("NO APROBADA", "APROBADA con cambios" o un
      veredicto vacío se toman como rechazo), y
    - el checklist trae los PUNTOS_CHECKLIST puntos, todos con CUMPLE o NO APLICA.
    Si rechaza sin observaciones, arma una para que el redactor sepa qué corregir y el usuario
    no vea observaciones vacías."""
    veredicto = extraer_campo(revision, "VEREDICTO").upper().strip(" .!¡:")
    marcas, fallidos = leer_checklist(revision)
    faltan = [str(n) for n in range(1, PUNTOS_CHECKLIST + 1) if n not in marcas]
    if veredicto == "APROBADA" and not fallidos and not faltan:
        return True, ""

    observaciones = extraer_campo(revision, "OBSERVACIONES", ["VEREDICTO", "CHECKLIST"])
    if observaciones.strip().lower().strip(" .") in ("", "ninguna"):
        observaciones = ""
    extra = []
    if veredicto not in ("APROBADA", "RECHAZADA"):
        extra.append(f"El revisor no devolvió un veredicto válido (APROBADA o RECHAZADA): '{veredicto or 'vacío'}'.")
    if fallidos:
        extra.append("Puntos del checklist que no cumplen:\n" + "\n".join(fallidos))
    if faltan:
        extra.append(f"El revisor no respondió los puntos {', '.join(faltan)} del checklist.")
    if not observaciones and not fallidos:
        extra.append("Revisá la pieza contra todas las reglas de la guía y corregí lo que no cumpla.")
    return False, "\n".join(([observaciones] if observaciones else []) + extra)


def nodo_guardar(estado: EstadoMarketing) -> dict:
    pedido_guardar = estado.get("guardar")   # True / False / None (ver pide_guardar)
    if not estado.get("aprobada"):
        msg = "No se guardó: la pieza no pasó la revisión de marca."
    elif pedido_guardar is False:
        msg = "No se guardó (el pedido pidió no guardarla)."
    elif not (pedido_guardar or GUARDAR_SIEMPRE):
        msg = "No se guardó (el pedido no lo pidió)."
    else:
        msg = guardar_pieza.invoke({"canal": estado["canal"], "segmento": estado["segmento"],
                                    "titulo": estado["titulo"], "texto": estado["texto"]})
    return {"resultado_guardado": msg, "pasos": [{"titulo": "4. Guardado", "contenido": msg}]}


# --------------------------------------------------------------------------
# Aristas condicionales
# --------------------------------------------------------------------------

def despues_del_estratega(estado: EstadoMarketing) -> str:
    return "fin" if estado.get("fuera_de_alcance") or estado.get("sin_canal") else "redactor"


def despues_del_revisor(estado: EstadoMarketing) -> str:
    if estado["aprobada"] or estado["borradores"] >= MAX_BORRADORES:
        return "guardar"
    return "redactor"   # vuelve a escribir con las observaciones


grafo = StateGraph(EstadoMarketing)
grafo.add_node("estratega", nodo_estratega)
grafo.add_node("redactor", nodo_redactor)
grafo.add_node("revisor", nodo_revisor)
grafo.add_node("guardar", nodo_guardar)
grafo.add_edge(START, "estratega")
grafo.add_conditional_edges("estratega", despues_del_estratega, {"redactor": "redactor", "fin": END})
grafo.add_edge("redactor", "revisor")
grafo.add_conditional_edges("revisor", despues_del_revisor, {"redactor": "redactor", "guardar": "guardar"})
grafo.add_edge("guardar", END)
flujo_marketing = grafo.compile()


# --------------------------------------------------------------------------
# Puntos de entrada
# --------------------------------------------------------------------------

# Pedir guardar la pieza: el verbo tiene que tener a la pieza como objeto. Con "guard" a secas,
# "guardarropas" o "guardá tus cosas" contaban como pedido de guardado.
_GUARDAR_PIEZA = (
    r"gu[aá]rd(?:[aá]|ar|e|en)l[oa]s?\b"                            # guardalo, guárdala, guardarlo
    r"|l[oa]s?\s+gu[aá]rd(?:[aá]s|es|en|ar|amos|emos)\b"            # lo guardás, la guardemos
    r"|gu[aá]rd\w*\s+(?:\w+\s+){0,3}?en\s+el\s+(?:historial|csv)\b"  # guardá la pieza en el historial
)
GUARDAR_SI = re.compile(rf"\b(?:{_GUARDAR_PIEZA})")
# "no lo guardes", "sin guardarla", "no hace falta guardarlo" (pero no "tips para no guardar ropa")
# Con GUARDAR_SIEMPRE = True, un "no guardar" que no se reconoce termina guardando la pieza, así
# que la negación acepta más formas que el pedido de guardar: "no guardar", "no guardes la pieza",
# "que no se guarde", "no la agregues al historial", "no hace falta que la guardes".
_NO_GUARDAR = (
    _GUARDAR_PIEZA
    + r"|se\s+gu[aá]rd\w*"                                                       # que no se guarde
    + r"|gu[aá]rd\w*(?=\s*(?:[.,;:!?)]|$|nada\b|(?:la|el|esta|esa|este|ese)\s+"
      r"(?:pieza|mensaje|post|publicaci[oó]n|e-?mail|mail|push|texto|borrador)\b))"  # no guardar.
    # solo imperativo, subjuntivo o infinitivo ("no la agregues", "no registrar"), no "no registran"
    + r"|(?:l[oa]s?\s+)?(?:agreg(?:ues|uen|á|ar\w*)|sum(?:es|en|á|ar\w*)|registr(?:es|en|á|ar\w*)"
      r"|carg(?:ues|uen|á|ar\w*))\s+(?:\w+\s+){0,2}?(?:al|en\s+el)\s+(?:historial|csv)\b"
)
GUARDAR_NO = re.compile(rf"\b(?:no|sin|nunca)\s+(?:\w+\s+){{0,3}}?(?:{_NO_GUARDAR})", flags=re.MULTILINE)


def pide_guardar(pedido: str) -> bool | None:
    """True si el pedido pide guardar la pieza, False si pide explícitamente NO guardarla,
    None si no dice nada (ahí decide GUARDAR_SIEMPRE)."""
    texto = pedido.lower()
    if GUARDAR_NO.search(texto):
        return False
    return True if GUARDAR_SI.search(texto) else None


def formatear_respuesta(estado: EstadoMarketing) -> str:
    if estado.get("fuera_de_alcance"):
        return ("Solo puedo ayudarte a crear piezas de comunicación para clientes de Hornero "
                "(posts de Instagram o LinkedIn, emails, mensajes de WhatsApp o push).")
    if estado.get("sin_canal"):
        return (f"No pude identificar el canal de la pieza. Decime en cuál de estos canales la "
                f"querés: {', '.join(canales_disponibles())}.")
    texto_md = estado["texto"].replace("\n", "  \n")   # en markdown, "  \n" fuerza el salto de línea
    partes = [f"**Canal:** {estado['canal']} · **Segmento:** {estado['segmento']}",
              f"**Título interno:** {estado['titulo']}",
              texto_md,
              f"*Imagen sugerida:* {estado['imagen']}"]
    if estado.get("aprobada"):
        partes.append(f"✅ Aprobada por la revisión de marca ({estado['borradores']} borrador/es).")
    else:
        obs = estado.get("observaciones", "").replace("\n", "  \n")
        partes.append(f"⚠️ No pasó la revisión de marca después de {estado['borradores']} borradores. "
                      f"Observaciones pendientes:  \n{obs}")
    partes.append(estado.get("resultado_guardado", ""))
    return "\n\n".join(partes)


def ejecutar_marketing(pedido: str) -> dict:
    """Corre el flujo completo. Devuelve {"respuesta", "detalle", "estado", "uso"}.
    "uso" tiene llamadas al LLM, tokens de entrada y salida, y segundos de punta a punta.
    La etapa 2 la reutiliza para integrar Marketing como un área más."""
    contador_uso.reiniciar()
    inicio = time.perf_counter()
    estado = flujo_marketing.invoke({"pedido": pedido, "guardar": pide_guardar(pedido), "pasos": []})
    uso = {"llamadas": contador_uso.llamadas, "tokens_entrada": contador_uso.entrada,
           "tokens_salida": contador_uso.salida, "segundos": round(time.perf_counter() - inicio, 1)}
    return {"respuesta": formatear_respuesta(estado), "detalle": estado.get("pasos", []),
            "estado": estado, "uso": uso}


def describir_uso(uso: dict) -> str:
    return (f"{uso['segundos']:.1f} segundos · {uso['llamadas']} llamadas al LLM · "
            f"{uso['tokens_entrada'] + uso['tokens_salida']} tokens "
            f"({uso['tokens_entrada']} de entrada, {uso['tokens_salida']} de salida)")


def responder(pregunta: str) -> dict:
    """Punto de entrada que usa main.py: {"respuesta": str, "detalle": [{"titulo", "contenido"}]}."""
    inicio = time.perf_counter()
    try:
        resultado = ejecutar_marketing(pregunta)
        respuesta, detalle = resultado["respuesta"], resultado["detalle"]
        uso = describir_uso(resultado["uso"])
    except Exception as e:  # que la app muestre el error en vez de romperse
        respuesta, detalle = f"Ocurrió un error al procesar el pedido: {e}", []
        uso = f"{time.perf_counter() - inicio:.1f} segundos (con error)"
    detalle.append({"titulo": "⏱ Tiempo y uso", "contenido": uso})
    return {"respuesta": respuesta, "detalle": detalle}