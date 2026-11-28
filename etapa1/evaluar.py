"""
evaluar.py: evaluación de la etapa 1 con varias corridas por consulta.

Un agente no se evalúa con una sola corrida: el mismo pedido puede salir bien una vez y mal
la siguiente. medir() repite cada consulta de prueba y cuenta en cuántas corridas funciona,
con las llamadas al LLM, los tokens y los segundos de cada una (ContadorUso de agentes.py).

Cómo correrlo (desde esta carpeta, con el venv activado):
    python evaluar.py                               # las 4 consultas de la consigna, 3 corridas cada una
    python evaluar.py --corridas 5 --consultas 3 4  # solo algunas consultas, con otra cantidad de corridas
    python evaluar.py --consultas fuera             # el caso fuera de alcance (el Mundial): funciona si lo rechaza
    python evaluar.py --consultas e1 e2 e3 --corridas 2   # pedidos que no están en la consigna

Las consultas de la consigna (1 a 4) se guardan en evaluacion_etapa1.csv, que se pisa en cada
corrida. Las demás van a evaluaciones/<consultas>.csv, que no se sube al repo (.gitignore).

"Funcionó" quiere decir que la pieza quedó aprobada (por el revisor y por los controles de
código). Aprobada no es lo mismo que bien escrita: el CSV de salida guarda el texto de cada
pieza y las observaciones, para leerlas y elegir casos buenos y malos para el informe.

Cada corrida parte del mismo historial: el CSV se restaura después de cada una, así una pieza
guardada en la corrida 1 no cambia lo que ve la corrida 2 (ni queda en el entregable).
Ojo con el límite diario de tokens de Groq: 4 consultas x 3 corridas consumen buena parte.
"""

import argparse
import re
import subprocess
import time
from pathlib import Path

import pandas as pd

from agentes import ejecutar_marketing
from tools import RUTA_CONTENIDOS

# Funciona = la pieza se aprobó (veredicto APROBADA, checklist completo y sin fallas de los
# controles por código). Para el caso fuera de alcance, funciona si el sistema lo rechaza.
CONSULTAS = {
    "1": ("Armá un post de Instagram para clientes Premium anunciando la nueva colección de "
          "muebles de exterior.", "aprobada"),
    "2": ("Escribí un email para recuperar a clientes que tuvieron problemas con una entrega.",
          "aprobada"),
    "3": ("Proponé un mensaje de WhatsApp para clientes Nuevos con un 15% de descuento en Cocina "
          "y guardalo.", "aprobada"),
    "4": ("Armá un push para clientes Recurrentes sobre aires acondicionados que no repita lo que "
          "ya les mandamos.", "aprobada"),
    "fuera": ("¿Quién va a ganar el Mundial?", "fuera_de_alcance"),
    # Pedidos que NO están en la consigna, para ver que los controles no rechazan de más con
    # canales, segmentos y datos distintos (LinkedIn, "Todos", envío sin cargo con fecha, tips)
    "e1": ("Escribí un post de LinkedIn contando que abrimos un nuevo centro de distribución en "
           "Neuquén.", "aprobada"),
    "e2": ("Armá un email para clientes Premium con envío sin cargo en compras de Living hasta el "
           "15/12.", "aprobada"),
    "e3": ("Hacé un post de Instagram para todos los clientes con tips para ordenar el placard.",
           "aprobada"),
}
ESPERA = 30   # segundos entre corridas, por el límite de tokens por minuto de Groq
ESPERA_MAX_429 = 600   # si Groq pide esperar más (límite diario), la corrida se anota como error


def espera_429(error: Exception) -> float | None:
    """Segundos que pide Groq en un 429 ("Please try again in 1m23.5s"); None si no es un 429."""
    texto = str(error)
    if "429" not in texto:
        return None
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", texto)
    if not m or not any(m.groups()):
        return 60.0
    h, mi, se = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + se + 5


def describir_error(error: Exception) -> str:
    """El error en una línea; en un 429 dice qué límite de Groq se alcanzó (por minuto o por día)."""
    texto = str(error)
    limite = re.search(r"\((T[PR][MD])\): Limit (\d+), Used (\d+)", texto)
    if limite:
        return f"error 429: límite {limite.group(1)} de Groq ({limite.group(3)}/{limite.group(2)} usados)"
    return f"error: {texto[:100]}"


def ejecutar_con_reintentos(pedido: str, intentos: int = 3) -> dict:
    """ejecutar_marketing, reintentando si Groq corta por límite de tokens por minuto."""
    for intento in range(intentos):
        try:
            return ejecutar_marketing(pedido)
        except Exception as e:
            espera = espera_429(e)
            if espera is None or espera > ESPERA_MAX_429 or intento == intentos - 1:
                raise
            print(f"  (Groq pidió esperar: {espera:.0f} s por el límite de tokens)")
            time.sleep(espera)


def version_del_codigo() -> str:
    """Commit con el que se corre la evaluación ("fd07ffc"; con "+cambios" si hay cambios sin
    commitear en el código), para que cada fila del CSV diga de qué versión sale."""
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    commit = git("rev-parse", "--short", "HEAD") or "sin git"
    cambios = git("status", "--porcelain", "--", "*.py")
    return commit + ("+cambios" if cambios else "")


def funciono(estado: dict, esperado: str) -> bool:
    if esperado == "fuera_de_alcance":
        return bool(estado.get("fuera_de_alcance"))
    return bool(estado.get("aprobada"))


def motivo(estado: dict) -> str:
    """Por qué no funcionó, en una línea (la primera observación del revisor)."""
    if estado.get("fuera_de_alcance"):
        return "fuera de alcance"
    if estado.get("sin_canal"):
        return "no se reconoció el canal"
    lineas = [l.strip(" -*") for l in str(estado.get("observaciones", "")).splitlines() if l.strip(" -*")]
    return lineas[0][:120] if lineas else ""


def medir(nombre: str, pedido: str, esperado: str, corridas: int = 3, espera: int = ESPERA,
          al_terminar_corrida=None) -> list[dict]:
    """Corre el mismo pedido varias veces y devuelve una fila por corrida. al_terminar_corrida(fila)
    se llama después de cada una, para guardarla enseguida: si la evaluación se corta (límite diario
    de Groq, Ctrl+C), las corridas que ya terminaron no se pierden."""
    print(f"\nConsulta {nombre}: {pedido}")
    csv_original = Path(RUTA_CONTENIDOS).read_bytes()
    filas = []
    for i in range(corridas):
        try:
            resultado = ejecutar_con_reintentos(pedido)
            estado, uso = resultado["estado"], resultado["uso"]
            ok = funciono(estado, esperado)
            fila = {"Consulta": nombre, "Corrida": i + 1, "Funcionó": ok, "Error": False,
                    "Borradores": estado.get("borradores", 0),
                    "Llamadas al LLM": uso["llamadas"], "Tokens entrada": uso["tokens_entrada"],
                    "Tokens salida": uso["tokens_salida"], "Segundos": uso["segundos"],
                    "Motivo": "" if ok else motivo(estado),
                    # La pieza y la revisión, para leerlas: aprobar no es lo mismo que estar bien
                    "Canal": estado.get("canal", ""), "Segmento": estado.get("segmento", ""),
                    "Título": estado.get("titulo", ""), "Texto": estado.get("texto", ""),
                    "Observaciones": estado.get("observaciones", "")}
        except Exception as e:   # p. ej. el límite diario de Groq: se registra y se sigue
            # Una corrida con error no dice nada del sistema: no cuenta en la tasa de éxito
            fila = {"Consulta": nombre, "Corrida": i + 1, "Funcionó": False, "Error": True, "Borradores": 0,
                    "Llamadas al LLM": 0, "Tokens entrada": 0, "Tokens salida": 0, "Segundos": 0.0,
                    "Motivo": describir_error(e), "Canal": "", "Segmento": "", "Título": "", "Texto": "",
                    "Observaciones": ""}
        finally:
            Path(RUTA_CONTENIDOS).write_bytes(csv_original)
        filas.append(fila)
        if al_terminar_corrida:
            al_terminar_corrida(fila)
        estado_txt = ("ERROR, no cuenta" if fila["Error"] else "funcionó" if fila["Funcionó"]
                      else "NO funcionó") + (f" ({fila['Motivo']})" if fila["Motivo"] else "")
        print(f"  Corrida {i + 1}: {estado_txt} · {fila['Borradores']} borrador/es · "
              f"{fila['Llamadas al LLM']} llamadas · "
              f"{fila['Tokens entrada'] + fila['Tokens salida']} tokens · {fila['Segundos']} s")
        if i < corridas - 1:
            time.sleep(espera)
    validas = [f for f in filas if not f["Error"]]
    aciertos = sum(f["Funcionó"] for f in validas)
    errores = f" ({corridas - len(validas)} con error, no cuentan)" if len(validas) < corridas else ""
    print(f"→ {aciertos}/{len(validas)} corridas funcionaron{errores}")
    return filas


def resumir(filas: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(filas)
    errores = df.groupby("Consulta", sort=False)["Error"].sum()
    df = df[~df["Error"]]   # los promedios y la tasa salen solo de las corridas que terminaron
    df["Tokens totales"] = df["Tokens entrada"] + df["Tokens salida"]
    resumen = df.groupby("Consulta", sort=False).agg(
        Corridas=("Corrida", "count"), Funcionaron=("Funcionó", "sum"),
        **{"Borradores (prom.)": ("Borradores", "mean"),
           "Llamadas (prom.)": ("Llamadas al LLM", "mean"),
           "Tokens (prom.)": ("Tokens totales", "mean"),
           "Segundos (prom.)": ("Segundos", "mean"),
           "Segundos (máx.)": ("Segundos", "max")})
    resumen = resumen.reindex(errores.index)
    resumen["Corridas"] = resumen["Corridas"].fillna(0).astype(int)
    resumen["Funcionaron"] = resumen["Funcionaron"].fillna(0).astype(int)
    resumen["Con error"] = errores.astype(int)
    resumen["Tasa de éxito"] = [f"{f / c:.0%}" if c else "-" for f, c in zip(resumen["Funcionaron"], resumen["Corridas"])]
    return resumen.round(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa la etapa 1 con varias corridas por consulta.")
    parser.add_argument("--corridas", type=int, default=3)
    parser.add_argument("--consultas", nargs="+", default=["1", "2", "3", "4"], choices=list(CONSULTAS))
    parser.add_argument("--espera", type=int, default=ESPERA)
    parser.add_argument("--salida", default=None,
                        help="CSV con una fila por corrida. Por defecto: evaluacion_etapa1.csv si son "
                             "consultas de la consigna; si no, evaluaciones/<consultas>.csv, para no "
                             "pisar los resultados de la consigna")
    args = parser.parse_args()
    if args.salida is None:
        de_la_consigna = all(c in {"1", "2", "3", "4"} for c in args.consultas)
        args.salida = ("evaluacion_etapa1.csv" if de_la_consigna
                       else f"evaluaciones/{'_'.join(args.consultas)}.csv")
    Path(args.salida).parent.mkdir(parents=True, exist_ok=True)
    version = version_del_codigo()
    print(f"Versión del código: {version} · salida: {args.salida}")

    filas = []
    guardar = lambda fila: (filas.append({"Versión": version, **fila}),
                            pd.DataFrame(filas).to_csv(args.salida, index=False))
    for n, nombre in enumerate(args.consultas):
        pedido, esperado = CONSULTAS[nombre]
        medir(nombre, pedido, esperado, args.corridas, args.espera, al_terminar_corrida=guardar)
        if n < len(args.consultas) - 1:
            time.sleep(args.espera)

    print("\nResumen:\n")
    print(resumir(filas).to_string())
    print(f"\nDetalle por corrida guardado en {args.salida}")
