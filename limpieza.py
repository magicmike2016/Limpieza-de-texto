"""
Limpieza de texto de un libro: tokenización, filtrado de stop words,
lematización/normalización y comparativa Stemming vs Lematización.

Uso:
    python limpieza.py                 # usa libro.txt
    python limpieza.py otro_libro.txt  # usa otro archivo

Salidas:
    libro_limpio.txt               -> texto normalizado (lemas en minúsculas)
    comparativa_stem_lema.csv      -> tabla Original / Stemming / Lema
"""
import re
import sys
from collections import Counter

import pandas as pd
import spacy
from nltk.stem import SnowballStemmer

ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "libro.txt"

# ---------------------------------------------------------------
# 0. Cargar modelo en español (lo descarga si no existe)
# ---------------------------------------------------------------
try:
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
except OSError:
    print("Descargando modelo...")
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

# ---------------------------------------------------------------
# Carga del texto
# ---------------------------------------------------------------
with open(ARCHIVO, "r", encoding="utf-8") as f:
    texto = f.read()

# Si el libro viene de Project Gutenberg, quitamos el encabezado y la licencia
inicio = re.search(r"\*\*\* ?START OF.*?\*\*\*", texto)
fin = re.search(r"\*\*\* ?END OF.*?\*\*\*", texto)
if inicio and fin:
    texto = texto[inicio.end():fin.start()]

print(f"Texto cargado con éxito. Longitud: {len(texto):,} caracteres.")

# Procesamos por párrafos con nlp.pipe: más rápido y evita el límite
# de 1,000,000 de caracteres que spaCy tiene por documento.
parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
docs = list(nlp.pipe(parrafos, batch_size=64))
tokens = [t for d in docs for t in d]

# ---------------------------------------------------------------
# 1. TOKENIZACIÓN
# ---------------------------------------------------------------
print(f"\n--- 1. Tokenización (Total tokens: {len(tokens):,}) ---")
print([t.text for t in tokens][:20])

# ---------------------------------------------------------------
# 2. FILTRADO DE STOP WORDS
# ---------------------------------------------------------------
def es_relevante(t):
    return not t.is_stop and not t.is_punct and not t.is_space and t.is_alpha

tokens_relevantes = [t.text for t in tokens if es_relevante(t)]
tokens_ruido = [t.text for t in tokens if t.is_stop]

print("\n--- 2. Filtrado de Stop Words ---")
print(f"Palabras eliminadas (Ruido): {tokens_ruido[:10]}...")
print(f"Palabras conservadas (Contenido): {tokens_relevantes[:10]}...")
print(f"Reducción de tamaño: de {len(tokens):,} a {len(tokens_relevantes):,} tokens "
      f"({100 * (1 - len(tokens_relevantes) / len(tokens)):.1f}% menos).")

# ---------------------------------------------------------------
# 3. LEMATIZACIÓN Y NORMALIZACIÓN FINAL
# ---------------------------------------------------------------
tokens_normalizados = []
cambios_interesantes = []

for t in tokens:
    if es_relevante(t):
        lema = t.lemma_.lower()
        tokens_normalizados.append(lema)
        if t.text.lower() != lema:
            cambios_interesantes.append(f"{t.text} ➡ {lema}")

print("\n--- 3. Lematización y Normalización ---")
print(f"Total de tokens procesados: {len(tokens_normalizados):,}")
print("Ejemplos de transformaciones (Palabra original ➡ Lema):")
print(cambios_interesantes[:10])
print("\nResultado final (Primeros 10 tokens):")
print(tokens_normalizados[:10])

print("\nVocabulario: "
      f"{len(set(w.lower() for w in tokens_relevantes)):,} palabras distintas antes de lematizar, "
      f"{len(set(tokens_normalizados)):,} después.")

print("\nLos 15 lemas más frecuentes:")
for lema, n in Counter(tokens_normalizados).most_common(15):
    print(f"  {lema:<15} {n:>6}")

with open("libro_limpio.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(tokens_normalizados))
print("\nTexto limpio guardado en libro_limpio.txt")

# ---------------------------------------------------------------
# 4. STEMMING VS LEMATIZACIÓN
# ---------------------------------------------------------------
stemmer = SnowballStemmer("spanish")

data_comparativa = []
for t in tokens:
    if not t.is_punct and not t.is_space:
        raiz_stem = stemmer.stem(t.text)
        lema = t.lemma_
        data_comparativa.append({
            "Original": t.text,
            "Stemming (Corte)": raiz_stem,
            "Lematización (Diccionario)": lema,
            "¿Coinciden?": raiz_stem == lema.lower(),
        })

df = pd.DataFrame(data_comparativa)

print("\n--- 4. Stemming vs Lematización ---")
# Ajusta esta lista con palabras que aparezcan en TU libro
palabras_interesantes = ["hombres", "caballero", "dijo", "fue", "eran", "ojos", "andantes"]
filtro = df[df["Original"].str.lower().isin(palabras_interesantes)].drop_duplicates("Original")
print(filtro.to_string(index=False))

print("\n--- Visualización completa de los primeros 10 tokens ---")
print(df.head(10).to_string(index=False))

print(f"\nCoincidencia stem = lema: {100 * df['¿Coinciden?'].mean():.1f}% de los tokens")

df.drop_duplicates("Original").to_csv("comparativa_stem_lema.csv", index=False, encoding="utf-8")
print("Comparativa guardada en comparativa_stem_lema.csv")
