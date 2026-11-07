"""
Vectorización de texto: Bag of Words vs TF-IDF sobre el libro completo.

Uso:
    python vectorizacion.py                 # usa libro.txt
    python vectorizacion.py otro_libro.txt

Salidas:
    vectorizacion_3d.png          -> palabras proyectadas en 3D (BoW vs TF-IDF)
    top_palabras_bow_tfidf.csv    -> ranking de palabras en cada representación
"""
import re
import sys

import matplotlib
matplotlib.use("Agg")  # permite guardar la gráfica aunque no haya pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "libro.txt"
N_PALABRAS_GRAFICA = 40  # cuántas palabras dibujar en el espacio 3D

# ---------------------------------------------------------
# 0. Modelo de spaCy con detector de oraciones
# ---------------------------------------------------------
try:
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
except OSError:
    from spacy.cli import download
    download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

# 'senter' separa oraciones sin el parser completo (mucho más rápido)
if "senter" in nlp.disabled:
    nlp.enable_pipe("senter")
else:
    nlp.add_pipe("sentencizer")

# ---------------------------------------------------------
# 1. Carga del texto (sin encabezado/licencia de Gutenberg)
# ---------------------------------------------------------
with open(ARCHIVO, "r", encoding="utf-8") as f:
    texto = f.read()

inicio = re.search(r"\*\*\* ?START OF.*?\*\*\*", texto)
fin = re.search(r"\*\*\* ?END OF.*?\*\*\*", texto)
if inicio and fin:
    texto = texto[inicio.end():fin.start()]

parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]

# ---------------------------------------------------------
# 2. Corpus lematizado: cada oración es un "documento"
# ---------------------------------------------------------
print("Procesando oraciones con spaCy (tarda unos minutos con un libro completo)...")
corpus_lematizado = []
for doc in nlp.pipe(parrafos, batch_size=64):
    for oracion in doc.sents:
        lemas_oracion = [
            token.lemma_.lower()
            for token in oracion
            if not token.is_punct and not token.is_space and not token.is_stop and token.is_alpha
        ]
        if lemas_oracion:
            corpus_lematizado.append(" ".join(lemas_oracion))

print(f"Total de oraciones procesadas: {len(corpus_lematizado):,}")
print("Ejemplo de oración lematizada:", corpus_lematizado[0])

# ---------------------------------------------------------
# 3. Vectorización completa
# ---------------------------------------------------------
# Bag of Words
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)

print("\n--- Matrices resultantes (oraciones x vocabulario) ---")
print(f"BoW:    {X_bow.shape}  | celdas no cero: {X_bow.nnz:,}")
print(f"TF-IDF: {X_tfidf.shape}  | celdas no cero: {X_tfidf.nnz:,}")
densidad = X_bow.nnz / (X_bow.shape[0] * X_bow.shape[1])
print(f"Densidad: {100 * densidad:.3f}%  -> matriz dispersa (casi todo es 0)")

# Ejemplo: cómo queda una oración en cada representación
i = 0
vocab = bow_vectorizer.get_feature_names_out()
fila_bow = X_bow[i].toarray().ravel()
fila_tfidf = X_tfidf[i].toarray().ravel()
idx = np.nonzero(fila_bow)[0]
print(f"\nOración {i}: '{corpus_lematizado[i]}'")
print(pd.DataFrame({
    "palabra": vocab[idx],
    "BoW (conteo)": fila_bow[idx],
    "TF-IDF (peso)": fila_tfidf[idx].round(3),
}).to_string(index=False))

# Ranking: palabras más frecuentes (BoW) vs más "importantes" (TF-IDF)
suma_bow = np.asarray(X_bow.sum(axis=0)).ravel()
suma_tfidf = np.asarray(X_tfidf.sum(axis=0)).ravel()
top_bow = vocab[suma_bow.argsort()[::-1][:20]]
top_tfidf = tfidf_vectorizer.get_feature_names_out()[suma_tfidf.argsort()[::-1][:20]]
ranking = pd.DataFrame({"Top BoW (frecuencia)": top_bow, "Top TF-IDF (importancia)": top_tfidf})
print("\n--- Top 20 palabras en cada representación ---")
print(ranking.to_string())
ranking.to_csv("top_palabras_bow_tfidf.csv", index_label="posición", encoding="utf-8")

# ---------------------------------------------------------
# 4. Visualización 3D (solo las N palabras más frecuentes)
# ---------------------------------------------------------
def graficar_palabras_3d(ax, matriz, vocabulario, titulo, color_puntos):
    # Transponer: filas = palabras, columnas = oraciones (contextos)
    matriz_palabras = matriz.T
    coords = PCA(n_components=3).fit_transform(matriz_palabras.toarray())
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    ax.scatter(x, y, z, c=color_puntos, s=80, edgecolors="k", alpha=0.8, depthshade=True)
    desplazamiento = (z.max() - z.min()) * 0.03
    for k, palabra in enumerate(vocabulario):
        ax.text(x[k], y[k], z[k] + desplazamiento, palabra, fontsize=9)

    ax.set_title(titulo)
    ax.set_xlabel("Comp. Principal 1")
    ax.set_ylabel("Comp. Principal 2")
    ax.set_zlabel("Comp. Principal 3")
    ax.plot([0, 0], [0, 0], [z.min(), z.max()], c="grey", ls="--", lw=0.5, alpha=0.3)
    ax.plot([x.min(), x.max()], [0, 0], [0, 0], c="grey", ls="--", lw=0.5, alpha=0.3)
    ax.plot([0, 0], [y.min(), y.max()], [0, 0], c="grey", ls="--", lw=0.5, alpha=0.3)


fig = plt.figure(figsize=(18, 8))

ax1 = fig.add_subplot(121, projection="3d")
bow_graf = CountVectorizer(max_features=N_PALABRAS_GRAFICA)
X_bow_graf = bow_graf.fit_transform(corpus_lematizado)
graficar_palabras_3d(ax1, X_bow_graf, bow_graf.get_feature_names_out(),
                     f"Espacio BoW 3D (Conteos) - top {N_PALABRAS_GRAFICA}", "orange")

ax2 = fig.add_subplot(122, projection="3d")
tfidf_graf = TfidfVectorizer(max_features=N_PALABRAS_GRAFICA)
X_tfidf_graf = tfidf_graf.fit_transform(corpus_lematizado)
graficar_palabras_3d(ax2, X_tfidf_graf, tfidf_graf.get_feature_names_out(),
                     f"Espacio TF-IDF 3D (Importancia) - top {N_PALABRAS_GRAFICA}", "teal")

plt.tight_layout()
plt.savefig("vectorizacion_3d.png", dpi=150)
print("\nGráfica guardada en vectorizacion_3d.png")
