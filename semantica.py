"""
Vectorización con semántica distribucional (Word2Vec) sobre el libro.

"Una palabra se conoce por las compañías que frecuenta" (J.R. Firth):
Word2Vec aprende un vector denso por palabra a partir de sus contextos,
de modo que palabras usadas en contextos parecidos quedan cerca.

Uso:
    python semantica.py                 # usa libro.txt
    python semantica.py otro_libro.txt

Salidas:
    corpus_oraciones.txt         -> texto procesado: una oración lematizada por línea
    word2vec_libro.model         -> modelo entrenado
    word2vec_pca_2d.png          -> mapa 2D de palabras y sus vecinos semánticos
    word2vec_similitud.png       -> mapa de calor de similitud coseno
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

ARCHIVO = sys.argv[1] if len(sys.argv) > 1 else "libro.txt"
CORPUS = "corpus_oraciones.txt"

# Palabras a explorar (en forma de lema). Ajústalas a tu libro.
PALABRAS_CLAVE = ["quijote", "sancho", "dulcinea", "caballero", "escudero",
                  "rocinante", "amor", "batalla", "castillo", "comer"]

# ---------------------------------------------------------
# 1. CARGA + LIMPIEZA -> lista de oraciones tokenizadas
#    (se guarda en corpus_oraciones.txt para no repetir spaCy)
# ---------------------------------------------------------
if os.path.exists(CORPUS):
    print(f"Usando corpus ya procesado: {CORPUS}")
else:
    import spacy
    try:
        nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except OSError:
        from spacy.cli import download
        download("es_core_news_sm")
        nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    if "senter" in nlp.disabled:
        nlp.enable_pipe("senter")
    else:
        nlp.add_pipe("sentencizer")

    with open(ARCHIVO, "r", encoding="utf-8") as f:
        texto = f.read()
    inicio = re.search(r"\*\*\* ?START OF.*?\*\*\*", texto)
    fin = re.search(r"\*\*\* ?END OF.*?\*\*\*", texto)
    if inicio and fin:
        texto = texto[inicio.end():fin.start()]
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]

    print("Lematizando oraciones con spaCy (tarda unos minutos)...")
    with open(CORPUS, "w", encoding="utf-8") as out:
        for doc in nlp.pipe(parrafos, batch_size=64):
            for oracion in doc.sents:
                lemas = [t.lemma_.lower() for t in oracion
                         if t.is_alpha and not t.is_stop and not t.is_punct]
                if len(lemas) > 1:
                    out.write(" ".join(lemas) + "\n")

with open(CORPUS, "r", encoding="utf-8") as f:
    oraciones = [linea.split() for linea in f if linea.strip()]
print(f"Oraciones: {len(oraciones):,} | Tokens: {sum(map(len, oraciones)):,}")

# ---------------------------------------------------------
# 2. ENTRENAMIENTO WORD2VEC
# ---------------------------------------------------------
modelo = Word2Vec(
    sentences=oraciones,
    vector_size=100,   # dimensiones de cada vector
    window=5,          # palabras de contexto a cada lado
    min_count=5,       # ignora palabras con menos de 5 apariciones
    sg=1,              # Skip-gram: funciona mejor en corpus pequeños
    epochs=20,
    workers=4,
    seed=42,
)
modelo.save("word2vec_libro.model")
wv = modelo.wv
print(f"Vocabulario aprendido: {len(wv):,} palabras, vectores de {wv.vector_size} dimensiones")

claves = [p for p in PALABRAS_CLAVE if p in wv]
faltan = set(PALABRAS_CLAVE) - set(claves)
if faltan:
    print(f"(No están en el vocabulario y se omiten: {sorted(faltan)})")

# ---------------------------------------------------------
# 3. EXPLORACIÓN SEMÁNTICA
# ---------------------------------------------------------
print("\n--- Vecinos más cercanos (similitud coseno) ---")
for p in claves:
    vecinos = ", ".join(f"{w} ({s:.2f})" for w, s in wv.most_similar(p, topn=6))
    print(f"{p:>10}: {vecinos}")

print("\n--- Similitud entre pares ---")
pares = [("quijote", "sancho"), ("quijote", "caballero"), ("sancho", "escudero"),
         ("dulcinea", "amor"), ("caballero", "comer")]
for a, b in pares:
    if a in wv and b in wv:
        print(f"{a} ~ {b}: {wv.similarity(a, b):.3f}")

print("\n--- Analogía: quijote - caballero + escudero = ? ---")
if all(w in wv for w in ["quijote", "caballero", "escudero"]):
    print(wv.most_similar(positive=["quijote", "escudero"], negative=["caballero"], topn=3))

print("\n--- ¿Cuál no encaja? ---")
grupo = [w for w in ["sancho", "quijote", "dulcinea", "comer"] if w in wv]
if len(grupo) >= 3:
    print(f"{grupo} -> {wv.doesnt_match(grupo)}")

# ---------------------------------------------------------
# 4. GRÁFICA 1: mapa 2D (PCA) de palabras clave + vecinos
# ---------------------------------------------------------
semillas = claves[:6]
colores = plt.cm.tab10.colors
palabras, grupos = [], []
for g, s in enumerate(semillas):
    for w in [s] + [v for v, _ in wv.most_similar(s, topn=7)]:
        if w not in palabras:
            palabras.append(w)
            grupos.append(g)

coords = PCA(n_components=2, random_state=42).fit_transform(np.array([wv[w] for w in palabras]))

fig, ax = plt.subplots(figsize=(13, 9))
for (x, y), w, g in zip(coords, palabras, grupos):
    es_semilla = w in semillas
    ax.scatter(x, y, color=colores[g], s=220 if es_semilla else 70,
               edgecolors="k", alpha=0.85, zorder=3)
    ax.annotate(w, (x, y), xytext=(5, 4), textcoords="offset points",
                fontsize=12 if es_semilla else 9,
                fontweight="bold" if es_semilla else "normal")
for g, s in enumerate(semillas):
    ax.scatter([], [], color=colores[g], label=s, s=80, edgecolors="k")
ax.legend(title="Palabra semilla", loc="best")
ax.axhline(0, c="grey", lw=0.5, ls="--")
ax.axvline(0, c="grey", lw=0.5, ls="--")
ax.set_title("Word2Vec: palabras clave y sus vecinos semánticos (PCA 2D)")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
plt.tight_layout()
plt.savefig("word2vec_pca_2d.png", dpi=150)
plt.close()
print("\nGráfica guardada en word2vec_pca_2d.png")

# ---------------------------------------------------------
# 5. GRÁFICA 2: mapa de calor de similitud coseno
# ---------------------------------------------------------
n = len(claves)
matriz = np.array([[wv.similarity(a, b) for b in claves] for a in claves])

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(matriz, cmap="viridis", vmin=matriz.min(), vmax=1)
ax.set_xticks(range(n), claves, rotation=45, ha="right")
ax.set_yticks(range(n), claves)
for i in range(n):
    for j in range(n):
        ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center",
                color="white" if matriz[i, j] < 0.6 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="Similitud coseno")
ax.set_title("Word2Vec: similitud entre palabras clave")
plt.tight_layout()
plt.savefig("word2vec_similitud.png", dpi=150)
plt.close()
print("Gráfica guardada en word2vec_similitud.png")
