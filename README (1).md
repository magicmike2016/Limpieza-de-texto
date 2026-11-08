# Procesamiento de lenguaje natural: Don Quijote de la Mancha

Proyecto de NLP sobre un libro completo en español: carga de datos, limpieza
y tres formas de vectorizar el texto.

**Libro:** *Don Quijote de la Mancha*, Miguel de Cervantes (dominio público,
obtenido de Project Gutenberg).

## Flujo del proyecto
| Etapa | Archivo | Qué hace |
|---|---|---|
| Carga + Limpieza | `limpieza.py` | Tokenización, stop words, lematización y comparativa Stemming vs Lematización |
| Vectorización 1 | `vectorizacion.py` | Bag of Words y TF-IDF, proyectados en 3D con PCA |
| Vectorización 2 | `semantica.py` | Semántica distribucional con Word2Vec: vecinos, analogías y similitud |

## Archivos de datos y resultados
| Archivo | Contenido |
|---|---|
| `libro.txt` | Libro original |
| `libro_limpio.txt` | Texto procesado: lemas normalizados |
| `corpus_oraciones.txt` | Texto procesado: una oración lematizada por línea |
| `requirements.txt` | Dependencias congeladas con `pip freeze` |
| `vectorizacion_3d.png` | Espacios BoW vs TF-IDF |
| `word2vec_pca_2d.png` | Mapa de palabras y sus vecinos semánticos |
| `word2vec_similitud.png` | Mapa de calor de similitud coseno |
| `top_palabras_bow_tfidf.csv`, `comparativa_stem_lema.csv` | Tablas de apoyo |

## Cómo reproducirlo
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python limpieza.py
python vectorizacion.py
python semantica.py
```
