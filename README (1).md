# Limpieza de texto: Don Quijote de la Mancha

Proyecto de NLP que aplica tokenización, filtrado de stop words, lematización
y una comparativa Stemming vs Lematización sobre un libro completo en español.

**Libro:** *Don Quijote de la Mancha*, Miguel de Cervantes (dominio público,
obtenido de Project Gutenberg).

## Archivos
| Archivo | Contenido |
|---|---|
| `limpieza.py` | Proceso de limpieza (normalización y lematización) |
| `requirements.txt` | Dependencias congeladas con `pip freeze` |
| `libro.txt` | Libro a procesar |
| `libro_limpio.txt` | Resultado: lemas normalizados |
| `comparativa_stem_lema.csv` | Tabla Original / Stem / Lema |

## Cómo reproducirlo
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python limpieza.py
```
