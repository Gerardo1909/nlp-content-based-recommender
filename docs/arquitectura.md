# Especificación Arquitectónica: Recomendador de Películas Basado en Contenido

**Rol:** Arquitecto de ML y NLP
**Fecha:** 2026-05-17
**Versión:** 2.0

> **Sobre esta versión.** v2.0 reescribe la v1.0 incorporando la revisión crítica registrada en
> `docs/hallazgos_arquitectura.md` (rol Evaluador, veredicto v1.0: *NO ENTREGABLE*). Cierra los
> **3 bloqueantes** (protocolo comparativo A/B con criterio de decisión, normalización
> verificable en embeddings, reproducibilidad formal) e implementa los 6 hallazgos mayores y
> menores. El principio rector de la revisión: **ninguna regla heurística se presenta como
> verdad; se presenta como hipótesis con su protocolo de validación**. Las fórmulas se expresan
> como snippets de pseudocódigo numpy/sklearn (ilustrativos de la especificación, no
> implementación) en lugar de notación LaTeX, para claridad de lectura y de auditoría. La tabla
> de trazabilidad de la Sección 13 mapea cada hallazgo a la sección que lo resuelve.

---

## 1. Análisis del Corpus y sus Limitaciones

### 1.1 Campos disponibles y utilidad

| Campo | Utilidad | Observaciones |
|---|---|---|
| `description` | **Alta** — sinopsis en español, fuente principal de señal semántica | Puede ser nula o muy corta (<20 tokens); es la columna más informativa |
| `keywords` | **Media** — enriquece el vocabulario temático | Mezcla español/inglés, separadas por comas; ruido ortográfico posible |
| `genre` | **Media-baja** — categoría temática de grano grueso | Mezcla español/inglés; útil como complemento, no como señal principal |
| `year` | **Baja para representación de contenido** | Candidata a re-ranking, no a feature de texto (ver Sección 10) |
| `director` | **Baja para contenido** | Candidata a boosting por co-ocurrencia (ver Sección 10) |
| `name` | **Referencial** — clave de join con `user_profiles.csv` | No aporta señal de contenido |

**Decisión (Estrategia A):** la representación concatena `description`, `keywords` y `genre`.
**Decisión (Estrategia B):** el conjunto de campos a concatenar **no se fija a priori**: es la
variable de un experimento de ablation (ver §3.3 y Sección 7, hallazgo 4).

### 1.2 Limitaciones estructurales que condicionan el diseño

1. **Nulos en `description`:** películas sin sinopsis carecen de señal de contenido. Se les
   asigna un vector degenerado (vector cero L2 en TF-IDF; embedding de string vacío en B) y se
   marcan internamente como `low_confidence`; sus recomendaciones se deprioran (Sección 6.3).

2. **Idioma mixto en `keywords`/`genre`:** términos en inglés dentro de texto español rompen la
   coherencia léxica. Se normaliza mayúsculas pero **no se traduce** (la traducción automática
   introduce ruido propio no medible aquí). El modelo multilingüe de B maneja esto de forma
   nativa; en A, `min_df>=2` filtra hápax ruidosos.

3. **Coincidencia imperfecta de títulos (historial → corpus) — umbral a calibrar
   (hallazgo 5).** Los títulos en `pelicula_1..5` pueden no coincidir exactamente con `name`.
   Se aplica coincidencia difusa por similitud de cadenas (ratio de Levenshtein normalizado en
   `[0, 1]`). **El umbral NO se fija por intuición.** Protocolo de calibración obligatorio:

   ```python
   # PROTOCOLO DE CALIBRACIÓN DEL UMBRAL DE FUZZY MATCHING
   # 1. Tomar los 14*5 = 70 títulos del historial.
   # 2. Para cada título, calcular el mejor match contra `name` y su score.
   # 3. Muestrear manualmente ~30-50 pares (estratificando por bandas de score:
   #    alto >0.9, gris 0.75-0.9, bajo <0.75) y etiquetar match correcto/incorrecto.
   # 4. Barrer thresholds en {0.80, 0.85, 0.90, 0.95} y reportar precisión/recall:
   for thr in [0.80, 0.85, 0.90, 0.95]:
       preds = [score >= thr for score in scores_muestra]
       precision, recall = prf(y_true_manual, preds)   # tabla en el notebook
   # 5. Elegir el umbral que maximice recall sin que precisión caiga por debajo
   #    de un mínimo defendible (objetivo: precisión >= 0.95 — un falso match
   #    contamina el perfil del usuario, es más caro que un falso negativo).
   # 6. LOGGING OBLIGATORIO: registrar todo match con score en zona gris
   #    [thr-0.05, thr+0.05] en una tabla de auditoría por usuario.
   ```

   Punto de partida sugerido: `0.85`. **El valor final lo fija la evidencia del paso 5, no este
   documento.** Si un título no supera el umbral calibrado, se descarta del historial efectivo y
   queda trazado en el log.

4. **Distribución desigual de longitud de sinopsis:** sinopsis cortas producen vectores TF-IDF
   menos estables; el embedding de frase (B) es más robusto a longitud variable. Se reporta el
   histograma de longitudes en el EDA y la cobertura efectiva (% de películas con sinopsis
   utilizable).

5. **14 perfiles sin feedback explícito:** no hay valoraciones, clicks ni tiempo de visionado.
   La evaluación off-line es necesariamente indirecta y su honestidad se trata en la Sección 11.

---

## 2. Estrategia A — Representación Dispersa Léxica (TF-IDF)

### 2.1 Naturaleza

Cada película es un vector disperso en un espacio de dimensión `|V|` (vocabulario); cada
dimensión es un término y su valor es el peso TF-IDF.

### 2.2 Construcción del vector

Texto de entrada por película (con preprocesamiento de la Sección 4):

```python
texto_pelicula = " ".join([description, keywords, genre])
```

Pesos TF-IDF y parámetros (la fórmula del peso queda expresada por la configuración del
vectorizador, no en notación matemática):

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# peso(t, d) = tf(t, d) * [ ln((N + 1) / (df(t) + 1)) + 1 ]   (smooth_idf=True),
# con tf sublineal: tf -> 1 + ln(tf). Vectores L2-normalizados (norm="l2").
vectorizer = TfidfVectorizer(
    min_df=2,            # descarta términos en <2 películas (hápax ruidosos, idioma mixto)
    max_df=0.85,         # descarta términos en >85% del corpus (stopwords de dominio)
    ngram_range=(1, 2),  # unigramas + bigramas ("crimen organizado")
    sublinear_tf=True,   # 1 + ln(tf)
    smooth_idf=True,     # +1 en numerador y denominador del idf
    norm="l2",
)
M = vectorizer.fit_transform(corpus_textos)   # M: matriz dispersa (N, |V|)

# CHEQUEO DE NORMA VERIFICABLE (hallazgo 2) — vale también para A:
import numpy as np
from sklearn.preprocessing import normalize
filas_no_nulas = np.asarray((M.power(2)).sum(axis=1)).ravel() > 0
normas = np.sqrt(np.asarray((M.power(2)).sum(axis=1)).ravel()[filas_no_nulas])
assert np.allclose(normas, 1.0, atol=1e-6)
```

### 2.3 Qué captura y qué pierde

| Captura | Pierde |
|---|---|
| Términos específicos del dominio cinematográfico español | Relaciones entre sinónimos ("asesino" ≠ "criminal" léxicamente) |
| Distinción de películas por vocabulario propio | Variación morfológica sin lematizar |
| Interpretabilidad total (qué términos pesan) | Contexto y ambigüedad |
| Eficiencia computacional (matriz dispersa) | Generalización a términos fuera del vocabulario (OOV) |

### 2.4 Adecuación al corpus

La sinopsis en español tiene vocabulario suficientemente variado para que TF-IDF diferencie
géneros y temáticas; la cobertura es alta porque las sinopsis son la fuente principal. El
problema OOV afecta a las queries de usuario y se mitiga con el preprocesamiento de la Sección 4
(lematización compartida corpus/query).

---

## 3. Estrategia B — Representación Densa Contextual (Sentence Embeddings)

### 3.1 Naturaleza

Cada película es un vector denso de dimensión fija `d=768`, producido por un transformer
entrenado para similitud semántica entre oraciones.

### 3.2 Modelo seleccionado

**`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`**: 50+ idiomas (incl. español),
768 dim, MPNet, fine-tuned con pares paráfrasis. La **revisión exacta del modelo se ancla** por
reproducibilidad (Sección 9):

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    revision="<commit_sha_fijado>",   # se fija el SHA del snapshot HF usado (Sección 9)
)
```

**Alternativa descartada:** `dccuchile/bert-base-spanish-wwm-cased` — requiere pooling manual y
no está fine-tuned para similitud; peor calibración para esta tarea.

### 3.3 Campo de entrada — decisión por ablation, no por supuesto (hallazgo 4)

La v1.0 excluía `keywords`/`genre` por "coherencia semántica" sin evidencia. v2.0 **no cierra
esta decisión**: la convierte en hipótesis a contrastar mediante un ablation mínimo, decidido
con las métricas de la Sección 7 (mismo protocolo, mismos usuarios/candidatos):

```python
# ABLATION B (decidido empíricamente, NO por intuición):
variantes_entrada_B = {
    "B_desc":      lambda r: r.description,
    "B_desc_kw_g": lambda r: " ".join([r.description, r.keywords, r.genre]),
}
# Hipótesis H0: añadir keywords/genre (idioma mixto) NO mejora MRR global ni Hit@5.
# Se ejecutan ambas variantes con el protocolo de la Sección 7 y se reporta la tabla
# segmentada por tipo_perfil. La variante ganadora se fija como "Estrategia B" final;
# la perdedora queda documentada como ablation en el informe.
```

### 3.4 Qué captura y qué pierde

| Captura | Pierde |
|---|---|
| Similitud semántica entre sinónimos y paráfrasis | Interpretabilidad (caja negra) |
| Contexto y ambigüedad | Términos de dominio sin representación en pretraining |
| Robustez a variación morfológica | Control fino sobre qué contribuye a la similitud |
| Queries en lenguaje natural sin preprocesamiento agresivo | Sinopsis nulas (vectores degenerados) |

**Costo computacional:** inferencia sobre ~4970 películas ≈ 2–5 min en CPU. Se **precalcula y
cachea** en disco (Sección 9).

### 3.5 Contraste entre Estrategias A y B

| Dimensión | Estrategia A (TF-IDF) | Estrategia B (Embeddings) |
|---|---|---|
| Espacio vectorial | Disperso, ~|V| ≈ 20.000–50.000 dim | Denso, 768 dim |
| Similitud capturada | Léxica (términos compartidos) | Semántica (significado compartido) |
| Manejo de OOV | No (falla con términos nuevos) | Sí (representación generalizable) |
| Idioma mixto | Problemático | Manejado nativamente |
| Interpretabilidad | Alta | Baja |
| Velocidad de indexación | Muy alta | Media (requiere cache) |
| Sensibilidad a nulos | Alta (vector vacío = vector nulo) | Media (embedding de string vacío) |

La elección **final** entre A y B no se decide en este documento: se decide con el protocolo
experimental de la Sección 7.

---

## 4. Pipeline de Datos

### 4.1 Preprocesamiento para Estrategia A (TF-IDF)

```
plots.csv
  → [1] Carga y limpieza de nulos
        description nula → ""  y marcar película como low_confidence
        keywords/genre nulos → ""
  → [2] Concatenación: texto = description + " " + keywords + " " + genre
  → [3] Normalización: lowercase; quitar caracteres especiales (conservar tildes y ñ);
        quitar dígitos (el año vive en la columna year)
  → [4] Tokenización: spaCy es_core_news_sm (preserva morfología española)
  → [5] Stopwords: lista NLTK español + dominio ("película","historia","film")
  → [6] Lematización: spaCy lemmatizer ("corriendo"→"correr", "amores"→"amor")
  → [7] Vectorización TF-IDF (params §2.2) + chequeo de norma L2 (§2.2)
  → Matriz M (N≈4970, |V|≈20k-50k)
```

### 4.2 Preprocesamiento para Estrategia B (Embeddings) — normalización verificable (hallazgo 2)

La v1.0 afirmaba que el modelo "devuelve embeddings normalizados de forma implícita". **Eso es
falso**: `sentence-transformers` NO normaliza por defecto. v2.0 exige normalización **explícita
y verificada**:

```
plots.csv
  → [1] Carga y limpieza de nulos: description nula → ""  y marcar low_confidence
  → [2] Texto = variante ganadora del ablation §3.3
  → [3] Truncado: respetar límite del modelo (max_seq_length); truncado controlado
  → [4] Sin lematización ni stopword removal (degrada el embedding)
  → [5] Inferencia CON NORMALIZACIÓN EXPLÍCITA
  → [6] Chequeo de norma verificable
  → [7] Persistencia en disco (cache)
```

```python
E = model.encode(
    textos,
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True,   # OBLIGATORIO Y EXPLÍCITO (no asumir comportamiento por defecto)
)
# CHEQUEO VERIFICABLE en el notebook (hallazgo 2):
import numpy as np
assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-3)

np.save("artifacts/embeddings_B.npy", E)   # cache; criterio de invalidación en Sección 9
```

### 4.3 Preprocesamiento de consultas de usuario

- **Estrategia A:** aplicar pasos [3]–[6] del pipeline A y proyectar con el vectorizador **ya
  ajustado sobre el corpus** (`transform`, nunca `fit` — ver protocolo anti-leakage, Sección 8).
- **Estrategia B:** pasar la query directamente a `model.encode(..., normalize_embeddings=True)`
  (sin preprocesamiento agresivo; el modelo fue entrenado con lenguaje natural).

---

## 5. Modelado de Usuario y Combinación de Señales

### 5.1 Vector del historial

```python
# H_e = subconjunto del historial que SÍ matchea el corpus (fuzzy score >= umbral calibrado §1.2.3)
V_hist = matriz_vectores[idx_peliculas_He]    # (n, dim), n = |H_e| <= 5
if n == 0:
    v_hist = None          # ninguna película matcheó -> modo query-only (alpha = 0)
else:
    v_hist = V_hist.mean(axis=0)   # centroide del historial efectivo
```

### 5.2 Vector de la query

```python
# Estrategia A: v_query in R^|V| (proyección transform del vectorizador del corpus)
# Estrategia B: v_query in R^768 (model.encode(query, normalize_embeddings=True))
v_query = representar_query(query, estrategia)   # misma estrategia que el corpus
```

### 5.3 Combinación lineal historial + query

```python
from sklearn.preprocessing import normalize

def combinar(v_hist, v_query, alpha):
    # Normalizar CADA señal antes de combinar: la escala no debe depender de la magnitud.
    vh = normalize(v_hist.reshape(1, -1))[0] if v_hist is not None else 0.0
    vq = normalize(v_query.reshape(1, -1))[0]
    v_user = alpha * vh + (1.0 - alpha) * vq
    v_user = normalize(v_user.reshape(1, -1))[0]   # renormalizar antes de medir similitud
    # CHEQUEO (hallazgo 2): la norma de v_user debe ser 1.
    assert abs(np.linalg.norm(v_user) - 1.0) < 1e-6
    return v_user
```

### 5.4 Determinación de α — hipótesis con respaldo, no constante mágica

Regla base por `tipo_perfil` (punto de partida, no verdad fija):

| tipo_perfil | α base | Razonamiento |
|---|---|---|
| `definido` | 0.6 | Historial coherente = señal fuerte; la query refina |
| `disperso` | 0.3 | Historial no converge; la query es más fiable |
| n = 0 | 0.0 | Solo query |

Refinamiento adaptativo a partir de la dispersión interna medida del historial:

```python
from sklearn.metrics.pairwise import cosine_similarity

def dispersion_historial(V_hist):       # sigma_H in [0, 1]
    if V_hist.shape[0] < 2:
        return 0.0
    S = cosine_similarity(V_hist)                  # (n, n)
    iu = np.triu_indices_from(S, k=1)
    return float(1.0 - S[iu].mean())               # 0 = coherente, 1 = disperso

def alpha_adaptativo(V_hist):
    sigma = dispersion_historial(V_hist)
    return float(np.clip(0.8 - 0.6 * sigma, 0.2, 0.8))   # ~0.8 coherente .. ~0.2 disperso
```

**Regla:** la regla adaptativa **prevalece** sobre la base cuando la dispersión medida
contradiga la etiqueta `tipo_perfil` (p. ej. un perfil `definido` con historial de hecho
heterogéneo). El criterio de aceptación de esta regla NO es declarativo: se valida con el
protocolo de la Sección 7 reportando métricas separadas por `tipo_perfil`.

### 5.5 Resolución de conflictos hist/query — políticas comparadas, no regla hardcodeada (h.6)

La v1.0 fijaba `α=0.4` ante conflicto sin evidencia. v2.0 define el conflicto y **especifica un
mini-experimento de arbitraje** cuya ganadora se decide con la Sección 7:

```python
def hay_conflicto(v_hist, v_query, theta=0.10):
    if v_hist is None:
        return False
    return float(cosine_similarity(v_hist.reshape(1,-1), v_query.reshape(1,-1))[0,0]) < theta

# POLÍTICAS DE ARBITRAJE A COMPARAR (mínimo 2; se evalúan con el protocolo §7,
# reportando impacto SEGMENTADO por tipo_perfil):
politicas_conflicto = {
    "history_first": 0.7,     # confía en el patrón de consumo
    "query_first":   0.2,     # confía en la intención declarada
    "blend_adapt":   None,    # usa alpha_adaptativo(V_hist) sin override
}
# Regla de decisión: se adopta la política con mayor MRR global; ante empate (<0.02)
# se prefiere la de mayor Hit@5 en perfiles 'disperso'. Se registra el conflicto y,
# para diagnóstico, se generan rankings parciales solo-historial y solo-query.
```

**Caso "sin señal" — centralidad semántica, NO popularidad (hallazgo 7).** No existe señal de
consumo real en el corpus, así que hablar de "popularidad" es incorrecto. Cuando `n == 0` y la
query es vacía o genérica, el fallback es **centralidad semántica**: las películas más cercanas
al centroide del corpus. Es un fallback explícitamente débil; por eso en este modo es
**obligatorio reportar diversidad intra-lista y cobertura** (Sección 7) para evidenciar si
degenera en recomendaciones genéricas/repetitivas.

```python
def fallback_centralidad_semantica(M):
    centroide = normalize(np.asarray(M.mean(axis=0)))     # NO es popularidad: es centralidad
    sims = cosine_similarity(centroide.reshape(1, -1), M)[0]
    top5 = np.argsort(-sims)[:5]
    flag = "fallback:centralidad_semantica (sin señal de usuario)"
    return top5, flag                                     # medir diversidad/cobertura en §7
```

---

## 6. Métrica de Similitud

### 6.1 Definición

```python
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(v_user.reshape(1, -1), M)[0]   # similitud usuario vs todo el corpus
# Con vectores L2-normalizados (garantizado por §2.2/§4.2/§5.3), coseno == producto punto:
# sims = M @ v_user   (equivalente y más rápido en Estrategia B)
```

### 6.2 Justificación frente a alternativas

| Métrica | Ventaja | Desventaja | Adecuación |
|---|---|---|---|
| **Coseno** | Invariante a magnitud; estándar para texto | No distingue magnitud de dirección | **Elegida** — ambas estrategias producen vectores L2-normalizados verificados |
| Euclidiana | Intuitiva | Sensible a magnitud; mala en alta dimensión | No adecuada |
| Producto punto | = coseno si está normalizado | Sin normalizar favorece textos largos | Equivalente post-normalización (se usa como atajo en B) |
| BM25 (solo A) | Buen ranking léxico | No extensible a embeddings ni a perfil vectorial de usuario | Descartada por incompatibilidad con el modelado de usuario |

### 6.3 Ranking final

```python
1. sims = cosine_similarity(v_user, M)            # todas las películas
2. excluir índices del historial efectivo H_e     # anti-leakage (Sección 8)
3. si hay >=5 candidatos sin marca: excluir low_confidence
4. top5 = primeras 5 por similitud descendente
```

---

## 7. Protocolo Experimental Comparativo A vs B  *(BLOQUEANTE 1 — cerrado)*

La v1.0 contrastaba A y B solo conceptualmente, sin criterio operativo de selección. v2.0 define
un protocolo replicable y auditable.

### 7.1 Condiciones idénticas para ambas estrategias

- **Mismos 14 usuarios**, **mismo pool de candidatos** (corpus completo menos exclusiones de la
  Sección 8), **mismas métricas**, **mismas semillas** (Sección 9).
- Cada estrategia produce su `v_user` con su propio espacio vectorial pero con el **mismo
  modelado de usuario** (Sección 5) y la **misma métrica** (Sección 6).

### 7.2 Métricas idénticas (mismas definiciones para A, B y baselines)

| Métrica | Definición | Interpreta |
|---|---|---|
| **Hit@5 / Hit@10** | Fracción de usuarios cuya película retirada (LOO, Sección 8) aparece en top-K | Recuperación básica |
| **MRR** | Promedio de 1/posición de la película retirada | Calidad del ranking |
| **Diversidad intra-lista** | Promedio de (1 − coseno) entre pares del top-5 | ¿Redundante o variado? |
| **Cobertura** | Fracción del corpus que aparece en algún top-5 | ¿Concentra o distribuye? |

### 7.3 Baselines obligatorios

| Baseline | Definición | Para qué sirve |
|---|---|---|
| **Aleatorio** | top-5 al azar con semilla fija (Sección 9) | Piso absoluto: ninguna estrategia útil debe perder contra esto |
| **Centralidad semántica** | top-5 por cercanía al centroide del corpus (§5.5) | Detecta si la estrategia solo está devolviendo "lo central/genérico" |

### 7.4 Matriz de resultados (plantilla que rellena el Ingeniero)

| Sistema | Hit@5 | Hit@10 | MRR | Div@5 | Cobertura |
|---|---|---|---|---|---|
| Baseline aleatorio | | | | | |
| Baseline centralidad | | | | | |
| Estrategia A (TF-IDF) | | | | | |
| Estrategia B (ablation `B_desc`) | | | | | |
| Estrategia B (ablation `B_desc_kw_g`) | | | | | |

Segmentación obligatoria: **repetir la matriz separando `tipo_perfil` ∈ {definido, disperso}**
y reportar también el subconjunto de casos "sin señal"/conflicto.

### 7.5 Regla de decisión explícita

```
1. Descartar toda estrategia que no supere ambos baselines en MRR global.
2. Entre las que sobreviven, elegir la de mayor MRR global.
3. Empate si |MRR_A - MRR_B| < 0.02  -> desempata mayor Hit@5 en perfiles 'disperso'
   (es el caso más difícil y más informativo de la calidad real del sistema).
4. La variante de ablation §3.3 que pierde se conserva como evidencia en el informe.
5. Documentar SIEMPRE ambas estrategias y su contraste (el enunciado lo exige);
   "elegir" no significa ocultar la otra, significa justificar cuál se recomienda.
```

---

## 8. Protocolo Anti-Leakage (Leave-One-Out)  *(hallazgo 8)*

La v1.0 excluía películas vistas del ranking pero no formalizaba el protocolo LOO. v2.0 lo fija
con asserts verificables.

### 8.1 Procedimiento por usuario

```
Para cada usuario u con historial efectivo H_e (|H_e| = n >= 2):
  para cada película p_held en H_e:
    perfil_reconstruido = H_e \ {p_held}          # 4 (o n-1) películas
    v_user = combinar(centroide(perfil_reconstruido), v_query, alpha)
    candidatos = corpus \ perfil_reconstruido     # p_held SÍ es candidato recuperable
    ranking  = top-K(sims(v_user, candidatos))
    registrar posición de p_held (o "no recuperada")
```

### 8.2 Checklist con asserts verificables (en el notebook)

```python
# A1. El vectorizador / embeddings del corpus se ajustan UNA vez sobre el corpus COMPLETO,
#     nunca re-fit por usuario (evita leakage de la partición de evaluación):
assert vectorizer_fue_fiteado_una_sola_vez_sobre_corpus_completo

# A2. La película retirada NO entra al perfil reconstruido:
assert p_held not in perfil_reconstruido

# A3. Las películas del perfil reconstruido NO aparecen en el ranking evaluado:
assert len(set(perfil_reconstruido) & set(ranking_idx)) == 0

# A4. La película retirada SÍ es candidata (no se la excluye del pool por error):
assert p_held in candidatos

# A5. v_user no se construyó usando p_held de ninguna forma:
assert p_held not in indices_usados_para_v_user
```

### 8.3 Trazabilidad

Tabla por usuario: `id`, `tipo_perfil`, `n` efectivo, `p_held`, posición recuperada, `alpha`
usado, flag de conflicto, flag de fallback. Esta tabla alimenta la segmentación de la Sección 7
y la discusión de la Sección 11.

---

## 9. Política de Reproducibilidad  *(BLOQUEANTE 3 — cerrado)*

### 9.1 Semillas

```python
import os, random, numpy as np
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)   # usado por el baseline aleatorio (Sección 7.3)
```

### 9.2 Versiones e identidad del modelo

- Dependencias **pinneadas**: el entorno se reconstruye desde `uv.lock` + `pyproject.toml`
  (`uv sync`); el notebook imprime las versiones efectivas de `numpy`, `scikit-learn`,
  `sentence-transformers`, `spaCy`, `nltk` y del modelo de spaCy `es_core_news_sm`.
- Identidad del modelo HF anclada por `revision="<commit_sha>"` (Sección 3.2); el notebook
  imprime el SHA efectivo cargado.

### 9.3 Orden determinista y artefactos cacheados

| Artefacto | Archivo | Criterio de invalidación |
|---|---|---|
| Embeddings B | `artifacts/embeddings_B.npy` | Cambia el modelo/revisión, la variante de §3.3 o el preprocesamiento |
| Matriz TF-IDF + vocab | `artifacts/tfidf_*.pkl` | Cambian params §2.2 o el preprocesamiento |
| Tabla de matches fuzzy | `artifacts/fuzzy_matches.csv` | Cambia el umbral calibrado §1.2.3 |

El pipeline se ejecuta en orden fijo: EDA → preprocesamiento → vectorización (con chequeos de
norma) → modelado de usuario → ranking → evaluación. Ejecutar el notebook de principio a fin
reproduce exactamente las tablas de la Sección 7.

---

## 10. Información No Utilizada y Vías de Mejora  *(hallazgo 9)*

Responde a la pregunta del bloque `evaluacion` del enunciado ("¿qué información no se está usando
y podría mejorar el sistema?").

| Feature no usada | Cómo se incorporaría | Criterio de validación |
|---|---|---|
| `director` | Boosting post-ranking: +δ a la similitud de películas cuyo director coincide con el de alguna película del historial efectivo (señal de afinidad de autor) | Re-ejecutar la matriz §7 con/sin boosting; aceptar solo si mejora MRR sin colapsar diversidad |
| `year` | Re-ranking suave por cercanía temporal a la mediana de años del historial (preferencia de época) | Igual que arriba; medir efecto en cobertura (riesgo: concentrar en una década) |
| Señal de consumo real | No existe en el corpus | Limitación estructural: ningún feature actual habilita popularidad real (justifica el fallback de §5.5 como centralidad, no popularidad) |

Estas mejoras se especifican pero **no se incorporan al sistema base**: cada una solo se adopta
si el experimento de la Sección 7 demuestra ganancia neta (evita sobre-diseño).

---

## 11. Estrategia de Evaluación Off-line

### 11.1 Proxy de relevancia

Leave-one-out sobre el historial (Sección 8): la película retirada se asume relevante. Supuesto
imperfecto y declarado: el usuario pudo ver algo que no le gustó.

### 11.2 Limitaciones honestas

- **No mide satisfacción real**; el objetivo del sistema es descubrir, el proxy premia
  reproducir el pasado (sesgo de confirmación).
- **14 perfiles = muestra pequeña:** alta varianza; las métricas agregadas **no son
  estadísticamente significativas** y se reportan como indicativas, con su desglose por usuario
  (Sección 8.3), no solo el promedio.
- **Perfiles dispersos son más difíciles de evaluar:** el LOO sobre un historial incoherente no
  tiene un "correcto" claro; por eso la Sección 7 reporta `disperso` por separado y desempata
  decisiones con ese segmento.

### 11.3 Mapeo a preguntas del enunciado

| Pregunta del enunciado | Dónde se responde |
|---|---|
| ¿Las recomendaciones tienen sentido para cada usuario? | Inspección cualitativa top-5 + Hit@5 (Sección 7) |
| ¿Casos donde acierta / falla? ¿A qué se deben? | Matriz segmentada por `tipo_perfil` (7.4) + trazas (8.3) |
| ¿Qué hace sin señal clara? | Fallback centralidad semántica + diversidad/cobertura (5.5, 7) |
| ¿La métrica es válida para todos los perfiles? | MRR comparado definido vs disperso (7.4, 11.2) |
| ¿Qué información no se usa y podría mejorar? | Sección 10 |
| ¿Limitaciones de un sistema puramente de contenido? | Sección 12 (no resuelve filtrado colaborativo, novedad, serendipia sin señal de consumo) |

---

## 12. Riesgos y Supuestos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Alta tasa de no-coincidencia de títulos | Historial vacío → query-only | Umbral fuzzy calibrado (§1.2.3) + log de matches dudosos |
| Sinopsis cortas/nulas | Vectores degenerados; falsa similitud alta | Flag `low_confidence`; reporte de cobertura efectiva |
| Query genérica/vacía en perfiles dispersos | `v_query` no informativo | α adaptativo + fallback centralidad con diversidad medida |
| Keywords/genre en inglés (Estrategia A) | OOV / bajo peso | Normalización de case + `min_df>=2` |
| Sobreajuste a perfiles "definidos" | Métricas optimistas | Reporte segmentado por `tipo_perfil` (Sección 7) |
| Falsa precisión de reglas (α, conflicto) | Decisiones no defendibles | Toda regla heurística pasa por el protocolo §7 antes de adoptarse |

**Supuestos — ahora explícitamente marcados como hipótesis a validar, no como hechos:**

- *(H1)* Las sinopsis en español distinguen géneros/temáticas → se verifica en el EDA y vía
  Hit@5 contra baseline aleatorio (Sección 7).
- *(H2)* `tipo_perfil` es consistente con el contenido real del historial → se contrasta con
  `sigma_H` medido (§5.4); si discrepa, prevalece la medida.
- *(H3)* El vocabulario de las queries es compatible con el del corpus (necesario para A) → se
  cuantifica el % de tokens OOV de cada query tras el preprocesamiento compartido.

---

## 13. Trazabilidad de Hallazgos → Resolución

| # | Hallazgo (severidad) | Sección que lo resuelve | Estado |
|---|---|---|---|
| 1 | Falta protocolo A/B con criterio de decisión (**bloqueante**) | Sección 7 (matriz, baselines, regla de decisión §7.5) | **Cerrado** |
| 2 | Supuesto falso de normalización en embeddings (**bloqueante**) | §3.2, §4.2, §5.3 (normalize_embeddings=True + asserts de norma) | **Cerrado** |
| 3 | Reproducibilidad insuficiente (**bloqueante**) | Sección 9 (semillas, versiones, revisión HF, cache) | **Cerrado** |
| 4 | Descartar keywords/genre en B sin validar (mayor) | §3.3 (ablation decidido por §7) | Cerrado |
| 5 | Umbral fuzzy sin calibrar (mayor) | §1.2.3 (protocolo precisión/recall + logging) | Cerrado |
| 6 | Conflicto hist/query hardcodeado (mayor) | §5.5 (3 políticas comparadas vía §7, segmentado) | Cerrado |
| 7 | Fallback "popularidad" mal definido (mayor) | §5.5 (renombrado centralidad semántica + diversidad/cobertura) | Cerrado |
| 8 | Leakage no formalizado (mayor) | Sección 8 (checklist + asserts A1–A5 + trazas) | Cerrado |
| 9 | Features no usadas sin plan (menor) | Sección 10 (director/year + criterio de validación) | Cerrado |

**Veredicto objetivo de la v1.0 a cerrar:** los 3 bloqueantes quedan resueltos con protocolos
verificables; las reglas heurísticas dejan de presentarse como verdades y pasan a ser hipótesis
con su validación. La especificación queda lista para la Fase 2 (Implementación).
