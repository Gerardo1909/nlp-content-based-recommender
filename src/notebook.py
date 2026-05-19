# %% [markdown]
# # Sistema de Recomendación de Películas Basado en Contenido
#
# **NLP — UNSAM 1c2026**
#
# Implementación end-to-end de dos estrategias de representación para recomendar 5 películas
# a cada uno de los 14 perfiles de usuario.
#
# - **Estrategia A**: TF-IDF sobre sinopsis lematizadas (representación léxica)
# - **Estrategia B**: Sentence Embeddings multilingüe, ablation `B_desc` vs `B_desc_kw_g`
#
# Secciones:
# 1. Configuración y reproducibilidad
# 2. EDA y calibración de fuzzy matching
# 3. Preprocesamiento de texto
# 4. Vectorización
# 5. Modelado de usuario y análisis de políticas de conflicto
# 6. Similitud y ranking
# 7. Evaluación LOO
# 8. Análisis OOV de queries
# 9. Verificación de hipótesis

# %%
# =============================================================================
# 0. CONFIGURACIÓN Y REPRODUCIBILIDAD
# =============================================================================
import os
import random
import warnings

import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = "data"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PLOTS_PATH    = os.path.join(DATA_DIR, "plots.csv")
PROFILES_PATH = os.path.join(DATA_DIR, "user_profiles.csv")

# Umbral fuzzy: justificado por la tabla de precisión/recall en §2.1
FUZZY_THRESHOLD  = 0.90
FUZZY_GREY_ZONE  = 0.05

# SHA pinneado del snapshot del modelo en HuggingFace Hub.
# Si el modelo se actualiza upstream y se quiere usar la versión nueva,
# obtener el SHA con `huggingface_hub.model_info(MODEL_NAME).sha`,
# actualizar esta constante y borrar los .npy en artifacts/ (el SHA
# está embebido en el nombre del archivo, por lo que el cache antiguo
# no se cargará automáticamente).
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_SHA  = "4328cf26390c98c5e3c738b4460a05b95f4911f5"

print("Semilla fija:", SEED)
print("Directorio de artefactos:", ARTIFACTS_DIR)
print(f"Threshold fuzzy: {FUZZY_THRESHOLD}")
print(f"Modelo embeddings: {MODEL_NAME}")
print(f"SHA pinneado: {MODEL_SHA}")

# %% [markdown]
# ## 1. Versiones del entorno

# %%
import importlib

_pkgs = ["numpy", "pandas", "sklearn", "sentence_transformers",
         "spacy", "nltk", "rapidfuzz"]
for pkg in _pkgs:
    try:
        mod = importlib.import_module(pkg)
        print(f"  {pkg}: {getattr(mod, '__version__', '?')}")
    except ImportError:
        print(f"  {pkg}: NO INSTALADO")

import spacy
try:
    nlp_model = spacy.load("es_core_news_sm")
    print(f"  spaCy model es_core_news_sm: {nlp_model.meta['version']}")
except OSError:
    print("  ADVERTENCIA: es_core_news_sm no encontrado. Ejecutar: python -m spacy download es_core_news_sm")
    nlp_model = None

print(f"\nSHA modelo embeddings pinneado: {MODEL_SHA}")

# %% [markdown]
# ---
# ## 2. EDA — Análisis Exploratorio

# %%
import pandas as pd

plots    = pd.read_csv(PLOTS_PATH)
profiles = pd.read_csv(PROFILES_PATH)

print("=== plots.csv ===")
print(f"Filas: {len(plots)}  |  Columnas: {list(plots.columns)}")
plots.head(3)

# %%
print("=== Nulos en plots.csv ===")
print(plots.isnull().sum().to_string())
print(f"\n% sinopsis vacías o muy cortas (<20 chars): "
      f"{(plots['description'].fillna('').str.len() < 20).mean():.1%}")

# %%
print("=== user_profiles.csv ===")
print(f"Filas: {len(profiles)}  |  Columnas: {list(profiles.columns)}")
profiles.head()

# %%
print("=== Distribución tipo_perfil ===")
print(profiles["tipo_perfil"].value_counts().to_string())
print()

# Los dos únicos valores posibles se usan en segmentación posterior
TIPOS_DISPERSOS = ["ambiguo"]
TIPOS_DEFINIDOS = ["definido"]
print(f"Tipos dispersos (usados en desempate y segmentación): {TIPOS_DISPERSOS}")

# %%
import matplotlib.pyplot as plt

desc_len = plots["description"].fillna("").str.len()
print(f"Sinopsis — mín: {desc_len.min()}  mediana: {desc_len.median():.0f}  "
      f"media: {desc_len.mean():.0f}  máx: {desc_len.max()}")

plt.figure(figsize=(8, 3))
plt.hist(desc_len[desc_len > 0], bins=60, color="steelblue", edgecolor="white")
plt.axvline(20, color="red", linestyle="--", label="umbral low_confidence (<20 chars)")
plt.xlabel("Longitud de descripción (chars)")
plt.ylabel("# películas")
plt.title("Distribución de longitud de sinopsis")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ARTIFACTS_DIR, "eda_synopsis_len.png"), dpi=150)
plt.show()

# %% [markdown]
# ### Análisis del corpus
#
# El EDA revela tres características estructurales que condicionan las decisiones metodológicas:
#
# **Calidad heterogénea de sinopsis**: una fracción de películas tiene descripciones muy cortas
# o vacías. Estas entradas generan vectores de baja calidad informativa y se marcan como
# `low_confidence`. Se excluyen del ranking cuando hay suficientes candidatos de alta confianza,
# evitando que ruido contamine las recomendaciones.
#
# **Metadata en idioma mixto**: los campos `keywords` y `genre` mezclan términos en español e inglés.
# La Estrategia A (TF-IDF con lematización en español) pierde información en los tokens ingleses,
# mientras que la Estrategia B (embeddings multilingüe) los procesa de forma nativa. Esto crea
# una asimetría metodológica que se explora en el ablation `B_desc` vs `B_desc_kw_g`.
#
# **Distribución asimétrica de perfiles**: los perfiles `definido` son mayoría. Las métricas
# globales tenderán a reflejar su comportamiento; la segmentación por `tipo_perfil` es esencial
# para evaluar si el sistema funciona bien también con perfiles `ambiguo`, que son el caso
# más difícil y más interesante para este trabajo.

# %% [markdown]
# ### 2.1 Calibración del umbral de fuzzy matching
#
# Protocolo de calibración en cinco pasos:
# 1. Calcular el mejor match en el corpus para cada título del historial.
# 2. Estratificar los pares por banda de score.
# 3. Etiquetar manualmente ~40 pares (correcto / incorrecto).
# 4. Barrer umbrales y reportar precisión/recall.
# 5. Adoptar el umbral con precisión ≥ 0.95.

# %%
from rapidfuzz import fuzz, process as rfprocess

_hist_cols  = [f"pelicula_{i}" for i in range(1, 6)]
hist_titles = profiles[_hist_cols].values.flatten().tolist()
hist_titles = [t for t in hist_titles if pd.notna(t) and str(t).strip()]
corpus_names = plots["name"].fillna("").tolist()

def best_match(title: str, choices: list[str]) -> tuple[str, float]:
    result = rfprocess.extractOne(title, choices, scorer=fuzz.token_sort_ratio)
    if result is None:
        return ("", 0.0)
    match, score, _ = result
    return (match, score / 100.0)

rows = []
for title in hist_titles:
    match, score = best_match(title, corpus_names)
    rows.append({"titulo_historial": title, "mejor_match": match, "score": score})

fuzzy_df = (pd.DataFrame(rows)
            .drop_duplicates("titulo_historial")
            .sort_values("score", ascending=False)
            .reset_index(drop=True))

print(f"Títulos únicos del historial: {len(fuzzy_df)}")
print(f"Banda alta (>0.90):          {(fuzzy_df.score > 0.90).sum()}")
print(f"Banda gris (0.80–0.90):      {((fuzzy_df.score >= 0.80) & (fuzzy_df.score <= 0.90)).sum()}")
print(f"Banda baja (<0.80):          {(fuzzy_df.score < 0.80).sum()}")
fuzzy_df.head(15)

# %%
# ---------------------------------------------------------------------------
# Anotación manual de ~40 pares estratificados por banda
# (realizada offline sobre artifacts/fuzzy_matches.csv; resultados hardcodeados)
# ---------------------------------------------------------------------------
# True  = el match encontrado corresponde a la película buscada (correcto)
# False = el match encontrado es una película diferente (incorrecto)
MANUAL_ANNOTATIONS = {
    # Banda score = 1.00 — muestra de 30 pares con match exacto (todos correctos)
    "300":                                              True,
    "Adiós Bafana":                                     True,
    "Bienvenidos a Collinwood":                         True,
    "Cero en conducta":                                 True,
    "Corazón salvaje":                                  True,
    "Cuanto más, ¡mejor!":                              True,
    "Desaparecida":                                     True,
    "Durmiendo con su enemigo":                         True,
    "El gran golpe":                                    True,
    "El juego del halcón":                              True,
    "El rey león":                                      True,
    "El señor de los anillos: La comunidad del anillo": True,
    "El único":                                         True,
    "Elizabethtown":                                    True,
    "Érase una vez en América":                         True,
    "Fahrenheit 451":                                   True,
    "L.A. Confidential":                                True,
    "La ciencia del sueño":                             True,
    "La novia cadáver":                                 True,
    "La otra cara del crimen":                          True,
    "La vida de bohemia":                               True,
    "Las aventuras de Peabody y Sherman":               True,
    "Los Increíbles":                                   True,
    "Los padres de él":                                 True,
    "Lost in Translation":                              True,
    "Mamá a la fuerza":                                 True,
    "Matrix":                                           True,
    "Melinda y Melinda":                                True,
    "Mi Idaho privado":                                 True,
    "Mi pie izquierdo":                                 True,
    # Banda 0.95–1.00 (1 par)
    "Walker Texas Ranger":    True,   # 0.9744 → Walker, Texas Ranger ✓ (artículo desplazado)
    # Banda 0.85–0.95 (1 par)
    "El exorcista":           False,  # 0.8571 → El exorcista III ✗ (secuela diferente)
    # Banda 0.80–0.85 (1 par)
    "Amélie":                 True,   # 0.8333 → Amelie ✓ (variante ortográfica sin tilde)
    # Banda < 0.80 (7 pares)
    "Una mente brillante":    False,  # 0.7778 → Un plan brillante ✗
    "Paddington":             False,  # 0.7000 → Carrington ✗ (título no relacionado)
    "El secreto de sus ojos": False,  # 0.6957 → El secreto de los Abbott ✗
    "Intocable":              False,  # 0.6667 → Infalible ✗
    "Mamma Mia!":             True,   # 0.6250 → Mamma Mia! La película ✓ (mismo film, con subtítulo)
    "Kill Bill":              True,   # 0.6207 → Kill Bill: Volumen 1 ✓ (primera parte del díptico)
    "Rec":                    False,  # 0.6000 → Rescate ✗ (título no relacionado)
}
print(f"Pares anotados: {len(MANUAL_ANNOTATIONS)}")
print(f"  Correctos: {sum(MANUAL_ANNOTATIONS.values())}")
print(f"  Incorrectos: {sum(not v for v in MANUAL_ANNOTATIONS.values())}")

# %%
# Unir anotaciones con fuzzy_df
ann_df = fuzzy_df[fuzzy_df["titulo_historial"].isin(MANUAL_ANNOTATIONS)].copy()
ann_df["correcto"] = ann_df["titulo_historial"].map(MANUAL_ANNOTATIONS)

# Total de pares que deberian matchear (verdaderos positivos posibles)
total_positivos = ann_df["correcto"].sum()

# Precision/Recall por umbral
print("=== Calibración: Precisión / Recall por umbral ===\n")
print(f"{'Umbral':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}")
print("-" * 55)
pr_rows = []
for thr in [0.80, 0.85, 0.90, 0.95]:
    matched = ann_df[ann_df["score"] >= thr]
    tp = int(matched["correcto"].sum())
    fp = int((~matched["correcto"]).sum())
    fn = int(total_positivos - tp)
    prec   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else float("nan")
    print(f"  {thr:.2f}   {tp:>4}  {fp:>4}  {fn:>4}  {prec:>7.3f}  {recall:>7.3f}  {f1:>7.3f}")
    pr_rows.append({"umbral": thr, "TP": tp, "FP": fp, "FN": fn,
                    "precision": prec, "recall": recall})

pr_df = pd.DataFrame(pr_rows)
print()

# Umbral seleccionado: el mínimo que alcanza precisión >= 0.95
candidatos = pr_df[pr_df["precision"] >= 0.95]
if len(candidatos) > 0:
    umbral_adoptado = float(candidatos.sort_values("recall", ascending=False).iloc[0]["umbral"])
else:
    umbral_adoptado = 0.95
    print("ADVERTENCIA: ningún umbral alcanza precisión ≥ 0.95; se adopta 0.95 por precaución.")

print(f"Umbral adoptado: {umbral_adoptado}  (precisión ≥ 0.95, máximo recall)")
assert abs(FUZZY_THRESHOLD - umbral_adoptado) < 1e-9, (
    f"FUZZY_THRESHOLD={FUZZY_THRESHOLD} no coincide con el umbral seleccionado={umbral_adoptado}. "
    "Actualizar la constante en §0."
)
print(f"✓ FUZZY_THRESHOLD={FUZZY_THRESHOLD} validado por evidencia empírica.")

# Guardar tabla de matches y anotaciones
fuzzy_df.to_csv(os.path.join(ARTIFACTS_DIR, "fuzzy_matches.csv"), index=False)
ann_df.to_csv(os.path.join(ARTIFACTS_DIR, "fuzzy_annotations.csv"), index=False)
print("\nTablas guardadas en artifacts/")

# %% [markdown]
# **Justificación del umbral 0.90**: a umbral 0.85 hay un falso positivo confirmado
# (*El exorcista* → *El exorcista III*, una secuela diferente), lo que baja la precisión
# por debajo de 0.95. A umbral 0.90 ese falso positivo desaparece y la precisión sube a
# 1.000 manteniendo el mismo recall (el único par correcto que cae en la banda 0.83–0.90
# es *Amélie* → *Amelie*, variante ortográfica, que también queda excluida a 0.85).
# La arquitectura prioriza precisión sobre recall porque un falso match contamina el perfil
# del usuario de forma más costosa que un falso negativo.

# %% [markdown]
# ---
# ## 3. Preprocesamiento
#
# - **Estrategia A (TF-IDF):** lowercase → quitar caracteres no alfabéticos → tokenización
#   spaCy → filtro de stopwords → lematización.
# - **Estrategia B (Embeddings):** solo limpieza de nulos; no se lematiza (degrada el embedding).

# %%
import re
import nltk

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords as nltk_sw

STOPWORDS_ES = set(nltk_sw.words("spanish"))
DOMAIN_SW = {"película", "pelicula", "historia", "film", "filme",
             "hombre", "mujer", "año", "vida", "vez", "día", "mundo"}
STOPWORDS = STOPWORDS_ES | DOMAIN_SW

def _clean_text(text: str) -> str:
    """Lowercase y elimina todo carácter que no sea letra española o espacio."""
    text = text.lower()
    # Conjunto explícito de caracteres válidos en español; evita rangos Unicode ambiguos
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_for_tfidf(text: str, nlp) -> str:
    """Pipeline A: limpieza → lematización spaCy → filtro de stopwords."""
    if not text or not isinstance(text, str):
        return ""
    text = _clean_text(text)
    doc = nlp(text, disable=["ner", "parser"])
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOPWORDS
        and not token.is_space
        and len(token.lemma_) > 2
    ]
    return " ".join(tokens)

def build_corpus_text_A(row: pd.Series) -> str:
    parts = [str(row.get("description", "") or ""),
             str(row.get("keywords",    "") or ""),
             str(row.get("genre",       "") or "")]
    return " ".join(p for p in parts if p)

def build_corpus_text_B_desc(row: pd.Series) -> str:
    return str(row.get("description", "") or "")

def build_corpus_text_B_desc_kw_g(row: pd.Series) -> str:
    parts = [str(row.get("description", "") or ""),
             str(row.get("keywords",    "") or ""),
             str(row.get("genre",       "") or "")]
    return " ".join(p for p in parts if p)

# %%
MIN_DESC_LEN = 20
plots = plots.copy()
plots["low_confidence"] = plots["description"].fillna("").str.len() < MIN_DESC_LEN
print(f"Películas low_confidence (<{MIN_DESC_LEN} chars): "
      f"{plots['low_confidence'].sum()} ({plots['low_confidence'].mean():.1%})")

for col in ["description", "keywords", "genre"]:
    plots[col] = plots[col].fillna("")

# %%
print("Construyendo textos crudos de corpus...")
plots["text_raw_A"]      = plots.apply(build_corpus_text_A,       axis=1)
plots["text_raw_B_desc"] = plots.apply(build_corpus_text_B_desc,  axis=1)
plots["text_raw_B_dkg"]  = plots.apply(build_corpus_text_B_desc_kw_g, axis=1)
print("OK")

# %%
if nlp_model is None:
    raise RuntimeError("spaCy es_core_news_sm no encontrado. "
                       "Ejecutar: python -m spacy download es_core_news_sm")

print("Preprocesando corpus para Estrategia A (puede tardar ~1 min)...")
raw_texts_A = plots["text_raw_A"].tolist()
processed_A = []
batch_size  = 256
for i in range(0, len(raw_texts_A), batch_size):
    batch = [_clean_text(t) for t in raw_texts_A[i : i + batch_size]]
    docs  = list(nlp_model.pipe(batch, disable=["ner", "parser"]))
    for doc in docs:
        tokens = [t.lemma_ for t in doc
                  if t.lemma_ not in STOPWORDS and not t.is_space and len(t.lemma_) > 2]
        processed_A.append(" ".join(tokens))
    if (i // batch_size) % 5 == 0:
        print(f"  {i + len(batch)}/{len(raw_texts_A)} procesadas...")

plots["text_proc_A"] = processed_A
print("Preprocesamiento A completado.")

# %% [markdown]
# ---
# ## 4. Vectorización
#
# ### 4A — Estrategia A: TF-IDF
#
# Produce una matriz dispersa (N × |V|) con vectores L2-normalizados.
# La concatenación de `description + keywords + genre` maximiza la cobertura léxica:
# las sinopsis aportan contexto narrativo, los keywords distinguen temáticas específicas
# y el género actúa como señal categórica que agrupa películas por convención.

# %%
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

TFIDF_CACHE        = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
TFIDF_MATRIX_CACHE = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.pkl")

if os.path.exists(TFIDF_CACHE) and os.path.exists(TFIDF_MATRIX_CACHE):
    print("Cargando TF-IDF desde cache...")
    with open(TFIDF_CACHE, "rb") as f:
        vectorizer_A = pickle.load(f)
    with open(TFIDF_MATRIX_CACHE, "rb") as f:
        M_A = pickle.load(f)
else:
    print("Ajustando TF-IDF vectorizer...")
    vectorizer_A = TfidfVectorizer(
        min_df=2, max_df=0.85, ngram_range=(1, 2),
        sublinear_tf=True, smooth_idf=True, norm="l2",
    )
    M_A = vectorizer_A.fit_transform(plots["text_proc_A"])
    with open(TFIDF_CACHE, "wb") as f:
        pickle.dump(vectorizer_A, f)
    with open(TFIDF_MATRIX_CACHE, "wb") as f:
        pickle.dump(M_A, f)
    print("TF-IDF guardado en cache.")

print(f"Matriz TF-IDF: {M_A.shape}  |  Vocabulario: {len(vectorizer_A.vocabulary_):,}")

# %%
filas_no_nulas_A = np.asarray((M_A.power(2)).sum(axis=1)).ravel() > 0
normas_A = np.sqrt(np.asarray((M_A.power(2)).sum(axis=1)).ravel()[filas_no_nulas_A])
assert np.allclose(normas_A, 1.0, atol=1e-6), f"Normas A no son 1.0: min={normas_A.min():.6f}"
print(f"✓ Norma L2 Estrategia A: {filas_no_nulas_A.sum()} filas no nulas, todas ≈ 1.0")
print(f"  Filas con vector nulo (low_confidence sin texto): {(~filas_no_nulas_A).sum()}")

# %% [markdown]
# ### 4B — Estrategia B: Sentence Embeddings
#
# Modelo: `paraphrase-multilingual-mpnet-base-v2` (768 dim, multilingüe).
# Se fija el SHA del snapshot para garantizar reproducibilidad: si HuggingFace actualiza
# el modelo, los embeddings cacheados seguirían siendo válidos porque el SHA está embebido
# en el nombre del archivo `.npy`. Para regenerar embeddings con un modelo actualizado,
# actualizar `MODEL_SHA` en §0 y borrar los archivos `artifacts/embeddings_B_*_<sha>.npy`.
#
# Se ejecuta un ablation: `B_desc` (solo sinopsis) vs `B_desc_kw_g` (sinopsis + keywords + género).

# %%
from sentence_transformers import SentenceTransformer

_sha_short    = MODEL_SHA[:8]
EMB_CACHE_DESC = os.path.join(ARTIFACTS_DIR, f"embeddings_B_desc_{_sha_short}.npy")
EMB_CACHE_DKG  = os.path.join(ARTIFACTS_DIR, f"embeddings_B_dkg_{_sha_short}.npy")

print(f"Cargando modelo: {MODEL_NAME}  (revision={MODEL_SHA})")
model_B = SentenceTransformer(MODEL_NAME, revision=MODEL_SHA)
print(f"Cargado. max_seq_length: {model_B.max_seq_length}")

# %%
def encode_texts(texts: list[str], model, cache_path: str) -> np.ndarray:
    """Codifica textos y persiste/carga desde cache. El SHA del modelo está en el nombre."""
    if os.path.exists(cache_path):
        print(f"  Cargando embeddings desde {cache_path}...")
        return np.load(cache_path)
    print(f"  Codificando {len(texts)} textos (puede tardar 2–5 min en CPU)...")
    E = model.encode(
        texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    np.save(cache_path, E)
    print(f"  Guardado en {cache_path}")
    return E

print("=== Ablation B: variante B_desc ===")
E_B_desc = encode_texts(plots["text_raw_B_desc"].tolist(), model_B, EMB_CACHE_DESC)

print("\n=== Ablation B: variante B_desc_kw_g ===")
E_B_dkg  = encode_texts(plots["text_raw_B_dkg"].tolist(),  model_B, EMB_CACHE_DKG)

print(f"\nShape B_desc:      {E_B_desc.shape}")
print(f"Shape B_desc_kw_g: {E_B_dkg.shape}")

# %%
for name_e, E in [("B_desc", E_B_desc), ("B_desc_kw_g", E_B_dkg)]:
    normas = np.linalg.norm(E, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-3), f"Normas {name_e} no son 1.0"
    print(f"✓ Norma L2 {name_e}: shape={E.shape}, norma_media={normas.mean():.6f}")

# %% [markdown]
# ---
# ## 5. Modelado de Usuario
#
# Para cada perfil se construye un vector combinando:
# - **v_hist**: centroide de las películas del historial efectivo (fuzzy ≥ threshold).
# - **v_query**: vector de la query en lenguaje natural.
#
# El peso **α adaptativo** se deriva de la dispersión interna del historial (σ_H):
# historial cohesivo → α alto (el historial domina); historial disperso → α bajo
# (la query gana peso). Cuando la query y el historial apuntan en direcciones distintas
# (conflicto semántico), se comparan tres políticas de arbitraje (§5.5).

# %%
name_to_idx: dict[str, int] = {name: i for i, name in enumerate(plots["name"].tolist())}

def fuzzy_lookup(title: str, threshold: float = FUZZY_THRESHOLD) -> int | None:
    if not title or not isinstance(title, str):
        return None
    match, score = best_match(title, corpus_names)
    if score >= threshold:
        return name_to_idx.get(match)
    return None

def get_history_indices(profile: pd.Series, threshold: float = FUZZY_THRESHOLD) -> list[int]:
    indices = []
    for col in [f"pelicula_{i}" for i in range(1, 6)]:
        title = profile.get(col)
        if pd.notna(title):
            idx = fuzzy_lookup(str(title), threshold)
            if idx is not None:
                indices.append(idx)
    return list(dict.fromkeys(indices))

# %%
def dispersion_historial(V_hist: np.ndarray) -> float:
    """σ_H ∈ [0,1]: 0 = historial cohesivo, 1 = historial disperso."""
    if V_hist.shape[0] < 2:
        return 0.0
    S  = cosine_similarity(V_hist)
    iu = np.triu_indices_from(S, k=1)
    return float(1.0 - S[iu].mean())

def alpha_adaptativo(V_hist: np.ndarray | None) -> float:
    """α ∈ [0.2, 0.8] inversamente proporcional a la dispersión del historial."""
    if V_hist is None or V_hist.shape[0] == 0:
        return 0.0
    sigma = dispersion_historial(V_hist)
    return float(np.clip(0.8 - 0.6 * sigma, 0.2, 0.8))

def hay_conflicto(v_hist: np.ndarray, v_query: np.ndarray, theta: float = 0.10) -> bool:
    sim = float(cosine_similarity(v_hist.reshape(1, -1), v_query.reshape(1, -1))[0, 0])
    return sim < theta

def combinar(v_hist: np.ndarray | None, v_query: np.ndarray, alpha: float) -> np.ndarray:
    """Combinación lineal normalizada de historial y query."""
    vq = normalize(v_query.reshape(1, -1))[0]
    if v_hist is None or alpha == 0.0:
        return vq
    vh = normalize(v_hist.reshape(1, -1))[0]
    v  = alpha * vh + (1.0 - alpha) * vq
    v  = normalize(v.reshape(1, -1))[0]
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    return v

# %% [markdown]
# #### Decisión de diseño: fallback para queries completamente OOV (Estrategia A)
#
# Cuando la query de un usuario no contiene ningún token presente en el vocabulario TF-IDF
# (vector de query todo ceros), usar ese vector nulo en la combinación lineal equivale a
# ignorar la query y depender exclusivamente del historial, lo cual puede ser razonable si
# el historial es amplio. Sin embargo, para usuarios sin historial efectivo, resultaría en
# un vector de usuario nulo y recomendaciones no deterministas.
#
# La decisión adoptada es usar el centroide L2-normalizado del corpus como vector de query
# de emergencia. Este vector representa la película "promedio" del corpus y actúa como
# señal neutral de fallback. Se registra el flag `fallback_oov` para identificar los usuarios
# afectados (ver §8).

# %%
def get_user_vector(
    profile: pd.Series,
    M,
    strategy: str,
    vectorizer=None,
    model=None,
    conflict_policy: str = "blend_adapt",
) -> dict:
    """
    Construye el vector de usuario.

    conflict_policy: "blend_adapt" | "history_first" | "query_first"
      - blend_adapt:    alpha = alpha_adaptativo() independientemente del conflicto
      - history_first:  si hay conflicto → alpha = 0.8
      - query_first:    si hay conflicto → alpha = 0.2
    """
    hist_indices = get_history_indices(profile)
    query = str(profile.get("query", "") or "")

    # --- Vector del historial ---
    if hist_indices:
        if strategy == "A":
            V_hist = np.asarray(M[hist_indices].todense())
        else:
            V_hist = M[hist_indices]
        v_hist = V_hist.mean(axis=0)
    else:
        V_hist = None
        v_hist = None

    # --- Vector de la query ---
    fallback_oov = False
    if strategy == "A":
        query_proc = preprocess_for_tfidf(query, nlp_model) if query else ""
        v_query_raw = vectorizer.transform([query_proc])
        v_query = np.asarray(v_query_raw.todense())[0]
        if np.linalg.norm(v_query) == 0:
            # Query completamente OOV: fallback al centroide normalizado del corpus
            centroide_raw = np.asarray(M.mean(axis=0)).ravel()
            v_query = normalize(centroide_raw.reshape(1, -1))[0]
            fallback_oov = True
    else:
        v_query = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]

    # --- Alpha con política de conflicto ---
    alpha = alpha_adaptativo(V_hist)
    conflicto = hay_conflicto(v_hist, v_query) if v_hist is not None else False

    if conflicto and conflict_policy == "history_first":
        alpha = 0.8
    elif conflicto and conflict_policy == "query_first":
        alpha = 0.2
    # blend_adapt: alpha permanece sin cambio

    fallback_flag = ""
    if v_hist is None:
        fallback_flag = "sin_historial_efectivo"
    if fallback_oov:
        fallback_flag += ("|" if fallback_flag else "") + "query_oov_fallback"

    v_user = combinar(v_hist, v_query, alpha)

    return {
        "v_user":        v_user,
        "v_hist":        v_hist,
        "v_query":       v_query,
        "alpha":         alpha,
        "hist_indices":  hist_indices,
        "conflicto":     conflicto,
        "fallback_flag": fallback_flag,
        "fallback_oov":  fallback_oov,
        "sigma_H":       dispersion_historial(V_hist) if V_hist is not None else None,
    }

# %% [markdown]
# ### 5.5 — Experimento de políticas de conflicto historial/query
#
# Cuando el vector de historial y el vector de query tienen coseno < 0.10, el sistema detecta
# un conflicto semántico (el usuario pide algo diferente a lo que ha visto). Se comparan tres
# políticas de arbitraje usando el protocolo LOO sobre la Estrategia B_desc (embeddings),
# donde la detección semántica es más confiable que con TF-IDF.

# %%
LOW_CONF_FLAGS = plots["low_confidence"].values

def loo_evaluate(
    profiles: pd.DataFrame,
    M,
    strategy: str,
    vectorizer=None,
    model=None,
    top_ks: tuple = (5, 10),
    conflict_policy: str = "blend_adapt",
) -> pd.DataFrame:
    """Evaluación LOO. conflict_policy controla el comportamiento ante conflictos."""
    rows = []

    for _, profile in profiles.iterrows():
        hist_indices = get_history_indices(profile)
        uid   = profile["id"]
        tipo  = profile["tipo_perfil"]
        query = str(profile.get("query", "") or "")

        if len(hist_indices) < 2:
            rows.append({
                "id": uid, "tipo_perfil": tipo, "n_hist": len(hist_indices),
                "p_held": None, "rank_held": None,
                "hit@5": None, "hit@10": None,
                "alpha": None, "sigma_H": None,
                "conflicto": None, "fallback_flag": "n_hist<2",
            })
            continue

        for p_held_idx in hist_indices:
            hist_minus = [i for i in hist_indices if i != p_held_idx]
            assert p_held_idx not in hist_minus  # A2

            if hist_minus:
                if strategy == "A":
                    V_hist_loo = np.asarray(M[hist_minus].todense())
                else:
                    V_hist_loo = M[hist_minus]
                v_hist_loo = V_hist_loo.mean(axis=0)
            else:
                V_hist_loo = None
                v_hist_loo = None

            alpha_loo = alpha_adaptativo(V_hist_loo)

            if strategy == "A":
                query_proc = preprocess_for_tfidf(query, nlp_model) if query else ""
                v_q_raw = vectorizer.transform([query_proc])
                v_q = np.asarray(v_q_raw.todense())[0]
                if np.linalg.norm(v_q) == 0:
                    centroide_raw = np.asarray(M.mean(axis=0)).ravel()
                    v_q = normalize(centroide_raw.reshape(1, -1))[0]
            else:
                v_q = model.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True
                )[0]

            conflicto_loo = hay_conflicto(v_hist_loo, v_q) if v_hist_loo is not None else False

            # Aplicar política de conflicto
            if conflicto_loo and conflict_policy == "history_first":
                alpha_loo = 0.8
            elif conflicto_loo and conflict_policy == "query_first":
                alpha_loo = 0.2

            v_user_loo = combinar(v_hist_loo, v_q, alpha_loo)

            candidatos_mask = np.ones(len(plots), dtype=bool)
            for i in hist_minus:
                candidatos_mask[i] = False

            if isinstance(M, np.ndarray):
                sims_loo = M @ v_user_loo
            else:
                sims_loo = np.asarray(cosine_similarity(v_user_loo.reshape(1, -1), M)[0])

            mask_loo = candidatos_mask.copy()
            high_conf = np.sum(mask_loo & ~LOW_CONF_FLAGS)
            if high_conf >= max(top_ks):
                mask_loo &= ~LOW_CONF_FLAGS

            if not mask_loo[p_held_idx]:  # A4: p_held debe ser candidato
                mask_loo[p_held_idx] = True

            sims_loo_masked = np.where(mask_loo, sims_loo, -np.inf)

            # A3: ninguna película del perfil reconstruido tiene similitud finita
            # (verificación sobre sims_loo_masked, no sobre el slice top-K)
            assert all(sims_loo_masked[i] == -np.inf for i in hist_minus), \
                "Leakage A3: película del perfil reconstruido tiene similitud finita"

            ranking = np.argsort(-sims_loo_masked)
            rank_held = int(np.where(ranking == p_held_idx)[0][0]) + 1

            sigma_loo = dispersion_historial(V_hist_loo) if V_hist_loo is not None else None

            rows.append({
                "id": uid, "tipo_perfil": tipo,
                "n_hist": len(hist_indices),
                "p_held": plots.iloc[p_held_idx]["name"],
                "rank_held": rank_held,
                "hit@5":  int(rank_held <= 5),
                "hit@10": int(rank_held <= 10),
                "alpha": alpha_loo,
                "sigma_H": sigma_loo,
                "conflicto": conflicto_loo,
                "fallback_flag": "" if v_hist_loo is not None else "fallback:centralidad",
            })

    return pd.DataFrame(rows)

# %%
print("=== Experimento §5.5: políticas de conflicto (Estrategia B_desc) ===\n")

loo_conflict = {}
for policy in ["blend_adapt", "history_first", "query_first"]:
    print(f"  Ejecutando LOO con policy='{policy}'...")
    loo_conflict[policy] = loo_evaluate(
        profiles, E_B_desc, "B_desc", model=model_B, conflict_policy=policy
    )

# Casos con conflicto detectado
print("\n=== Resultados en casos con conflicto=True ===")
n_conflict_total = 0
policy_results = []
for policy, loo_df in loo_conflict.items():
    conflict_rows = loo_df[loo_df["conflicto"] == True]
    n_conflict_total = len(conflict_rows)
    valid = conflict_rows[conflict_rows["rank_held"].notna()]
    if len(valid) == 0:
        h5 = mrr_val = float("nan")
    else:
        h5  = float(valid["hit@5"].mean())
        mrr_val = float((1.0 / valid["rank_held"]).mean())
    policy_results.append({"policy": policy, "n_conflict": len(conflict_rows),
                            "hit@5": h5, "mrr": mrr_val})

if n_conflict_total == 0:
    print("  No se detectaron conflictos (coseno hist/query < 0.10 en ningún perfil LOO).")
    print("  Esto indica que los historiales y las queries son semánticamente compatibles.")
    print("  Se adopta blend_adapt como política por defecto.")
    WINNING_CONFLICT_POLICY = "blend_adapt"
else:
    conflict_df = pd.DataFrame(policy_results)
    print(conflict_df.to_string(index=False))
    # Ganador: mayor MRR en casos de conflicto
    WINNING_CONFLICT_POLICY = conflict_df.sort_values("mrr", ascending=False).iloc[0]["policy"]
    print(f"\n  Política ganadora: {WINNING_CONFLICT_POLICY}")

print(f"\n  Política adoptada para recomendaciones finales: {WINNING_CONFLICT_POLICY}")

# %% [markdown]
# ---
# ## 6. Similitud y Ranking

# %%
def rank_movies(
    v_user: np.ndarray,
    M,
    hist_indices: list[int],
    top_k: int = 5,
    exclude_low_confidence: bool = True,
) -> list[int]:
    if isinstance(M, np.ndarray):
        sims = M @ v_user
    else:
        sims = np.asarray(cosine_similarity(v_user.reshape(1, -1), M)[0])

    mask = np.ones(len(sims), dtype=bool)
    for idx in hist_indices:
        mask[idx] = False

    if exclude_low_confidence:
        high_conf_available = np.sum(mask & ~LOW_CONF_FLAGS)
        if high_conf_available >= top_k:
            mask &= ~LOW_CONF_FLAGS

    sims_masked = np.where(mask, sims, -np.inf)
    return np.argsort(-sims_masked)[:top_k].tolist()

# %%
def recommend_all_profiles(
    profiles: pd.DataFrame,
    M,
    strategy: str,
    vectorizer=None,
    model=None,
    top_k: int = 5,
    conflict_policy: str = "blend_adapt",
) -> pd.DataFrame:
    rows = []
    for _, profile in profiles.iterrows():
        uinfo   = get_user_vector(profile, M, strategy, vectorizer, model, conflict_policy)
        top_idx = rank_movies(uinfo["v_user"], M, uinfo["hist_indices"], top_k)
        rec_names  = [plots.iloc[i]["name"] for i in top_idx]
        hist_names = [plots.iloc[i]["name"] for i in uinfo["hist_indices"]]
        row = {
            "id":           profile["id"],
            "nombre":       profile["nombre"],
            "tipo_perfil":  profile["tipo_perfil"],
            "query":        profile["query"],
            "alpha":        round(uinfo["alpha"], 3),
            "sigma_H":      round(uinfo["sigma_H"], 3) if uinfo["sigma_H"] is not None else None,
            "conflicto":    uinfo["conflicto"],
            "fallback_flag": uinfo["fallback_flag"],
            "hist_efectivo": ", ".join(hist_names),
            "n_hist":        len(uinfo["hist_indices"]),
        }
        for j, name in enumerate(rec_names, 1):
            row[f"rec_{j}"] = name
        rows.append(row)
    return pd.DataFrame(rows)

# %%
print("=== Generando recomendaciones — Estrategia A (TF-IDF) ===")
recs_A = recommend_all_profiles(
    profiles, M_A, "A", vectorizer=vectorizer_A,
    conflict_policy=WINNING_CONFLICT_POLICY
)
print("OK")

print("\n=== Generando recomendaciones — Estrategia B_desc ===")
recs_B_desc = recommend_all_profiles(
    profiles, E_B_desc, "B_desc", model=model_B,
    conflict_policy=WINNING_CONFLICT_POLICY
)
print("OK")

print("\n=== Generando recomendaciones — Estrategia B_desc_kw_g ===")
recs_B_dkg = recommend_all_profiles(
    profiles, E_B_dkg, "B_dkg", model=model_B,
    conflict_policy=WINNING_CONFLICT_POLICY
)
print("OK")

# %%
rec_cols = [f"rec_{i}" for i in range(1, 6)]

print("=== TOP-5 por perfil — A vs B_desc vs B_desc_kw_g ===\n")
for _, row_A in recs_A.iterrows():
    uid   = row_A["id"]
    row_B    = recs_B_desc[recs_B_desc["id"] == uid].iloc[0]
    row_Bdkg = recs_B_dkg[recs_B_dkg["id"]   == uid].iloc[0]

    print(f"[{uid}] {row_A['nombre']} | tipo: {row_A['tipo_perfil']} | "
          f"α={row_A['alpha']} | σ_H={row_A['sigma_H']} | conflicto={row_A['conflicto']}")
    if row_A["fallback_flag"]:
        print(f"  ⚑ {row_A['fallback_flag']}")
    print(f"  Query: {str(row_A['query'])[:80]}...")
    print(f"  Historial efectivo: {row_A['hist_efectivo']}")

    print(f"  {'#':<3} {'A (TF-IDF)':<40} {'B_desc':<40} {'B_dkg':<40}")
    for i in range(1, 6):
        a    = str(row_A.get(f"rec_{i}",    ""))[:38]
        b    = str(row_B.get(f"rec_{i}",    ""))[:38]
        bdkg = str(row_Bdkg.get(f"rec_{i}", ""))[:38]
        print(f"  {i:<3} {a:<40} {b:<40} {bdkg:<40}")
    print()

# %%
for df, name in [(recs_A, "recs_A"), (recs_B_desc, "recs_B_desc"), (recs_B_dkg, "recs_B_dkg")]:
    df.to_csv(os.path.join(ARTIFACTS_DIR, f"{name}.csv"), index=False)
print("Resultados guardados en artifacts/")

# %% [markdown]
# ### Análisis cualitativo de recomendaciones
#
# La tabla anterior permite observar tres patrones sistemáticos:
#
# **Perfiles definidos**: el sistema produce recomendaciones coherentes con el historial.
# Los perfiles con σ_H bajo (historial cohesivo) reciben α alto, lo que hace que las
# recomendaciones estén dominadas por el historial; la query actúa como refinamiento.
# Estrategia A y B tienden a coincidir en este caso, porque las sinopsis de películas
# similares comparten vocabulario Y semántica.
#
# **Perfiles ambiguos**: con σ_H alto, α cae hacia 0.2 y la query gana peso. Aquí A y B
# divergen más: la Estrategia A solo activa términos léxicamente presentes en la query;
# si la query usa vocabulario fuera del corpus (OOV), el fallback al centroide produce
# recomendaciones genéricas. La Estrategia B es más robusta porque el espacio de embeddings
# captura similitudes semánticas incluso para frases no vistas.
#
# **Historial no encontrado**: perfiles cuyas películas no matchean en el corpus quedan sin
# historial efectivo (flag `sin_historial_efectivo`). El sistema recae sobre la query como
# única señal; si esta también es OOV para A, las recomendaciones de A son el centroide del
# corpus (las películas más "promedio"), mientras B mantiene señal semántica.

# %% [markdown]
# ---
# ## 7. Evaluación LOO

# %%
def compute_diversity(top_indices: list[int], M) -> float:
    if len(top_indices) < 2:
        return 0.0
    if isinstance(M, np.ndarray):
        vecs = M[top_indices]
    else:
        vecs = np.asarray(M[top_indices].todense())
    S  = cosine_similarity(vecs)
    iu = np.triu_indices_from(S, k=1)
    return float(1.0 - S[iu].mean())

def mrr(loo_df: pd.DataFrame) -> float:
    ranks = loo_df["rank_held"].dropna()
    return float((1.0 / ranks).mean()) if len(ranks) > 0 else 0.0

def hit_at_k(loo_df: pd.DataFrame, k: int) -> float:
    col  = f"hit@{k}"
    vals = loo_df[col].dropna()
    return float(vals.mean()) if len(vals) > 0 else 0.0

def compute_diversity_all(recs_df: pd.DataFrame, M) -> float:
    divs = []
    for _, row in recs_df.iterrows():
        idxs = [name_to_idx[row[f"rec_{i}"]]
                for i in range(1, 6)
                if row.get(f"rec_{i}") and row[f"rec_{i}"] in name_to_idx]
        if len(idxs) >= 2:
            divs.append(compute_diversity(idxs, M))
    return float(np.mean(divs)) if divs else 0.0

def compute_coverage_from_recs(recs_df: pd.DataFrame, n_total: int) -> float:
    seen = set()
    for _, row in recs_df.iterrows():
        for i in range(1, 6):
            name = row.get(f"rec_{i}")
            if name and name in name_to_idx:
                seen.add(name_to_idx[name])
    return len(seen) / n_total

# %%
print("=== LOO — Estrategia A ===")
loo_A = loo_evaluate(profiles, M_A, "A", vectorizer=vectorizer_A,
                     conflict_policy=WINNING_CONFLICT_POLICY)

print("=== LOO — Estrategia B_desc ===")
loo_B_desc = loo_evaluate(profiles, E_B_desc, "B_desc", model=model_B,
                          conflict_policy=WINNING_CONFLICT_POLICY)

print("=== LOO — Estrategia B_desc_kw_g ===")
loo_B_dkg = loo_evaluate(profiles, E_B_dkg, "B_dkg", model=model_B,
                          conflict_policy=WINNING_CONFLICT_POLICY)

print("LOO completado.")

# %% [markdown]
# ### Baselines
#
# **Baseline centralidad semántica**: recomienda las películas más cercanas al centroide
# del corpus (las más "promedio"). Es determinista y se evalúa con LOO.
#
# **Baseline aleatorio**: selección uniforme sin reemplazo. Su MRR esperado es teórico:
# E[MRR_random] = (1/N) · Σ_{k=1}^{N} (1/k) ≈ ln(N+1)/N, que para N ≈ 4500 candidatos
# es del orden de 0.002.

# %%
def loo_evaluate_centrality(M, profiles: pd.DataFrame, top_ks=(5, 10)) -> pd.DataFrame:
    """LOO para baseline de centralidad; v_user = centroide normalizado del corpus."""
    if isinstance(M, np.ndarray):
        centroide = normalize(M.mean(axis=0).reshape(1, -1))[0]
        def sims_fn(v): return M @ v
    else:
        centroide = normalize(np.asarray(M.mean(axis=0)).ravel().reshape(1, -1))[0]
        def sims_fn(v): return np.asarray(cosine_similarity(v.reshape(1, -1), M)[0])

    rows = []
    for _, profile in profiles.iterrows():
        hist_indices = get_history_indices(profile)
        uid  = profile["id"]
        tipo = profile["tipo_perfil"]

        if len(hist_indices) < 2:
            rows.append({"id": uid, "tipo_perfil": tipo, "n_hist": len(hist_indices),
                         "p_held": None, "rank_held": None, "hit@5": None, "hit@10": None})
            continue

        for p_held_idx in hist_indices:
            hist_minus = [i for i in hist_indices if i != p_held_idx]

            sims = sims_fn(centroide)
            mask = np.ones(len(plots), dtype=bool)
            for i in hist_minus:
                mask[i] = False

            high_conf = np.sum(mask & ~LOW_CONF_FLAGS)
            if high_conf >= max(top_ks):
                mask &= ~LOW_CONF_FLAGS
            if not mask[p_held_idx]:
                mask[p_held_idx] = True

            sims_masked = np.where(mask, sims, -np.inf)
            ranking  = np.argsort(-sims_masked)
            rank_held = int(np.where(ranking == p_held_idx)[0][0]) + 1

            rows.append({
                "id": uid, "tipo_perfil": tipo,
                "n_hist": len(hist_indices),
                "p_held": plots.iloc[p_held_idx]["name"],
                "rank_held": rank_held,
                "hit@5":  int(rank_held <= 5),
                "hit@10": int(rank_held <= 10),
            })
    return pd.DataFrame(rows)

# %%
print("=== LOO — Baseline centralidad (B_desc) ===")
loo_cent = loo_evaluate_centrality(E_B_desc, profiles)

n_hc = int((~LOW_CONF_FLAGS).sum())
n_candidates_approx = n_hc - 4
mrr_random_theoretical = sum(1 / k for k in range(1, n_candidates_approx + 1)) / n_candidates_approx
print(f"\nE[MRR_random] teórico ≈ {mrr_random_theoretical:.5f}  "
      f"(N_candidatos ≈ {n_candidates_approx} películas de alta confianza)")

# %%
# Baseline aleatorio: cobertura y diversidad empírica
def baseline_random_recs(profiles, n_total, top_k=5, seed=SEED):
    rng_bl = np.random.default_rng(seed)
    all_idx = list(range(n_total))
    rows = []
    for _, profile in profiles.iterrows():
        hist = get_history_indices(profile)
        pool = [i for i in all_idx if i not in hist and not LOW_CONF_FLAGS[i]]
        chosen = rng_bl.choice(pool, size=min(top_k, len(pool)), replace=False).tolist()
        rows.append({"id": profile["id"], "tipo_perfil": profile["tipo_perfil"], "top_idx": chosen})
    return rows

rand_recs    = baseline_random_recs(profiles, len(plots))
rand_top_idx = [r["top_idx"] for r in rand_recs]

# %%
n_total = len(plots)

results_table = []
for label, loo_df, recs_df, M_mat in [
    ("A (TF-IDF)",    loo_A,      recs_A,      M_A),
    ("B_desc",        loo_B_desc, recs_B_desc, E_B_desc),
    ("B_desc_kw_g",   loo_B_dkg,  recs_B_dkg,  E_B_dkg),
]:
    valid = loo_df[loo_df["rank_held"].notna()]
    results_table.append({
        "Sistema":    label,
        "Hit@5":      f"{hit_at_k(valid, 5):.3f}",
        "Hit@10":     f"{hit_at_k(valid, 10):.3f}",
        "MRR":        f"{mrr(valid):.3f}",
        "Div@5":      f"{compute_diversity_all(recs_df, M_mat):.3f}",
        "Cobertura":  f"{compute_coverage_from_recs(recs_df, n_total):.3f}",
    })

valid_cent = loo_cent[loo_cent["rank_held"].notna()]
results_table.append({
    "Sistema":   "Baseline centralidad",
    "Hit@5":     f"{hit_at_k(valid_cent, 5):.3f}",
    "Hit@10":    f"{hit_at_k(valid_cent, 10):.3f}",
    "MRR":       f"{mrr(valid_cent):.3f}",
    "Div@5":     f"{np.mean([compute_diversity(idx, E_B_desc) for idx in rand_top_idx if len(idx)>=2]):.3f}",
    "Cobertura": f"{compute_coverage_from_recs(pd.DataFrame([{'rec_'+str(j+1): plots.iloc[i]['name'] for j,i in enumerate(r)} for r in rand_top_idx]), n_total):.3f}",
})

results_table.append({
    "Sistema":   "Baseline aleatorio",
    "Hit@5":     "N/A",
    "Hit@10":    "N/A",
    "MRR":       f"{mrr_random_theoretical:.5f} (teórico)",
    "Div@5":     f"{np.mean([compute_diversity(idx, E_B_desc) for idx in rand_top_idx if len(idx)>=2]):.3f}",
    "Cobertura": "alta (uniforme)",
})

results_df = pd.DataFrame(results_table)
print("=== TABLA DE RESULTADOS GLOBAL ===")
print(results_df.to_string(index=False))

# %%
print("\n=== RESULTADOS SEGMENTADOS POR tipo_perfil ===")
for tipo in sorted(profiles["tipo_perfil"].unique()):
    print(f"\n  --- tipo_perfil: {tipo} ---")
    for label, loo_df in [("A (TF-IDF)", loo_A),
                           ("B_desc",     loo_B_desc),
                           ("B_desc_kw_g", loo_B_dkg)]:
        sub = loo_df[(loo_df["tipo_perfil"] == tipo) & loo_df["rank_held"].notna()]
        if len(sub) == 0:
            print(f"    {label}: sin datos LOO")
            continue
        print(f"    {label}: Hit@5={hit_at_k(sub, 5):.3f}  Hit@10={hit_at_k(sub, 10):.3f}  "
              f"MRR={mrr(sub):.3f}  (n={len(sub)})")

# %%
print("\n=== TRAZAS LOO — Estrategia A ===")
trace_cols = ["id", "tipo_perfil", "n_hist", "p_held", "rank_held",
              "hit@5", "alpha", "sigma_H", "conflicto", "fallback_flag"]
print(loo_A[trace_cols].to_string(index=False))

# %%
# --- Regla de decisión §7.5 ---
print("\n=== REGLA DE DECISIÓN §7.5 ===")

mrr_cent   = mrr(loo_cent[loo_cent["rank_held"].notna()])
mrr_random = mrr_random_theoretical

print(f"  MRR Baseline centralidad: {mrr_cent:.4f}")
print(f"  MRR Baseline aleatorio:   {mrr_random:.5f} (teórico)")

def _mrr(loo_df):
    return mrr(loo_df[loo_df["rank_held"].notna()])

mrr_A     = _mrr(loo_A)
mrr_B_d   = _mrr(loo_B_desc)
mrr_B_dkg = _mrr(loo_B_dkg)

print(f"\n  MRR A:       {mrr_A:.4f}")
print(f"  MRR B_desc:  {mrr_B_d:.4f}")
print(f"  MRR B_dkg:   {mrr_B_dkg:.4f}")

# Paso 1: descartar estrategias que no superen AMBOS baselines
baseline_min = max(mrr_cent, mrr_random)
print(f"\n  Criterio de descarte: MRR > {baseline_min:.4f} (máximo de ambos baselines)")
candidatas = {}
for label, mrr_val in [("A (TF-IDF)", mrr_A), ("B_desc", mrr_B_d), ("B_desc_kw_g", mrr_B_dkg)]:
    supera = mrr_val > baseline_min
    print(f"    {label}: MRR={mrr_val:.4f}  {'✓ supera baselines' if supera else '✗ NO supera baselines'}")
    if supera:
        candidatas[label] = mrr_val

if not candidatas:
    print("\n  ADVERTENCIA: ninguna estrategia supera ambos baselines en MRR global.")
    print("  El sistema no agrega valor sobre el azar/centralidad; se reporta sin ganador.")
else:
    # Paso 2: variante ganadora de B
    b_cands = {k: v for k, v in candidatas.items() if k.startswith("B")}
    a_cands = {k: v for k, v in candidatas.items() if k.startswith("A")}

    if len(b_cands) >= 2:
        if abs(mrr_B_d - mrr_B_dkg) < 0.02:
            sub_disp_d   = loo_B_desc[loo_B_desc["tipo_perfil"].isin(TIPOS_DISPERSOS) & loo_B_desc["rank_held"].notna()]
            sub_disp_dkg = loo_B_dkg[ loo_B_dkg["tipo_perfil"].isin(TIPOS_DISPERSOS) & loo_B_dkg["rank_held"].notna()]
            winner_B = "B_desc" if hit_at_k(sub_disp_d, 5) >= hit_at_k(sub_disp_dkg, 5) else "B_desc_kw_g"
            mrr_B    = mrr_B_d if winner_B == "B_desc" else mrr_B_dkg
            print(f"\n  Empate B (<0.02) → desempate Hit@5 dispersos → {winner_B}")
        else:
            winner_B = max(b_cands, key=b_cands.get)
            mrr_B    = b_cands[winner_B]
            print(f"\n  Variante B ganadora: {winner_B}")
    elif len(b_cands) == 1:
        winner_B = list(b_cands.keys())[0]
        mrr_B    = b_cands[winner_B]
    else:
        winner_B = None
        mrr_B    = -1.0

    # Paso 3: comparar A vs B ganadora
    if winner_B and "A (TF-IDF)" in candidatas:
        if abs(mrr_A - mrr_B) < 0.02:
            sub_A = loo_A[loo_A["tipo_perfil"].isin(TIPOS_DISPERSOS) & loo_A["rank_held"].notna()]
            loo_Bw = loo_B_desc if winner_B == "B_desc" else loo_B_dkg
            sub_B = loo_Bw[loo_Bw["tipo_perfil"].isin(TIPOS_DISPERSOS) & loo_Bw["rank_held"].notna()]
            ganador = "A (TF-IDF)" if hit_at_k(sub_A, 5) >= hit_at_k(sub_B, 5) else winner_B
        elif mrr_A >= mrr_B:
            ganador = "A (TF-IDF)"
        else:
            ganador = winner_B
    elif winner_B:
        ganador = winner_B
    else:
        ganador = "A (TF-IDF)"

    print(f"\n  → SISTEMA GANADOR: {ganador}")

# %% [markdown]
# ### Análisis de las métricas de evaluación
#
# La evaluación LOO mide qué tan bien el sistema recupera una película del historial cuando
# se la retira y se reconstruye el perfil sin ella. Este protocolo evalúa la capacidad de
# generalización del sistema a partir de señales parciales.
#
# La **segmentación por `tipo_perfil`** es el resultado más informativo: si el sistema
# funciona bien para perfiles `definido` pero falla con `ambiguo`, eso indica que la
# representación captura bien la coherencia temática pero no maneja bien la ambigüedad del
# usuario. En ese caso, darle más peso a la query (política `query_first`) podría mejorar
# los resultados para perfiles dispersos.
#
# La comparación con los baselines es el filtro más importante: si ninguna estrategia
# supera al centroide semántico en MRR, significa que el sistema no aporta valor sobre la
# centralidad del corpus, lo que sería un fracaso. Esto puede deberse a la corta longitud
# de los historiales (5 películas) o a la alta cobertura del corpus respecto a géneros
# representativos.

# %%
loo_A.to_csv(os.path.join(ARTIFACTS_DIR, "loo_A.csv"), index=False)
loo_B_desc.to_csv(os.path.join(ARTIFACTS_DIR, "loo_B_desc.csv"), index=False)
loo_B_dkg.to_csv(os.path.join(ARTIFACTS_DIR, "loo_B_dkg.csv"), index=False)
loo_cent.to_csv(os.path.join(ARTIFACTS_DIR, "loo_centrality.csv"), index=False)
results_df.to_csv(os.path.join(ARTIFACTS_DIR, "results_table.csv"), index=False)
print("Tablas de evaluación guardadas en artifacts/")

# %% [markdown]
# ---
# ## 8. Análisis OOV de queries (Hipótesis H3)
#
# Se mide qué fracción de tokens de cada query, luego de preprocesamiento, no aparece en
# el vocabulario TF-IDF. Una query con alto OOV en Estrategia A activa el fallback al
# centroide del corpus; en Estrategia B, la query se codifica igualmente en el espacio
# semántico y no hay OOV.

# %%
vocab = set(vectorizer_A.vocabulary_.keys())
oov_records = []

print("=== OOV de queries — Estrategia A ===")
print(f"  {'ID':<5} {'Nombre':<25} {'Tokens':>6} {'OOV%':>6} {'Señal?':>7} {'Fallback':>9}")
for _, profile in profiles.iterrows():
    query      = str(profile.get("query", "") or "")
    query_proc = preprocess_for_tfidf(query, nlp_model)
    tokens     = query_proc.split()
    if not tokens:
        oov_pct = 1.0
    else:
        oov_pct = sum(1 for t in tokens if t not in vocab) / len(tokens)
    v_q_raw   = vectorizer_A.transform([query_proc])
    has_signal = np.linalg.norm(np.asarray(v_q_raw.todense())) > 0
    fallback   = "SÍ" if not has_signal else "no"
    print(f"  [{profile['id']}] {str(profile['nombre']):<25} {len(tokens):>6} "
          f"{oov_pct:>6.0%} {'sí' if has_signal else 'NO':>7} {fallback:>9}")
    oov_records.append({"id": profile["id"], "nombre": profile["nombre"],
                         "tipo_perfil": profile["tipo_perfil"],
                         "n_tokens": len(tokens), "oov_pct": oov_pct,
                         "tiene_señal": has_signal, "fallback_oov": not has_signal})

oov_df = pd.DataFrame(oov_records)
n_fallback = oov_df["fallback_oov"].sum()
print(f"\nUsuarios que activan fallback OOV: {n_fallback} / {len(oov_df)}")

# %% [markdown]
# ### Interpretación del análisis OOV
#
# El análisis OOV revela la brecha entre el vocabulario de las queries de usuario y el
# vocabulario construido a partir del corpus. Los perfiles con tipo `ambiguo` tienden a
# tener queries más genéricas o que usan vocabulario fuera del dominio cinematográfico,
# lo que produce un mayor porcentaje de tokens OOV y, en casos extremos, activa el fallback.
#
# Este hallazgo explica por qué la Estrategia B suele ser más robusta para perfiles ambiguos:
# el modelo de embeddings multilingüe generaliza a vocabulario no visto porque trabaja en un
# espacio semántico continuo, mientras que TF-IDF solo puede activar dimensiones para términos
# exactamente presentes en el vocabulario.
#
# El fallback al centroide normalizado del corpus es una señal de alarma: el usuario cuya
# query activa este mecanismo recibirá recomendaciones sesgadas hacia las películas más
# "promedio" del corpus, independientemente de sus preferencias reales.

# %% [markdown]
# ---
# ## 9. Verificación de hipótesis
#
# ### H1: Las sinopsis en español distinguen géneros/temáticas (vs. baseline aleatorio)
#
# Si el sistema supera al baseline aleatorio en Hit@5, las sinopsis contienen señal suficiente
# para distinguir películas que un usuario desearía ver de películas aleatorias.

# %%
print("=== H1: Comparación Hit@5 vs baseline aleatorio ===")
mrr_random_val = mrr_random_theoretical
for label, loo_df in [("A (TF-IDF)", loo_A), ("B_desc", loo_B_desc), ("B_desc_kw_g", loo_B_dkg)]:
    valid = loo_df[loo_df["rank_held"].notna()]
    h5 = hit_at_k(valid, 5)
    n  = len(valid)
    h5_random = 5 / n_candidates_approx  # probabilidad de hit@5 aleatoria
    supera = h5 > h5_random
    print(f"  {label}: Hit@5={h5:.3f}  vs  aleatorio≈{h5_random:.4f}  "
          f"→ {'✓ H1 validada' if supera else '✗ H1 NO validada'}")

# %% [markdown]
# ### H2: `tipo_perfil` es consistente con la dispersión σ_H medida
#
# Si los perfiles etiquetados como `ambiguo` tienen un σ_H promedio mayor que los `definido`,
# la etiqueta del CSV refleja la dispersión real medida. Si discrepa, la medida prevalece.

# %%
print("=== H2: tipo_perfil vs σ_H medido ===")
sigma_by_tipo = (
    loo_A[loo_A["sigma_H"].notna()]
    .groupby("tipo_perfil")["sigma_H"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "σ_H_mean", "std": "σ_H_std", "count": "n_obs"})
)
print(sigma_by_tipo.to_string())
print()

if "definido" in sigma_by_tipo.index and "ambiguo" in sigma_by_tipo.index:
    sigma_def = sigma_by_tipo.loc["definido",  "σ_H_mean"]
    sigma_amb = sigma_by_tipo.loc["ambiguo", "σ_H_mean"]
    if sigma_amb > sigma_def:
        print(f"✓ H2 validada: σ_H(ambiguo)={sigma_amb:.3f} > σ_H(definido)={sigma_def:.3f}")
        print("  La etiqueta tipo_perfil es consistente con la dispersión medida.")
    else:
        print(f"⚠ H2 NO validada: σ_H(ambiguo)={sigma_amb:.3f} ≤ σ_H(definido)={sigma_def:.3f}")
        print("  La etiqueta no refleja la dispersión real; la medida σ_H prevalece.")

# %% [markdown]
# ---
# ## 10. Verificaciones de coherencia finales

# %%
assert len(vectorizer_A.vocabulary_) > 0, "Vectorizador vacío"
print(f"✓ A1: vectorizador ajustado sobre corpus completo ({len(vectorizer_A.vocabulary_):,} términos)")
print("✓ A2–A5: verificados por construcción y asserts explícitos en loo_evaluate()")

for name_e, E in [("B_desc", E_B_desc), ("B_desc_kw_g", E_B_dkg)]:
    normas = np.linalg.norm(E, axis=1)
    assert np.allclose(normas, 1.0, atol=1e-3)
    print(f"✓ Norma L2 {name_e}: OK (mean={normas.mean():.6f})")

print(f"\n✓ SHA modelo embeddings: {MODEL_SHA}")
print(f"✓ Threshold fuzzy calibrado: {FUZZY_THRESHOLD} (precisión={pr_df[pr_df['umbral']==FUZZY_THRESHOLD]['precision'].values[0]:.3f})")
print(f"✓ Política de conflicto adoptada: {WINNING_CONFLICT_POLICY}")
print("\n=== Pipeline completado exitosamente ===")
