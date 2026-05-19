# Trabajo Práctico Final — Introducción al Procesamiento del Lenguaje Natural
**Universidad Nacional de San Martín (UNSAM) · Primer cuatrimestre 2026**

**Integrantes:** Emiliano Churruca · Gerardo Toboso · Juan Serrano

---

## Descripción del proyecto

Sistema de recomendación de películas **basado en contenido** que, dado el historial de visualización y una consulta en lenguaje natural de cada usuario, produce una lista de 5 películas relevantes.

El trabajo contrasta dos estrategias de representación de texto:

| Estrategia | Técnica | Espacio vectorial |
|---|---|---|
| **A — TF-IDF** | Representación léxica dispersa sobre sinopsis lematizadas | ~20.000–50.000 dimensiones |
| **B — Embeddings** | Sentence Transformers multilingüe (`paraphrase-multilingual-mpnet-base-v2`) | 768 dimensiones densas |

Ambas estrategias se evalúan bajo las mismas condiciones (protocolo Leave-One-Out, métricas MRR / Hit@5 / diversidad / cobertura) y se comparan entre sí y contra baselines explícitos.

---

## Datos

| Archivo | Descripción |
|---|---|
| `data/plots.csv` | Corpus principal: ~4.970 películas con sinopsis en español, keywords, género, año y director. Fuente: `mathigatti/spanish_imdb_synopsis`. |
| `data/user_profiles.csv` | 14 perfiles de usuario simulados, cada uno con historial de 5 películas y una query en lenguaje natural. Los perfiles varían en cohesión: algunos tienen preferencias claras (`definido`), otros son dispersos. |

> Los archivos de datos están versionados en el repositorio y **no requieren descarga adicional** para ejecutar el notebook.

---

## Estructura del repositorio

```
.
├── data/
│   ├── plots.csv            # Corpus de películas (fuente de verdad del sistema)
│   └── user_profiles.csv    # Perfiles de usuario con historial y query
│
├── docs/
│   └── arquitectura.md      # Especificación arquitectónica v2.0: diseño de las dos
│                            # estrategias, pipeline de datos, modelado de usuario,
│                            # protocolo experimental A/B y política de reproducibilidad
│
├── prompts/
│   ├── enunciado.xml        # Enunciado oficial del trabajo: objetivo, datos, preguntas
│   │                        # clave y entregables. Fuente de verdad del TP.
│   ├── guia_de_uso.xml      # Manual del flujo multi-agente: qué rol usar en cada fase
│   │                        # (diseño → implementación → redacción → evaluación)
│   └── roles/
│       ├── 01_arquitecto.xml         # Rol: diseño conceptual y matemático
│       ├── 02_ingeniero.xml          # Rol: implementación del notebook reproducible
│       ├── 03_redactor_cientifico.xml # Rol: redacción del informe PDF (≤10 páginas)
│       └── 04_evaluador.xml          # Rol: control de calidad crítico (simula al docente)
│
├── src/
│   ├── notebook.py          # Notebook en formato Jupytext (fuente editable, versionable)
│   ├── notebook.ipynb       # Notebook ejecutable generado desde notebook.py
│   └── download.py          # Script auxiliar de descarga de datos desde HuggingFace Hub
│
├── artifacts/               # (generado al ejecutar el notebook, no versionado)
│   ├── embeddings_B.npy     # Embeddings precomputados cacheados (Estrategia B)
│   ├── tfidf_*.pkl          # Matriz TF-IDF y vocabulario cacheados (Estrategia A)
│   └── fuzzy_matches.csv    # Tabla de matches fuzzy entre títulos del historial y el corpus
│
├── pyproject.toml           # Dependencias del proyecto (gestionadas con uv)
├── uv.lock                  # Lock file para reproducibilidad exacta del entorno
└── .gitignore
```

### Por qué cada cosa está donde está

- **`data/`** contiene únicamente los datos de entrada estables; no se mezclan con código ni artefactos generados.
- **`docs/`** aloja la especificación arquitectónica —el documento de diseño que precede a la implementación y que el Evaluador valida antes de codificar.
- **`prompts/`** centraliza los prompts de sistema usados durante el desarrollo con asistencia de IA. Separarlos del código los mantiene versionables y reutilizables como plantillas para cada fase del flujo.
- **`src/`** contiene el código ejecutable. Se usa Jupytext para mantener `notebook.py` como fuente editable en texto plano (diff limpio en git) y generar `notebook.ipynb` para ejecución.
- **`artifacts/`** está excluido del control de versiones porque sus contenidos se regeneran al ejecutar el notebook; son costosos de recomputar (especialmente los embeddings) y por eso se cachean en disco con criterios de invalidación explícitos.

---

## Cómo empezar

### 1. Requisitos previos

- Python >= 3.13
- [`uv`](https://docs.astral.sh/uv/) instalado globalmente

### 2. Clonar y configurar el entorno

```bash
git clone https://github.com/Gerardo1909/nlp-content-based-recommender
cd nlp-content-based-recommender

# Crear el entorno virtual e instalar dependencias pinneadas
uv sync
```

### 3. Convertir el notebook (Jupytext → .ipynb)

El archivo fuente es `src/notebook.py` (formato Jupytext). Para generar el `.ipynb` ejecutable:

```bash
uv run jupytext --to notebook src/notebook.py
```

### 4. Ejecutar el notebook

```bash
uv run jupyter notebook src/notebook.ipynb
```

O desde VS Code / cualquier entorno Jupyter, seleccionando el kernel del `.venv` generado por `uv`.

### 5. Ejecución end-to-end

El notebook está organizado en secciones que deben ejecutarse en orden:

1. **Configuración y reproducibilidad** — fija semillas, constantes y directorios
2. **Versiones del entorno** — imprime versiones exactas de cada dependencia
3. **EDA y calibración de fuzzy matching** — análisis exploratorio del corpus y calibración del umbral de coincidencia difusa de títulos
4. **Preprocesamiento de texto** — limpieza, lematización (spaCy) y construcción del texto de entrada por estrategia
5. **Vectorización** — construcción de la matriz TF-IDF (A) y cómputo/cache de embeddings (B)
6. **Modelado de usuario** — combinación lineal historial + query, α adaptativo por dispersión, análisis de conflicto
7. **Similitud y ranking** — similitud coseno, exclusión de vistas y películas de baja confianza
8. **Evaluación LOO** — protocolo Leave-One-Out con métricas MRR, Hit@5/10, diversidad y cobertura, segmentado por `tipo_perfil`
9. **Análisis OOV de queries** — cobertura vocabular de las queries sobre el vocabulario TF-IDF
10. **Verificación de hipótesis** — contraste A vs B con regla de decisión explícita

> La primera ejecución descarga el modelo de embeddings desde HuggingFace Hub (~1 GB) y tarda entre 2 y 5 minutos en CPU. Las siguientes ejecuciones cargan desde `artifacts/`.

---

## Flujo de desarrollo (multi-agente con IA)

El desarrollo se estructuró en cuatro fases con roles especializados definidos en `prompts/roles/`. Cada rol se usa en una sesión separada con el contexto de `enunciado.xml` como fuente de verdad:

```
Fase 1 — Diseño:      Arquitecto → Evaluador → Arquitecto (hasta cerrar bloqueantes)
Fase 2 — Implementación: Ingeniero → ejecución local → Evaluador
Fase 3 — Redacción:   Redactor Científico → Evaluador → Redactor
Fase 4 — Revisión:    Evaluador global → Redactor (guion de exposición)
```

Ver `prompts/guia_de_uso.xml` para el detalle de entradas, salidas y principios de cada paso.

---

## Stack tecnológico

| Herramienta | Uso |
|---|---|
| `pandas` | Carga y manipulación del corpus y perfiles |
| `scikit-learn` | TF-IDF (Estrategia A), similitud coseno, normalización |
| `spaCy` (`es_core_news_sm`) | Tokenización y lematización en español |
| `nltk` | Stopwords en español |
| `sentence-transformers` | Embeddings multilingüe (Estrategia B) |
| `rapidfuzz` | Fuzzy matching de títulos entre perfiles y corpus |
| `numpy` | Álgebra vectorial, cache de embeddings |
| `matplotlib` | Visualizaciones del EDA y resultados |
| `jupytext` | Notebook versionable en texto plano |
| `uv` | Gestión de entorno y dependencias con lock file |

---

## Entregables del trabajo práctico

- [x] Dos estrategias de representación contrastadas con análisis explícito de diferencias
- [ ] Notebook reproducible con outputs ejecutados (`src/notebook.ipynb`)
- [ ] Informe en PDF (máximo 10 páginas sin código ni outputs)
- [ ] Exposición de 10 minutos con los principales resultados
