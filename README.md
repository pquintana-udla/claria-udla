# CLARIA — Clinical Learning Analytics & Risk Intelligence

**Sistema de Alertas Tempranas para Estudiantes de Odontología Clínica III · UDLA Ecuador**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)](https://fastapi.tiangolo.com/)
[![ML](https://img.shields.io/badge/ML-Logistic%20Regression%20%2B%20Random%20Forest-orange)]()
[![AUC](https://img.shields.io/badge/AUC%20Ensemble-0.9977-brightgreen)]()

---

## ¿Qué es CLARIA?

CLARIA es un prototipo de machine learning para la identificación temprana de estudiantes en riesgo académico en el nivel Clínica III (octavo semestre) de la carrera de Odontología de la UDLA. El sistema analiza 62 variables operativas clínicas registradas a la **semana 6 del semestre** y genera una clasificación tripartita de riesgo para cada estudiante activo.

Desarrollado como trabajo de titulación de la **Maestría en Inteligencia Artificial Aplicada (MIA) · UDLA 2026**.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│  CAPA 1: Pipeline ML (run_pipeline_2026_2.py)           │
│  ├── Preprocesamiento + SMOTE (NumPy puro)              │
│  ├── Regresión Logística    AUC = 0.9967                │
│  ├── Random Forest (80 árboles)  AUC = 0.9982          │
│  └── Ensamble AUC-ponderado     AUC = 0.9977           │
├─────────────────────────────────────────────────────────┤
│  CAPA 2: Backend API (main.py)                          │
│  ├── FastAPI + SQLite                                   │
│  └── 8 endpoints REST (/api/summary, /api/students...)  │
├─────────────────────────────────────────────────────────┤
│  CAPA 3: Dashboard Web (static/index.html)              │
│  ├── Vista General  · KPIs + gráfico de dona            │
│  ├── Vista Cohorte  · tabla de 98 estudiantes           │
│  └── Vista Individual · gauge + perfil de riesgo        │
└─────────────────────────────────────────────────────────┘
```

---

## Resultados — Cohorte Prospectiva 2026-2

Aplicación sobre **98 estudiantes activos** (semana 6 del semestre):

| Nivel de Riesgo | Estudiantes | Porcentaje |
|---|---|---|
| 🔴 ALTO RIESGO | 82 | 83.7% |
| 🟡 MODERADO RIESGO | 6 | 6.1% |
| 🟢 BAJO RIESGO | 10 | 10.2% |

> La clasificación se basa en `p_riesgo = 1 − P(aprueba)` calculada por el ensamble.  
> Umbral ALTO: ≥ 0.65 · Umbral MODERADO: 0.40–0.64 · Umbral BAJO: < 0.40

---

## Estructura del Repositorio

```
claria-udla/
├── main.py                        # Servidor FastAPI (8 endpoints REST)
├── pipeline_results.json          # Resultados del pipeline ML (98 estudiantes)
├── CLARIA_Dashboard_Estatico.html # Dashboard standalone (sin servidor)
├── static/
│   └── index.html                 # Dashboard dinámico (consume la API)
├── requirements.txt               # Dependencias Python
├── render.yaml                    # Configuración despliegue Render.com
└── Procfile                       # Comando de inicio
```

---

## Instalación Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/pquintana-udla/claria-udla.git
cd claria-udla

# 2. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Iniciar el servidor
uvicorn main:app --reload --port 8000

# 4. Abrir el dashboard
# Navegar a: http://localhost:8000
```

---

## API Endpoints

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servidor |
| `GET /api/summary` | KPIs globales de la cohorte |
| `GET /api/students` | Lista de 98 estudiantes con scores |
| `GET /api/students/{id}` | Perfil detallado por estudiante |
| `GET /api/metrics` | Métricas de los 3 modelos (AUC, F1, Accuracy) |
| `GET /api/risk-distribution` | Distribución de riesgo |
| `GET /api/top-risk` | Top estudiantes por score descendente |
| `GET /api/feature-importance` | Importancia de variables del modelo |

---

## Tecnologías

- **ML**: NumPy · Pandas (implementación desde cero, sin scikit-learn)
- **API**: FastAPI · Uvicorn · SQLite · aiofiles
- **Frontend**: HTML5 · Chart.js · CSS3 (vanilla, sin frameworks)
- **Privacidad**: Anonimización SHA-256 con sal institucional (sin PII en dashboard)
- **Despliegue**: Render.com (free tier)

---

## Autor

**Pablo Alfredo Quintana Ramírez**  
Maestría en Inteligencia Artificial Aplicada · UDLA Ecuador · 2026  
📧 pabloq2k@gmail.com
