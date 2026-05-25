"""
CLARIA — Backend FastAPI con SQLite
Clinical Learning Analytics & Risk Intelligence for Academic institutions
UDLA | Maestría en IA Aplicada | Pablo Quintana Ramírez

Ejecutar localmente:
    pip install fastapi uvicorn[standard] aiofiles
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Abrir en navegador: http://localhost:8000
API docs:           http://localhost:8000/docs
"""

import json
import sqlite3
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ─── RUTAS ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DB_PATH     = BASE_DIR / "claria.db"
RESULTS_PATH = BASE_DIR / "pipeline_results.json"
STATIC_DIR  = BASE_DIR / "static"

# ─── BASE DE DATOS ────────────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Crea las tablas y carga los datos del pipeline_results.json."""
    if not RESULTS_PATH.exists():
        raise RuntimeError(f"No se encontró {RESULTS_PATH}")

    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    conn = get_conn()
    c = conn.cursor()

    # Tabla de métricas del modelo
    c.execute("DROP TABLE IF EXISTS metrics")
    c.execute("""
        CREATE TABLE metrics (
            model TEXT PRIMARY KEY,
            auc REAL, accuracy REAL, f1 REAL,
            tp INTEGER, fp INTEGER, fn INTEGER, tn INTEGER
        )
    """)
    m = data["metrics"]
    c.execute("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?)",
              ("Logistic Regression", m["lr_auc"], m["lr_acc"], m["lr_f1"],
               m["lr_tp"], m["lr_fp"], m["lr_fn"], m["lr_tn"]))
    c.execute("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?)",
              ("Random Forest (80 árboles)", m["rf_auc"], m["rf_acc"], m["rf_f1"],
               m["rf_tp"], m["rf_fp"], m["rf_fn"], m["rf_tn"]))
    c.execute("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?)",
              ("Ensemble AUC-ponderado", m["ens_auc"], m["ens_acc"], m["ens_f1"],
               m["ens_tp"], m["ens_fp"], m["ens_fn"], m["ens_tn"]))

    # Tabla de importancia de features
    c.execute("DROP TABLE IF EXISTS features")
    c.execute("CREATE TABLE features (rank INTEGER, feature TEXT, importance REAL)")
    for i, fi in enumerate(data["feature_importance"], 1):
        c.execute("INSERT INTO features VALUES (?,?,?)",
                  (i, fi["feature"], fi["importance"]))

    # Tabla de estudiantes
    c.execute("DROP TABLE IF EXISTS students")
    students = data["students"]
    if students:
        cols = list(students[0].keys())
        col_defs = ", ".join([f'"{col}" TEXT' for col in cols])
        c.execute(f"CREATE TABLE students ({col_defs})")
        for s in students:
            vals = [str(v) if v is not None else "" for v in s.values()]
            placeholders = ",".join(["?" for _ in cols])
            c.execute(f"INSERT INTO students VALUES ({placeholders})", vals)

    # Tabla de distribución de riesgo
    c.execute("DROP TABLE IF EXISTS risk_dist")
    c.execute("CREATE TABLE risk_dist (nivel TEXT PRIMARY KEY, cantidad INTEGER)")
    for nivel, cnt in data["riesgo_dist"].items():
        c.execute("INSERT INTO risk_dist VALUES (?,?)", (nivel, cnt))

    conn.commit()
    conn.close()
    print(f"✅ BD inicializada: {len(students)} estudiantes cargados")
    return data["metrics"]

# ─── LIFESPAN ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando CLARIA — cargando base de datos...")
    init_db()
    yield
    print("🛑 CLARIA detenido.")

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CLARIA API",
    description="Clinical Learning Analytics & Risk Intelligence — UDLA CAO",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API ENDPOINTS ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "app": "CLARIA", "version": "2.0.0"}


@app.get("/api/summary")
def get_summary():
    """Resumen general del dashboard: KPIs, distribución de riesgo, histograma."""
    conn = get_conn()
    c = conn.cursor()

    # Distribución de riesgo
    dist = {row["nivel"]: row["cantidad"] for row in c.execute("SELECT * FROM risk_dist")}

    # Métricas del ensamble
    ens = dict(c.execute(
        "SELECT * FROM metrics WHERE model = 'Ensemble AUC-ponderado'"
    ).fetchone())

    # Estadísticas generales de estudiantes
    total = c.execute("SELECT COUNT(*) as n FROM students").fetchone()["n"]
    avg_avance = c.execute("SELECT AVG(CAST(pct_avance AS REAL)) FROM students").fetchone()[0]
    avg_ipc = c.execute("SELECT AVG(CAST(ipc AS REAL)) FROM students").fetchone()[0]

    # Top 10 mayor riesgo
    top10 = [dict(r) for r in c.execute("""
        SELECT estudiante_id, pct_avance, ipc, prob_Ensemble, nivel_riesgo, rank_riesgo
        FROM students ORDER BY CAST(rank_riesgo AS INTEGER) LIMIT 10
    """)]

    # Histograma avance
    with open(RESULTS_PATH) as f:
        raw = json.load(f)
    hist = raw.get("pct_hist", {})

    conn.close()
    return {
        "total_estudiantes": total,
        "semestre": "2026-2",
        "semana_datos": 6,
        "riesgo_dist": dist,
        "avg_pct_avance": round(avg_avance or 0, 4),
        "avg_ipc": round(avg_ipc or 0, 4),
        "ensemble_metrics": ens,
        "top10_mayor_riesgo": top10,
        "histograma_avance": hist,
    }


@app.get("/api/students")
def get_students(
    riesgo: Optional[str] = Query(None, description="Filtrar: ALTO, MODERADO, BAJO"),
    search: Optional[str] = Query(None, description="Buscar por ID"),
    limit: int = Query(100, le=200),
    offset: int = Query(0, ge=0),
):
    """Lista completa de estudiantes con predicciones. Soporta filtro y búsqueda."""
    conn = get_conn()
    c = conn.cursor()
    conditions = []
    params = []
    if riesgo:
        conditions.append("nivel_riesgo = ?")
        params.append(riesgo.upper())
    if search:
        conditions.append("estudiante_id LIKE ?")
        params.append(f"%{search}%")
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    total = c.execute(f"SELECT COUNT(*) as n FROM students {where}", params).fetchone()["n"]
    rows = [dict(r) for r in c.execute(
        f"SELECT * FROM students {where} ORDER BY CAST(rank_riesgo AS INTEGER) LIMIT ? OFFSET ?",
        params + [limit, offset]
    )]
    conn.close()
    return {"total": total, "offset": offset, "limit": limit, "students": rows}


@app.get("/api/students/{student_id}")
def get_student(student_id: str):
    """Perfil individual de un estudiante con todas sus variables."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM students WHERE estudiante_id = ?", (student_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Estudiante {student_id} no encontrado")
    return dict(row)


@app.get("/api/metrics")
def get_metrics():
    """Métricas comparativas de los tres modelos."""
    conn = get_conn()
    rows = [dict(r) for r in conn.execute("SELECT * FROM metrics ORDER BY auc DESC")]
    with open(RESULTS_PATH) as f:
        raw = json.load(f)
    conn.close()
    return {
        "models": rows,
        "n_train": raw["metrics"]["n_train"],
        "n_pred": raw["metrics"]["n_pred"],
        "n_features": raw["metrics"]["n_features"],
        "semestre": raw["metrics"]["semestre"],
        "semana_datos": raw["metrics"]["semana_datos"],
        "weights": {"lr": raw["metrics"]["w_lr"], "rf": raw["metrics"]["w_rf"]},
    }


@app.get("/api/features")
def get_features(top: int = Query(15, le=62)):
    """Importancia de variables (|coeficientes LR| normalizados)."""
    conn = get_conn()
    rows = [dict(r) for r in conn.execute(
        f"SELECT rank, feature, importance FROM features ORDER BY rank LIMIT ?", (top,)
    )]
    conn.close()
    return {"features": rows}


@app.post("/api/reload")
def reload_data():
    """Recarga los datos desde pipeline_results.json (útil tras re-ejecutar el pipeline)."""
    try:
        init_db()
        return {"status": "ok", "message": "Datos recargados correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── FRONTEND (sirve el HTML) ──────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "CLARIA API activa. Docs en /docs"})
