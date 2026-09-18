# Prescription OCR — FastAPI backend

Flask → FastAPI refactor. Same routes, same request/response JSON shapes,
same OCR pipeline (EasyOCR → TrOCR → RapidFuzz `correct.py` → Gemini
validation). `correct.py` (project root) is untouched.

## Folder structure

```
prescription-ocr/
├── app/
│   ├── main.py                 # FastAPI app, lifespan startup, router mounting
│   ├── config.py                # .env loading, GEMINI_MODEL, MEDICINE_PREFIXES
│   ├── schemas.py                # Pydantic request bodies
│   ├── models/
│   │   └── registry.py          # ModelRegistry: loads EasyOCR + TrOCR ONCE at startup
│   ├── routes/
│   │   ├── pages.py              # GET /, /favicon.ico, chrome devtools probe
│   │   ├── ocr.py                # POST /process
│   │   ├── reference.py          # POST /translate_references
│   │   └── pharmacy.py           # POST /nearby_pharmacies, /route_to_pharmacy
│   ├── services/
│   │   ├── ocr_service.py        # recognize(), region detection, local extraction
│   │   ├── gemini_service.py     # whole-image Gemini pass + translation + merge
│   │   ├── reference_service.py  # medicine name normalization + local uses/side-effects
│   │   ├── pharmacy_service.py   # Overpass nearby search + OSRM routing
│   │   └── fda_service.py        # openFDA lookup (kept for parity; unused by routes today)
│   └── utils/
│       ├── text.py               # is_prefix/is_dosage/line grouping/buy links
│       └── geo.py                # haversine distance, lat/lon parsing & validation
├── correct.py                     # unchanged — YOUR existing file, goes at project root
├── info_txt.txt                   # optional local reference fallbacks (if you use one)
├── templates/index.html          # built React app's index.html goes here
├── static/                       # built React app's static/ assets go here
├── src/api.js                    # unchanged — same endpoints/shapes
├── requirements.txt
├── Dockerfile
└── .env / .env.example
```

`correct.py` sits at the project root (same level as `app/`), exactly where
your old `app.py` expected it — `app/models/registry.py` just does
`from correct import correct, DICTIONARY`, no `sys.path` hack, no
`crnn-pytorch/` subfolder needed.

Model loading is centralized in `app/models/registry.py` and triggered exactly
once from `app/main.py`'s `lifespan` handler at process startup — mirroring
the original module-level `reader = easyocr.Reader(...)` / `model = ...`
loading, but under FastAPI's startup lifecycle instead of import time.

## What changed vs. the original app.py

- Flask → FastAPI + Uvicorn; `@app.route` → `APIRouter` per feature area.
- `request.files["image"]` → `UploadFile = File(...)`.
- `request.get_json()` + manual validation → Pydantic request models
  (`app/schemas.py`), with the **same manual `parse_float`/bounds-checking**
  preserved so error messages and status codes match the original exactly
  (deliberately *not* using FastAPI's automatic 422 validation, to keep the
  API contract identical).
- `jsonify(...)` → `JSONResponse(...)` (identical JSON output).
- `render_template("index.html")` → `Jinja2Templates` + `StaticFiles` mount
  (put your built React app's `index.html` in `templates/` and its
  `static/` assets in `static/`, same as before).
- Model loading moved from "at import time" to `lifespan` startup so it
  happens once per worker process, deterministically, before the app starts
  accepting requests.
- `/translate_references` now catches Gemini errors and returns a JSON
  `502 {"error": ...}` instead of an unhandled 500 (the only intentional
  behavior improvement — every other route's error handling is preserved
  as-is).
- **Not changed:** `crnn-pytorch/correct.py`, the OCR/TrOCR pipeline logic,
  region-detection heuristics, medicine-name correction, Gemini prompts, or
  any response field names/shapes.

`src/api.js` needs **no changes** — all four endpoints keep the same paths,
HTTP methods, and JSON shapes.

## requirements.txt / Dockerfile changes

- Added `fastapi`, `uvicorn[standard]`, `python-multipart` (required for
  `UploadFile` form parsing), `jinja2`, `pydantic`. Removed `flask`.
- `torch`, `transformers`, `easyocr`, `Pillow`, `requests`, `rapidfuzz`
  (used by `correct.py`) are unchanged.
- **Note:** I didn't have your original `requirements.txt` in front of me —
  double check pinned versions above against what you were already using,
  especially `torch`/`transformers`/`easyocr`, and adjust if you pin
  differently for GPU vs CPU wheels.
- Dockerfile: same base image (`python:3.11`), same `WORKDIR /app`, same
  `libgl1`/`libglib2.0-0` install, same `COPY . .` + `pip install` flow you
  already had — only the final `CMD` changed, from `["python", "app.py"]`
  to `["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]`.

## Local setup & test

```bash
# 1. Create/activate a virtualenv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Bring in your unmodified files (not included in this package)
#    - correct.py at the project root (+ its DICTIONARY data)
#    - info_txt.txt (optional local reference fallbacks)
#    - .env with GEMINI_API_KEY=...

# 4. Build the React frontend and copy its output in, same as before:
#    cp react-app/build/index.html templates/index.html
#    cp -r react-app/build/static/* static/

# 5. Run the dev server (auto-reload)
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload

# 6. Smoke-test the routes
curl http://localhost:7860/
curl -X POST http://localhost:7860/process -F "image=@/path/to/prescription.jpg"
curl -X POST http://localhost:7860/translate_references \
  -H "Content-Type: application/json" \
  -d '{"items":[{"name":"paracetamol","uses":"fever","side_effects":"nausea"}],"target_language":"Hindi"}'
curl -X POST http://localhost:7860/nearby_pharmacies \
  -H "Content-Type: application/json" \
  -d '{"lat":18.9634,"lon":72.8281,"radius_m":2000}'
curl -X POST http://localhost:7860/route_to_pharmacy \
  -H "Content-Type: application/json" \
  -d '{"start":{"lat":18.9634,"lon":72.8281},"end":{"lat":18.9700,"lon":72.8300}}'

# Interactive API docs (auto-generated by FastAPI):
#   http://localhost:7860/docs
```

## Docker (local)

```bash
docker build -t prescription-ocr .
docker run -p 7860:7860 --env-file .env prescription-ocr
curl http://localhost:7860/
```

## Deploy to Hugging Face Spaces (Docker SDK)

```bash
# In your Space repo (Space SDK = Docker), with this project's contents at the root:
git add .
git commit -m "Refactor to FastAPI"
git push

# In the Space's Settings → Repository secrets, set:
#   GEMINI_API_KEY = <your key>
# The Space builds the Dockerfile and exposes port 7860 automatically.
```

## Deploy to Render

```bash
# 1. Push this repo to GitHub.
# 2. In Render: New + → Web Service → connect the repo → Environment: Docker.
#    Render auto-detects the Dockerfile and sets $PORT; the CMD above reads it.
# 3. Add an environment variable:
#    GEMINI_API_KEY = <your key>
# 4. Deploy. Render will build the image and run:
#      uvicorn app.main:app --host 0.0.0.0 --port $PORT
```
