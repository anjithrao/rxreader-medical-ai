FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Build the FAISS index (and the BM25 document pickle) once, at image build
# time, instead of on every container start — downloads the bge-small
# embedding model and embeds app/data/medicine_docs/*.txt. Needs network
# access during the build; does NOT need GEMINI_API_KEY.
RUN python -m app.data.build_index

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
