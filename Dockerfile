# Dockerfile untuk Hugging Face Spaces (PeekMyFruit)
FROM python:3.10-slim

# Buat user non-root (disarankan oleh Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies dulu (agar cache build efisien)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Salin seluruh file project (app.py, templates/, static/, models/)
COPY --chown=user . /app

# Hugging Face Spaces mengharuskan aplikasi berjalan di port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "-w", "1", "--timeout", "180", "app:app"]
