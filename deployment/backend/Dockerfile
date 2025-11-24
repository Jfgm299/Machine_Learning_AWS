# Dockerfile

# 1. Imagen Base: ¡ACTUALIZADA! Usamos una imagen oficial de Python 3.11 ligera (slim).
FROM python:3.11-slim

# 2. Directorio de Trabajo:
WORKDIR /app

# 3. Copiar e Instalar Dependencias: 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el Código:
COPY . .

# 5. Puerto:
EXPOSE 8000

# 6. Comando de Ejecución (Punto de Entrada):
# Gunicorn con 4 workers de Uvicorn.
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]