# 1) Imagen base ligera con Python 3.12
 FROM python:3.12-slim
 
# 2) Ajustes de Python
 ENV PYTHONDONTWRITEBYTECODE=1 \
     PYTHONUNBUFFERED=1
 
# 3) (Opcional) Paquetes del sistema si alguna dependencia lo requiere
 # RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
# 4) Directorio de trabajo dentro del contenedor
 WORKDIR /app
 
# 5) Instalar dependencias
 COPY requirements.txt .
 RUN pip install --no-cache-dir -r requirements.txt
 
# 6) Este comando permite copiar el código de la aplicación ( El que se encuentra en app.py)
 COPY . .
 
# 7) Crear y usar un usuario no-root (buena práctica de seguridad)
 RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app
 USER appuser
 
# 8) Documentar el puerto interno
 EXPOSE 8000
 
# 9) Comando por defecto: lanzar Gunicorn en 0.0.0.0:8000 
 CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
