# Backend: FastAPI + Pyomo/HiGHS engine.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY backend /app/backend
RUN pip install /app/backend

EXPOSE 8000
CMD ["uvicorn", "gb_battery.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
