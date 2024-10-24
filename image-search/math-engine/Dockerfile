FROM python:3.9.7
WORKDIR /app
ADD app/requirements.txt .
RUN pip install -r requirements.txt
ADD app /app
ENV FLASK_APP=/app/app.py
ENTRYPOINT ["gunicorn","-w 4", "-b 0.0.0.0:8001", "app:app"]