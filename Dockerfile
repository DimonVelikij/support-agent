FROM python:3.12

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

ARG HOST=0.0.0.0
ARG PORT=5000

ENV HOST=$HOST
ENV PORT=$PORT

CMD ["sh", "-c", "gunicorn --bind ${HOST}:${PORT} app:app"]