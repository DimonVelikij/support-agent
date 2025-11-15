FROM python:3.12

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

ARG HOST=0.0.0.0
ARG PORT=8000

ENV HOST=$HOST
ENV PORT=$PORT

CMD ["sh", "-c", "uvicorn --proxy-headers --host $HOST --port $PORT api:app"]