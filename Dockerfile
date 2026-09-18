FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY avr25d ./avr25d
COPY tools ./tools
COPY web ./web

RUN pip install --no-cache-dir .

ENV HOST=0.0.0.0
ENV PORT=8080
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["sh", "-c", "avr25d serve --host ${HOST} --port ${PORT}"]