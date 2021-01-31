FROM python:3.7-slim as production

ENV PYTHONUNBUFFERED=1
WORKDIR /app/

COPY requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY . /app/

EXPOSE 8000

