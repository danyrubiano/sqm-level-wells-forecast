FROM hub.docker.prod.walmart.com/library/python:3.11-slim-buster
ARG PROJECT_DIRECTORY="/usr/src/app"

WORKDIR ${PROJECT_DIRECTORY}

RUN apt update && apt-get install g++ -yy && pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
RUN pip install .