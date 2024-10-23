#
# Build trade-executor as a Docker container for backtesting notebooks
#
# See https://stackoverflow.com/a/71786211/315168 for the recipe
#

# See official Python Docker images
# https://hub.docker.com/_/python/
FROM python:3.11.10-bullseye

# pysha3 does not yet work on Python 3.11
# FROM python:3.11.1-slim-buster

# Passed from Github Actions
ARG GIT_VERSION_TAG=unspecified
ARG GIT_COMMIT_MESSAGE=unspecified
ARG GIT_VERSION_HASH=unspecified

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install curl gcc -y \
    && curl -sSL https://install.python-poetry.org | python - --version 1.8.1

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /trading-strategy

# Set in Github Actions
RUN echo $GIT_VERSION_TAG > GIT_VERSION_TAG.txt
RUN echo $GIT_COMMIT_MESSAGE > GIT_COMMIT_MESSAGE.txt
RUN echo $GIT_VERSION_HASH > GIT_VERSION_HASH.txt

# package source code
COPY . .

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi -E web-server -E execution -E qstrader



