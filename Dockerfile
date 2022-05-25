#
# Build trade-executor as a Docker container
#
# See https://stackoverflow.com/a/71786211/315168 for the recipe
#
FROM python:3.10.4-slim-buster

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install curl -y \
    && curl -sSL https://install.python-poetry.org | python - --version 1.1.13

WORKDIR /usr/src/trade-executor

# master tracked dependencies
COPY ../web3-ethereum-defi /usr/src
COPY ../trading-strategy /usr/src

# package source code
COPY ./spec ./
COPY ./tradexecutor ./
COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Webhook port 3456
EXPOSE 3456

CMD ["poetry", "run", "trade-executor"]