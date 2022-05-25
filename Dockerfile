#
# Build trade-executor as a Docker container
#
# See https://stackoverflow.com/a/71786211/315168 for the recipe
#
FROM python:3.10.4

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install curl -y \
    && curl -sSL https://install.python-poetry.org | python - --version 1.1.13

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /usr/src/trade-executor

# package source code
COPY . .

RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Webhook port 3456
EXPOSE 3456

CMD ["poetry", "run", "trade-executor"]

ENTRYPOINT ["scripts/docker-entrypoint.sh"]