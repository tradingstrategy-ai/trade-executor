#
# Build trade-executor as a Docker container for live treading
#
# See https://stackoverflow.com/a/71786211/315168 for the recipe
#
FROM python:3.10.8

# Passed from Github Actions
ARG GIT_VERSION_TAG=unspecified
ARG GIT_COMMIT_MESSAGE=unspecified
ARG GIT_VERSION_HASH=unspecified

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get install curl -y \
    && curl -sSL https://install.python-poetry.org | python - --version 1.4.2

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /usr/src/trade-executor

RUN echo $GIT_VERSION_TAG > GIT_VERSION_TAG.txt
RUN echo $GIT_COMMIT_MESSAGE > GIT_COMMIT_MESSAGE.txt
RUN echo $GIT_VERSION_HASH > GIT_VERSION_HASH.txt

# package source code
COPY . .

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi -E web-server -E execution -E quantstats

# Pyramid HTTP server for webhooks at port 3456
EXPOSE 3456

# Use --quiet to supress Skipping virtualenv creation, as specified in config file.
# use --directory so we can use -w and -v switches with Docker run
# https://stackoverflow.com/questions/74564601/poetry-echos-skipping-virtualenv-creation-as-specified-in-config-file-when-r
CMD ["poetry", "run", "--directory=/usr/src/trade-executor", "--quiet", "trade-executor"]

ENTRYPOINT ["/usr/src/trade-executor/scripts/docker-entrypoint.sh"]