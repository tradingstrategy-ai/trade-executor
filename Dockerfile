#
# Build trade-executor as a Docker container for live treading
#
# See https://stackoverflow.com/a/71786211/315168 for the recipe
#
FROM python:3.11.10

# Passed from Github Actions
ARG GIT_VERSION_TAG=unspecified
ARG GIT_COMMIT_MESSAGE=unspecified
ARG GIT_VERSION_HASH=unspecified

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

# curl and jq needed for the health checks
# node.js and g++ libssl1.0.0 libssl-dev needed for enzyme below - remove when enzyme dep has been factored out
# https://github.com/nodejs/node-gyp/issues/1195#issuecomment-371954099
# https://stackoverflow.com/questions/40075271/gmpy2-not-installing-mpir-h-not-found
RUN apt-get update && apt-get install -y curl jq ca-certificates gnupg libmpfr-dev libmpc-dev
RUN curl -sSL https://install.python-poetry.org | python - --version 1.8.3
RUN rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /usr/src/trade-executor

# Read by VersionInfo class in trade-executor
RUN echo $GIT_VERSION_TAG > GIT_VERSION_TAG.txt
RUN echo $GIT_COMMIT_MESSAGE > GIT_COMMIT_MESSAGE.txt
RUN echo $GIT_VERSION_HASH > GIT_VERSION_HASH.txt

# package source code
COPY . .

RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi --all-extras

# Anvil is needed for the transaction simulation e.g. by trade-executor enzyme-deploy-vault command
# For the latest pindowns check Github test workflow
ENV PATH="${PATH}:/root/.foundry/bin"
RUN curl -L https://foundry.paradigm.xyz | bash
RUN foundryup --install v0.3.0

# trade-executor /api
# Pyramid HTTP server for webhooks at port 3456
EXPOSE 3456

# Speed up Python process startup
RUN python -m compileall .
RUN python -m compileall /usr/local/lib/python3.11

# Use --quiet to supress Skipping virtualenv creation, as specified in config file.
# use --directory so we can use -w and -v switches with Docker run
# https://stackoverflow.com/questions/74564601/poetry-echos-skipping-virtualenv-creation-as-specified-in-config-file-when-r
# https://github.com/python-poetry/poetry/issues/8077
CMD ["poetry", "run", "--quiet", "--directory", "/usr/src/trade-executor", "trade-executor"]

ENTRYPOINT ["/usr/src/trade-executor/scripts/docker-entrypoint.sh"]