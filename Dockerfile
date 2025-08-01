#
# Build trade-executor as a Docker container for live treading
#
# See https://stackoverflow.com/a/71786211/315168 for the recipe
#
# To test building the image: docker build .
#
#
FROM python:3.11.10

# Passed from Github Actions
ARG GIT_VERSION_TAG=unspecified
ARG GIT_COMMIT_MESSAGE=unspecified
ARG GIT_VERSION_HASH=unspecified

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

# curl and jq needed for the health checks
# chromium needed for Kaleido/Plotly to render chart images
# node.js and g++ libssl1.0.0 libssl-dev needed for enzyme below - remove when enzyme dep has been factored out
# https://github.com/nodejs/node-gyp/issues/1195#issuecomment-371954099
# https://stackoverflow.com/questions/40075271/gmpy2-not-installing-mpir-h-not-found
RUN apt-get update && apt-get install -y curl jq chromium ca-certificates gnupg libmpfr-dev libmpc-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python Poetry - pinned version
RUN curl -sSL https://install.python-poetry.org | python - --version 1.8.3

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

# Clean Poetry cache to reduce image size
RUN poetry cache clear pypi --all --no-interaction

# Anvil is needed for the transaction simulation e.g. by trade-executor enzyme-deploy-vault command
# For the latest pindowns check Github test workflow
ENV PATH="${PATH}:/root/.foundry/bin"
RUN curl -L https://foundry.paradigm.xyz | bash
RUN foundryup --install v1.2.1

# We need to install Lagoon dependencies if we want to deploy Vault.sol contract.
# Otherwise trade-executor lagoon-deploy-vault command will fail.
# Needed only for installing the dependencies, not during run time
RUN (cd deps/web3-ethereum-defi/contracts/lagoon-v0 && forge soldeer install)

# Speed up Python process startup
RUN rm -rf ./tests
RUN python -m compileall -q .
RUN python -m compileall -q /usr/local/lib/python3.11

# trade-executor /api
# Pyramid HTTP server for webhooks at port 3456
EXPOSE 3456

# Use --quiet to supress Skipping virtualenv creation, as specified in config file.
# use --directory so we can use -w and -v switches with Docker run
# https://stackoverflow.com/questions/74564601/poetry-echos-skipping-virtualenv-creation-as-specified-in-config-file-when-r
# https://github.com/python-poetry/poetry/issues/8077
CMD ["poetry", "run", "--quiet", "--directory", "/usr/src/trade-executor", "trade-executor"]

ENTRYPOINT ["/usr/src/trade-executor/scripts/docker-entrypoint.sh"]
