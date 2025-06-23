# Running with Docker

**Note**: [See the docuemtation for up-to-date documentation how to run live trade execution instances](https://tradingstrategy.ai/docs/running/strategy-deployment.html).

The trade executor is packaged as a Docker container.
Multiple strategy executors can use the same container image. 
The container is run by a docker-compose.
Different strategies are configured by their environment variables.

# Environment

There is a mapping of 1 strategy : 1 container : 1 domain : 1 internal TCP/IP port : 1 domain name. 

- Each strategy executor runs as its own container
- Port 3456 is exposed for the executor webhook integration - mapped in `docker-compose.yml`
- All executor parameters must be passed as environment variables
- The application files are copied to `/usr/src/trade-executor`
- Work dir is `/usr/src/trade-executor`
- You need to configure a domain name for each strategy executions
- Local `/state` and `/cache` are mapped to `/usr/src/trade-executor` - note that these folders are **shared across instances**
  and trade executor application code must deal with having specific state files for each strategy

# Running

You need to first login to Github Container Registry.

* [Check the latest released version from Github](https://github.com/tradingstrategy-ai/trade-executor/pkgs/container/trade-executor)

* You can start `trade-executor` binary as:

```shell
export TRADE_EXECUTOR_VERSION=v50
docker run ghcr.io/tradingstrategy-ai/trade-executor:$TRADE_EXECUTOR_VERSION --help
```

You can start a shell within the container as:

```shell
docker run -ti --entrypoint /bin/bash ghcr.io/tradingstrategy-ai/trade-executor:$TRADE_EXECUTOR_VERSION 
```

# Building locally

Build the Docker image from the local source code and tags it as `latest` for your local usage:

```shell
docker build -t ghcr.io/tradingstrategy-ai/trade-executor:latest .
docker run ghcr.io/tradingstrategy-ai/trade-executor:latest --help 
```

To pop open a Bash shell:

```shell
docker run -it --entrypoint /bin/bash ghcr.io/tradingstrategy-ai/trade-executor:latest --
```

This image is referred in `docker-container.yml`.
