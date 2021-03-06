# Running with Docker

The trade executor is packaged as a Docker container.
Multiple strategy executors can use the same container image. 
The container is run by a docker-compose.
Different strategies are configured by their environment variables.

# Using your own strategies

In these instructions, we only cover the bundled example strategies and `docker-compose.yml` recipes for them.
But nothing prevents you to pass your own trading strategies through Docker volume mapping or by building your own container.  

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

# Building

Builds the Docker image and tags it as `latest`:

```shell
docker build -t trading-strategy/trade-executor:latest . 
```

This image is referred in `docker-container.yml`.

# Running

You can start `trade-executor` binary as:

```shell
docker run -ti trading-strategy/trade-executor --help
```

# Configuring and launching strategy executors

## Creating environment file for Docker

First you need to prepare an environment variable file that is going to be 
passed to `docker-compose`. This file contains secrets and
is spliced together for multiple configuration files

- Generic secrets: `~/secrets.env`
- Strategy specific secrets: `~/quickswap-momentum.secrets.env`
- Generic options

This option splicing is done by a configuration helper script like `configurations/quickswap-momentum.sh`

```shell
# Creates ~/quickswap-momentum.env
bash configuration/quickswap-momentum.sh 
```

## Launching a strategy using docker-compose

The project comes with a `docker-compose.yml` with configurations for example strategies.

Now when we have created `~/quickswap-momentum.env` we can launch the executor.

Build the container using instructions above.

First do a pre-check:

```shell
# Load universe data and run timestamp check
docker-compose run quickswap-momentum check-universe

# Confirm cached files are available on the local file system 
ls -lha cache/quickswap-momentum

# Check walelt
docker-compose run quickswap-momentum check-wallet
```

Then start the strategy execution:

```shell
docker-compose up --no-deps -d quickswap-momentum start --trade-immediately
```

Or starting on foreground:

```shell
docker-compose up quickswap-momentum
```

Then check the webhook status:


```shell
curl -I http://localhost:19002
```

This executor 

- Maps a host port for the webhook access - each strategy execution gets its own port
- This port is mapped to the Internet through a Caddy reverse proxy

## All together

Single liner:

```shell
poetry shell
bash configurations/quickswap-momentum.sh && \
  docker-compose up --no-deps -d quickswap-momentum start --trade-immediately 
```

# Stopping the executor

TODO

# Troubleshooting the container

## Opening a bash in the container

You can do bash

```shell
docker run -it --entrypoint /bin/bash trading-strategy/trade-executor 
```

Or:

```shell
docker-compose run --entrypoint /bin/bash quickswap-momentum 
```

## Loading env file in the host environment

To load an `.env` file in the bash:

```shell
export $(cat ~/quickswap-momentum.env |  grep -v '#' | sed 's/\r$//' | awk '/=/ {print $1}' )
```

## Checking ports inside the container

Make sure a container is first running.

Open a shell inside a running container.

```shell
docker-compose exec -it quickswap-momentum /bin/bash 
```

Then curl within the container:

```shell
curl -i http://localhost:3456
```

# Publishing at DockerHub

TODO
