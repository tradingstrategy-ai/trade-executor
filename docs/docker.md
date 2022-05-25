# Running with Docker

- The trade executor is packaged as a Docker container

# Environment

- Port 3456 is exposed for the executor webhook integration
- All executor parameters must be passed as environment variables
- The application files are copied to `/usr/src/trade-executor`
- Work dir is `/usr/src/trade-executor`
- You need to configure a domain name for each strategy executions
- Local `/state` and `/cache` are mapped to `/usr/src/trade-executor` - note that these folders are **shared across instances**
  and trade executor application code must deal with having specific state files for each strategy

# Building

```shell
docker build -t trading-strategy/trade-executor . 
```

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

## Launching a strategty using docker-compose

The project comes with a `docker-compose.yml` with configurations for example strategies.

Now when we have created `~/quickswap-momentum.env` we can launch the executor.

```shell
docker-compose build quickswap-momentum
docker-compose up --no-deps -d quickswap-momentum
```

This executor 

- Maps a host port for the webhook access - each strategy execution gets its own port
- This port is mapped to the Internet through a Caddy reverse proxy

# Troubleshooting the container

You can do bash

```shell
docker run -it --entrypoint /bin/bash trading-strategy/trade-executor 
```