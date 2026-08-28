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

For an image-level debugging shell that does not need production files, you can
use:

```shell
docker run -ti --entrypoint /bin/bash ghcr.io/tradingstrategy-ai/trade-executor:$TRADE_EXECUTOR_VERSION 
```

This standalone container does **not** have the production Compose service's
state or cache volumes. Do not use it for production repairs or migrations.

## Production maintenance shell

Run production repairs and migrations through the deployment's Docker Compose
project so that the container receives the same image, environment, state
volume, and cache volume as the `hyper-ai` service.

Stop the live service before any command that may write state. The maintenance
shell is a separate one-off container; opening it does not stop the running
executor or prevent concurrent state writes.

```shell
docker compose stop hyper-ai
docker compose run --entrypoint /bin/bash hyper-ai --
```

Inside the shell, first confirm that the authoritative production state is
mounted:

```shell
test -s state/hyper-ai.json
```

Run repository scripts directly from the image with Poetry. For example, the
Hyper AI closed-position profitability repair is previewed and applied with:

```shell
poetry run python scripts/hyper-ai/repair-closed-position-profitability.py state/hyper-ai.json
poetry run python scripts/hyper-ai/repair-closed-position-profitability.py state/hyper-ai.json --write
```

Run the preview again after writing; an idempotent repair should report no
remaining changes. The repair creates a numbered backup before replacing the
state and also refreshes the compressed state copy.

For other repair and migration scripts:

1. Ensure the deployed image contains the intended script revision. A missing
   script means the deployment image must be updated; do not download a script
   from another revision into a production container.
2. Read the script's `--help` and its specific runbook. Preview first when the
   script supports it. Do not add `--write` unless that script documents the
   option.
3. Confirm the authoritative input path and the backup behaviour before any
   mutation.
4. Run the documented post-migration validation before restarting the service.

Leave the maintenance shell and restart the service:

```shell
exit
docker compose start hyper-ai
```

Finally, check that the executor starts cleanly and that its production state
endpoint loads. Keep the numbered backup until the result has been verified.

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
