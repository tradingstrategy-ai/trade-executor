# Running with Docker

- The trade executor is packaged as a Docker container

# Environment

- Port 3456 is exposed for the executor webhook integration
- All executor parameters must be passed as environment variables
- The application files are copied to `/usr/src/trade-executor`

# Building

```shell
docker build -t trading-strategy/trade-executor . 
```

# Running

```shell
docker run -ti trading-strategy/trade-executor --help
```

# Launching exeuctors

