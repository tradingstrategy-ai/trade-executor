# Executor configurations

This folder contains configuration scripts for different trading strategy executions

- Strategy Python file
- Used Python hot wallet
- Discord webhooks 

These scripts are used to create `.env` file that you can use with `docker-compose` or directly `source` in shell.

The configuration is created by splicing together

- Shared secret variables (e.g. JSON-RPC endpoints)
- Strategy specific secret variables (e.g. hot wallet private key)
- Public variables (e.g. strategy icon, name and description, gas pricing parameters)
