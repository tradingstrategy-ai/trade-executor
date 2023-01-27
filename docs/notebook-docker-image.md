# Docker image for Jupyter Notebooks

Docker image for Jupyter Notebook is combines
- Visual Studio Code for editing notebooks
- Easy installatino of Docker containers


## Setting up Visual Studio Code

- Install the [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers)

## Building image by hand

```shell
docker build \
  -t ghcr.io/tradingstrategy-ai/trading-strategy-notebook:latest \
  --file notebook.dockerfile \
  .
```

## Links

- https://code.visualstudio.com/docs/devcontainers/containers

- https://stackoverflow.com/questions/63998873/vscode-how-to-run-a-jupyter-notebook-in-a-docker-container-over-a-remote-serve
