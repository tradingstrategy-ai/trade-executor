# Visual Studio Code Dev Container for Jupyter Notebooks

These instructions are for using [Trading Strategy](https://tradingstrategy.ai)
backtesting and decentralised finance research environment in [Microsoft Visual Studio Code](https://code.visualstudio.com/).

![img_1.png](vscode-splash.png)

[Microsoft Visual Studio Code](https://code.visualstudio.com/), a popular editor for Jupyter notebooks.
[Dev Container](https://code.visualstudio.com/docs/devcontainers/containers) is a Visual Studio Code
feature to easily distribute ready-made development environments to users. Visual Studio Code
comes with powerful editing features for [Jupyter Notebooks](https://jupyter.org/), a programming
file format for data research.
Dev Containers work on any operating system (Windows, macOS, Linux).
Dev Container users a special `.devcontainer` configuration format in supported Git repositories.

The Trading Strategy Dev Container is a pre-made development environment
for quant finance research in decentralised finance. 

It combines

- Trading Strategy framework and libraries
- Ready set up Python environment with correct Python interpreter and Jupyter Notebook kernel
- Visual Studio Code plugins and settings needed to run and
  edit these notebooks, saving them to your local disk
- Example notebooks ready available in the project explorer
- Apple Silicon (Macbook M1) friendliness

## Prerequisites

- Existing basic knowledge of Python programming, Jupyter notebooks and data science and trading
- The set up will download 2 GB+ data, so we do not recommend to try this 
  over a mobile connection

## Setting up Visual Studio Code

- [Install Visual Studio code](https://code.visualstudio.com/)
- [Install Docker desktop](https://www.docker.com/products/docker-desktop/)
- Install the [Dev Containers extension](https://code.visualstudio.com/docs/devcontainers/containers)
  within Visual Studio Code

## Checkout the repository from Github

After you are done with the local software installation steps above,
you can check out the repository using Visual Studio Code.

![img.png](git-clone.png)

Press `F1` to bring up the command palette (`fn` + `F1` on Macs)

Choose `Clone from Github`.

Paste in the repository URL: `https://github.com/tradingstrategy-ai/trade-executor.git`.

It will now ask you for the destination folder on your hard disk. 
Choose any folder name you like, e.g. `my-fabulous-trading-strategy`.

Open the folder after cloning is complete.

![img.png](open-folder.png)

## Start the Dev Container

When the cloned Github project opens, you get a pop-up *Reopen in container*.

![img.png](open-in-dev-container.png)

Click it and Visual Studio Code will build the development environment for you.
This will take 2 - 15 minutes depening on your Internet connection speed.

![img.png](dev-container-building.png)

You can also manually execute this action by pressing `F1` to bring up the command palette (`fn` + `F1` on Macs)
and finding *Reopen in container* action.

## Using the container

After the container is started, open Terminal in Visual Studio Code (*View > Terminal*).
Press *New Terminal* button to open a new terminal window within your Dev Cointainer.

![img.png](new-termianl.png)

Paste in the following command:

```shell
scripts/set-up-examples.sh 
```

![img.png](cloning-examples.png)

This will create `examples` folder and copies all example notebooks [from the Trading Strategy documentation](https://tradingstrategy.ai/docs/)
there.

### Running an example notebook

Here are short instructions how to edit and run notebooks.

Open `examples/synthetic-ema.ipynb`

Edit the backtesting period in the first code cell:

![img_1.png](backtesting-period.png)


Set to 

```python
start_at = datetime.datetime(2022, 1, 1)
end_at = datetime.datetime(2023, 1, 1)
```

Then press Run all:

![img_1.png](run-all.png)

Now scroll to the bottom of the notebook and see you have updated results for 2022 - 2023:

![img_1.png](run-all-results.png)

## Next steps

Instead of randomly clicking example notebooks around, 
we suggest [you start with Getting started documentation](https://tradingstrategy.ai/docs/programming/code-examples/getting-started.html).  

## Using command line Python

You can also use Dev Container environment as normal Python development environment.

If you open Visual Studio Code terminal and run `python` command
it comes with Trading Strategy packages installed.

![img_1.png](command-line-python.png)

## Troubleshooting

### No space left on device error

Make sure you clean up old Docker images, containers and volumes in your Docker for Desktop
to reclaim disk apce.

### Manual build

Building the Docker image by hand:

```shell
docker build --file .devcontainer/Dockerfile .
```
## Further reading

- https://code.visualstudio.com/docs/devcontainers/containers
- https://stackoverflow.com/questions/63998873/vscode-how-to-run-a-jupyter-notebook-in-a-docker-container-over-a-remote-serve
- https://keestalkstech.com/2022/08/jupyter-notebooks-vscode-dev-container-with-puppeteer-support/
- https://marioscalas.medium.com/using-python-and-poetry-inside-a-dev-container-33c80bc5a22c
