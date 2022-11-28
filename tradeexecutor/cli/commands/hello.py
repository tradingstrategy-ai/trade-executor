from tradeexecutor.cli.commands.app import app


@app.command()
def hello():
    """Check that the application loads without doing anything."""
    print("Hello blockchain")
