# Pyodide integration MVP 

- Try to see if Pyodide is ready for the main stage
- Run various code in browser
    
# Example

Build a wheel distribution for Pyodide to load:

```shell
poetry build
cp dist/trade_executor-0.3-py3-none-any.whl scripts/pyodide/
```

Start the web server:

```shell
cd scripts/pyodide
python -m http.server 9000
```

Then

http://localhost:9000