# Pyodide integration MVP 

- Try to see if Pyodide is ready for the main stage
- Run various code in browser
    
# Example

Prepare a wheel distribution for Pyodide:

```shell
poetry build
co dist/
```

Start the web server:

```shell
cd scripts/pyodide
python -m http.server 9000
```

Then

http://localhost:9000