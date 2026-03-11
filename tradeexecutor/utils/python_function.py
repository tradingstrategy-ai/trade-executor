"""Python function utilities.

- Hash Python function bodies

"""

import ast
import hashlib
import inspect
import textwrap


def _remove_docstring(node):
    '''
    Removes all the doc strings in a FunctionDef or ClassDef as node.
    Arguments:
        node (ast.FunctionDef or ast.ClassDef): The node whose docstrings to
            remove.
    '''
    if not (isinstance(node, ast.FunctionDef) or
            isinstance(node, ast.ClassDef)):
        return

    if len(node.body) != 0:
        docstr = node.body[0]
        if isinstance(docstr, ast.Expr) and isinstance(docstr.value, ast.Str):
            node.body.pop(0)


# Cache for hash_function results, keyed by (func_id, char_length, bytecode_only).
# The same function is hashed many times during indicator cache key generation
# (once per indicator combination), and inspect.getsource + tokeniser is expensive.
# Each entry stores (func_reference, hash_result) — the strong reference to func
# prevents GC and guards against id() reuse in multiprocessing child processes.
_hash_function_cache: dict[tuple, tuple[object, str]] = {}


def hash_function(func, char_length=8, bytecode_only=False):
    """Produces a hash for the code in the given function.

    See https://stackoverflow.com/a/49998190/315168

    Hash is not guaranteed to be the same across different environments / scenarios.
    Read the function for more info.

    :param char_length:
        How many characters you want in your hash,
        to reduce the hash size.

    :param bytecode_only:
        Don't attempt to hash with sanitised source.

        We will also autodetect this when the source is not available.

    :return:
        Part of hex hash of the function body
    """

    assert callable(func), f"Not a function: {func}"

    cache_key = (id(func), char_length, bytecode_only)
    cached = _hash_function_cache.get(cache_key)
    if cached is not None:
        cached_func, cached_result = cached
        if cached_func is func:
            return cached_result

    try:
        func_str = inspect.getsource(func)
    except OSError:
        # When using https://github.com/cloudpipe/cloudpickle with multiprocessing.
        # Fall back to function bytecode
        # File "/Users/moo/.pyenv/versions/3.11.5/lib/python3.11/inspect.py", line 1081, in findsource
        #   raise OSError('could not get source code')
        bytecode_only = True


    if not bytecode_only:
        func_str = textwrap.dedent(func_str)
        # Heurestics if this is a lambda function - in this case ast will fail
        # Account for ending new line (may or may not be there?)
        lambda_like = len(func_str.split("\n")) in (1, 2) and "lambda" in func_str

        if not lambda_like:
            module = ast.parse(func_str)
            assert len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef)

            # Clear function name so it doesn't affect the hash
            func_node = module.body[0]
            func_node.name = ""

            # Clear all the doc strings
            for node in ast.walk(module):
                _remove_docstring(node)

            # Convert the ast to a string for hashing
            ast_str = ast.dump(module, annotate_fields=False).encode("utf-8")
            # Produce the hash
            fhash = hashlib.sha256(ast_str)
        else:
            # Handle lambda special case
            fhash = hashlib.sha256(func_str.encode("utf-8"))
    else:
        fhash = hashlib.sha256(func.__code__.co_code)

    result = fhash.hexdigest()[0:char_length]
    _hash_function_cache[cache_key] = (func, result)
    return result



