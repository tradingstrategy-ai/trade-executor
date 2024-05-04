"""Python function utilities.

- Hash Python function bodies

"""

import ast
import hashlib
import inspect


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


def hash_function(func, char_length=8):
    """Produces a hash for the code in the given function.

    See https://stackoverflow.com/a/49998190/315168

    :return:
        Part of hex hash of the function body
    """

    assert callable(func), f"Not a function: {func}"

    func_str = inspect.getsource(func)
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
    return fhash.hexdigest()[0:char_length]
