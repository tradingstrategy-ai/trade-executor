"""Python helpers."""

def get_object_full_name(o) -> str:
    # https://stackoverflow.com/a/2020083/315168
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__ # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__