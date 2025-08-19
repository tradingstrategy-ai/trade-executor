"""Get Jupyter notebook filename.

- The code lifted from ipynbname project by Mark McPherson https://github.com/msm1089/ipynbname

- Do the complex dance to figure out the Jupyter notebook filename

.. warning::

    TODO: Does not work in Pycharm
"""

import os
import json
import os
import sys
import urllib.error
import urllib.request
from itertools import chain
from pathlib import Path, PurePath
from typing import Generator, Tuple, Union

import ipykernel
from jupyter_core.paths import jupyter_runtime_dir
from traitlets.config import MultipleInstanceError


FILE_ERROR = "Can't identify the notebook {}."
CONN_ERROR = "Unable to access server;\n" \
           + "ipynbname requires either no security or token based security."


def _list_maybe_running_servers(runtime_dir=None) -> Generator[dict, None, None]:
    """ Iterate over the server info files of running notebook servers.
    """
    if runtime_dir is None:
        runtime_dir = jupyter_runtime_dir()
    runtime_dir = Path(runtime_dir)

    if runtime_dir.is_dir():
        # Get notebook configuration files, sorted to check the more recently modified ones first
        for file_name in sorted(
            chain(
                runtime_dir.glob('nbserver-*.json'),  # jupyter notebook (or lab 2)
                runtime_dir.glob('jpserver-*.json'),  # jupyterlab 3
            ),
            key=os.path.getmtime,
            reverse=True,
        ):
            try:
                yield json.loads(file_name.read_bytes())
            except json.JSONDecodeError as err:
                # Sometimes we encounter empty JSON files. Ignore them.
                pass


def _get_kernel_id() -> str:
    """ Returns the kernel ID of the ipykernel.
    """
    connection_file = Path(ipykernel.get_connection_file()).stem
    print("connection_file", connection_file)
    kernel_id = connection_file.split('-', 1)[1]
    return kernel_id


def _get_sessions(srv):
    """ Given a server, returns sessions, or HTTPError if access is denied.
        NOTE: Works only when either there is no security or there is token
        based security. An HTTPError is raised if unable to connect to a
        server.
    """
    try:
        qry_str = ""
        token = srv['token']
        if token:
            qry_str = f"?token={token}"
        if not token and "JUPYTERHUB_API_TOKEN" in os.environ:
            token = os.environ["JUPYTERHUB_API_TOKEN"]
        url = f"{srv['url']}api/sessions{qry_str}"
        # Use a timeout in case this is a stale entry.
        with urllib.request.urlopen(url, timeout=0.5) as req:
            return json.load(req)
    except Exception:
        raise urllib.error.HTTPError(CONN_ERROR)


def _find_nb_path() -> Union[Tuple[dict, PurePath], Tuple[None, None]]:
    kernel_id = _get_kernel_id()
    print("_find_nb_path", kernel_id)
    for srv in _list_maybe_running_servers():
        try:
            print(kernel_id, srv)
            sessions = _get_sessions(srv)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return srv, PurePath(sess['notebook']['path'])
        except Exception:
            pass  # There may be stale entries in the runtime directory
    return None, None


def get_notebook_file_from_within_notebook(_globals) -> Path:
    """Return the notebook file.

    - Must be called from within the notebook
    - Compatibility limits
        - Visual Studio Code
        - ipython
        - Jupyter-based systems through ipynbname

    Example:

    .. code-block:: python

        # Get notebook filename stem
        RUN_ID = get_notebook_file_from_within_notebook(globals())


    :return:
        Full path to the currently executed notebook
    """

    assert type(_globals) is dict, "Must pass globals() to get_notebook_file_from_within_notebook()"

    if "__vsc_ipynb_file__" in _globals:
        # Visual Studio Code
        return Path(_globals['__vsc_ipynb_file__'])
    elif sys.argv[0].endswith("ipynb"):
        # Ipython
        return Path(sys.argv[0])
    else:
        # Jupyter-based systems
        import ipynbname
        return ipynbname.path()


def get_notebook_id(_globals) -> str:
    """Return the base name of notebook"""
    return get_notebook_file_from_within_notebook(_globals).stem


def path() -> Path:
    """ Returns the absolute path of the notebook,
        or raises a FileNotFoundError exception if it cannot be determined.
    """
    srv, path = _find_nb_path()
    if srv and path:
        root_dir = Path(srv.get('root_dir') or srv['notebook_dir'])
        return root_dir / path
    raise FileNotFoundError(FILE_ERROR.format('path'))
