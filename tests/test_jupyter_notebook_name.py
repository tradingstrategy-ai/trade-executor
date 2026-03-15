import warnings

from tradeexecutor.utils import jupyter_notebook_name


def test_get_notebook_id_falls_back_when_ipynbname_missing(monkeypatch):
    monkeypatch.delenv("TRADEEXECUTOR_NOTEBOOK_ID", raising=False)
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    monkeypatch.setattr(jupyter_notebook_name.sys, "argv", ["python"])
    monkeypatch.setattr(jupyter_notebook_name, "_find_notebook_path_from_parent_process", lambda: None)
    monkeypatch.setattr(jupyter_notebook_name, "_get_kernel_id", lambda: "abcdef123456")
    original_import_module = jupyter_notebook_name.importlib.import_module

    monkeypatch.setattr(
        jupyter_notebook_name.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)) if name == "ipynbname" else original_import_module(name),
    )

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        notebook_id = jupyter_notebook_name.get_notebook_id({})

    assert notebook_id == "unknown-notebook-abcdef12"
    assert any("using fallback id" in str(w.message) for w in captured)


def test_get_notebook_id_prefers_explicit_env_override(monkeypatch):
    monkeypatch.setenv("TRADEEXECUTOR_NOTEBOOK_ID", "nb-from-env")

    notebook_id = jupyter_notebook_name.get_notebook_id({})

    assert notebook_id == "nb-from-env"
