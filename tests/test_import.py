"""The geometry and the reference need only numpy: importing labelfield must not import torch,
so a numpy-only consumer (a decoder, a store reader) can depend on it."""
import subprocess
import sys


def test_import_does_not_pull_in_torch():
    code = ("import sys, labelfield; "
            "labelfield.Grid((2, 2, 2)); labelfield.Mapping.corner((4, 4, 4), (2, 2, 2)); "
            "assert 'torch' not in sys.modules, 'torch was imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_torch_names_resolve_on_first_use():
    import labelfield
    assert callable(labelfield.to_labels)
    assert "torch" in labelfield.available_backends()
