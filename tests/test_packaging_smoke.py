"""
test_packaging_smoke.py
-----------------------
Packaging integrity smoke tests for RamanPL_2D.

These tests verify that the package imports cleanly and that all required
data files and public entry points are reachable after installation.
They are distinct from unit tests: they catch packaging failures such as
missing data files, broken import paths, or editable-install assumptions.
"""

import importlib
import importlib.util
import sys
import pytest  # noqa: E402

def test_import_top_level_package():
    import ramanpl
    assert ramanpl.__version__ == "0.6.3"


def test_import_public_entry_points():
    import ramanpl
    # Trigger lazy loading of every public name in __all__
    for name in ramanpl.__all__:
        obj = getattr(ramanpl, name)
        assert obj is not None, f"ramanpl.{name} resolved to None"


def test_package_data_materials_json_present():
    import importlib.resources as pkg_resources
    # Confirm raman_materials.json is accessible as packaged data
    try:
        # Python 3.9+ path
        ref = pkg_resources.files("ramanpl").joinpath("raman_materials.json")
        assert ref.is_file(), "raman_materials.json not found in installed package"
    except AttributeError:
        # Fallback for older importlib.resources API
        import ramanpl
        from pathlib import Path
        pkg_dir = Path(ramanpl.__file__).parent
        assert (pkg_dir / "raman_materials.json").exists(), (
            "raman_materials.json not found in package directory"
        )


def test_base_install_does_not_require_ramanspy():
    # The ramanpl package must be fully importable even when ramanspy is absent.
    # This is always true for the base install; guard against accidental hard imports.
    import ramanpl
    import ramanpl.integration.ramanspy_adapter as adapter
    # has_ramanspy() must return a bool without raising
    result = adapter.has_ramanspy()
    assert isinstance(result, bool)


def test_optional_ramanspy_import_path_is_guarded_cleanly():
    # Importing the integration subpackage must never raise even when ramanspy
    # is absent. The guard inside ramanspy_adapter.has_ramanspy() covers this.
    try:
        from ramanpl.integration import ramanspy_bridge
        from ramanpl.integration import ramanspy_translate
    except ImportError as exc:
        raise AssertionError(
            f"ramanpl.integration submodule raised ImportError without ramanspy: {exc}"
        ) from exc


def test_ml_namespace_importable_in_base_install():
    import ramanpl.ml
    import ramanpl.ml.clustering
    assert callable(getattr(ramanpl.ml, "pca_reduce"))
    assert callable(getattr(ramanpl.ml, "kmeans_cluster"))


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is not None,
    reason="scikit-learn is installed; test only runs in base install",
)
def test_ml_functions_raise_clean_error_without_sklearn():
    import pandas as pd
    from ramanpl.ml.clustering import kmeans_cluster, pca_reduce

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    with pytest.raises(ImportError, match=r"\[ml\]"):
        pca_reduce(df, n_components=1)

    with pytest.raises(ImportError, match=r"\[ml\]"):
        kmeans_cluster(df, n_clusters=2)


def test_descriptors_top_level_import_works():
    from ramanpl import descriptors
    assert callable(descriptors.build_feature_row)
    assert callable(descriptors.validate_peak_pairs)


def test_tqdm_importable():
    import tqdm
    from tqdm.auto import tqdm as _t
    assert hasattr(_t, "update")
