import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from app.analysis import load_centroids
from tests.pdf_builder import build_pdf
from tests.test_analyze_document import KeywordEmbedder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_centroids.py"


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_centroids", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "get_embedder", KeywordEmbedder)
    return module


def test_manifest_becomes_one_centroid_per_category(script: ModuleType, tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "motor.pdf").write_bytes(build_pdf(["Manual do motor."]))
    (docs / "contrato.pdf").write_bytes(build_pdf(["Cláusulas do contrato."]))
    (docs / "quebrado.pdf").write_bytes(b"isto nao e um pdf")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "File,Label\ndocs/motor.pdf,Tecnico\ndocs/contrato.pdf,Juridico\ndocs/quebrado.pdf,Juridico\n,Tecnico\n",
        encoding="utf-8",
    )
    output = tmp_path / "out" / "centroids.npz"

    assert script.main(["--manifest", str(manifest), "--output", str(output)]) == 0

    centroids = load_centroids(output)
    assert centroids is not None
    assert centroids.labels == ("Juridico", "Tecnico")
    assert centroids.dimensions == KeywordEmbedder.dimensions


def test_manifest_without_the_expected_columns_is_refused(script: ModuleType, tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("documento,tipo\na.pdf,Tecnico\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="header"):
        script.main(["--manifest", str(manifest), "--output", str(tmp_path / "c.npz")])


def test_nothing_is_written_when_no_document_can_be_read(script: ModuleType, tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,category\nnao-existe.pdf,Tecnico\n", encoding="utf-8")
    output = tmp_path / "c.npz"

    assert script.main(["--manifest", str(manifest), "--output", str(output)]) == 1
    assert not output.exists()
