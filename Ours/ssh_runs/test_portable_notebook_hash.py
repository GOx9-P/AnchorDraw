"""Generated profiles must not depend on checkout line endings."""

from pathlib import Path

import export_notebooks


def test_all_ports_render_identically_with_lf_and_crlf_notebooks(monkeypatch) -> None:
    original_read_bytes = Path.read_bytes
    inventories = (
        export_notebooks.PORTS,
        export_notebooks.DIAGNOSTIC_PORTS,
        export_notebooks.METRIC_PORTS,
    )
    for inventory in inventories:
        for profile, (notebook_rel, cell_ids) in inventory.items():
            notebook_path = export_notebooks.OURS / notebook_rel
            canonical = original_read_bytes(notebook_path).replace(b"\r\n", b"\n")
            expected = export_notebooks.render_one(profile, notebook_rel, cell_ids)
            for checkout_bytes in (canonical, canonical.replace(b"\n", b"\r\n")):
                with monkeypatch.context() as patch:
                    patch.setattr(
                        Path,
                        "read_bytes",
                        lambda path, data=checkout_bytes: (
                            data if path == notebook_path else original_read_bytes(path)
                        ),
                    )
                    assert export_notebooks.render_one(profile, notebook_rel, cell_ids) == expected
