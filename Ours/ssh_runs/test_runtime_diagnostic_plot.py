"""Exercise the terminal plot with the mask and measurement schema it receives."""

import ast
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROFILE = Path(__file__).parent / "diagnostics" / "diag_ours_sd15_lcm_runtime_smoke.py"


def _plot_helpers():
    tree = ast.parse(PROFILE.read_text(encoding="utf-8"))
    names = {"_scatter_anchor_markers", "save_per_layer_contact_sheet"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in functions} == names
    namespace = {"np": np, "plt": plt}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(PROFILE), "exec"), namespace)
    return namespace


class _Mask:
    def __init__(self, values):
        self.values = values

    def squeeze(self):
        return _Mask(self.values.squeeze())

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def test_runtime_contact_sheet_accepts_batched_mask_and_anchor_schema(tmp_path):
    helpers = _plot_helpers()
    measurement = {
        "anchor_x": 2.0, "anchor_y": 3.0,
        "topk_center_x": 4.0, "topk_center_y": 5.0,
        "topk_anchor_x": 6.0, "topk_anchor_y": 7.0,
        "bbox_center_x": 8.0, "bbox_center_y": 9.0,
        "runtime_reference_x": 10.0, "runtime_reference_y": 11.0,
    }
    figure, axis = plt.subplots()
    try:
        helpers["_scatter_anchor_markers"](axis, measurement)
        actual = [tuple(collection.get_offsets()[0]) for collection in axis.collections]
        assert actual == [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0),
                          (8.0, 9.0), (10.0, 11.0)]
    finally:
        plt.close(figure)

    layer = SimpleNamespace(layer_name="attn2", values=np.ones((12, 12)), spatial_size=12)
    destination = tmp_path / "contact.png"
    helpers["save_per_layer_contact_sheet"](
        original=np.zeros((12, 12, 3)),
        step_image=np.zeros((12, 12, 3)),
        mask=_Mask(np.ones((1, 12, 12))),
        aggregate_heatmap=np.ones((12, 12)),
        aggregate_measurement=measurement,
        layer_maps=[layer],
        layer_measurements={"attn2": measurement},
        title="diagnostic", destination=destination,
    )
    assert destination.is_file()
