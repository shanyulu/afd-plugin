from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _runner_module():
    path = (
        Path(__file__).parents[3]
        / "scripts"
        / "qwen36_v026"
        / "run_capability_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("qwen36_capability_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("argv", "node_suffix"),
    [
        (
            [
                "--topology",
                "1a1f",
                "--gate-side",
                "ffn",
                "--mode",
                "eager",
                "--batch-size",
                "1",
            ],
            "test_qwen3_6_afd_1a1f_eager_matches_native_tp1",
        ),
        (
            [
                "--topology",
                "2a2f",
                "--gate-side",
                "ffn",
                "--mode",
                "eager",
                "--batch-size",
                "2",
            ],
            "test_qwen3_6_afd_2a2f_tp2_eager_matches_native_tp2",
        ),
        (
            [
                "--topology",
                "2a2f",
                "--gate-side",
                "ffn",
                "--mode",
                "graph",
                "--batch-size",
                "1",
            ],
            "test_qwen3_6_afd_2a2f_tp2_graph_matches_native[b1]",
        ),
    ],
)
def test_capability_runner_exposes_checked_gates_only(argv, node_suffix):
    runner = _runner_module()

    args = runner._parser().parse_args(argv)

    assert runner._node_id(args).endswith(node_suffix)


def test_capability_runner_rejects_attention_gate_and_fake_flags():
    runner = _runner_module()
    parser = runner._parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--topology",
                "1a1f",
                "--gate-side",
                "attention",
                "--mode",
                "eager",
                "--batch-size",
                "1",
            ],
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["--compare-native"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--cleanup"])


def test_capability_runner_rejects_ungated_combination():
    runner = _runner_module()
    args = runner._parser().parse_args(
        [
            "--topology",
            "2a2f",
            "--gate-side",
            "ffn",
            "--mode",
            "graph",
            "--batch-size",
            "2",
        ],
    )

    with pytest.raises(ValueError, match="batch-size 1"):
        runner._node_id(args)
