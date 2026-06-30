from __future__ import annotations

from scripts.progress_monitor import checkpoint_targets, crossed_checkpoints


def test_checkpoint_targets_round_up_to_prompt_counts():
    assert checkpoint_targets(16, [5, 20, 100]) == {5: 1, 20: 4, 100: 16}


def test_crossed_checkpoints_ignores_completed():
    crossed = crossed_checkpoints(4, 16, [5, 20, 100], completed={5})

    assert crossed == [20]
