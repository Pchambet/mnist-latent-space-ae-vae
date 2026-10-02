from mnist_latent.figures import first_win, ratio_versus, relative_change, versus_identity
from mnist_latent.report import replace_blocks


def test_first_win_is_where_the_model_stays_below_the_reference():
    amps = [0.0, 0.5, 1.0, 2.0]
    assert first_win([0.3, 0.2, 0.1, 0.1], [0.0, 0.1, 0.2, 0.4], amps) == 1.0
    # A win at 0.5 that is lost at 1.0 does not count.
    assert first_win([0.3, 0.05, 0.3, 0.1], [0.0, 0.1, 0.2, 0.4], amps) == 2.0
    assert first_win([0.3, 0.3, 0.3, 0.5], [0.0, 0.1, 0.2, 0.4], amps) is None


def test_replace_blocks_only_touches_the_marked_body():
    text = "intro\n<!-- BEGIN:t -->\nold\nrows\n<!-- END:t -->\noutro"
    out = replace_blocks(text, {"t": "| new |", "absent": "x"})
    assert out == "intro\n<!-- BEGIN:t -->\n| new |\n<!-- END:t -->\noutro"
    empty = "<!-- BEGIN:t -->\n<!-- END:t -->"
    assert replace_blocks(empty, {"t": "| new |"}) == "<!-- BEGIN:t -->\n| new |\n<!-- END:t -->"
    assert replace_blocks(replace_blocks(empty, {"t": "a"}), {"t": "b"}) == (
        "<!-- BEGIN:t -->\nb\n<!-- END:t -->"
    )


def test_wording_follows_the_numbers():
    assert (
        versus_identity(0.0003, [0.0002, 0.0004], 0.0113) == "marginally worse than doing nothing"
    )
    assert versus_identity(-0.005, [-0.006, -0.004], 0.0113) == "better than doing nothing"
    assert versus_identity(0.0001, [-0.0001, 0.0003], 0.0113).startswith("statistically")
    assert relative_change(0.0108, 0.0116) == "7% lower"
    assert relative_change(0.0120, 0.0100) == "20% higher"
    assert ratio_versus(0.0042, 0.0098) == "2.3× lower than"
    assert ratio_versus(0.0110, 0.0098) == "worse than"
