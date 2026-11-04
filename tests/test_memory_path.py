"""Memory path selection (CPU only): the 32 GB fit is the default below 64 GiB, selectable on any card."""
import pytest

from namgpu import state


@pytest.mark.parametrize("want,card,path,how", [
    ("", 31.4, "fit", "auto"),          # RTX 5090
    ("auto", 47.4, "fit", "auto"),      # 48 GB cards
    ("", 63.9, "fit", "auto"),
    ("", 64.0, "full", "auto"),
    ("auto", 94.9, "full", "auto"),     # 96 GB cards
    ("fit", 94.9, "fit", "set"),
    ("FULL", 31.4, "full", "set"),
])
def test_resolve(want, card, path, how):
    r = state.resolve_memory_path(want, card)
    assert (r["path"], r["how"]) == (path, how)
    assert r["reason"]


def test_bad_setting():
    with pytest.raises(ValueError):
        state.resolve_memory_path("small", 31.4)


def test_scratch_sets_are_not_state_fields():
    # released arrays must be step locals, never fields of the NAM state tables
    specs = set(state.all_specs())
    released = set(state.STEP_SCRATCH) | set(state.HISTORY_SCRATCH) | set(state.PHYS_SCRATCH)
    for v in state.UNIT_SCRATCH.values():
        released |= set(v)
    assert not (released & specs), sorted(released & specs)
