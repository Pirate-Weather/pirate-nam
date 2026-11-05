"""The live launcher runs at most NAMGPU_FOLLOW_PREP_MAX preparers at once; a waited-on window jumps the queue.

Breakage it prevents: one --follow door ran 6 preparers together (load 74 on a 30.7-core machine)."""
import sys
import time

from namgpu.nest import follow


class W:
    def __init__(self, key):
        self.key = key

    def as_dict(self):
        return {"key": self.key}


def test_prepare_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("NAMGPU_FOLLOW_PREP_MAX", "2")

    def argv(w, landing, d):
        return [sys.executable, "-c", f"import time,pathlib; time.sleep(0.6); pathlib.Path(r'{d}/READY.json').write_text('{{}}')"]

    L = follow.ProcessLauncher(tmp_path, argv, None, check=False)
    for i, landing in enumerate((300.0, 100.0, 200.0, 400.0)):
        L.prepare(f"k{i}", W(f"k{i}"), landing, 0.0)
    assert len(L._live()) == 2 and set(L.pending) == {"k2", "k3"}
    assert L.status("k3", 0.0) == "preparing"
    # waiting on a queued window starts it at once, beside the two running
    L.wait_ready("k3", W("k3"), 0.0, poll_s=0.1)
    assert L.status("k3", 0.0) == "ready"
    t0 = time.time()
    while L.status("k2", 0.0) != "ready":
        assert time.time() - t0 < 20
        assert len(L._live()) <= 2
        time.sleep(0.1)
    assert all(L.status(f"k{i}", 0.0) == "ready" for i in range(4))
    L.stop()
