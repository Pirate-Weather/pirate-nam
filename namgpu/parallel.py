"""Multi-card runtime: one process per card (rank), an NCCL communicator over cupy.cuda.nccl.

MULTICARD-PLAN.md section 6. Started by tools/run_forecast.py --cards N: every rank pins CUDA_VISIBLE_DEVICES to its
card, rank 0 writes the NCCL unique id to a file, the others read it. With one card ctx.comm is None and nothing here
runs, so the one-card plan, shapes and arithmetic are unchanged.

Every collective is enqueued on the caller's stream (ctx.stream): no host sync. barrier() is the one exception
(host sync, used at init and at the end only).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _dt(nccl, dtype):
    import numpy as np
    d = np.dtype(dtype)
    return {np.dtype(np.float32): nccl.NCCL_FLOAT32, np.dtype(np.float64): nccl.NCCL_FLOAT64,
            np.dtype(np.int32): nccl.NCCL_INT32, np.dtype(np.uint32): nccl.NCCL_UINT32,
            np.dtype(np.int64): nccl.NCCL_INT64, np.dtype(np.uint64): nccl.NCCL_UINT64,
            np.dtype(np.uint8): nccl.NCCL_UINT8}[d]


class Comm:
    def __init__(self, rank: int, nranks: int, idfile, timeout_s: float = 600.0):
        from cupy.cuda import nccl
        self.rank, self.nranks = int(rank), int(nranks)
        self._nccl = nccl
        idfile = Path(idfile)
        if self.rank == 0:
            uid = nccl.get_unique_id()
            tmp = idfile.with_name(idfile.name + ".tmp")
            # bytes in current CuPy (a tuple of ints in older ones): keep the type through the file
            tmp.write_text(uid.hex() if isinstance(uid, (bytes, bytearray)) else json.dumps(list(uid)))
            os.replace(tmp, idfile)
        else:
            t0 = time.time()
            while not idfile.exists():
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"parallel: rank {rank} waited {timeout_s} s for the NCCL id {idfile}")
                time.sleep(0.05)
            txt = idfile.read_text().strip()
            uid = tuple(json.loads(txt)) if txt.startswith("[") else bytes.fromhex(txt)
        self.nccl = nccl.NcclCommunicator(self.nranks, uid, self.rank)

    # ---- point to point -----------------------------------------------------------------------------------
    def sendrecv(self, pairs, stream):
        """pairs: [(sendbuf or None, recvbuf or None, peer)], all in one NCCL group on `stream`."""
        nccl, s = self._nccl, stream.ptr
        nccl.groupStart()
        try:
            for snd, rcv, peer in pairs:
                if snd is not None:
                    self.nccl.send(snd.data.ptr, snd.size, _dt(nccl, snd.dtype), int(peer), s)
                if rcv is not None:
                    self.nccl.recv(rcv.data.ptr, rcv.size, _dt(nccl, rcv.dtype), int(peer), s)
        finally:
            nccl.groupEnd()

    # ---- collectives --------------------------------------------------------------------------------------
    def allgather(self, sendbuf, recvbuf, stream):
        self.nccl.allGather(sendbuf.data.ptr, recvbuf.data.ptr, sendbuf.size, _dt(self._nccl, sendbuf.dtype),
                            stream.ptr)

    def bcast(self, buf, root: int, stream):
        self.nccl.bcast(buf.data.ptr, buf.size, _dt(self._nccl, buf.dtype), int(root), stream.ptr)

    def allreduce(self, sendbuf, recvbuf, op: str, stream):
        nccl = self._nccl
        o = {"sum": nccl.NCCL_SUM, "max": nccl.NCCL_MAX, "min": nccl.NCCL_MIN}[op]
        self.nccl.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, sendbuf.size, _dt(nccl, sendbuf.dtype), o,
                            stream.ptr)

    def barrier(self, stream) -> int:
        """Host barrier (init and end only): a 1-int allreduce, then a stream sync. Returns nranks."""
        import cupy as cp
        with stream:
            x = cp.ones((1,), dtype=cp.int32)
            self.allreduce(x, x, "sum", stream)
        stream.synchronize()
        return int(x.get()[0])


def get(ctx):
    """The communicator when more than one card runs, else None (one card: every multi-card step is a no-op)."""
    c = getattr(ctx, "comm", None)
    return c if (c is not None and c.nranks > 1) else None
