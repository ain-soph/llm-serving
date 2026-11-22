"""serving/bin/gpu-top on fake gpu-lock state directories: holders, queue, dead pids, half-written files, the
mutex, concurrent rewrites. Never calls gpu-lock; the only processes started are sleep and a fake `gpu-lock` script
in the temporary directory."""
import contextlib
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "serving/bin/gpu-top"
# the installed script runs under the system Python (3.9 on macOS): test it there when present
PY = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
POOL = 10000


def load(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


gt = load(SCRIPT, "gpu_top")


def tree(d):
    """Names, sizes, mtimes and contents of everything under d (to prove nothing was written)."""
    out = {}
    for p in sorted(Path(d).rglob("*")):
        st = p.stat()
        out[str(p.relative_to(d))] = (p.is_dir(), st.st_mtime_ns, None if p.is_dir() else p.read_bytes())
    return out


class GpuTopTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="gpu-top.")
        self.home = Path(self._tmp.name)
        bin_ = self.home / "llm-serving/bin"
        bin_.mkdir(parents=True)
        (bin_ / "gpu-pool").write_text(f"#!/bin/sh\necho {POOL}\n")
        (bin_ / "gpu-pool").chmod(0o755)
        self.locks = self.home / "llm-serving/.locks"
        for sub in ("holders", "queue", "turns", "forced"):
            (self.locks / sub).mkdir(parents=True)
        self.procs = []
        self.now = time.time()
        self._pool = gt.GPU_POOL
        gt.GPU_POOL = bin_ / "gpu-pool"

    def tearDown(self):
        gt.GPU_POOL = self._pool
        for p in self.procs:
            with contextlib.suppress(ProcessLookupError):
                p.kill()
            p.wait()
        self._tmp.cleanup()

    # ---- helpers
    def spawn(self, *argv):
        p = subprocess.Popen(list(argv) or ["/bin/sleep", "60"])
        self.procs.append(p)
        return p

    def dead_pid(self):
        p = subprocess.Popen(["/usr/bin/true"])
        p.wait()
        return p.pid

    def holder(self, name, mb, shell=None, agent=None, age=60, beat=10, raw=None):
        since = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.now - age))
        text = raw if raw is not None else f"{mb} since {since}\nchain: {shell or ''}\nagent: {agent or ''}\n"
        p = self.locks / "holders" / name
        p.write_text(text)
        os.utime(p, (self.now - beat, self.now - beat))
        return p

    def ticket(self, name, mb, waited=60, poll=5):
        p = self.locks / "queue" / f"{int(self.now - waited)}-{name}"
        p.write_text(f"{mb}\n")
        os.utime(p, (self.now - poll, self.now - poll))
        return p

    def run_top(self, *args):
        env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(self.home), "GPU_LOCK_DIR": str(self.locks),
               "GPU_LOCK_FAKE_WIRED": "5000"}
        return subprocess.run([PY, "-I", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)

    def snap(self):
        return gt.build(self.locks, swap=False, now=self.now)

    # ---- tests
    def test_holders_queue_and_prediction(self):
        live = self.spawn()
        self.holder("alive_h", 4000, shell=live.pid, agent=f"{live.pid} sleep", age=125)
        self.holder("dead_h", 2000, shell=self.dead_pid(), agent=f"{self.dead_pid()} claude")
        self.holder("old_h", 3000, beat=gt.STALE + 400)            # stale: dropped by gpu-lock's next prune
        self.ticket("big", 5000, waited=600)                       # does not fit; may be passed (< 20 min)
        self.ticket("small", 3000, waited=599)                     # fits: ready
        self.ticket("small2", 2000, waited=598)                    # small takes the room first
        self.ticket("gone", 1000, waited=597, poll=gt.TICKET_STALE + 60)
        s = self.snap()
        self.assertTrue(s["state"]["consistent"])
        self.assertEqual((s["pool_mb"], s["reserved_mb"], s["free_mb"], s["stale_reserved_mb"]),
                         (POOL, 6000, 4000, 3000))
        h = {x["name"]: x for x in s["holders"]}
        self.assertTrue(h["alive_h"]["alive"])
        self.assertIn("sleep", h["alive_h"]["command"])
        self.assertAlmostEqual(h["alive_h"]["held_s"], 125, delta=1)
        self.assertFalse(h["dead_h"]["alive"])
        self.assertTrue(h["old_h"]["stale"])
        q = {x["name"]: x for x in s["queue"]}
        self.assertEqual([x["name"] for x in s["queue"]], ["big", "small", "small2", "gone"])
        self.assertEqual({n: x["status"] for n, x in q.items()},
                         {"big": "waiting", "small": "ready", "small2": "waiting", "gone": "stale"})
        self.assertIn("needs 2000 MB, 1000 MB free", q["small2"]["reason"])
        self.assertAlmostEqual(q["big"]["waited_s"], 600, delta=1)
        text = "\n".join(t for t, _ in gt.render(s, 160))
        for needle in ("pool 10000 MB", "free 4000 MB", "HOLDERS (3)", "QUEUE (4)", "DEAD", "stale", "ready: fits"):
            self.assertIn(needle, text)

    def test_queue_rules_mirror_gpu_lock(self):
        now = 10_000
        # nobody passes a request that waited longer than BACKFILL_MAX
        q = [(now - gt.BACKFILL_MAX - 5, "big", 9000), (now - 10, "small", 1000)]
        ok, why = gt.may_take("small", POOL, {"h": 5000}, q, {}, now)
        self.assertFalse(ok)
        self.assertIn("nobody may pass big", why)
        # a waiter whose turn lapsed (TURN_MAX) no longer keeps its room
        q = [(now - 50, "a", 3000), (now - 40, "b", 3000)]
        self.assertFalse(gt.may_take("b", POOL, {"h": 5000}, q, {"a": now - 10}, now)[0])
        self.assertTrue(gt.may_take("b", POOL, {"h": 5000}, q, {"a": now - gt.TURN_MAX - 5}, now)[0])
        # a growing holder: its own share counts as room
        self.assertTrue(gt.may_take("h", POOL, {"h": 5000}, [(now, "h", POOL)], {}, now)[0])
        self.assertFalse(gt.may_take("h", POOL, {"h": 5000, "x": 1}, [(now, "h", POOL)], {}, now)[0])

    def test_deadlocks_flagged(self):
        a, b = self.spawn(), self.spawn()
        self.holder("x", 5000, shell=a.pid)
        self.holder("y", 5000, shell=b.pid)
        self.ticket("x", 8000)                                     # both want to grow into each other
        self.ticket("y", 8000)
        s = self.snap()
        self.assertEqual({q["name"]: q["status"] for q in s["queue"]}, {"x": "deadlock", "y": "deadlock"})
        self.assertEqual({q["name"]: q["grow_from"] for q in s["queue"]}, {"x": 5000, "y": 5000})

    def test_waiting_process_and_self_wait_deadlock(self):
        fake = self.home / "fakebin/gpu-lock"                      # a stand-in that only sleeps
        fake.parent.mkdir()
        fake.write_text("#!/bin/sh\nsleep 30\n")
        fake.chmod(0o755)
        shell = self.spawn("/bin/sh", "-c", f"{fake} acquire w 8000; true")   # `; true` keeps sh as the parent
        self.holder("h", 5000, shell=shell.pid)                    # the same script already holds 5000
        self.ticket("w", 8000)
        for _ in range(100):
            waiting = gt.waiting_procs(gt.processes()).get("w")
            if waiting:
                break
            time.sleep(0.05)
        self.assertTrue(waiting)
        q = self.snap()["queue"][0]
        self.assertEqual(q["pids"], waiting)
        self.assertEqual(q["status"], "deadlock")
        self.assertIn("exit 4", q["reason"])

    def test_read_only_and_busy_mutex(self):
        live = self.spawn()
        self.holder("h", 4000, shell=live.pid)
        self.ticket("w", 7000)
        (self.locks / "turns/w").write_text(f"{int(self.now)}\n")
        before = tree(self.locks)
        r = self.run_top("--json", "--no-swap")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(json.loads(r.stdout)["state"]["consistent"])
        self.run_top("--once", "--no-swap")
        self.assertEqual(tree(self.locks), before)
        # gpu-lock holds its mutex: shown as busy, without waiting for it or touching it
        (self.locks / "mutex").mkdir()
        before = tree(self.locks)
        t0 = time.time()
        r = self.run_top("--json", "--no-swap")
        self.assertLess(time.time() - t0, 10)
        s = json.loads(r.stdout)
        self.assertTrue(s["state"]["busy"])
        self.assertLessEqual(s["state"]["mutex_age_s"], 5)
        self.assertEqual([h["name"] for h in s["holders"]], ["h"])   # last read still shown
        self.assertEqual(tree(self.locks), before)
        self.assertIn("STATE BUSY", self.run_top("--once", "--no-swap").stdout)

    def test_half_written_and_vanishing_files(self):
        self.holder("h", 0, raw="")                                # caught between truncate and write
        self.ticket("w", 1000).write_text("")
        s = self.snap()
        self.assertTrue(s["state"]["busy"])                        # retried, still partial: not a consistent read
        self.assertTrue(s["holders"][0]["partial"])
        self.assertEqual(s["holders"][0]["mb"], POOL)               # gpu-lock's mb_of: no number = the whole pool
        self.assertTrue(s["queue"][0]["partial"])
        # concurrent rewrites, creations and deletions never break a read
        stop = threading.Event()

        def churn():
            i = 0
            while not stop.is_set():
                i += 1
                p = self.locks / "holders" / f"c{i % 3}"
                with open(p, "w") as f:
                    f.write("2000 since 2026-11-21 10:00:00\n")
                    f.flush()
                    f.write("chain: 1\nagent: 1 x\n")
                t = self.locks / "queue" / f"{1000 + i % 5}-q{i % 5}"
                t.write_text("1500\n")
                if i % 2:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(p)
                        os.remove(t)

        th = threading.Thread(target=churn)
        th.start()
        try:
            for _ in range(25):
                s = gt.build(self.locks, swap=False)
                for h in s["holders"]:
                    self.assertIn(h["mb"], (2000, POOL))
                for q in s["queue"]:
                    self.assertIn(q["mb"], (1500, 1000, POOL))
        finally:
            stop.set()
            th.join()

    def test_text_mode_pipe_and_missing_dir(self):
        self.holder("h", POOL, shell=self.dead_pid())
        r = self.run_top("--no-swap")                              # stdout is a pipe: prints once and exits
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("HOLDERS (1)", r.stdout)
        self.assertRegex(r.stdout, r"h\s+all\s")
        self.assertIn("leaked share", r.stdout)
        self.assertIn("memory (measured, not reservations): wired 5000 MB", r.stdout)
        os.rename(self.locks, self.home / "gone")
        r = self.run_top("--json", "--no-swap")
        self.assertEqual(r.returncode, 1)
        self.assertIn("error", json.loads(r.stdout))
        self.assertEqual(self.run_top("-d", "0").returncode, 2)

    def test_llama_swap_events_and_client_rule(self):
        def msg(t, data):
            return b"data:" + json.dumps({"type": t, "data": json.dumps(data)}).encode() + b"\n\n"
        reqs = [{"model": "m1", "req_path": "/v1/chat/completions", "elapsed_ms": 4200, "remote_ip": "127.0.0.1",
                 "req_headers": {"X-Forwarded-For": "9.9.9.9, 127.0.0.1", "Authorization": "Bearer xyz"}},
                {"model": "m2", "req_path": "/v1/embeddings", "elapsed_ms": 10, "remote_ip": "127.0.0.1"}]
        buf = (b"HTTP/1.1 200 OK\r\n\r\nevent:message\n" + msg("modelStatus", [{"id": "a", "state": "ready"},
                                                                              {"id": "b", "state": "stopped"}])
               + msg("pauseChanged", {"paused": True}) + msg("inflight", {"operation": "snapshot", "requests": reqs})
               + b"data:{not json\n")
        got = gt.parse_events(buf)
        self.assertEqual(got["models"], [{"id": "a", "state": "ready"}])
        self.assertTrue(got["paused"])
        self.assertEqual([(r["model"], r["remote"], r["elapsed_s"]) for r in got["inflight"]],
                         [("m1", True, 4), ("m2", False, 0)])
        self.assertNotIn("xyz", json.dumps(got))                   # request headers are never passed on
        self.assertEqual(gt.parse_events(b"HTTP/1.1 200 OK\r\n\r\n" + msg("pauseChanged", {"paused": False})),
                         {"paused": False})
        inflight = load(ROOT / "serving/bin/lsw-inflight", "lsw_inflight")
        self.assertEqual(gt.LOCAL, inflight.LOCAL)
        for r in reqs + [{"remote_ip": "1.1.1.1"}, {"req_headers": {"X-Real-Ip": "8.8.4.4"}}, {}]:
            self.assertEqual(gt.client(r), inflight.client(r))

    def test_constants_match_gpu_lock(self):
        src = (ROOT / "serving/bin/gpu-lock").read_text()
        for name in ("BASE_TOL", "STALE", "TICKET_STALE", "BACKFILL_MAX", "TURN_MAX"):
            self.assertEqual(int(re.search(rf"^{name}=(\d+)", src, re.M).group(1)), getattr(gt, name), name)
        self.assertEqual(int(re.search(r"\|\| POOL=(\d+)", src).group(1)), gt.DEFAULT_POOL)
        self.assertIn("> 30 ))", src)                              # its stale-mutex age
        self.assertEqual(gt.MUTEX_STALE, 30)


if __name__ == "__main__":
    unittest.main()
