"""shell.run no longer waits for backgrounded children, first-class background
jobs (shell.jobs / shell.output / shell.kill), session-owned job cleanup,
persistent cwd/env and the child-environment hygiene (no Ventri runtime)."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest

from ventri_agent.tools import shell
from ventri_agent.tools.registry import Risk, ToolContext, ToolError, call_handler

pytestmark = pytest.mark.anyio


def tools(tmp_path: Path, **kw):
    (tmp_path / "wd").mkdir(exist_ok=True)
    kw.setdefault("timeout", 20)
    return {t.name: t for t in shell.make_tools(shell.ShellConfig(cwd=str(tmp_path / "wd"), **kw))}


def ctx(tmp_path: Path, sid: str = "sj", call_id: str = "call_j") -> ToolContext:
    return ToolContext(sid, None, tmp_path, call_id=call_id)  # type: ignore[arg-type]


async def call(t, args, tc):
    return await call_handler(t, t.parse(args), tc)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                        check=False).stdout.strip()
    return bool(st) and not st.startswith("Z")


async def gone(pid: int, timeout: float = 3.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if not alive(pid):
            return True
        await anyio.sleep(0.05)
    return False


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("VENTRI_HOME", str(tmp_path / "home"))
    yield
    for jobs in list(shell._FALLBACK.values()):
        jobs.on_dispose = "kill"
        jobs.close()
    shell._FALLBACK.clear()


# ------------------------------------------------------------- no stall
async def test_backgrounded_child_does_not_block_run(tmp_path):
    t = tools(tmp_path)["shell.run"]
    pidfile = tmp_path / "wd" / "bg.pid"
    t0 = time.monotonic()
    with anyio.fail_after(5):
        # the child inherits stdout (no redirect): the old pipe reader waited for its EOF
        out = await call(t, {"command": f"sleep 30 & echo $! > {pidfile}; echo launched"}, ctx(tmp_path))
    assert time.monotonic() - t0 < 3 and out == "exit code 0\nlaunched\n"
    pid = int(pidfile.read_text())
    assert alive(pid)                       # a deliberate `&` survives the call
    os.kill(pid, 9)
    with anyio.fail_after(5):
        out = await call(t, {"command": f"nohup sleep 30 >/dev/null 2>&1 & echo $! > {pidfile}; echo ok"},
                         ctx(tmp_path))
    assert out.endswith("ok\n")
    os.kill(int(pidfile.read_text()), 9)


async def test_background_child_output_after_exit_is_harmless(tmp_path):
    t = tools(tmp_path)["shell.run"]
    with anyio.fail_after(5):
        out = await call(t, {"command": "(sleep 0.3; echo late) & echo now"}, ctx(tmp_path))
    assert out == "exit code 0\nnow\n"
    await anyio.sleep(0.5)                  # the late write goes to the unlinked spool, no SIGPIPE/crash


async def test_spool_disk_bounded(tmp_path):
    sp = shell._Spool(1000, 500)
    sp.SLACK = 0
    os.write(sp.f.fileno(), b"H" * 1000 + b"x" * 5000)
    sp.check()
    assert os.fstat(sp.f.fileno()).st_size == 500
    os.write(sp.f.fileno(), b"END")
    text = sp.capture().text()
    sp.close()
    assert text.startswith("H" * 1000) and text.endswith("x" * 497 + "END")
    assert "4503 bytes dropped" in text


# ------------------------------------------------------------- jobs
async def test_background_job_lifecycle(tmp_path):
    ts = tools(tmp_path)
    tc = ctx(tmp_path)
    out = await call(ts["shell.run"], {"command": "echo ready; sleep 1; echo two; sleep 30",
                                       "background": True}, tc)
    assert "background job 1 started" in out and "ready" in out
    assert "1: running" in await call(ts["shell.jobs"], {}, tc)
    out = await call(ts["shell.output"], {"job": "1", "pattern": "two", "wait": 5}, tc)
    assert "pattern matched" in out and "two" in out and "ready" not in out   # since=last
    assert "(no new output)" in await call(ts["shell.output"], {"job": "1"}, tc)
    assert "ready\ntwo" in await call(ts["shell.output"], {"job": "1", "since": "start"}, tc)
    log = tmp_path / "artifacts" / "job-1.txt"
    assert log.read_text() == "ready\ntwo\n"          # artifact.read can page it
    pid = shell._FALLBACK["sj"].jobs["1"].proc.pid
    out = await call(ts["shell.kill"], {"job": "1"}, tc)
    assert "killed by signal 15" in out and await gone(pid)
    with pytest.raises(ToolError, match="no job"):
        await call(ts["shell.output"], {"job": "9"}, tc)


async def test_background_job_exit_and_wait(tmp_path):
    ts = tools(tmp_path)
    tc = ctx(tmp_path)
    out = await call(ts["shell.run"], {"command": "false", "background": True}, tc)
    assert "exited with code 1" in out              # an instant failure is visible at start
    await call(ts["shell.run"], {"command": "sleep 1; echo done; exit 4", "background": True}, tc)
    t0 = time.monotonic()
    out = await call(ts["shell.output"], {"job": "2", "wait": 10}, tc)
    assert time.monotonic() - t0 < 5 and "exited with code 4" in out and "done" in out


async def test_jobs_are_session_scoped_and_killed_on_close(tmp_path):
    ts = tools(tmp_path)
    await call(ts["shell.run"], {"command": "sleep 30 & sleep 30", "background": True}, ctx(tmp_path, "a"))
    with pytest.raises(ToolError, match="no job"):
        await call(ts["shell.kill"], {"job": "1"}, ctx(tmp_path, "b"))   # another session's job
    jobs = shell._FALLBACK["a"]
    pgid = jobs.jobs["1"].proc.pid
    jobs.close()
    assert not shell._group_alive(pgid)             # the whole group, incl. the inner `&` child


async def test_jobs_keep_on_dispose(tmp_path):
    ts = tools(tmp_path, jobs_on_dispose="keep")
    await call(ts["shell.run"], {"command": "sleep 30", "background": True}, ctx(tmp_path, "k"))
    jobs = shell._FALLBACK["k"]
    pid = jobs.jobs["1"].proc.pid
    jobs.close()
    assert alive(pid)
    os.killpg(pid, 9)


async def test_job_limit_and_log_ring(tmp_path):
    ts = tools(tmp_path, max_jobs=1, job_log_bytes=20_000, job_keep_bytes=1_000)
    tc = ctx(tmp_path)
    await call(ts["shell.run"], {"command": "seq 1 20000; sleep 30", "background": True}, tc)
    with pytest.raises(ToolError, match="already running"):
        await call(ts["shell.run"], {"command": "true", "background": True}, tc)
    out = await call(ts["shell.output"], {"job": "1", "since": "start"}, tc)
    assert "earlier output dropped" in out and out.rstrip().endswith("20000")
    assert (tmp_path / "artifacts" / "job-1.txt").stat().st_size < 2_000


async def test_job_permissions_shape(tmp_path):
    ts = tools(tmp_path)
    run = ts["shell.run"]
    assert run.risk == Risk.EXTERNAL and not run.grantable and run.default_action == "ask"
    assert run.describe_call(run.parse({"command": "x", "background": True}))["background"] == "true"
    assert ts["shell.output"].risk == Risk.READ and ts["shell.jobs"].risk == Risk.READ
    assert ts["shell.kill"].risk == Risk.WRITE_LOCAL and ts["shell.kill"].default_action == "allow"


async def test_session_dispose_kills_jobs(tmp_path):
    from ventri_agent.sessions import SESSION_KEYS

    from .harness import Env
    from .harness import call as tcall
    assert shell.ShellJobs in SESSION_KEYS
    script = [{"tool_calls": [tcall("shell.run", {"command": "sleep 60", "background": True})]},
              {"tool_calls": [tcall("shell.jobs")]},
              {"content": "started"}]
    async with Env(tmp_path, script) as env:
        await env.kernel.plugin(shell.shell, {"cwd": str(tmp_path / "wd"), "policy": "allow"})
        s = await env.open()
        r = await s.turn("start a job")
        assert r.text == "started"
        jobs = s.ctx.get(shell.ShellJobs)
        assert isinstance(jobs, shell.ShellJobs) and not shell._FALLBACK
        pid = jobs.jobs["1"].proc.pid
        assert alive(pid)
        await s.suspend()                     # scope disposed -> the job's group is killed
        assert await gone(pid)


# ------------------------------------------------------------- persist
async def test_persist_cwd_and_env(tmp_path):
    ts = tools(tmp_path, persist=True)
    tc = ctx(tmp_path)
    (tmp_path / "wd" / "sub").mkdir()
    await call(ts["shell.run"], {"command": "cd sub && export FOO=bar && export MY_TOKEN=x"}, tc)
    out = await call(ts["shell.run"], {"command": "pwd; echo foo=$FOO tok=$MY_TOKEN"}, tc)
    assert str((tmp_path / "wd" / "sub").resolve()) in out
    assert "foo=bar" in out and "tok=\n" in out + "\n" and "tok=x" not in out
    out = await call(ts["shell.run"], {"command": "exit 3"}, tc)
    assert out.startswith("exit code 3")            # the exit trap keeps the status
    out = await call(ts["shell.run"], {"command": "pwd", "cwd": "."}, tc)
    assert out.rstrip().endswith("sub")
    plain = tools(tmp_path)["shell.run"]           # without persist nothing carries over
    out = await call(plain, {"command": "pwd; echo foo=$FOO"}, ctx(tmp_path, "other"))
    assert "foo=\n" in out


# ------------------------------------------------------------- env hygiene
def test_clean_env_hides_runtime(tmp_path, monkeypatch):
    rt = tmp_path / "rt"
    (rt / "bin").mkdir(parents=True)
    env = {"PATH": f"{rt / 'bin'}:/usr/bin:/bin", "VENTRI_DS_KEY_FILE": "/x", "VENTRI_HOME": "/h",
           "PYTHONHOME": str(rt), "PYTHONPATH": f"{sys.prefix}/lib", "HOME": "/root", "DEEPSEEK_API_KEY": "k",
           "LANG": "C.UTF-8"}
    out = shell.clean_env(env, hide_paths=[str(rt)])
    assert out["PATH"] == "/usr/bin:/bin"
    assert set(out) == {"PATH", "HOME", "LANG"}
    kept = shell.clean_env(env, hide_runtime=False)
    assert kept["PATH"].startswith(str(rt)) and "PYTHONHOME" in kept and "VENTRI_HOME" not in kept


def test_clean_env_strips_own_venv(monkeypatch):
    if sys.prefix == sys.base_prefix:
        pytest.skip("not running from a virtualenv")
    venv_bin = os.path.join(sys.prefix, "bin")
    out = shell.clean_env({"PATH": f"{venv_bin}:/usr/bin", "VIRTUAL_ENV": sys.prefix})
    assert out == {"PATH": "/usr/bin"}


async def test_base_env_file(tmp_path):
    envf = tmp_path / "orig.env"
    envf.write_bytes(b"PATH=/usr/bin:/bin\0ORIG=yes\0MULTI=a\nb\0")
    t = tools(tmp_path, base_env_file=str(envf))["shell.run"]
    out = await call(t, {"command": "echo orig=$ORIG; echo \"$MULTI\"; echo path=$PATH"}, ctx(tmp_path))
    assert "orig=yes" in out and "a\nb" in out and "path=/usr/bin:/bin" in out
