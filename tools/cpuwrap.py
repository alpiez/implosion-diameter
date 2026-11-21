"""Run a command and report its CPU (user+sys of all reaped descendants) on stderr.

Mounted into the submission container by tools/eval.py so we measure the
container's CPU ourselves (podman here does not enforce or report limits).
"""
import resource
import subprocess
import sys
import time

t = time.perf_counter()
rc = subprocess.run(sys.argv[1:]).returncode
r = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"@@CPUWRAP cpu={r.ru_utime + r.ru_stime:.6f} wall={time.perf_counter() - t:.3f} "
      f"maxrss_kb={r.ru_maxrss} rc={rc}", file=sys.stderr)
sys.exit(rc)
