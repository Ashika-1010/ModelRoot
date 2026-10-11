# ModelRoot — repository root Makefile
#
# Environment assumptions:
#   - Python 3.11+ is available on PATH (or activated via a virtual environment).
#   - The package has been installed with:  pip install -e "core[dev]"
#   - All commands are designed to be run from the repository root.
#   - On Windows, use a POSIX-compatible make (e.g. from Git for Windows / MSYS2)
#     or GNU Make via Chocolatey / Scoop.  Alternatively, run the individual
#     commands listed under each target directly.
#
# Targets:
#   check      — full fast-CI gate: ruff lint, mypy strict, pytest (no slow/docker/e2e)
#   lint       — ruff check + format check only
#   type       — mypy strict type check on core/ only
#   test       — pytest excluding slow/docker/e2e markers
#   test-slow  — pytest including slow tests (excludes docker and e2e)

.PHONY: check lint type test test-slow

# ---------------------------------------------------------------------------
# check: full gate (mirrors what CI runs on every push)
# ---------------------------------------------------------------------------
check: lint type
	pytest -m "not slow and not docker and not e2e" \
	       --cov=modelroot \
	       --cov-report=term-missing \
	       --cov-fail-under=0 \
	       core/tests/

# ---------------------------------------------------------------------------
# lint: ruff check + format verification
# ---------------------------------------------------------------------------
lint:
	ruff check core/src/ core/tests/
	ruff format --check core/src/ core/tests/

# ---------------------------------------------------------------------------
# type: strict mypy on the core package
# ---------------------------------------------------------------------------
type:
	mypy --config-file core/pyproject.toml core/src/modelroot/

# ---------------------------------------------------------------------------
# test: fast tests (no slow / docker / e2e markers)
# ---------------------------------------------------------------------------
test:
	pytest -m "not slow and not docker and not e2e" core/tests/

# ---------------------------------------------------------------------------
# test-slow: all tests that are not docker/e2e (includes slow)
# ---------------------------------------------------------------------------
test-slow:
	pytest -m "not docker and not e2e" core/tests/
