"""``python -m bench.validate`` entry point — thin alias to ``bench.validator.main``."""

from __future__ import annotations

from bench.validator import main

if __name__ == "__main__":
    raise SystemExit(main())
