.PHONY: bench-tiny bench-small bench-bump bench-machine-id

# Machine ID slug used in bench/baselines/v<version>-<machine-id>/.
# Format: <os>-<arch>-<ram-rounded>g — stable enough to disambiguate
# laptop vs. cloud baselines without leaking hostnames.
MACHINE_ID := $(shell uv run python -c "import platform, psutil; r=int(psutil.virtual_memory().total/(1024**3)); print(f'{platform.system().lower()}-{platform.machine().lower()}-{r}g')")

# factrix version (post-`cz bump`, so `bench-bump` writes under the new version).
VERSION := $(shell uv run python -c "import factrix; print(factrix.__version__)")

BENCH_ENV := OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

bench-machine-id:
	@echo $(MACHINE_ID)

bench-tiny:
	$(BENCH_ENV) uv run python -m bench --target tiny --output out/bench-tiny

bench-small:
	$(BENCH_ENV) uv run python -m bench --target small --output out/bench-small --cold-cache

# Reference-baseline rerun for the release flow. Run AFTER `cz bump`
# (so VERSION resolves to the new version) and BEFORE pushing the
# release tag — the resulting JSONL is then committed and amended
# into the release commit. See bench/README.md "Release flow" and
# bench/baselines/README.md.
bench-bump:
	mkdir -p bench/baselines/v$(VERSION)-$(MACHINE_ID)
	$(BENCH_ENV) uv run python -m bench --target small \
		--output bench/baselines/v$(VERSION)-$(MACHINE_ID) --cold-cache
	@echo "Baseline written to bench/baselines/v$(VERSION)-$(MACHINE_ID)/"
	@echo "Next: git add bench/baselines/v$(VERSION)-$(MACHINE_ID) && git commit --amend --no-edit"
