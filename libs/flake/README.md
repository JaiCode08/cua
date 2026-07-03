# cua-driver flakiness

This package adds stepwise trajectory replay and multi-run flakiness analysis to
`cua-driver`. It provides:

- `cua-driver continue_replay` for cache-driven replay until completion or a miss;
- `cua-driver flakiness` for strict and forgiving multi-run execution;
- exact, perceptual, crop, and downsample screenshot matching;
- async verifier `reset(driver)` and `verify(driver)` hooks;
- bounded agent recovery on forgiving-mode cache misses;
- cache hit, miss, and intervention accounting;
- JSONL logs, screenshots, action artifacts, verifier output, summaries, and HTML reports;
- structured failure categories and nonzero exit codes for failed batches;
- ownership-aware cleanup and configurable timeouts.

For this assessment build, the CLI integration is implemented in the Python wrapper at
`libs/cua-driver/python/src/cua_driver/__main__.py`, avoiding a native driver rebuild while preserving the `cua-driver` trajectory format and tool calls.
## Replay behavior

- `strict` uses exact screenshot matching. It never calls an agent and fails on the first
  cache miss.
- `forgiving-perceptual` compares perceptual hashes.
- `forgiving-crop` compares the region around the previous action target.
- `forgiving-downsample` compares 64-by-64 grayscale observations.

Forgiving modes perform cache lookup first and call the agent only on a miss.
`--max-agent-interventions` limits recovery per run. Replayed and agent-generated actions can
extend the cache for subsequent runs.

## Running

From the repository root:

```powershell
uv sync
cua-driver status

cua-driver flakiness `
  --trajectory libs\flake\trajectories\browser_form `
  --verifier libs\flake\examples\browser_form\verifier.py `
  --runs 3 `
  --mode strict `
  --max-agent-interventions 0 `
  --output-dir libs\flake\out\browser_form `
  --report libs\flake\out\browser_form\browser_form_report.html
```

Forgiving runs additionally accept `--agent`, for example `--agent claude`.

## Example tasks

### Browser form

A local `cua-bench-ui` window collects a name, email address, and comment. The verifier checks
`window.__submitted` and the submitted values. Window geometry is fixed, and the benchmark CSS
suppresses caret animation to keep exact screenshots deterministic.

### CSV edit

The verifier creates an isolated temporary CSV, redirects the recorded launch action to that
fixture, parses the saved file, and confirms that a cell contains `999`. The fixture and owned
application are removed during teardown.

### ZIP folder

The trajectory creates `Desktop\test_folder`, writes a text file, and creates
`Desktop\test_folder.zip`. The verifier removes only these task-owned paths during reset and
requires the ZIP to exist with nonzero size. Verification polls for archive creation to settle.

## Recorded results

Results below are from reports generated under `libs/flake/out`.

| Output | Mode | Passes | Cache hit rate | Interventions |
| --- | --- | ---: | ---: | ---: |
| `browser_form` | strict | 3/3 | 100% | 0 |
| `browser_form_c` | forgiving-crop | 10/10 | 100% | 0 |
| `csv_edit` | strict | 0/1 | 0% | 0 |
| `csv_edit_ds` | forgiving-downsample | 8/10 | 86.6% | 7 |
| `zip_folder` | strict | 0/1 | 0% | 0 |
| `zip_folder_fp` | forgiving-perceptual | 10/10 | 81.4% | 6 |

### Strict CSV and ZIP results

Both trajectories begin with `launch_app`. Their recorded post-launch observations are
desktop-scoped, and the current driver path records a structured `desktop_capture_fallback`
warning when it captures the corresponding live observation. Exact strict matching detects
pixel differences caused by launch timing, foreground state, and desktop/window composition,
so both runs stop at step 2 without invoking an agent.

This is expected strict-mode behavior: the runs identify that these launch states are not
exactly reproducible. The forgiving runs demonstrate recovery under visual variation without
weakening strict matching thresholds.

## Artifacts

Each run writes:

```text
out/<task>/
  summary.json
  metadata.json
  report.html
  runs/run_001/
    environment.json
    log.jsonl
    verifier_result.json
    turn-00001/
      action.json
      app_state.json
      screenshot.png
```

Reports include the reproduction command, per-run timeline, failure category, verifier result,
and links to logs and screenshots.

## Validation

```powershell
python -m pytest libs\flake\tests -q
ruff check libs\flake
ruff format --check libs\flake
```

The current suite has 50 passing tests. It covers the public cache, replay, verifier, report,
failure-classification, and wait interfaces using generated trajectory fixtures. The results
above were produced by real GUI runs. An optional integration script is available at
`libs/flake/scripts/browser_form_integration.ps1`.

## Known constraints

- Visual replay assumes compatible application geometry, DPI, fonts, theme, and desktop image.
- Windows foreground-lock or integrity boundaries may reject activation; these failures are
  reported explicitly.
- Agent recovery is nondeterministic and is therefore bounded per run.
- Runs share one interactive desktop and driver daemon and should execute serially.
