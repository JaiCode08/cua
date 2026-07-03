# ZIP folder example

Goal: create `~/Desktop/test_folder`, add a text file, and create `~/Desktop/test_folder.zip`.

The verifier checks the real output, not UI text. It requires:

- `test_folder.zip` exists;
- `test_folder.zip` has nonzero size.

## Current committed trajectory

`libs/flake/trajectories/zip_folder` contains four replayable actions: launch Command Prompt,
focus it with a pixel click, type one deterministic command chain, and press Return. The command
creates the folder and text file, invokes local PowerShell `Compress-Archive`, and prints
`ZIP_READY`. It does not use Explorer or external resources.

The verifier follows the project spec and does not inspect archive members. Its success
criterion is that `test_folder.zip` exists and has nonzero size. It polls for up to ten seconds
so the temporary zero-byte state observed while `Compress-Archive` starts is not reported as a
false failure.

The verifier treats the launched `cmd.exe` process as owned, closes only its returned PID, and
removes only the exact task paths `Desktop/test_folder` and `Desktop/test_folder.zip`. It does
not kill unrelated shell or Explorer processes or delete similarly named files.

## Recording an Explorer variant

Windows Explorer context menus vary by Windows release, locale, installed shell extensions, and whether the compact or classic menu is shown. A trajectory recorded on one machine can therefore measure Explorer menu drift rather than replay correctness.

## Record with Explorer

```powershell
$trajectory = "libs\flake\trajectories\zip_folder"
.\.venv\Scripts\cua-driver.exe recording start --output-dir $trajectory
```

Using cua-driver actions only:

1. Launch Explorer.
2. Create `Desktop\test_folder`.
3. Create a text file inside it and enter non-empty content.
4. Create `Desktop\test_folder.zip` using the local Explorer ZIP command.
5. Stop recording:

```powershell
.\.venv\Scripts\cua-driver.exe recording stop
```

Prefer keyboard shortcuts and stable coordinates over context-menu element indices. Element-index-only actions are rejected because their references do not survive a new window snapshot.

## Deterministic shell-assisted variant

For diagnosing replay infrastructure independently from Explorer context-menu nondeterminism, a task setup may create the folder/file and invoke PowerShell `Compress-Archive` outside the replay action sequence. This remains a real file workflow and produces a real ZIP that the same verifier inspects. It must not be presented as GUI trajectory replay, and shell commands are not part of the replay-action allowlist.

Example fixture operation:

```powershell
$desktop = [Environment]::GetFolderPath('Desktop')
$folder = Join-Path $desktop 'test_folder'
New-Item -ItemType Directory -Force $folder | Out-Null
Set-Content (Join-Path $folder 'notes.txt') 'non-empty test content'
Compress-Archive -LiteralPath $folder -DestinationPath (Join-Path $desktop 'test_folder.zip')
```

## Run

```powershell
.\.venv\Scripts\cua-driver.exe flakiness `
  --trajectory libs\flake\trajectories\zip_folder `
  --verifier libs\flake\examples\zip_folder\verifier.py `
  --runs 1 `
  --mode strict `
  --max-agent-interventions 0 `
  --report "out\zip-folder\strict report.html"
```
