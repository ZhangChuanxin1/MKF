# Generated HLS fixtures

The four bit-exact P1/16-party fixtures are not stored in Git. Generate them
from the checked-in Go references:

```powershell
pwsh -NoProfile -File scripts/generate_fixtures.ps1 -AllowLargeOutput
```

The command writes under `fixtures/generated/` and creates a SHA-256 manifest.
Generation is sequential and can consume substantial memory, disk and time.
The output directory must be empty before generation.

Expected subdirectories:

- `kklss_ckks_p1_n16`
- `mkf_ckks_p1_n16`
- `kklss_bfv_p1_n16`
- `mkf_bfv_p1_n16`

Do not add these payloads with ordinary Git. If fixed fixtures must be shared,
publish a separately checksummed release asset and retain the generator source
revision and manifest.
