param(
    [Parameter(Mandatory = $true)]
    [string]$ImageName,

    [Parameter(Mandatory = $true)]
    [string]$AdapterImageName,

    [Parameter(Mandatory = $true)]
    [string]$ImageTag
)

$ErrorActionPreference = "Stop"
$image = "${ImageName}:${ImageTag}"
$adapterImage = "${AdapterImageName}:${ImageTag}"

docker build --target production --provenance=false -t $image .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker run --rm `
    --mount "type=bind,source=$((Get-Location).Path)\README.md,target=/app/README.md,readonly" `
    $image sh -lc "timeout 300s python -m pip install --default-timeout=30 --no-cache-dir -r requirements-ci.txt && python scripts/check_architecture.py && ruff check . && timeout 270s python -u -m unittest discover -s test -t . -v"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker build --target test --provenance=false -t "${adapterImage}-test" ./flyai-adapter
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

docker build --target production --provenance=false -t $adapterImage ./flyai-adapter
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
