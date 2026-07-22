[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Image,
    [string]$LogDirectory = "_ci-logs"
)

$ErrorActionPreference = "Stop"
$results = [System.Collections.Generic.List[object]]::new()
$resultFile = Join-Path $LogDirectory "stages.json"
$containerName = "fusion-api-ci-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT"
$failureExitCode = 0

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

function Save-Results {
    @($results) | ConvertTo-Json -Depth 4 | Out-File -LiteralPath $resultFile -Encoding utf8
}

function Add-SkippedStage {
    param([string]$Name, [string]$DisplayName)
    $results.Add([pscustomobject]@{
        name = $Name; display_name = $DisplayName; status = "skipped"
        exit_code = $null; started_at = $null; finished_at = $null; duration_seconds = 0
        log_file = $null
    })
    Save-Results
}

function Invoke-DockerStage {
    param([string]$Name, [string]$DisplayName, [string[]]$DockerArguments)
    $ErrorActionPreference = "Continue"
    $started = [DateTimeOffset]::UtcNow
    $logFile = Join-Path $LogDirectory "$Name.log"
    Write-Host "::group::$DisplayName"
    & docker @DockerArguments 2>&1 | ForEach-Object {
        $line = $_.ToString()
        Write-Host $line
        $line | Add-Content -LiteralPath $logFile -Encoding utf8
    }
    $exitCode = $LASTEXITCODE
    Write-Host "::endgroup::"
    $finished = [DateTimeOffset]::UtcNow
    $results.Add([pscustomobject]@{
        name = $Name; display_name = $DisplayName
        status = $(if ($exitCode -eq 0) { "success" } else { "failure" })
        exit_code = $exitCode
        started_at = $started.ToString("o"); finished_at = $finished.ToString("o")
        duration_seconds = [math]::Round(($finished - $started).TotalSeconds, 2)
        log_file = $logFile
    })
    Save-Results
    return $exitCode
}

try {
    $failureExitCode = Invoke-DockerStage "docker-build" "Docker build" @(
        "build", "--target", "production", "-t", $Image, "."
    )
    if ($failureExitCode -ne 0) {
        Add-SkippedStage "ci-dependencies" "Install CI dependencies"
        Add-SkippedStage "architecture" "Architecture check"
        Add-SkippedStage "ruff" "Ruff"
        Add-SkippedStage "unit-tests" "Unit tests"
        exit $failureExitCode
    }

    & docker create --name $containerName $Image sleep infinity | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & docker start $containerName | Out-Null
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    $stages = @(
        @("ci-dependencies", "Install CI dependencies", "timeout 300s python -m pip install --default-timeout=30 --no-cache-dir -r requirements-ci.txt"),
        @("architecture", "Architecture check", "python scripts/check_architecture.py"),
        @("ruff", "Ruff", "ruff check ."),
        @("unit-tests", "Unit tests", "timeout 270s python -u -m unittest discover -s test -t . -v")
    )
    for ($index = 0; $index -lt $stages.Count; $index++) {
        $stage = $stages[$index]
        $failureExitCode = Invoke-DockerStage $stage[0] $stage[1] @(
            "exec", $containerName, "sh", "-lc", $stage[2]
        )
        if ($failureExitCode -ne 0) {
            for ($remaining = $index + 1; $remaining -lt $stages.Count; $remaining++) {
                Add-SkippedStage $stages[$remaining][0] $stages[$remaining][1]
            }
            exit $failureExitCode
        }
    }
}
finally {
    $ErrorActionPreference = "Continue"
    docker container inspect $containerName 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { docker rm -f $containerName 2>$null | Out-Null }
}

exit $failureExitCode
