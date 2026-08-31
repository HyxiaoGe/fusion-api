param(
    [Parameter(Mandatory = $true)]
    [string]$ImageName,

    [Parameter(Mandatory = $true)]
    [string]$AdapterImageName,

    [Parameter(Mandatory = $true)]
    [string]$ImageTag
)

docker image rm "${ImageName}:${ImageTag}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "清理 API 镜像失败，继续执行后续清理"
}

docker image rm "${AdapterImageName}:${ImageTag}" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "清理 FlyAI Adapter 镜像失败，继续执行后续清理"
}

docker image rm "${AdapterImageName}:${ImageTag}-test" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "清理 FlyAI Adapter 测试镜像失败，继续执行后续清理"
}

$pruneArgs = @(
    "builder", "prune",
    "--force",
    "--max-used-space", "10gb",
    "--reserved-space", "4gb",
    "--min-free-space", "30gb"
)
docker @pruneArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warning "清理默认 builder 的缓存失败"
}

if (
    [string]::IsNullOrWhiteSpace($env:RUNNER_TEMP) -or
    [string]::IsNullOrWhiteSpace($env:DOCKER_CONFIG)
) {
    throw "RUNNER_TEMP 或 DOCKER_CONFIG 未配置，拒绝清理 Docker 凭据目录"
}

$runnerTempPath = (Resolve-Path -LiteralPath $env:RUNNER_TEMP -ErrorAction Stop).Path
$dockerConfigPath = [System.IO.Path]::GetFullPath($env:DOCKER_CONFIG)
$dockerConfigParent = Split-Path -Parent $dockerConfigPath
$dockerConfigParentPath = (Resolve-Path -LiteralPath $dockerConfigParent -ErrorAction Stop).Path
$dockerConfigLeaf = Split-Path -Leaf $dockerConfigPath

if (
    -not [string]::Equals(
        $dockerConfigParentPath,
        $runnerTempPath,
        [System.StringComparison]::OrdinalIgnoreCase
    ) -or
    $dockerConfigLeaf -notlike ".docker-*"
) {
    throw "DOCKER_CONFIG 不在 RUNNER_TEMP 的 .docker-* 隔离目录中，拒绝删除: $dockerConfigPath"
}

if (Test-Path -LiteralPath $dockerConfigPath) {
    Remove-Item -LiteralPath $dockerConfigPath -Recurse -Force
}

exit 0
