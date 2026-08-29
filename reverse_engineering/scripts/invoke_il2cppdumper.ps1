[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameRoot,

    [string]$WorkspaceRoot = 'B:\CodexTools\DemonBluffReverseEngineering'
)

$ErrorActionPreference = 'Stop'

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$reverseEngineeringRoot = Split-Path -Parent $scriptDirectory
$lockPath = Join-Path $reverseEngineeringRoot 'toolchain\toolchain.lock.json'
$lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
$tool = $lock.tools.il2cpp_dumper

$gameAssembly = Join-Path $GameRoot 'GameAssembly.dll'
$metadata = Join-Path $GameRoot 'Demon Bluff_Data\il2cpp_data\Metadata\global-metadata.dat'
foreach ($requiredPath in @($gameAssembly, $metadata)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required input not found: $requiredPath"
    }
}

$downloadDirectory = Join-Path $WorkspaceRoot 'downloads'
$toolDirectory = Join-Path $WorkspaceRoot ("tools\Il2CppDumper-v{0}" -f $tool.version)
$archivePath = Join-Path $downloadDirectory $tool.archive_name
$executablePath = Join-Path $toolDirectory 'Il2CppDumper.exe'
New-Item -ItemType Directory -Force -Path $downloadDirectory, $toolDirectory | Out-Null

if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    $temporaryArchive = "$archivePath.download"
    Invoke-WebRequest -Uri $tool.archive_url -OutFile $temporaryArchive
    Move-Item -LiteralPath $temporaryArchive -Destination $archivePath
}

$archiveHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash
if ($archiveHash -ne $tool.archive_sha256) {
    throw "Il2CppDumper archive hash mismatch: expected $($tool.archive_sha256), got $archiveHash"
}
if ((Get-Item -LiteralPath $archivePath).Length -ne $tool.archive_size) {
    throw "Il2CppDumper archive size mismatch"
}

if (-not (Test-Path -LiteralPath $executablePath -PathType Leaf)) {
    Expand-Archive -LiteralPath $archivePath -DestinationPath $toolDirectory -Force
}

$configPath = Join-Path $toolDirectory 'config.json'
$tool.config | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $configPath -Encoding utf8

$gameAssemblyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $gameAssembly).Hash.ToLowerInvariant()
$metadataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $metadata).Hash.ToLowerInvariant()
$buildId = "{0}_{1}" -f $gameAssemblyHash.Substring(0, 12), $metadataHash.Substring(0, 12)
$outputDirectory = Join-Path $WorkspaceRoot ("artifacts\{0}\il2cppdumper-v{1}" -f $buildId, $tool.version)
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

& $executablePath $gameAssembly $metadata $outputDirectory
if ($LASTEXITCODE -ne 0) {
    throw "Il2CppDumper failed with exit code $LASTEXITCODE"
}

foreach ($expectedOutput in @('dump.cs', 'script.json', 'il2cpp.h', 'stringliteral.json')) {
    $outputPath = Join-Path $outputDirectory $expectedOutput
    if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
        throw "Il2CppDumper did not produce $expectedOutput"
    }
}

Write-Output "build_id=$buildId"
Write-Output "output=$outputDirectory"
