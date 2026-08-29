[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameRoot,

    [ValidateSet('import', 'analyze', 'all')]
    [string]$Stage = 'all',

    [string]$WorkspaceRoot = 'B:\CodexTools\DemonBluffReverseEngineering',

    [string]$GhidraRoot = 'C:\Users\BMO\.cache\ghidra\tools\ghidra_12.1.3_PUBLIC',

    [int]$AnalysisTimeoutSeconds = 3600,

    [int]$MaxCpu = 4
)

$ErrorActionPreference = 'Stop'
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$reverseEngineeringRoot = Split-Path -Parent $scriptDirectory
$lock = Get-Content -LiteralPath (Join-Path $reverseEngineeringRoot 'toolchain\toolchain.lock.json') -Raw | ConvertFrom-Json
$ghidra = $lock.tools.ghidra
$dumper = $lock.tools.il2cpp_dumper

$headless = Join-Path $GhidraRoot 'support\analyzeHeadless.bat'
$propertiesPath = Join-Path $GhidraRoot 'Ghidra\application.properties'
if (-not (Test-Path -LiteralPath $headless -PathType Leaf)) {
    throw "Ghidra headless launcher not found: $headless"
}
$applicationProperties = Get-Content -LiteralPath $propertiesPath
$versionLine = $applicationProperties | Where-Object { $_ -like 'application.version=*' } | Select-Object -First 1
if (($versionLine -split '=', 2)[1] -ne $ghidra.version) {
    throw "Installed Ghidra does not match pinned version $($ghidra.version)"
}

$gameAssembly = Join-Path $GameRoot 'GameAssembly.dll'
$metadata = Join-Path $GameRoot 'Demon Bluff_Data\il2cpp_data\Metadata\global-metadata.dat'
$gameAssemblyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $gameAssembly).Hash.ToLowerInvariant()
$metadataHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $metadata).Hash.ToLowerInvariant()
$buildId = "{0}_{1}" -f $gameAssemblyHash.Substring(0, 12), $metadataHash.Substring(0, 12)
$manifestPath = Join-Path $reverseEngineeringRoot "manifests\builds\$buildId.json"
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "No checked-in build manifest for $buildId"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.inputs.game_assembly.sha256 -ne $gameAssemblyHash.ToUpperInvariant() -or
    $manifest.inputs.global_metadata.sha256 -ne $metadataHash.ToUpperInvariant()) {
    throw 'Build inputs do not match the checked-in manifest'
}

$artifactRoot = Join-Path $WorkspaceRoot "artifacts\$buildId"
$dumperOutput = Join-Path $artifactRoot ("il2cppdumper-v{0}" -f $dumper.version)
$scriptJson = Join-Path $dumperOutput 'script.json'
$extractionManifestPath = Join-Path $reverseEngineeringRoot "manifests\extractions\${buildId}_il2cppdumper-v$($dumper.version).json"
if (-not (Test-Path -LiteralPath $extractionManifestPath -PathType Leaf)) {
    throw "Missing extraction manifest: $extractionManifestPath"
}
$extractionManifest = Get-Content -LiteralPath $extractionManifestPath -Raw | ConvertFrom-Json
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $scriptJson).Hash -ne $extractionManifest.outputs.script_json.sha256) {
    throw 'script.json does not match the checked-in extraction manifest'
}

$projectRoot = Join-Path $artifactRoot ("ghidra-{0}" -f $ghidra.version)
$projectDirectory = Join-Path $projectRoot 'project'
$projectName = "DemonBluff_$buildId"
$programName = 'GameAssembly.dll'
$ghidraScripts = Join-Path $reverseEngineeringRoot 'ghidra_scripts'
New-Item -ItemType Directory -Force -Path $projectDirectory, (Join-Path $projectRoot 'logs') | Out-Null

if ($Stage -in @('import', 'all')) {
    $arguments = @(
        $projectDirectory,
        $projectName,
        '-import', $gameAssembly,
        '-overwrite',
        '-scriptPath', $ghidraScripts,
        '-preScript', 'ImportIl2CppSymbols.java', $scriptJson,
        '-max-cpu', $MaxCpu,
        '-log', (Join-Path $projectRoot 'logs\import.log'),
        '-scriptlog', (Join-Path $projectRoot 'logs\import-script.log')
    )
    if ($Stage -eq 'import') {
        $arguments += '-noanalysis'
    }
    else {
        $arguments += @('-analysisTimeoutPerFile', $AnalysisTimeoutSeconds)
    }
    & $headless @arguments
    if ($LASTEXITCODE -ne 0) { throw "Ghidra import failed with exit code $LASTEXITCODE" }
}

if ($Stage -eq 'analyze') {
    $arguments = @(
        $projectDirectory,
        $projectName,
        '-process', $programName,
        '-analysisTimeoutPerFile', $AnalysisTimeoutSeconds,
        '-max-cpu', $MaxCpu,
        '-log', (Join-Path $projectRoot 'logs\analysis.log'),
        '-scriptlog', (Join-Path $projectRoot 'logs\analysis-script.log')
    )
    & $headless @arguments
    if ($LASTEXITCODE -ne 0) { throw "Ghidra analysis failed with exit code $LASTEXITCODE" }
}

Write-Output "build_id=$buildId"
Write-Output "ghidra_project=$projectDirectory\$projectName.gpr"
