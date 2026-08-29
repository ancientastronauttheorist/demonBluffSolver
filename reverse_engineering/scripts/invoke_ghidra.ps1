[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameRoot,

    [ValidateSet('import', 'analyze', 'all', 'export-core')]
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
$importSummaryPath = Join-Path $projectRoot 'logs\import-summary.json'
$analysisSummaryPath = Join-Path $projectRoot 'logs\analysis-summary.json'

function Assert-AnalysisSummary {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Ghidra analysis did not write its completion summary: $Path"
    }
    $analysisSummary = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($analysisSummary.program -ne $programName -or $analysisSummary.analysis_timeout_occurred) {
        throw "Incomplete Ghidra analysis: $($analysisSummary | ConvertTo-Json -Compress)"
    }
}

if ($Stage -in @('import', 'all')) {
    Remove-Item -LiteralPath $importSummaryPath -Force -ErrorAction SilentlyContinue
    if ($Stage -eq 'all') {
        Remove-Item -LiteralPath $analysisSummaryPath -Force -ErrorAction SilentlyContinue
    }
    $arguments = @(
        $projectDirectory,
        $projectName,
        '-import', $gameAssembly,
        '-overwrite',
        '-scriptPath', $ghidraScripts,
        '-preScript', 'ImportIl2CppSymbols.java', $scriptJson, $importSummaryPath,
        '-max-cpu', $MaxCpu,
        '-log', (Join-Path $projectRoot 'logs\import.log'),
        '-scriptlog', (Join-Path $projectRoot 'logs\import-script.log')
    )
    if ($Stage -eq 'import') {
        $arguments += '-noanalysis'
    }
    else {
        $arguments += @(
            '-analysisTimeoutPerFile', $AnalysisTimeoutSeconds,
            '-postScript', 'RecordAnalysisCompletion.java', $analysisSummaryPath
        )
    }
    & $headless @arguments
    if ($LASTEXITCODE -ne 0) { throw "Ghidra import failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $importSummaryPath -PathType Leaf)) {
        throw 'Ghidra symbol import did not write its completion summary'
    }
    $importSummary = Get-Content -LiteralPath $importSummaryPath -Raw | ConvertFrom-Json
    if ($importSummary.cancelled -or
        $importSummary.method_labels -le 0 -or
        $importSummary.unique_functions -le 0 -or
        $importSummary.metadata_labels -le 0 -or
        $importSummary.string_labels -le 0) {
        throw "Incomplete Ghidra symbol import: $($importSummary | ConvertTo-Json -Compress)"
    }
    if ($Stage -eq 'all') {
        Assert-AnalysisSummary -Path $analysisSummaryPath
    }
}

if ($Stage -eq 'analyze') {
    Remove-Item -LiteralPath $analysisSummaryPath -Force -ErrorAction SilentlyContinue
    $arguments = @(
        $projectDirectory,
        $projectName,
        '-process', $programName,
        '-analysisTimeoutPerFile', $AnalysisTimeoutSeconds,
        '-max-cpu', $MaxCpu,
        '-scriptPath', $ghidraScripts,
        '-postScript', 'RecordAnalysisCompletion.java', $analysisSummaryPath,
        '-log', (Join-Path $projectRoot 'logs\analysis.log'),
        '-scriptlog', (Join-Path $projectRoot 'logs\analysis-script.log')
    )
    & $headless @arguments
    if ($LASTEXITCODE -ne 0) { throw "Ghidra analysis failed with exit code $LASTEXITCODE" }
    Assert-AnalysisSummary -Path $analysisSummaryPath
}

if ($Stage -eq 'export-core') {
    $targetPath = Join-Path $reverseEngineeringRoot 'targets\gameplay_core.json'
    $targets = Get-Content -LiteralPath $targetPath -Raw | ConvertFrom-Json
    if ($targets.build_id -ne $buildId) {
        throw "Core export targets belong to $($targets.build_id), not $buildId"
    }
    & python (Join-Path $scriptDirectory 'validate_ghidra_targets.py') `
        --targets $targetPath `
        --script-json $scriptJson
    if ($LASTEXITCODE -ne 0) {
        throw 'Ghidra target validation failed'
    }
    $exportDirectory = Join-Path $projectRoot 'exports\gameplay-core'
    New-Item -ItemType Directory -Force -Path $exportDirectory | Out-Null
    $resolvedProjectRoot = [IO.Path]::GetFullPath($projectRoot)
    $resolvedExportDirectory = [IO.Path]::GetFullPath($exportDirectory)
    $expectedPrefix = $resolvedProjectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedExportDirectory.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean export directory outside project root: $resolvedExportDirectory"
    }
    Get-ChildItem -LiteralPath $resolvedExportDirectory -File -Filter '*.c' -ErrorAction SilentlyContinue |
        Remove-Item -Force
    $summaryPath = Join-Path $resolvedExportDirectory '_export_summary.json'
    Remove-Item -LiteralPath $summaryPath -Force -ErrorAction SilentlyContinue
    $arguments = @(
        $projectDirectory,
        $projectName,
        '-process', $programName,
        '-noanalysis',
        '-scriptPath', $ghidraScripts,
        '-postScript', 'ExportFunctionDecompilations.java', $exportDirectory, $targetPath,
        '-log', (Join-Path $projectRoot 'logs\export-core.log'),
        '-scriptlog', (Join-Path $projectRoot 'logs\export-core-script.log')
    )
    & $headless @arguments
    if ($LASTEXITCODE -ne 0) { throw "Ghidra export failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw 'Ghidra export did not write its completion summary'
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    $expectedCount = @($targets.functions).Count
    if ($summary.requested -ne $expectedCount -or
        $summary.processed -ne $expectedCount -or
        $summary.exported -ne $expectedCount -or
        $summary.failed -ne 0 -or
        $summary.cancelled) {
        throw "Incomplete Ghidra export: $($summary | ConvertTo-Json -Compress)"
    }
    foreach ($target in $targets.functions) {
        $safeName = $target.name -replace '[^A-Za-z0-9_.-]', '_'
        $outputPath = Join-Path $resolvedExportDirectory "$safeName.c"
        if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf) -or
            (Get-Item -LiteralPath $outputPath).Length -eq 0) {
            throw "Missing or empty Ghidra export: $outputPath"
        }
    }
    $actualExportCount = @(
        Get-ChildItem -LiteralPath $resolvedExportDirectory -File -Filter '*.c'
    ).Count
    if ($actualExportCount -ne $expectedCount) {
        throw "Expected $expectedCount distinct C exports, found $actualExportCount"
    }
    Write-Output "decompiled_core=$exportDirectory"
}

Write-Output "build_id=$buildId"
Write-Output "ghidra_project=$projectDirectory\$projectName.gpr"
