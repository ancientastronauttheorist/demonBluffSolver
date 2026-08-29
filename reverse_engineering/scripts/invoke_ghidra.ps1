[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameRoot,

    [ValidateSet(
        'import',
        'analyze',
        'all',
        'export-core',
        'build-types',
        'typed-import',
        'typed-analyze',
        'typed-all',
        'typed-export'
    )]
    [string]$Stage = 'all',

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string]$TargetSet = 'gameplay_core',

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

$typedStages = @('build-types', 'typed-import', 'typed-analyze', 'typed-all', 'typed-export')
$isTypedStage = $Stage -in $typedStages
$typedInputRoot = Join-Path $artifactRoot 'typed-headers'
$normalizedHeaderPath = Join-Path $typedInputRoot 'il2cpp_ghidra.h'
$prototypeHeaderPath = Join-Path $typedInputRoot 'il2cpp_target_prototypes.h'
$alignmentManifestPath = Join-Path $typedInputRoot 'il2cpp_alignments.json'
$normalizationSummaryPath = Join-Path $typedInputRoot 'normalization-summary.json'
$gdtDirectory = Join-Path $typedInputRoot 'gdt'
$gdtPath = Join-Path $gdtDirectory "il2cpp-types-$buildId.gdt"
$gdtSummaryPath = Join-Path $typedInputRoot 'gdt-build-summary.json'
$typedProjectRoot = Join-Path $artifactRoot ("ghidra-{0}-typed" -f $ghidra.version)
$typedProjectDirectory = Join-Path $typedProjectRoot 'project'
$typedProjectName = "DemonBluff_${buildId}_typed"
$typedImportSummaryPath = Join-Path $typedProjectRoot 'logs\import-summary.json'
$typedAnalysisSummaryPath = Join-Path $typedProjectRoot 'logs\analysis-summary.json'
$targetInfos = @()
$targetByName = [Collections.Generic.Dictionary[string, object]]::new([StringComparer]::Ordinal)
$expectedPrototypeSignatures = [Collections.Generic.Dictionary[string, string]]::new([StringComparer]::Ordinal)
$expectedPrototypeNames = @()

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

function Get-Sha256Lower {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Assert-StringArrayEqual {
    param(
        [Parameter(Mandatory = $true)][object[]]$Actual,
        [Parameter(Mandatory = $true)][object[]]$Expected,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if ($Actual.Count -ne $Expected.Count) {
        throw "$Description count mismatch: expected $($Expected.Count), found $($Actual.Count)"
    }
    for ($index = 0; $index -lt $Expected.Count; $index++) {
        if ([string]$Actual[$index] -cne [string]$Expected[$index]) {
            throw "$Description mismatch at index $index`: expected '$($Expected[$index])', found '$($Actual[$index])'"
        }
    }
}

function Get-SignatureParameterCount {
    param([Parameter(Mandatory = $true)][string]$Signature)
    $openIndex = $Signature.IndexOf('(')
    $closeIndex = $Signature.LastIndexOf(')')
    if ($openIndex -lt 0 -or $closeIndex -le $openIndex) {
        throw "Malformed target signature: $Signature"
    }
    $parameters = $Signature.Substring($openIndex + 1, $closeIndex - $openIndex - 1).Trim()
    if ($parameters.Length -eq 0 -or $parameters -ceq 'void') {
        return 0
    }
    $count = 1
    $depth = 0
    foreach ($character in $parameters.ToCharArray()) {
        if ($character -eq '(') {
            $depth++
        }
        elseif ($character -eq ')') {
            $depth--
            if ($depth -lt 0) {
                throw "Malformed target signature: $Signature"
            }
        }
        elseif ($character -eq ',' -and $depth -eq 0) {
            $count++
        }
    }
    if ($depth -ne 0) {
        throw "Malformed target signature: $Signature"
    }
    return $count
}

function Assert-ArtifactMatchesSummary {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$SummaryEntry,
        [Parameter(Mandatory = $true)][string]$ExpectedName
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing typed-header artifact: $Path"
    }
    $file = Get-Item -LiteralPath $Path
    if ($file.Length -le 0) {
        throw "Empty typed-header artifact: $Path"
    }
    if ([string]$SummaryEntry.name -cne $ExpectedName -or
        [long]$SummaryEntry.size -ne $file.Length -or
        [string]$SummaryEntry.sha256 -ine (Get-Sha256Lower -Path $Path)) {
        throw "Typed-header artifact does not match its summary: $Path"
    }
}

function Assert-SummaryNotOlderThan {
    param(
        [Parameter(Mandatory = $true)][string]$SummaryPath,
        [Parameter(Mandatory = $true)][string[]]$InputPaths,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
        throw "Missing $Description summary: $SummaryPath"
    }
    $summaryTime = (Get-Item -LiteralPath $SummaryPath).LastWriteTimeUtc
    foreach ($inputPath in $InputPaths) {
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
            throw "Missing input needed to validate ${Description}: $inputPath"
        }
        if ($summaryTime -lt (Get-Item -LiteralPath $inputPath).LastWriteTimeUtc) {
            throw "Stale $Description summary '$SummaryPath'; input is newer: $inputPath"
        }
    }
}

function Invoke-HeadlessWithChecks {
    param(
        [Parameter(Mandatory = $true)][object[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description,
        [switch]$LargeHeap
    )
    $hadHeapOverride = Test-Path -LiteralPath 'Env:GHIDRA_HEADLESS_MAXMEM'
    $previousHeapOverride = $env:GHIDRA_HEADLESS_MAXMEM
    try {
        if ($LargeHeap) {
            $env:GHIDRA_HEADLESS_MAXMEM = '12G'
        }
        & $headless @Arguments
        $headlessExitCode = $LASTEXITCODE
    }
    finally {
        if ($LargeHeap) {
            if ($hadHeapOverride) {
                $env:GHIDRA_HEADLESS_MAXMEM = $previousHeapOverride
            }
            else {
                Remove-Item -LiteralPath 'Env:GHIDRA_HEADLESS_MAXMEM' -ErrorAction SilentlyContinue
            }
        }
    }
    if ($headlessExitCode -ne 0) {
        throw "$Description failed with exit code $headlessExitCode"
    }
}

function Assert-ImportSummary {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description did not write its completion summary: $Path"
    }
    $summary = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($summary.cancelled -or
        $summary.method_labels -le 0 -or
        $summary.unique_functions -le 0 -or
        $summary.metadata_labels -le 0 -or
        $summary.string_labels -le 0) {
        throw "Incomplete ${Description}: $($summary | ConvertTo-Json -Compress)"
    }
}

function Get-ApplySummaryPath {
    param([Parameter(Mandatory = $true)]$TargetInfo)
    return Join-Path $typedProjectRoot ("logs\apply-{0}-summary.json" -f $TargetInfo.BaseName)
}

if ($isTypedStage) {
    $targetDirectory = Join-Path $reverseEngineeringRoot 'targets'
    $targetPaths = [Collections.Generic.List[string]]::new()
    foreach ($targetFile in Get-ChildItem -LiteralPath $targetDirectory -File -Filter '*.json') {
        $targetPaths.Add($targetFile.FullName)
    }
    $targetPaths.Sort([StringComparer]::Ordinal)
    if ($targetPaths.Count -eq 0) {
        throw "No checked-in Ghidra target sets found in $targetDirectory"
    }

    foreach ($targetPath in $targetPaths) {
        $targetData = Get-Content -LiteralPath $targetPath -Raw | ConvertFrom-Json
        $targetFileName = [IO.Path]::GetFileName($targetPath)
        $targetBaseName = [IO.Path]::GetFileNameWithoutExtension($targetPath)
        if ([int]$targetData.schema_version -ne 1) {
            throw "Unsupported target schema in $targetPath"
        }
        if ([string]$targetData.build_id -cne $buildId) {
            throw "Target set $targetFileName belongs to $($targetData.build_id), not $buildId"
        }
        $targetFunctions = @($targetData.functions)
        if ($targetFunctions.Count -eq 0) {
            throw "Target set has no functions: $targetPath"
        }
        if ($targetByName.ContainsKey($targetBaseName)) {
            throw "Duplicate target-set basename: $targetBaseName"
        }

        $sevenArgumentCount = 0
        foreach ($targetFunction in $targetFunctions) {
            $signature = [string]$targetFunction.signature
            $signatureMatch = [regex]::Match(
                $signature,
                '\b([A-Za-z_][A-Za-z0-9_]*)\s*\('
            )
            if (-not $signatureMatch.Success) {
                throw "Could not recover a prototype identifier from '$signature' in $targetPath"
            }
            $prototypeName = $signatureMatch.Groups[1].Value
            if ($expectedPrototypeSignatures.ContainsKey($prototypeName)) {
                if ($expectedPrototypeSignatures[$prototypeName] -cne $signature) {
                    throw "Conflicting target signatures for prototype $prototypeName"
                }
            }
            else {
                $expectedPrototypeSignatures.Add($prototypeName, $signature)
            }
            if ((Get-SignatureParameterCount -Signature $signature) -eq 7) {
                $sevenArgumentCount++
            }
        }

        & python (Join-Path $scriptDirectory 'validate_ghidra_targets.py') `
            --targets $targetPath `
            --script-json $scriptJson
        if ($LASTEXITCODE -ne 0) {
            throw "Ghidra target validation failed: $targetPath"
        }

        $targetInfo = [pscustomobject]@{
            Path = $targetPath
            FileName = $targetFileName
            BaseName = $targetBaseName
            Data = $targetData
            FunctionCount = $targetFunctions.Count
            SevenArgumentCount = $sevenArgumentCount
            Sha256 = Get-Sha256Lower -Path $targetPath
        }
        $targetInfos += $targetInfo
        $targetByName.Add($targetBaseName, $targetInfo)
    }

    $prototypeNameList = [Collections.Generic.List[string]]::new()
    foreach ($prototypeName in $expectedPrototypeSignatures.Keys) {
        $prototypeNameList.Add($prototypeName)
    }
    $prototypeNameList.Sort([StringComparer]::Ordinal)
    $expectedPrototypeNames = @($prototypeNameList)

    if ($Stage -eq 'typed-export' -and -not $targetByName.ContainsKey($TargetSet)) {
        throw "Unknown checked-in target-set basename '$TargetSet'"
    }
}

function Assert-TargetSetList {
    param(
        [Parameter(Mandatory = $true)][object[]]$Reported,
        [Parameter(Mandatory = $true)][string]$Description
    )
    if ($Reported.Count -ne $targetInfos.Count) {
        throw "$Description target-set count mismatch: expected $($targetInfos.Count), found $($Reported.Count)"
    }
    for ($index = 0; $index -lt $targetInfos.Count; $index++) {
        if ([string]$Reported[$index].name -cne [string]$targetInfos[$index].FileName -or
            [string]$Reported[$index].sha256 -ine [string]$targetInfos[$index].Sha256) {
            throw "$Description target-set mismatch at index $index"
        }
    }
}

function Assert-NormalizationState {
    if (-not (Test-Path -LiteralPath $normalizationSummaryPath -PathType Leaf)) {
        throw "Missing normalization success summary: $normalizationSummaryPath"
    }
    $summary = Get-Content -LiteralPath $normalizationSummaryPath -Raw | ConvertFrom-Json
    if ([int]$summary.schema_version -ne 1 -or
        -not [bool]$summary.success -or
        [string]$summary.build_id -cne $buildId) {
        throw "Invalid or wrong-build normalization summary: $($summary | ConvertTo-Json -Compress)"
    }
    Assert-TargetSetList -Reported @($summary.target_sets) -Description 'Normalization summary'
    if ([int]$summary.prototype_count -ne $expectedPrototypeNames.Count -or
        [int]$summary.alignment_count -le 0 -or
        [int]$summary.inheritance_rewrite_count -le 0) {
        throw "Incomplete normalization summary: $($summary | ConvertTo-Json -Compress)"
    }
    Assert-ArtifactMatchesSummary `
        -Path $normalizedHeaderPath `
        -SummaryEntry $summary.normalized_header `
        -ExpectedName 'il2cpp_ghidra.h'
    Assert-ArtifactMatchesSummary `
        -Path $prototypeHeaderPath `
        -SummaryEntry $summary.prototype_header `
        -ExpectedName 'il2cpp_target_prototypes.h'
    Assert-ArtifactMatchesSummary `
        -Path $alignmentManifestPath `
        -SummaryEntry $summary.alignment_manifest `
        -ExpectedName 'il2cpp_alignments.json'

    $alignmentManifest = Get-Content -LiteralPath $alignmentManifestPath -Raw | ConvertFrom-Json
    if ([int]$alignmentManifest.schema_version -ne 1 -or
        [string]$alignmentManifest.build_id -cne $buildId -or
        [int]$alignmentManifest.alignment -ne 8 -or
        [int]$alignmentManifest.alignment_count -ne @($alignmentManifest.names).Count -or
        [int]$alignmentManifest.alignment_count -ne [int]$summary.alignment_count -or
        [int]$alignmentManifest.inheritance_rewrite_count -ne [int]$summary.inheritance_rewrite_count -or
        [int]$alignmentManifest.prototype_count -ne $expectedPrototypeNames.Count) {
        throw "Invalid typed-header alignment manifest: $alignmentManifestPath"
    }
    Assert-TargetSetList -Reported @($alignmentManifest.inputs.target_sets) -Description 'Alignment manifest'
    Assert-StringArrayEqual `
        -Actual @($alignmentManifest.prototype_names) `
        -Expected @($expectedPrototypeNames) `
        -Description 'Alignment-manifest prototype names'
    if ([string]$alignmentManifest.inputs.il2cpp_h_sha256 -ine
        (Get-Sha256Lower -Path (Join-Path $dumperOutput 'il2cpp.h'))) {
        throw 'Normalized header was not produced from the current extracted il2cpp.h'
    }
    if ([string]$alignmentManifest.outputs.normalized_header_sha256 -ine
            (Get-Sha256Lower -Path $normalizedHeaderPath) -or
        [string]$alignmentManifest.outputs.prototype_header_sha256 -ine
            (Get-Sha256Lower -Path $prototypeHeaderPath)) {
        throw 'Alignment manifest output hashes do not match the current typed headers'
    }
}

function Assert-GdtState {
    Assert-NormalizationState
    if (-not (Test-Path -LiteralPath $gdtSummaryPath -PathType Leaf)) {
        throw "Missing GDT success summary: $gdtSummaryPath"
    }
    $summary = Get-Content -LiteralPath $gdtSummaryPath -Raw | ConvertFrom-Json
    if ([int]$summary.schema_version -ne 1 -or
        -not [bool]$summary.success -or
        [string]$summary.build_id -cne $buildId -or
        -not [bool]$summary.critical_layouts_validated -or
        [int]$summary.data_type_count -le 0 -or
        [int]$summary.alignment_count -le 0 -or
        [int]$summary.function_definition_count -ne $expectedPrototypeNames.Count) {
        throw "Invalid or incomplete GDT summary: $($summary | ConvertTo-Json -Compress)"
    }
    $alignmentManifest = Get-Content -LiteralPath $alignmentManifestPath -Raw | ConvertFrom-Json
    if ([int]$summary.alignment_count -ne [int]$alignmentManifest.alignment_count -or
        [string]$summary.normalized_header_sha256 -ine (Get-Sha256Lower -Path $normalizedHeaderPath) -or
        [string]$summary.prototype_header_sha256 -ine (Get-Sha256Lower -Path $prototypeHeaderPath)) {
        throw 'GDT summary does not match the current normalized inputs'
    }
    $resolvedGdtPath = [IO.Path]::GetFullPath($gdtPath)
    if ([IO.Path]::GetFullPath([string]$summary.gdt_path) -ine $resolvedGdtPath) {
        throw "GDT summary points at the wrong archive: $($summary.gdt_path)"
    }
    $parserOutputPath = "${gdtPath}_CParser.out"
    foreach ($requiredOutput in @($gdtPath, $parserOutputPath)) {
        if (-not (Test-Path -LiteralPath $requiredOutput -PathType Leaf) -or
            (Get-Item -LiteralPath $requiredOutput).Length -le 0) {
            throw "Missing or empty GDT build output: $requiredOutput"
        }
    }
    Assert-SummaryNotOlderThan `
        -SummaryPath $gdtSummaryPath `
        -InputPaths @($gdtPath, $parserOutputPath) `
        -Description 'GDT build'
}

function Assert-ApplySummary {
    param([Parameter(Mandatory = $true)]$TargetInfo)
    $summaryPath = Get-ApplySummaryPath -TargetInfo $TargetInfo
    Assert-SummaryNotOlderThan `
        -SummaryPath $summaryPath `
        -InputPaths @($gdtPath, $TargetInfo.Path) `
        -Description "ApplyGdtSignatures/$($TargetInfo.BaseName)"
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ([int]$summary.schema_version -ne 1 -or
        [string]$summary.program -cne $programName -or
        [string]$summary.target_build_id -cne $buildId -or
        [int]$summary.requested -ne $TargetInfo.FunctionCount -or
        [int]$summary.applied -ne $TargetInfo.FunctionCount -or
        [int]$summary.validated -ne $TargetInfo.FunctionCount -or
        [int]$summary.unique_function_definitions -ne $TargetInfo.FunctionCount -or
        [int]$summary.seven_argument_targets -ne $TargetInfo.SevenArgumentCount -or
        [int]$summary.imported_datatypes -lt 0 -or
        [int]$summary.preserved_labels -lt $TargetInfo.FunctionCount -or
        [string]$summary.calling_convention -cne '__fastcall' -or
        [bool]$summary.cancelled) {
        throw "Incomplete typed-signature application for $($TargetInfo.BaseName): $($summary | ConvertTo-Json -Compress)"
    }
}

function Assert-TypedImportState {
    $typedProjectFile = Join-Path $typedProjectDirectory "$typedProjectName.gpr"
    if (-not (Test-Path -LiteralPath $typedProjectFile -PathType Leaf)) {
        throw "Missing typed Ghidra project: $typedProjectFile"
    }
    Assert-ImportSummary -Path $typedImportSummaryPath -Description 'typed Ghidra symbol import'
    foreach ($targetInfo in $targetInfos) {
        Assert-ApplySummary -TargetInfo $targetInfo
    }
}

function Assert-TypedAnalysisState {
    Assert-AnalysisSummary -Path $typedAnalysisSummaryPath
    $applySummaryPaths = @(
        foreach ($targetInfo in $targetInfos) {
            Get-ApplySummaryPath -TargetInfo $targetInfo
        }
    )
    Assert-SummaryNotOlderThan `
        -SummaryPath $typedAnalysisSummaryPath `
        -InputPaths $applySummaryPaths `
        -Description 'typed Ghidra analysis'
}

function Invoke-BuildTypes {
    New-Item -ItemType Directory -Force -Path `
        $typedInputRoot, `
        $gdtDirectory, `
        (Join-Path $typedInputRoot 'logs'), `
        (Join-Path $typedInputRoot 'builder-projects') | Out-Null

    Remove-Item -LiteralPath $normalizationSummaryPath -Force -ErrorAction SilentlyContinue
    $normalizerArguments = @(
        (Join-Path $scriptDirectory 'normalize_il2cpp_header.py'),
        '--il2cpp-h', (Join-Path $dumperOutput 'il2cpp.h')
    )
    foreach ($targetInfo in $targetInfos) {
        $normalizerArguments += @('--targets', $targetInfo.Path)
    }
    $normalizerArguments += @(
        '--extraction-manifest', $extractionManifestPath,
        '--output-dir', $typedInputRoot
    )
    & python @normalizerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "il2cpp.h normalization failed with exit code $LASTEXITCODE"
    }
    Assert-NormalizationState

    Remove-Item -LiteralPath $gdtSummaryPath -Force -ErrorAction SilentlyContinue
    $builderProjectDirectory = Join-Path $typedInputRoot 'builder-projects'
    $builderProjectName = "TypeBuilder_${buildId}_$([Guid]::NewGuid().ToString('N'))"
    $arguments = @(
        $builderProjectDirectory,
        $builderProjectName,
        '-import', $gameAssembly,
        '-readOnly',
        '-deleteProject',
        '-noanalysis',
        '-scriptPath', $ghidraScripts,
        '-preScript', 'BuildIl2CppTypeArchive.java',
            $normalizedHeaderPath,
            $prototypeHeaderPath,
            $alignmentManifestPath,
            $gdtDirectory,
            $gdtSummaryPath
    )
    foreach ($targetInfo in $targetInfos) {
        $arguments += $targetInfo.Path
    }
    $arguments += @(
        '-max-cpu', $MaxCpu,
        '-log', (Join-Path $typedInputRoot 'logs\gdt-build.log'),
        '-scriptlog', (Join-Path $typedInputRoot 'logs\gdt-build-script.log')
    )
    Invoke-HeadlessWithChecks `
        -Arguments $arguments `
        -Description 'Ghidra IL2CPP type-archive build' `
        -LargeHeap
    Assert-GdtState
    Write-Output "typed_gdt=$gdtPath"
}

function Add-ApplyPreScripts {
    param([Parameter(Mandatory = $true)][Collections.Generic.List[object]]$Arguments)
    foreach ($targetInfo in $targetInfos) {
        $Arguments.Add('-preScript')
        $Arguments.Add('ApplyGdtSignatures.java')
        $Arguments.Add($gdtPath)
        $Arguments.Add($targetInfo.Path)
        $Arguments.Add((Get-ApplySummaryPath -TargetInfo $targetInfo))
    }
}

function Remove-TypedApplySummaries {
    foreach ($targetInfo in $targetInfos) {
        Remove-Item -LiteralPath (Get-ApplySummaryPath -TargetInfo $targetInfo) `
            -Force `
            -ErrorAction SilentlyContinue
    }
}

function Invoke-TypedImport {
    param([Parameter(Mandatory = $true)][bool]$Analyze)
    Assert-GdtState
    New-Item -ItemType Directory -Force -Path `
        $typedProjectDirectory, `
        (Join-Path $typedProjectRoot 'logs') | Out-Null
    Remove-Item -LiteralPath $typedImportSummaryPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $typedAnalysisSummaryPath -Force -ErrorAction SilentlyContinue
    Remove-TypedApplySummaries

    $argumentList = [Collections.Generic.List[object]]::new()
    foreach ($argument in @(
        $typedProjectDirectory,
        $typedProjectName,
        '-import', $gameAssembly,
        '-overwrite',
        '-scriptPath', $ghidraScripts,
        '-preScript', 'ImportIl2CppSymbols.java', $scriptJson, $typedImportSummaryPath
    )) {
        $argumentList.Add($argument)
    }
    Add-ApplyPreScripts -Arguments $argumentList
    $operationName = if ($Analyze) { 'typed-all' } else { 'typed-import' }
    if ($Analyze) {
        foreach ($argument in @(
            '-analysisTimeoutPerFile', $AnalysisTimeoutSeconds,
            '-postScript', 'RecordAnalysisCompletion.java', $typedAnalysisSummaryPath
        )) {
            $argumentList.Add($argument)
        }
    }
    else {
        $argumentList.Add('-noanalysis')
    }
    foreach ($argument in @(
        '-max-cpu', $MaxCpu,
        '-log', (Join-Path $typedProjectRoot "logs\$operationName.log"),
        '-scriptlog', (Join-Path $typedProjectRoot "logs\$operationName-script.log")
    )) {
        $argumentList.Add($argument)
    }

    Invoke-HeadlessWithChecks `
        -Arguments @($argumentList) `
        -Description "Ghidra $operationName" `
        -LargeHeap
    Assert-TypedImportState
    if ($Analyze) {
        Assert-TypedAnalysisState
    }
}

function Invoke-TypedAnalyze {
    Assert-GdtState
    Assert-TypedImportState
    Remove-Item -LiteralPath $typedAnalysisSummaryPath -Force -ErrorAction SilentlyContinue
    Remove-TypedApplySummaries

    $argumentList = [Collections.Generic.List[object]]::new()
    foreach ($argument in @(
        $typedProjectDirectory,
        $typedProjectName,
        '-process', $programName,
        '-scriptPath', $ghidraScripts
    )) {
        $argumentList.Add($argument)
    }
    Add-ApplyPreScripts -Arguments $argumentList
    foreach ($argument in @(
        '-analysisTimeoutPerFile', $AnalysisTimeoutSeconds,
        '-max-cpu', $MaxCpu,
        '-postScript', 'RecordAnalysisCompletion.java', $typedAnalysisSummaryPath,
        '-log', (Join-Path $typedProjectRoot 'logs\typed-analyze.log'),
        '-scriptlog', (Join-Path $typedProjectRoot 'logs\typed-analyze-script.log')
    )) {
        $argumentList.Add($argument)
    }

    Invoke-HeadlessWithChecks `
        -Arguments @($argumentList) `
        -Description 'Ghidra typed analysis' `
        -LargeHeap
    Assert-TypedImportState
    Assert-TypedAnalysisState
}

function Invoke-TypedExport {
    param([Parameter(Mandatory = $true)]$TargetInfo)
    Assert-GdtState
    Assert-TypedImportState
    Assert-TypedAnalysisState

    $exportDirectory = Join-Path $typedProjectRoot ("exports\{0}" -f $TargetInfo.BaseName)
    New-Item -ItemType Directory -Force -Path $exportDirectory | Out-Null
    $resolvedTypedProjectRoot = [IO.Path]::GetFullPath($typedProjectRoot)
    $resolvedExportDirectory = [IO.Path]::GetFullPath($exportDirectory)
    $expectedPrefix = $resolvedTypedProjectRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) +
        [IO.Path]::DirectorySeparatorChar
    if (-not $resolvedExportDirectory.StartsWith($expectedPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean typed export directory outside typed project root: $resolvedExportDirectory"
    }
    Get-ChildItem -LiteralPath $resolvedExportDirectory -File -Filter '*.c' -ErrorAction SilentlyContinue |
        Remove-Item -Force
    $summaryPath = Join-Path $resolvedExportDirectory '_export_summary.json'
    Remove-Item -LiteralPath $summaryPath -Force -ErrorAction SilentlyContinue
    $arguments = @(
        $typedProjectDirectory,
        $typedProjectName,
        '-process', $programName,
        '-noanalysis',
        '-scriptPath', $ghidraScripts,
        '-postScript', 'ExportFunctionDecompilations.java', $resolvedExportDirectory, $TargetInfo.Path,
        '-log', (Join-Path $typedProjectRoot ("logs\typed-export-{0}.log" -f $TargetInfo.BaseName)),
        '-scriptlog', (Join-Path $typedProjectRoot ("logs\typed-export-{0}-script.log" -f $TargetInfo.BaseName))
    )
    Invoke-HeadlessWithChecks `
        -Arguments $arguments `
        -Description "Ghidra typed export/$($TargetInfo.BaseName)" `
        -LargeHeap
    if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
        throw "Typed Ghidra export did not write its completion summary: $summaryPath"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ([int]$summary.requested -ne $TargetInfo.FunctionCount -or
        [int]$summary.processed -ne $TargetInfo.FunctionCount -or
        [int]$summary.exported -ne $TargetInfo.FunctionCount -or
        [int]$summary.failed -ne 0 -or
        [bool]$summary.cancelled) {
        throw "Incomplete typed Ghidra export: $($summary | ConvertTo-Json -Compress)"
    }
    foreach ($target in @($TargetInfo.Data.functions)) {
        $safeName = $target.name -replace '[^A-Za-z0-9_.-]', '_'
        $outputPath = Join-Path $resolvedExportDirectory "$safeName.c"
        if (-not (Test-Path -LiteralPath $outputPath -PathType Leaf) -or
            (Get-Item -LiteralPath $outputPath).Length -le 0) {
            throw "Missing or empty typed Ghidra export: $outputPath"
        }
    }
    $actualExportCount = @(
        Get-ChildItem -LiteralPath $resolvedExportDirectory -File -Filter '*.c'
    ).Count
    if ($actualExportCount -ne $TargetInfo.FunctionCount) {
        throw "Expected $($TargetInfo.FunctionCount) distinct typed C exports, found $actualExportCount"
    }
    Write-Output "decompiled_typed_$($TargetInfo.BaseName)=$resolvedExportDirectory"
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

if ($Stage -eq 'build-types') {
    Invoke-BuildTypes
}

if ($Stage -eq 'typed-import') {
    Invoke-TypedImport -Analyze $false
}

if ($Stage -eq 'typed-analyze') {
    Invoke-TypedAnalyze
}

if ($Stage -eq 'typed-all') {
    Invoke-BuildTypes
    Invoke-TypedImport -Analyze $true
}

if ($Stage -eq 'typed-export') {
    Invoke-TypedExport -TargetInfo $targetByName[$TargetSet]
}

Write-Output "build_id=$buildId"
Write-Output "ghidra_project=$projectDirectory\$projectName.gpr"
if ($isTypedStage -and $Stage -ne 'build-types') {
    Write-Output "typed_ghidra_project=$typedProjectDirectory\$typedProjectName.gpr"
}
