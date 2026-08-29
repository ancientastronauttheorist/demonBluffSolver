[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$GameRoot,

    [string]$WorkspaceRoot = 'B:\CodexTools\DemonBluffReverseEngineering'
)

$ErrorActionPreference = 'Stop'
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$reverseEngineeringRoot = Split-Path -Parent $scriptDirectory
$lock = Get-Content -LiteralPath (Join-Path $reverseEngineeringRoot 'toolchain\toolchain.lock.json') -Raw | ConvertFrom-Json
$cpp2il = $lock.tools.cpp2il
$sdk = $lock.tools.dotnet_sdk
$ilspy = $lock.tools.ilspycmd

$gameAssembly = Join-Path $GameRoot 'GameAssembly.dll'
$metadata = Join-Path $GameRoot 'Demon Bluff_Data\il2cpp_data\Metadata\global-metadata.dat'
foreach ($requiredPath in @($gameAssembly, $metadata)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required input not found: $requiredPath"
    }
}

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

$downloadDirectory = Join-Path $WorkspaceRoot 'downloads'
$sdkDirectory = Join-Path $WorkspaceRoot ("tools\dotnet-sdk-{0}" -f $sdk.version)
$sdkArchive = Join-Path $downloadDirectory $sdk.archive_name
$dotnet = Join-Path $sdkDirectory 'dotnet.exe'
New-Item -ItemType Directory -Force -Path $downloadDirectory, $sdkDirectory | Out-Null

if (-not (Test-Path -LiteralPath $dotnet -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $sdkArchive -PathType Leaf)) {
        $temporaryArchive = "$sdkArchive.download"
        Invoke-WebRequest -Uri $sdk.archive_url -OutFile $temporaryArchive
        Move-Item -LiteralPath $temporaryArchive -Destination $sdkArchive
    }
    if ((Get-Item -LiteralPath $sdkArchive).Length -ne $sdk.archive_size) {
        throw '.NET SDK archive size mismatch'
    }
    $sdkHash = (Get-FileHash -Algorithm SHA512 -LiteralPath $sdkArchive).Hash
    if ($sdkHash -ne $sdk.archive_sha512) {
        throw ".NET SDK archive hash mismatch: expected $($sdk.archive_sha512), got $sdkHash"
    }
    Expand-Archive -LiteralPath $sdkArchive -DestinationPath $sdkDirectory -Force
}
if ((& $dotnet --version) -ne $sdk.version) {
    throw "Installed .NET SDK does not match pinned version $($sdk.version)"
}

$sourceDirectory = Join-Path $WorkspaceRoot ("sources\Cpp2IL-{0}" -f $cpp2il.commit.Substring(0, 12))
if (-not (Test-Path -LiteralPath (Join-Path $sourceDirectory '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $sourceDirectory) | Out-Null
    & git clone --filter=blob:none --no-checkout $cpp2il.repository $sourceDirectory
    if ($LASTEXITCODE -ne 0) { throw 'Cpp2IL clone failed' }
}
& git -C $sourceDirectory fetch origin $cpp2il.commit --depth 1
if ($LASTEXITCODE -ne 0) { throw 'Cpp2IL fetch failed' }
& git -C $sourceDirectory checkout --detach $cpp2il.commit
if ($LASTEXITCODE -ne 0) { throw 'Cpp2IL checkout failed' }
if ((& git -C $sourceDirectory rev-parse HEAD) -ne $cpp2il.commit) {
    throw 'Cpp2IL checkout does not match the pinned commit'
}

$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'
& $dotnet build (Join-Path $sourceDirectory 'Cpp2IL\Cpp2IL.csproj') -c Release
if ($LASTEXITCODE -ne 0) { throw 'Cpp2IL build failed' }
$cpp2ilDll = Join-Path $sourceDirectory 'Cpp2IL\bin\Release\net10.0\Cpp2IL.dll'

$artifactRoot = Join-Path $WorkspaceRoot "artifacts\$buildId"
$cpp2ilOutput = Join-Path $artifactRoot ("cpp2il-{0}\dll_il_recovery" -f $cpp2il.commit.Substring(0, 12))
$runtimeDirectory = Join-Path $WorkspaceRoot 'runtime'
New-Item -ItemType Directory -Force -Path $cpp2ilOutput, $runtimeDirectory | Out-Null
Push-Location $runtimeDirectory
try {
    & $dotnet $cpp2ilDll --game-path $GameRoot --exe-name 'Demon Bluff' --output-as $cpp2il.output_format --output-to $cpp2ilOutput
    if ($LASTEXITCODE -ne 0) { throw 'Cpp2IL recovery failed' }
}
finally {
    Pop-Location
}

$recoveredAssembly = Join-Path $cpp2ilOutput 'Assembly-CSharp.dll'
if (-not (Test-Path -LiteralPath $recoveredAssembly -PathType Leaf)) {
    throw 'Cpp2IL did not produce Assembly-CSharp.dll'
}

$ilspyArchive = Join-Path $downloadDirectory $ilspy.archive_name
$ilspyDirectory = Join-Path $WorkspaceRoot ("tools\ilspycmd-{0}" -f $ilspy.version)
$ilspyDll = Join-Path $ilspyDirectory 'tools\net10.0\any\ilspycmd.dll'
if (-not (Test-Path -LiteralPath $ilspyArchive -PathType Leaf)) {
    $temporaryArchive = "$ilspyArchive.download"
    Invoke-WebRequest -Uri $ilspy.archive_url -OutFile $temporaryArchive
    Move-Item -LiteralPath $temporaryArchive -Destination $ilspyArchive
}
if ((Get-Item -LiteralPath $ilspyArchive).Length -ne $ilspy.archive_size) {
    throw 'ILSpyCmd archive size mismatch'
}
$ilspyHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $ilspyArchive).Hash
if ($ilspyHash -ne $ilspy.archive_sha256) {
    throw "ILSpyCmd archive hash mismatch: expected $($ilspy.archive_sha256), got $ilspyHash"
}
if (-not (Test-Path -LiteralPath $ilspyDll -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path $ilspyDirectory | Out-Null
    Expand-Archive -LiteralPath $ilspyArchive -DestinationPath $ilspyDirectory -Force
}

$sourceOutput = Join-Path $artifactRoot ("ilspy-{0}\cpp2il-{1}" -f $ilspy.version, $cpp2il.commit.Substring(0, 12))
New-Item -ItemType Directory -Force -Path $sourceOutput | Out-Null
& $dotnet $ilspyDll --disable-updatecheck --ignore-decompilation-errors --nested-directories -p -r $cpp2ilOutput -o $sourceOutput $recoveredAssembly
if ($LASTEXITCODE -ne 0) { throw 'ILSpyCmd decompilation failed' }

Write-Output "build_id=$buildId"
Write-Output "recovered_dll=$cpp2ilOutput"
Write-Output "decompiled_source=$sourceOutput"
