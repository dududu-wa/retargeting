param(
    [Parameter(Mandatory = $true)][string]$ManifestCsv,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [Parameter(Mandatory = $true)][string]$GmrRoot,
    [string]$Robot = "unitree_g1",
    [string]$PythonBin = "python",
    [int]$Limit = 0,
    [switch]$Overwrite,
    [switch]$DryRun,
    [switch]$StopOnError,
    [string]$LogDir = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyScript = Join-Path $scriptDir "batch_convert_hdm05_to_g1.py"

if (!(Test-Path -LiteralPath $pyScript)) {
    throw "Cannot find $pyScript"
}

$argsList = @(
    $pyScript,
    "--manifest_csv", $ManifestCsv,
    "--output_dir", $OutputDir,
    "--gmr_root", $GmrRoot,
    "--robot", $Robot,
    "--python_bin", $PythonBin
)

if ($Limit -gt 0) {
    $argsList += @("--limit", "$Limit")
}
if ($Overwrite) {
    $argsList += "--overwrite"
}
if ($DryRun) {
    $argsList += "--dry_run"
}
if ($StopOnError) {
    $argsList += "--stop_on_error"
}
if ($LogDir -ne "") {
    $argsList += @("--log_dir", $LogDir)
}

& $PythonBin @argsList
exit $LASTEXITCODE
