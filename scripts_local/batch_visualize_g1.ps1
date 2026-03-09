param(
    [Parameter(Mandatory = $true)][string]$MotionDir,
    [Parameter(Mandatory = $true)][string]$VideoDir,
    [Parameter(Mandatory = $true)][string]$GmrRoot,
    [string]$Robot = "unitree_g1",
    [string]$PythonBin = "python",
    [int]$MaxFiles = 0
)

if (!(Test-Path -LiteralPath $MotionDir)) {
    throw "Motion directory not found: $MotionDir"
}

$visScript = Join-Path $GmrRoot "scripts/vis_robot_motion.py"
if (!(Test-Path -LiteralPath $visScript)) {
    throw "Visualization script not found: $visScript"
}

New-Item -ItemType Directory -Path $VideoDir -Force | Out-Null

$resolvedMotionDir = (Resolve-Path -LiteralPath $MotionDir).Path
$motionFiles = Get-ChildItem -Path $MotionDir -Recurse -Filter *.pkl | Sort-Object FullName
if ($motionFiles.Count -eq 0) {
    throw "No .pkl files found under $MotionDir"
}

$done = 0
$failed = 0
$total = $motionFiles.Count

foreach ($file in $motionFiles) {
    if ($MaxFiles -gt 0 -and $done -ge $MaxFiles) {
        break
    }

    $relative = $file.FullName.Substring($resolvedMotionDir.Length).TrimStart('\', '/')
    $videoName = ($relative -replace "[\\/]", "__") -replace "\.pkl$", ".mp4"
    $videoPath = Join-Path $VideoDir $videoName

    $index = $done + $failed + 1
    Write-Host "[viz] [$index/$total] $($file.FullName) -> $videoPath"

    & $PythonBin $visScript `
        --robot $Robot `
        --robot_motion_path $file.FullName `
        --record_video `
        --video_path $videoPath `
        --num_loops 1

    if ($LASTEXITCODE -eq 0) {
        $done += 1
    } else {
        $failed += 1
    }
}

Write-Host "[viz] finished done=$done failed=$failed video_dir=$VideoDir"
if ($failed -gt 0) {
    exit 1
}
