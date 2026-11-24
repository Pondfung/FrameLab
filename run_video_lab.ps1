$ErrorActionPreference = 'Stop'
$repository = $PSScriptRoot
$videoPython = Join-Path $repository '.runtime\venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $videoPython -PathType Leaf)) {
    throw 'Run python setup_runtime.py deps before starting the video tool.'
}
$site = Join-Path $repository '.runtime\venv\Lib\site-packages'
$cache = Join-Path $repository '.runtime\cupy-cache'
$previousPythonPath = $env:PYTHONPATH
$previousCuPyCache = $env:CUPY_CACHE_DIR
try {
    if ($site -match '[^\x00-\x7f]') {
        # Windows NVRTC can fail to read include files through a Unicode path.
        $bytes = [Text.Encoding]::UTF8.GetBytes($site.ToLowerInvariant())
        $sha = [Security.Cryptography.SHA256]::Create()
        try { $id = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '').Substring(0, 16) }
        finally { $sha.Dispose() }
        $aliasRoot = Join-Path $env:LOCALAPPDATA "VideoLab\$id"
        if ($aliasRoot -match '[^\x00-\x7f]') { throw 'VideoLab needs an ASCII LocalAppData path for Windows NVRTC.' }
        $alias = Join-Path $aliasRoot 'site-packages'
        New-Item -ItemType Directory -Path $aliasRoot -Force | Out-Null
        if (-not (Test-Path -LiteralPath $alias)) {
            try { New-Item -ItemType Junction -Path $alias -Target $site | Out-Null }
            catch { if (-not (Test-Path -LiteralPath $alias)) { throw } }
        }
        $link = Get-Item -LiteralPath $alias
        if ($link.LinkType -ne 'Junction' -or @($link.Target)[0] -ne $site) {
            throw "The VideoLab dependency alias points to an unexpected directory: $alias"
        }
        $env:PYTHONPATH = $alias
        $cache = Join-Path $aliasRoot 'cupy-cache'
    }
    $env:CUPY_CACHE_DIR = $cache
    & $videoPython (Join-Path $repository 'video_lab.py') @args
    $status = $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
    $env:CUPY_CACHE_DIR = $previousCuPyCache
}
exit $status
