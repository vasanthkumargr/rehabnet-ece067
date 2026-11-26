# Start RehabNet on this computer. Usage (PowerShell, in the project folder):
#   .\run_app.ps1          -> http://localhost:8000   (camera works in this computer's browser)
#   .\run_app.ps1 -Https   -> https://<this-computer-ip>:8443 for phones/tablets on the same Wi-Fi
# First run: creates the Python environment, installs packages, builds the exercise references and demo data.
param([switch]$Https)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating Python environment..."
    python -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt
}
if (-not (Test-Path "app\data\references\squat_both.pkl")) {
    Write-Host "Building exercise references (needs data\processed\rehab24_reps.pkl)..."
    .venv\Scripts\python -P app\build_references.py
}
if (-not (Test-Path "app\data\rehabnet.db")) {
    Write-Host "Creating demo accounts..."
    .venv\Scripts\python -P app\seed.py
    Write-Host "Demo sign-in details: app\data\DEMO_ACCOUNTS.md"
}
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } | Select-Object -First 1).IPAddress
Write-Host ""
if ($Https) {
    if (-not (Test-Path "app\data\cert.pem")) { .venv\Scripts\python tools\make_cert.py }
    Write-Host "RehabNet is starting:  https://localhost:8443"
    if ($ip) { Write-Host "On a phone on the same Wi-Fi open: https://$($ip):8443  (accept the one-time certificate warning)" }
    Write-Host "Press Ctrl+C to stop."
    .venv\Scripts\python -m uvicorn app.server.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile app\data\key.pem --ssl-certfile app\data\cert.pem
} else {
    Write-Host "RehabNet is starting:  http://localhost:8000"
    Write-Host "(For phones on the same Wi-Fi run: .\run_app.ps1 -Https)"
    Write-Host "Press Ctrl+C to stop."
    .venv\Scripts\python -m uvicorn app.server.main:app --host 0.0.0.0 --port 8000
}
