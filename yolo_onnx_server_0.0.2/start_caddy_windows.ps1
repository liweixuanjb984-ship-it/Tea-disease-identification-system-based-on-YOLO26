$ErrorActionPreference = "Stop"

if (!(Test-Path ".\caddy.exe")) {
    Write-Host "Downloading Caddy..."
    Invoke-WebRequest -Uri "https://caddyserver.com/api/download?os=windows&arch=amd64" -OutFile ".\caddy.exe"
}

Write-Host "Starting Caddy HTTPS reverse proxy..."
.\caddy.exe run --config .\Caddyfile
