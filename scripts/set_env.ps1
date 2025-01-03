# PowerShell script to set environment variables
param(
    [string]$envFile = ".env"
)

if (!(Test-Path $envFile)) {
    Write-Host "Error: .env file not found at $envFile"
    exit 1
}

Write-Host "Setting environment variables from $envFile..."

Get-Content $envFile | ForEach-Object {
    if ($_ -match '^([^#][^=]+)=(.*)$') {
        $name = $matches[1].Trim()
        $value = $matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($name, $value, [System.EnvironmentVariableTarget]::User)
        Write-Host "Set $name"
    }
}

Write-Host "Environment variables have been set successfully!" 