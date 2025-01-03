# PowerShell script to check environment variables
$requiredVars = @(
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "S3_BUCKET",
    "EC2_IP",
    "EC2_KEY"
)

$missingVars = @()

foreach ($var in $requiredVars) {
    $value = [System.Environment]::GetEnvironmentVariable($var, [System.EnvironmentVariableTarget]::User)
    if ([string]::IsNullOrEmpty($value)) {
        $missingVars += $var
    } else {
        Write-Host "${var}: [Set]"
    }
}

if ($missingVars.Count -gt 0) {
    Write-Host "`nMissing environment variables:"
    foreach ($var in $missingVars) {
        Write-Host "- $var"
    }
    exit 1
} else {
    Write-Host "`nAll required environment variables are set!"
} 