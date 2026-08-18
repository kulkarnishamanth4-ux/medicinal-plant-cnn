# =========================================================
# HerbScan AI - Basic Security Scanner
# Performs SAST (Static Application Security Testing) and SCA (Software Composition Analysis)
# =========================================================

Write-Host "Starting HerbScan AI Security Scan..." -ForegroundColor Cyan

# 1. Check for hardcoded Supabase Secrets
Write-Host "`n[1/3] Scanning for hardcoded Supabase secrets..." -ForegroundColor Yellow
$secretRegex = "eyJhbGciOiJIUzI1NiIs"
$foundSecrets = Get-ChildItem -Path . -Filter *.py -Recurse | Select-String -Pattern $secretRegex

if ($foundSecrets.Count -gt 0) {
    Write-Host "❌ DANGER: Hardcoded Supabase keys found!" -ForegroundColor Red
    $foundSecrets | ForEach-Object { Write-Host "   Found in: $($_.Path):$($_.LineNumber)" -ForegroundColor Red }
    Write-Host "   Move these to .streamlit/secrets.toml immediately." -ForegroundColor Red
} else {
    Write-Host "✅ No hardcoded Supabase keys detected in Python files." -ForegroundColor Green
}

# 2. Check if secrets.toml is ignored
Write-Host "`n[2/3] Verifying .gitignore configuration..." -ForegroundColor Yellow
$gitignoreContent = Get-Content -Path .gitignore -ErrorAction SilentlyContinue
if ($gitignoreContent -match ".streamlit/secrets.toml") {
    Write-Host "✅ secrets.toml is properly excluded in .gitignore." -ForegroundColor Green
} else {
    Write-Host "❌ WARNING: .streamlit/secrets.toml is NOT in .gitignore. You might leak secrets!" -ForegroundColor Red
}

# 3. Check for vulnerable dependencies
Write-Host "`n[3/3] Scanning requirements.txt for vulnerabilities (requires 'safety' package)..." -ForegroundColor Yellow
if (Get-Command safety -ErrorAction SilentlyContinue) {
    safety check -r requirements.txt
    if ($LASTEXITCODE -eq 0) {
         Write-Host "✅ Dependencies are secure." -ForegroundColor Green
    } else {
         Write-Host "❌ Vulnerable dependencies found! Please update them." -ForegroundColor Red
    }
} else {
    Write-Host "⚠️ The 'safety' package is not installed. Run 'pip install safety' to enable dependency scanning." -ForegroundColor DarkYellow
}

Write-Host "`nSecurity scan complete." -ForegroundColor Cyan
