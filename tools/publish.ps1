# publish.ps1 - branch, commit satu berkas per commit, push, Issue, dan PR draft.
# Jalankan dari root repo: powershell -ExecutionPolicy Bypass -File tools\publish.ps1
# Aman dijalankan ulang: memakai Issue yang sudah ada, tidak push ke main kecuali commit pertama.
$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $PSScriptRoot)
$repo = "alfzilham/BacteriaCV"
$branch = "feat/initial-monorepo"

$trailer1 = "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
$trailer2 = "Claude-Session: https://claude.ai/code/session_01YQPXpgXQPXFy8BHqZAAjW4"

function Fail([string]$m) { Write-Host "GAGAL: $m" -ForegroundColor Red; exit 1 }

function Commit-One([string]$path, [string]$type, [string]$summary) {
    git add -- $path | Out-Null
    $msg = $type + ": " + $summary + "`n`n" + $trailer1 + "`n" + $trailer2
    git commit -m $msg --quiet
    if ($LASTEXITCODE -ne 0) { Fail "commit $path" }
}

# 1. main: repo kosong, .gitignore jadi commit pertama.
if ((git branch --show-current) -ne "main") { git checkout -b main 2>&1 | Out-Null }
git rev-parse --verify HEAD 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Commit-One ".gitignore" "chore" "tambah .gitignore monorepo"
    git push -u origin main
    if ($LASTEXITCODE -ne 0) { Fail "push main" }
}

# 2. Branch kerja.
git checkout -b $branch 2>&1 | Out-Null
if ((git branch --show-current) -ne $branch) { Fail "tidak berada di $branch" }

# 3. Satu commit per berkas.
$files = git ls-files --others --exclude-standard
foreach ($f in $files) {
    if ($f -like "apps/site/*") { $t = "feat"; $s = "tambah $f ke situs" }
    elseif ($f -like "apps/api/*") { $t = "chore"; $s = "pindahkan $f ke apps/api" }
    elseif ($f -like "apps/mobile/*") { $t = "chore"; $s = "pindahkan $f ke apps/mobile" }
    elseif ($f -like "docs/*") { $t = "docs"; $s = "tambah $f" }
    elseif ($f -like ".github/*") { $t = "chore"; $s = "tambah CI $f" }
    else { $t = "chore"; $s = "tambah $f" }
    Commit-One $f $t $s
}
git diff --check main..HEAD
git push -u origin $branch
if ($LASTEXITCODE -ne 0) { Fail "push $branch" }

# 4. Issue: pakai yang sudah ada bila judulnya sama.
$title = "feat: bangun ulang BacteriaCV sebagai monorepo MVC"
$existing = gh issue list --repo $repo --state open --search $title --json number,title | ConvertFrom-Json
$hit = $existing | Where-Object { $_.title -eq $title } | Select-Object -First 1
if ($hit) { $issueNo = $hit.number }
else {
    $url = gh issue create --repo $repo --title $title --body-file docs\review\ISSUE_BODY.md
    $issueNo = ($url -split "/")[-1]
}
Write-Host "Issue: #$issueNo"

# 5. PR draft. Body lewat file sementara (bukan --body) supaya tidak terpecah.
$robot = [char]::ConvertFromUtf32(0x1F916)
$body = (Get-Content docs\review\PR_BODY.md -Raw) + "`n`nCloses #" + $issueNo + "`n`n" + $robot + " Generated with [Claude Code](https://claude.com/claude-code)`n`nhttps://claude.ai/code/session_01YQPXpgXQPXFy8BHqZAAjW4`n"
$tmp = Join-Path $env:TEMP "bcv_pr_body.md"
[System.IO.File]::WriteAllText($tmp, $body, (New-Object System.Text.UTF8Encoding($false)))
gh pr create --repo $repo --draft --base main --head $branch --title "feat: monorepo MVC untuk BacteriaCV (site, api, mobile)" --body-file $tmp
