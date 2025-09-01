# PowerShell deployment script for Windows Server
# Industry Classification Code Mapper - Windows Deployment Script

param(
    [string]$Domain = "localhost",
    [switch]$Production = $false,
    [switch]$InstallDocker = $false,
    [switch]$Force = $false
)

# Set execution policy and error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Color functions for output
function Write-Success { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-Info { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Cyan }
function Write-Warning { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-Step { param($Message) Write-Host "🚀 $Message" -ForegroundColor Blue }

Write-Step "Starting Industry Classification Code Mapper deployment on Windows Server"
Write-Info "Domain: $Domain"
Write-Info "Production Mode: $Production"
Write-Info "Current Directory: $(Get-Location)"

# Check if running as Administrator
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = (New-Object Security.Principal.WindowsPrincipal($currentUser)).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin -and ($InstallDocker -or $Force)) {
    Write-Error "This script requires Administrator privileges for Docker installation."
    Write-Info "Please run PowerShell as Administrator or remove the -InstallDocker flag."
    exit 1
}

# Function to check if a command exists
function Test-Command {
    param($CommandName)
    return [bool](Get-Command -Name $CommandName -ErrorAction SilentlyContinue)
}

# Install Docker Desktop if requested and not present
if ($InstallDocker -or (-not (Test-Command "docker"))) {
    Write-Step "Installing Docker Desktop for Windows..."
    
    if (-not $isAdmin) {
        Write-Error "Administrator privileges required to install Docker Desktop."
        Write-Info "Please either:"
        Write-Info "1. Run PowerShell as Administrator with -InstallDocker flag"
        Write-Info "2. Install Docker Desktop manually from https://www.docker.com/products/docker-desktop"
        exit 1
    }
    
    try {
        # Download Docker Desktop installer
        $dockerUrl = "https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe"
        $dockerInstaller = "$env:TEMP\DockerDesktopInstaller.exe"
        
        Write-Info "Downloading Docker Desktop..."
        Invoke-WebRequest -Uri $dockerUrl -OutFile $dockerInstaller
        
        Write-Info "Installing Docker Desktop (this may take several minutes)..."
        Start-Process -FilePath $dockerInstaller -ArgumentList "install", "--quiet" -Wait
        
        Write-Success "Docker Desktop installation completed"
        Write-Warning "Please restart your computer and re-run this script after Docker Desktop is fully started"
        exit 0
    }
    catch {
        Write-Error "Failed to install Docker Desktop: $($_.Exception.Message)"
        Write-Info "Please install Docker Desktop manually from https://www.docker.com/products/docker-desktop"
        exit 1
    }
}

# Check if Docker is available
if (-not (Test-Command "docker")) {
    Write-Error "Docker is not installed or not in PATH"
    Write-Info "Please install Docker Desktop from https://www.docker.com/products/docker-desktop"
    Write-Info "Or run this script with -InstallDocker flag as Administrator"
    exit 1
}

# Check Docker status
try {
    $dockerVersion = docker version --format "{{.Server.Version}}" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker daemon is not running"
        Write-Info "Please start Docker Desktop and wait for it to be ready"
        exit 1
    }
    Write-Success "Docker is running (version: $dockerVersion)"
}
catch {
    Write-Error "Cannot connect to Docker daemon. Please ensure Docker Desktop is running."
    exit 1
}

# Check for Docker Compose
if (-not (Test-Command "docker-compose")) {
    Write-Error "Docker Compose is not available"
    Write-Info "Please ensure Docker Desktop is properly installed with Docker Compose"
    exit 1
}

# Verify required files exist
$requiredFiles = @(
    "Dockerfile",
    "requirements.txt",
    "app.py",
    "nginx.conf",
    "docker-compose.windows.yml"
)

Write-Step "Checking required files..."
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "Required file missing: $file"
        exit 1
    }
    Write-Success "Found: $file"
}

# Check for data files
$dataFiles = @("Localised ISIC.xlsx", "isco_index.xlsx")
foreach ($file in $dataFiles) {
    if (Test-Path $file) {
        Write-Success "Found data file: $file"
    } else {
        Write-Warning "Optional data file missing: $file (feature may be limited)"
    }
}

# Create necessary directories
Write-Step "Creating directories..."
$directories = @("outputs", "ssl", "logs", "logs\nginx")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Success "Created directory: $dir"
    }
}

# Update nginx configuration if domain is specified
if ($Domain -ne "localhost" -and $Domain -ne "") {
    Write-Step "Updating nginx configuration for domain: $Domain"
    try {
        $nginxConf = Get-Content "nginx.conf"
        $nginxConf = $nginxConf -replace "server_name localhost;", "server_name $Domain;"
        $nginxConf | Set-Content "nginx.conf"
        Write-Success "Updated nginx.conf with domain: $Domain"
    }
    catch {
        Write-Warning "Could not update nginx.conf: $($_.Exception.Message)"
    }
}

# Select compose file based on production mode
$composeFile = if ($Production) { "docker-compose.prod.yml" } else { "docker-compose.windows.yml" }

if (-not (Test-Path $composeFile)) {
    Write-Error "Compose file not found: $composeFile"
    exit 1
}

Write-Step "Using Docker Compose file: $composeFile"

# Stop existing containers
Write-Step "Stopping existing containers..."
try {
    docker-compose -f $composeFile down --remove-orphans 2>$null
}
catch {
    Write-Info "No existing containers to stop"
}

# Build and start services
Write-Step "Building application container..."
try {
    docker-compose -f $composeFile build --no-cache
    if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }
    Write-Success "Application container built successfully"
}
catch {
    Write-Error "Failed to build container: $($_.Exception.Message)"
    exit 1
}

Write-Step "Starting services..."
try {
    docker-compose -f $composeFile up -d
    if ($LASTEXITCODE -ne 0) { throw "Docker compose up failed" }
    Write-Success "Services started successfully"
}
catch {
    Write-Error "Failed to start services: $($_.Exception.Message)"
    exit 1
}

# Wait for services to be ready
Write-Step "Waiting for services to start..."
Start-Sleep -Seconds 30

# Check service status
Write-Step "Checking service status..."
try {
    $services = docker-compose -f $composeFile ps --format "table {{.Name}}\t{{.State}}\t{{.Ports}}"
    Write-Host $services
}
catch {
    Write-Warning "Could not retrieve service status"
}

# Health check
Write-Step "Performing health checks..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -TimeoutSec 10 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Success "Application health check passed"
    }
}
catch {
    Write-Warning "Application health check failed, but services may still be starting"
}

# Get server information
try {
    $serverInfo = Get-ComputerInfo
    $ipAddress = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.*" }).IPAddress | Select-Object -First 1
    if (-not $ipAddress) {
        $ipAddress = "localhost"
    }
}
catch {
    $ipAddress = "localhost"
}

# Display deployment results
Write-Host "`n" -ForegroundColor Green
Write-Success "🎉 Deployment completed successfully!"
Write-Host ""

Write-Host "🌐 Access your application at:" -ForegroundColor Blue
if ($Production) {
    Write-Host "   Production URL:  http://$Domain" -ForegroundColor Green
    if ($ipAddress -ne "localhost") {
        Write-Host "   IP Access:       http://$ipAddress" -ForegroundColor Green
    }
} else {
    Write-Host "   Direct Access:   http://localhost:8501" -ForegroundColor Green
    if ($ipAddress -ne "localhost") {
        Write-Host "   Network Access:  http://${ipAddress}:8501" -ForegroundColor Green
    }
    Write-Host "   With Nginx:      http://localhost" -ForegroundColor Green
}

Write-Host ""
Write-Host "📊 Management Commands:" -ForegroundColor Blue
Write-Host "   View Status:     docker-compose -f $composeFile ps" -ForegroundColor White
Write-Host "   View Logs:       docker-compose -f $composeFile logs -f" -ForegroundColor White
Write-Host "   Restart App:     docker-compose -f $composeFile restart isic-mapper" -ForegroundColor White
Write-Host "   Stop Services:   docker-compose -f $composeFile down" -ForegroundColor White
Write-Host "   Update App:      .\deploy-windows.ps1 -Domain $Domain $(if($Production){'-Production'})" -ForegroundColor White

Write-Host ""
Write-Host "📁 Data Directories:" -ForegroundColor Blue
Write-Host "   User Outputs:    $(Join-Path (Get-Location) 'outputs')" -ForegroundColor White
Write-Host "   Application Logs: $(Join-Path (Get-Location) 'logs')" -ForegroundColor White

# Windows Firewall configuration
Write-Host ""
Write-Host "🔥 Windows Firewall:" -ForegroundColor Blue
if ($isAdmin) {
    try {
        # Check if rules already exist
        $existingRules = Get-NetFirewallRule -DisplayName "Code Mapper*" -ErrorAction SilentlyContinue
        
        if (-not $existingRules) {
            Write-Info "Creating Windows Firewall rules..."
            New-NetFirewallRule -DisplayName "Code Mapper HTTP" -Direction Inbound -Protocol TCP -LocalPort 80 -Action Allow | Out-Null
            New-NetFirewallRule -DisplayName "Code Mapper HTTPS" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow | Out-Null
            New-NetFirewallRule -DisplayName "Code Mapper App" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow | Out-Null
            Write-Success "Windows Firewall rules created"
        } else {
            Write-Success "Windows Firewall rules already exist"
        }
    }
    catch {
        Write-Warning "Could not configure Windows Firewall automatically: $($_.Exception.Message)"
        Write-Info "Please manually allow ports 80, 443, and 8501 in Windows Firewall"
    }
} else {
    Write-Info "Run as Administrator to automatically configure Windows Firewall"
    Write-Info "Or manually allow inbound connections on ports: 80, 443, 8501"
}

# SSL/HTTPS setup information
if ($Production -and ($Domain -ne "localhost")) {
    Write-Host ""
    Write-Host "🔐 HTTPS Setup (Production):" -ForegroundColor Blue
    Write-Host "   1. Obtain SSL certificate for $Domain" -ForegroundColor White
    Write-Host "   2. Place certificate files in .\ssl\ directory:" -ForegroundColor White
    Write-Host "      - cert.pem (certificate)" -ForegroundColor White
    Write-Host "      - key.pem (private key)" -ForegroundColor White
    Write-Host "   3. Uncomment HTTPS section in nginx.conf" -ForegroundColor White
    Write-Host "   4. Restart nginx: docker-compose -f $composeFile restart nginx" -ForegroundColor White
}

Write-Host ""
Write-Success "Windows deployment completed successfully! 🎉"

# Display final logs
Write-Host ""
Write-Host "📝 Recent application logs:" -ForegroundColor Blue
try {
    docker-compose -f $composeFile logs --tail=10 isic-mapper
}
catch {
    Write-Warning "Could not retrieve application logs"
}

Write-Host ""
Write-Info "For troubleshooting, check the full logs with: docker-compose -f $composeFile logs -f"