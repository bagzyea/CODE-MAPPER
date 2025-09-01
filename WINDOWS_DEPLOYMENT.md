# Windows Server Deployment Guide
## Industry Classification Code Mapper

Complete guide for deploying the Industry Classification Code Mapper on Windows Server 2019/2022 using Docker.

## 🎯 Quick Start

### One-Command Deployment
```powershell
# Open PowerShell as Administrator and run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\deploy-windows.ps1 -InstallDocker -Domain "your-domain.com" -Production
```

### Standard Deployment (Docker already installed)
```powershell
.\deploy-windows.ps1 -Domain "localhost"
```

## 📋 Prerequisites

### System Requirements
- **OS**: Windows Server 2019/2022 or Windows 10/11 Pro
- **RAM**: 8GB minimum (16GB recommended for production)
- **Storage**: 50GB available disk space
- **Network**: Static IP address (recommended for server deployment)
- **PowerShell**: Version 5.1 or later

### Required Software
- **Docker Desktop** or **Docker Engine** (script can install automatically)
- **PowerShell** (pre-installed on Windows Server)
- **Windows Features**: Hyper-V and Containers (enabled automatically by Docker Desktop)

## 🚀 Installation Methods

### Method 1: Automated Installation (Recommended)

**Step 1: Download and Prepare**
```powershell
# Clone repository (if not already done)
git clone https://github.com/YOUR_USERNAME/CODE-MAPPER.git
cd CODE-MAPPER

# Set PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Step 2: Run Deployment Script**
```powershell
# For development/testing (localhost access)
.\deploy-windows.ps1

# For production with domain
.\deploy-windows.ps1 -Domain "your-domain.com" -Production

# With automatic Docker installation (requires Admin)
.\deploy-windows.ps1 -InstallDocker -Domain "your-domain.com" -Production
```

### Method 2: Manual Installation

**Step 1: Install Docker Desktop**
1. Download Docker Desktop from https://www.docker.com/products/docker-desktop
2. Run installer as Administrator
3. Restart computer when prompted
4. Start Docker Desktop and wait for it to be ready

**Step 2: Verify Docker Installation**
```powershell
# Check Docker version
docker --version
docker-compose --version

# Test Docker is running
docker run hello-world
```

**Step 3: Deploy Application**
```powershell
# Create required directories
New-Item -ItemType Directory -Path "outputs","ssl","logs","logs\nginx" -Force

# Build and start services
docker-compose -f docker-compose.windows.yml build
docker-compose -f docker-compose.windows.yml up -d
```

## ⚙️ Configuration

### Domain Configuration

**For Custom Domain:**
```powershell
# Update nginx.conf automatically
.\deploy-windows.ps1 -Domain "myapp.company.com"

# Or manually edit nginx.conf
# Change: server_name localhost;
# To:     server_name myapp.company.com;
```

### SSL/HTTPS Setup

**1. Obtain SSL Certificate:**
```powershell
# Option A: Use Let's Encrypt with Win-ACME
# Download from: https://www.win-acme.com/
# Follow setup instructions for your domain

# Option B: Use existing certificate
# Place your certificate files in the ssl\ directory
```

**2. Configure SSL Files:**
```
ssl\
├── cert.pem        # SSL certificate
└── key.pem         # Private key
```

**3. Enable HTTPS:**
```powershell
# Edit nginx.conf - uncomment HTTPS server block
# Restart nginx
docker-compose -f docker-compose.windows.yml restart nginx
```

### Windows Firewall Configuration

**Automatic (if running as Administrator):**
```powershell
.\deploy-windows.ps1  # Firewall rules created automatically
```

**Manual Configuration:**
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Create inbound rules for ports:
   - Port 80 (HTTP)
   - Port 443 (HTTPS)
   - Port 8501 (Direct app access)

### Resource Configuration

**Memory Limits (docker-compose.windows.yml):**
```yaml
deploy:
  resources:
    limits:
      memory: 4G      # Adjust based on server capacity
      cpus: '2.0'     # Adjust based on server cores
```

## 🔧 Windows-Specific Features

### Windows Service Integration

**Create Windows Service (Optional):**
```powershell
# Install NSSM (Non-Sucking Service Manager)
# Download from: https://nssm.cc/download

# Create service
nssm install "Code-Mapper" "C:\Program Files\Docker\Docker\resources\bin\docker-compose.exe"
nssm set "Code-Mapper" AppParameters "-f docker-compose.windows.yml up"
nssm set "Code-Mapper" AppDirectory "C:\path\to\CODE-MAPPER"
nssm set "Code-Mapper" Description "Industry Classification Code Mapper"

# Start service
nssm start "Code-Mapper"
```

### Windows Authentication Integration

**For enterprise environments with Active Directory:**

1. **Configure IIS as reverse proxy (alternative to nginx):**
```xml
<!-- web.config for IIS -->
<system.webServer>
  <rewrite>
    <rules>
      <rule name="ReverseProxyInboundRule1" stopProcessing="true">
        <match url="(.*)" />
        <action type="Rewrite" url="http://localhost:8501/{R:1}" />
      </rule>
    </rules>
  </rewrite>
</system.webServer>
```

2. **Enable Windows Authentication in IIS**
3. **Configure authentication forwarding**

### Performance Optimization for Windows

**Docker Desktop Settings:**
1. Open Docker Desktop Settings
2. Go to Resources → Advanced
3. Increase Memory to 8GB+
4. Increase CPU cores to 4+
5. Set disk image location to fastest drive (SSD)

**Windows Server Optimization:**
```powershell
# Disable Windows Search (if not needed)
Set-Service -Name "WSearch" -StartupType Disabled
Stop-Service -Name "WSearch"

# Configure Windows Update for manual
# Set active hours to avoid automatic restarts

# Optimize virtual memory
# Set paging file to system-managed or fixed size
```

## 📊 Monitoring & Management

### Windows Event Logs

**View Application Events:**
```powershell
# Get Docker-related events
Get-EventLog -LogName Application -Source Docker -Newest 50

# Get system events
Get-EventLog -LogName System -Newest 50 | Where-Object {$_.Source -like "*Docker*"}
```

### Performance Monitoring

**Built-in Windows Tools:**
```powershell
# Resource usage
Get-Counter "\Processor(_Total)\% Processor Time"
Get-Counter "\Memory\Available MBytes"

# Docker container stats
docker stats

# Disk usage
Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, Size, FreeSpace
```

**Task Scheduler for Monitoring:**
```powershell
# Create scheduled task for health checks
$action = New-ScheduledTaskAction -Execute "curl" -Argument "http://localhost:8501/_stcore/health"
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "CodeMapperHealthCheck" -Action $action -Trigger $trigger
```

### Log Management

**PowerShell Log Rotation:**
```powershell
# Create log rotation script
$logRotationScript = @"
Get-ChildItem -Path ".\logs" -Recurse -File | Where-Object {
    $_.LastWriteTime -lt (Get-Date).AddDays(-30)
} | Remove-Item -Force
"@

# Save and schedule it
$logRotationScript | Out-File -FilePath "log-rotation.ps1"
# Add to Task Scheduler
```

## 🔄 Backup & Recovery

### Automated Backup Script

```powershell
# backup-windows.ps1
param(
    [string]$BackupPath = "C:\Backups\CodeMapper"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupFolder = Join-Path $BackupPath $timestamp

# Create backup directory
New-Item -ItemType Directory -Path $backupFolder -Force

# Backup outputs
Copy-Item -Path ".\outputs" -Destination $backupFolder -Recurse

# Backup configuration
$configFiles = @("docker-compose*.yml", "nginx.conf", ".env", "ssl\*")
foreach ($pattern in $configFiles) {
    Copy-Item -Path $pattern -Destination $backupFolder -Recurse -ErrorAction SilentlyContinue
}

# Backup Docker volumes
docker run --rm -v code-mapper_outputs:/data -v ${backupFolder}:/backup alpine tar czf /backup/docker-volumes.tar.gz /data

# Cleanup old backups (keep 30 days)
Get-ChildItem -Path $BackupPath -Directory | Where-Object {
    $_.CreationTime -lt (Get-Date).AddDays(-30)
} | Remove-Item -Recurse -Force

Write-Host "Backup completed: $backupFolder"
```

**Schedule Backup:**
```powershell
# Create scheduled task for daily backup
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File .\backup-windows.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At "02:00"
Register-ScheduledTask -TaskName "CodeMapperBackup" -Action $action -Trigger $trigger -User "System"
```

## 🚨 Troubleshooting

### Common Windows Issues

**1. Docker Desktop Not Starting**
```powershell
# Check Windows features
Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All, Containers-DisposableClientVM

# Enable if needed
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All, Containers-DisposableClientVM -All

# Restart computer
Restart-Computer
```

**2. Port Conflicts**
```powershell
# Check what's using a port
netstat -ano | findstr :8501
netstat -ano | findstr :80

# Kill process if needed
taskkill /PID <process_id> /F
```

**3. Permissions Issues**
```powershell
# Fix Docker permissions
# Run PowerShell as Administrator
Add-LocalGroupMember -Group "docker-users" -Member $env:USERNAME

# Log out and back in
```

**4. Memory Issues**
```powershell
# Check memory usage
Get-Process | Sort-Object WorkingSet -Descending | Select-Object -First 10

# Increase Docker memory limit in Docker Desktop settings
# Or reduce container memory limits in docker-compose file
```

**5. Slow Performance**
```powershell
# Check disk performance
Get-Counter "\PhysicalDisk(*)\Avg. Disk Queue Length"

# Check if Windows Defender is scanning Docker files
# Add Docker installation directory to exclusions
Add-MpPreference -ExclusionPath "C:\Program Files\Docker"
Add-MpPreference -ExclusionPath "C:\ProgramData\DockerDesktop"
```

### Application-Specific Issues

**File Upload Problems:**
```powershell
# Check disk space
Get-WmiObject -Class Win32_LogicalDisk

# Check folder permissions
Get-Acl .\outputs

# Fix permissions
icacls .\outputs /grant Everyone:F /T
```

**Model Loading Issues:**
```powershell
# Check if model files exist
Test-Path ".\isic_classifier_final_*"

# Check container logs
docker-compose -f docker-compose.windows.yml logs isic-mapper
```

## 🔐 Security Considerations

### Windows Server Security

**Basic Security Setup:**
```powershell
# Enable Windows Firewall
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True

# Disable unnecessary services
$servicesToDisable = @("Fax", "XblAuthManager", "XblGameSave")
foreach ($service in $servicesToDisable) {
    Set-Service -Name $service -StartupType Disabled -ErrorAction SilentlyContinue
}

# Configure automatic updates
# Use Group Policy or Windows Update for Business
```

**Container Security:**
```powershell
# Run security scan on images
docker scan code-mapper_isic-mapper

# Update base images regularly
docker-compose -f docker-compose.windows.yml pull
docker-compose -f docker-compose.windows.yml build --no-cache
```

### Network Security

**Restrict Network Access:**
```powershell
# Create firewall rule for specific IPs only
New-NetFirewallRule -DisplayName "Code Mapper - Specific IPs" -Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow -RemoteAddress "192.168.1.0/24"
```

## 📈 Scaling on Windows

### Multiple Instances

```powershell
# Scale application instances
docker-compose -f docker-compose.windows.yml up -d --scale isic-mapper=3

# Use nginx for load balancing
# Update nginx.conf with multiple upstream servers
```

### Windows Container Scaling

```powershell
# Use Docker Swarm mode
docker swarm init
docker stack deploy -c docker-compose.windows.yml code-mapper-stack
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks

**Weekly:**
```powershell
# Update Windows
Install-Module PSWindowsUpdate
Get-WindowsUpdate -Install -AutoReboot

# Clean Docker
docker system prune -f

# Check logs for errors
docker-compose -f docker-compose.windows.yml logs --since 7d | Select-String "ERROR"
```

**Monthly:**
```powershell
# Update Docker images
docker-compose -f docker-compose.windows.yml pull
docker-compose -f docker-compose.windows.yml up -d

# Review Windows Event Logs
Get-EventLog -LogName Application -EntryType Error -Newest 100

# Update SSL certificates (if using Let's Encrypt)
# Run Win-ACME renewal task
```

### Getting Help

**Logs to Check:**
1. **Application logs**: `docker-compose logs isic-mapper`
2. **Windows Event Logs**: Event Viewer → Windows Logs → Application
3. **Docker logs**: Docker Desktop → Troubleshoot → Get support
4. **IIS logs** (if using): `%SystemDrive%\inetpub\logs\LogFiles\W3SVC1\`

**Diagnostic Commands:**
```powershell
# System health check
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, TotalPhysicalMemory

# Docker health check
docker version
docker-compose -f docker-compose.windows.yml config

# Network connectivity
Test-NetConnection -ComputerName localhost -Port 8501
Test-NetConnection -ComputerName your-domain.com -Port 80
```

---

**Last Updated**: September 2024  
**Compatibility**: Windows Server 2019/2022, Windows 10/11 Pro  
**Docker**: Desktop 4.0+, Engine 20.10+