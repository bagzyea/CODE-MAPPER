@echo off
REM Windows Batch Deployment Script for CODE-MAPPER
REM Industry Classification Code Mapper

setlocal enabledelayedexpansion

echo ===============================================
echo    Industry Classification Code Mapper
echo         Windows Deployment Script
echo ===============================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from:
    echo https://www.docker.com/products/docker-desktop
    echo.
    echo Or run the PowerShell script with -InstallDocker flag:
    echo powershell -ExecutionPolicy Bypass -File deploy-windows.ps1 -InstallDocker
    pause
    exit /b 1
)

echo ✅ Docker is installed and available

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker daemon is not running
    echo Please start Docker Desktop and wait for it to be ready
    pause
    exit /b 1
)

echo ✅ Docker daemon is running

REM Check required files
echo.
echo Checking required files...
set "missing_files="
if not exist "Dockerfile" set "missing_files=!missing_files! Dockerfile"
if not exist "requirements.txt" set "missing_files=!missing_files! requirements.txt"
if not exist "app.py" set "missing_files=!missing_files! app.py"
if not exist "docker-compose.windows.yml" set "missing_files=!missing_files! docker-compose.windows.yml"

if defined missing_files (
    echo ERROR: Missing required files:!missing_files!
    pause
    exit /b 1
)

echo ✅ All required files found

REM Create directories
echo.
echo Creating directories...
if not exist "outputs" mkdir outputs
if not exist "ssl" mkdir ssl
if not exist "logs" mkdir logs
if not exist "logs\nginx" mkdir "logs\nginx"
echo ✅ Directories created

REM Stop existing containers
echo.
echo Stopping existing containers...
docker-compose -f docker-compose.windows.yml down --remove-orphans >nul 2>&1

REM Build application
echo.
echo Building application container...
docker-compose -f docker-compose.windows.yml build --no-cache
if %errorlevel% neq 0 (
    echo ERROR: Failed to build application container
    pause
    exit /b 1
)

echo ✅ Application container built successfully

REM Start services
echo.
echo Starting services...
docker-compose -f docker-compose.windows.yml up -d
if %errorlevel% neq 0 (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)

echo ✅ Services started successfully

REM Wait for services
echo.
echo Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service status
echo.
echo Checking service status...
docker-compose -f docker-compose.windows.yml ps

REM Try health check
echo.
echo Performing health check...
curl -s -f http://localhost:8501/_stcore/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Application health check passed
) else (
    echo ⚠️  Application health check failed, but services may still be starting
)

REM Get local IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set "local_ip=%%a"
    set "local_ip=!local_ip: =!"
    goto :found_ip
)
:found_ip

echo.
echo ===============================================
echo           🎉 Deployment Complete!
echo ===============================================
echo.
echo 🌐 Access your application at:
echo    Direct Access:   http://localhost:8501
if defined local_ip (
    echo    Network Access:  http://!local_ip!:8501
)
echo    With Nginx:      http://localhost
echo.
echo 📊 Management Commands:
echo    View Status:     docker-compose -f docker-compose.windows.yml ps
echo    View Logs:       docker-compose -f docker-compose.windows.yml logs -f
echo    Restart App:     docker-compose -f docker-compose.windows.yml restart isic-mapper
echo    Stop Services:   docker-compose -f docker-compose.windows.yml down
echo.
echo 📁 Data Directories:
echo    User Outputs:    %CD%\outputs
echo    Application Logs: %CD%\logs
echo.
echo For advanced configuration and production deployment,
echo use the PowerShell script: deploy-windows.ps1
echo.
echo Press any key to view recent logs...
pause >nul

echo.
echo Recent application logs:
docker-compose -f docker-compose.windows.yml logs --tail=10 isic-mapper

echo.
echo Deployment completed successfully!
pause