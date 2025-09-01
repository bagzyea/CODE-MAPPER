# Getting Started - Industry Classification Code Mapper

Quick start guide to get your application running in minutes.

## 🚀 Quick Start (Choose Your Platform)

### Windows Server/Desktop
```powershell
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CODE-MAPPER.git
cd CODE-MAPPER

# 2. Run deployment (PowerShell - Recommended)
.\deploy-windows.ps1

# OR run deployment (Batch file - Simple)
.\deploy-windows.bat
```

### Linux Server
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CODE-MAPPER.git
cd CODE-MAPPER

# 2. Run deployment
chmod +x deploy.sh
./deploy.sh
```

## 📋 What You Need

### Minimum Requirements
- **Windows**: Windows 10/11 Pro or Windows Server 2019/2022
- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+, or RHEL 8+
- **RAM**: 4GB (8GB recommended)
- **Storage**: 10GB free space
- **Docker**: Will be installed automatically if needed

### Data Files (Required)
Make sure these files are in your project directory:
- `Localised ISIC.xlsx` - ISIC classification data
- `isco_index.xlsx` - ISCO classification data (optional)

## 🎯 Access Your Application

After deployment, access your application at:
- **Development**: http://localhost:8501
- **With Nginx**: http://localhost or http://your-server-ip
- **Production**: http://your-domain.com

## 📱 How to Use

### 1. Upload Files
- Support for Excel (.xlsx) and CSV files
- Required column: `INDUSTRY` (for ISIC) or `OCCUPATION` (for ISCO)
- Optional: `DESCRIPTION` column for better accuracy

### 2. Choose Classification Type
- **ISIC**: Industry classification (manufacturing, retail, etc.)
- **ISCO**: Occupation classification (jobs, roles, etc.)

### 3. Select Model
- **Embedding Model**: Fast, good accuracy
- **Fine-tuned Model**: Best accuracy (if available)

### 4. Process Files
- Single file or batch processing
- Real-time progress tracking
- Download processed results

## 🛠️ Common Commands

### View Application Status
```bash
# Linux
docker-compose ps

# Windows
docker-compose -f docker-compose.windows.yml ps
```

### View Logs
```bash
# Linux
docker-compose logs -f

# Windows  
docker-compose -f docker-compose.windows.yml logs -f
```

### Restart Application
```bash
# Linux
docker-compose restart

# Windows
docker-compose -f docker-compose.windows.yml restart
```

### Stop Application
```bash
# Linux
docker-compose down

# Windows
docker-compose -f docker-compose.windows.yml down
```

## 📁 File Structure

```
CODE-MAPPER/
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Linux deployment
├── docker-compose.windows.yml  # Windows deployment
├── nginx.conf                  # Web server configuration
├── deploy.sh                   # Linux deployment script
├── deploy-windows.ps1          # Windows PowerShell deployment
├── deploy-windows.bat          # Windows batch deployment
├── outputs/                    # User-generated files
├── logs/                       # Application logs
└── ssl/                        # SSL certificates (for HTTPS)
```

## 🔧 Basic Configuration

### Change Port (if 8501 is in use)
Edit the compose file and change:
```yaml
ports:
  - "8502:8501"  # Change 8502 to your preferred port
```

### Enable HTTPS
1. Place SSL certificates in `ssl/` directory:
   - `cert.pem` (certificate)
   - `key.pem` (private key)
2. Uncomment HTTPS section in `nginx.conf`
3. Restart: `docker-compose restart nginx`

### Custom Domain
```bash
# Linux
./deploy.sh your-domain.com true

# Windows
.\deploy-windows.ps1 -Domain "your-domain.com" -Production
```

## 🆘 Need Help?

### Check Logs First
```bash
# Application logs
docker-compose logs isic-mapper

# All service logs
docker-compose logs -f
```

### Common Issues

**Port Already in Use:**
```bash
# Find what's using port 8501
netstat -tulpn | grep :8501    # Linux
netstat -ano | findstr :8501   # Windows
```

**Docker Not Running:**
- **Windows**: Start Docker Desktop
- **Linux**: `sudo systemctl start docker`

**Permission Denied:**
```bash
# Linux
sudo chown -R $USER:$USER outputs/ logs/

# Windows (as Administrator)
icacls outputs /grant Everyone:F /T
```

### Get More Help
- 📖 **Detailed Guides**:
  - [Windows Deployment Guide](WINDOWS_DEPLOYMENT.md)
  - [Production Deployment Guide](DEPLOYMENT_GUIDE_PRODUCTION.md)
- 🐛 **Report Issues**: Create an issue in the GitHub repository
- 💬 **Questions**: Check the main README.md for additional information

## 🎉 Next Steps

Once your application is running:
1. **Upload a test file** to verify functionality
2. **Configure your domain** for external access
3. **Set up SSL/HTTPS** for production use
4. **Configure backups** for user data
5. **Monitor performance** and logs

---

**Need production deployment?** See [DEPLOYMENT_GUIDE_PRODUCTION.md](DEPLOYMENT_GUIDE_PRODUCTION.md)  
**Deploying on Windows?** See [WINDOWS_DEPLOYMENT.md](WINDOWS_DEPLOYMENT.md)