# ISIC Code Mapper - Local Server Deployment Guide

## Prerequisites

### Server Requirements
- **OS**: Ubuntu 20.04+ or CentOS 8+ (recommended)
- **RAM**: Minimum 4GB (8GB+ recommended for multiple users)
- **Storage**: 20GB+ available space
- **Network**: Static IP address on your office network

### Software Requirements
- Docker & Docker Compose (will be installed by script)
- Open ports: 80 (HTTP), 8501 (Streamlit), 443 (HTTPS optional)

## Quick Deployment

### 1. Copy Files to Server
```bash
# Copy entire project to your server
scp -r UBOS-ISIC-CODE-MAPPER/ user@server-ip:/home/user/
```

### 2. Run Deployment Script
```bash
cd UBOS-ISIC-CODE-MAPPER/
chmod +x deploy.sh
./deploy.sh
```

### 3. Access Application
- **Direct access**: `http://YOUR-SERVER-IP:8501`
- **Through nginx**: `http://YOUR-SERVER-IP`

## Manual Deployment Steps

If you prefer manual setup:

### 1. Install Docker
```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker $USER
```

### 2. Start Services
```bash
# Simple deployment (Streamlit only)
docker-compose up -d isic-mapper

# Full deployment (with nginx)
docker-compose up -d
```

## Configuration Options

### Environment Variables
Create `.env` file:
```bash
# Server configuration
SERVER_PORT=8501
SERVER_ADDRESS=0.0.0.0

# File upload limits
MAX_UPLOAD_SIZE=100MB

# Email settings (optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your-email@company.com
EMAIL_PASSWORD=your-app-password
```

### Nginx Configuration
Edit `nginx.conf` to:
- Change server name to your domain/IP
- Enable HTTPS (uncomment SSL section)
- Adjust file upload limits

### SSL/HTTPS Setup (Optional)
```bash
# Generate self-signed certificate
mkdir ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem -out ssl/cert.pem

# Update nginx.conf to enable HTTPS section
# Restart: docker-compose restart nginx
```

## User Management & File Access

### Shared File Directory
- All user outputs saved to: `./outputs/`
- Each user gets a folder: `./outputs/username/`
- Files persist after container restarts

### Backup Strategy
```bash
# Backup user data
tar -czf isic-backup-$(date +%Y%m%d).tar.gz outputs/

# Restore from backup
tar -xzf isic-backup-YYYYMMDD.tar.gz
```

## Monitoring & Maintenance

### Check Application Status
```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f isic-mapper

# Check resource usage
docker stats
```

### Common Commands
```bash
# Restart application
docker-compose restart

# Update application
git pull
docker-compose build --no-cache
docker-compose up -d

# Stop all services
docker-compose down

# Clean up old images
docker system prune -a
```

## Network Access Setup

### Firewall Configuration
```bash
# Ubuntu/Debian
sudo ufw allow 80/tcp
sudo ufw allow 8501/tcp
sudo ufw allow 443/tcp  # if using HTTPS

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=8501/tcp
sudo firewall-cmd --reload
```

### Router Configuration
- Ensure server has static IP
- Configure port forwarding if needed for external access
- Consider VPN access for security

## Security Recommendations

1. **Firewall**: Only allow necessary ports
2. **Updates**: Keep Docker and OS updated
3. **Backups**: Regular automated backups
4. **Monitoring**: Set up log monitoring
5. **Access Control**: Use nginx auth or VPN

## Troubleshooting

### Common Issues

**Port Already in Use**
```bash
sudo lsof -i :8501
sudo kill -9 PID
```

**Permission Denied**
```bash
sudo chown -R $(whoami):$(whoami) outputs/
sudo chmod -R 755 outputs/
```

**Container Won't Start**
```bash
docker-compose logs isic-mapper
docker-compose down && docker-compose up -d
```

**Out of Memory**
```bash
# Check memory usage
free -h
docker stats

# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## Performance Optimization

### For Multiple Users
- Increase server RAM to 8GB+
- Use SSD storage
- Configure nginx caching
- Set up load balancing if needed

### Batch Processing Optimization
```bash
# In docker-compose.yml, add environment variables:
environment:
  - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
  - PYTHONPATH=/app
```

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify network connectivity
3. Check file permissions
4. Review firewall settings
5. Monitor system resources

---

**Last Updated**: $(date)
**Version**: 1.0