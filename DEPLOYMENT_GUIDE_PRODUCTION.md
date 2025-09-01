# ISIC Code Mapper - Production Deployment Guide

Comprehensive guide for deploying the ISIC Code Mapper application to production servers using Docker.

## 🚀 Quick Start

### One-Line Deployment
```bash
curl -fsSL https://raw.githubusercontent.com/your-repo/UBOS-ISIC-CODE-MAPPER/main/deploy.sh | bash
```

### Standard Deployment
```bash
git clone <repository-url>
cd UBOS-ISIC-CODE-MAPPER
chmod +x deploy.sh
./deploy.sh your-domain.com true  # true for production mode
```

## 📋 Prerequisites

### Server Requirements
- **OS**: Ubuntu 20.04+, Debian 11+, CentOS 8+, or RHEL 8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 20GB available disk space
- **Network**: Public IP address (for web deployment)
- **Access**: sudo privileges

### Required Data Files
Ensure these files are in your project directory:
- `Localised ISIC.xlsx` - ISIC classification data
- `isco_index.xlsx` - ISCO classification data (optional)
- `isic_classifier_final_*/` - Fine-tuned model directory (optional)

## 🐳 Docker Deployment Options

### Option 1: Development Mode
```bash
# Simple deployment with direct access
./deploy.sh localhost false
# Access: http://your-server-ip:8501
```

### Option 2: Production Mode with Nginx
```bash
# Production deployment with reverse proxy
./deploy.sh your-domain.com true
# Access: http://your-domain.com
```

## 🔧 Manual Installation

### Step 1: Install Docker & Docker Compose

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login to apply docker group changes
```

**CentOS/RHEL:**
```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

### Step 2: Deploy Application

**Development Deployment:**
```bash
# Create directories
mkdir -p outputs ssl logs logs/nginx

# Start services
docker-compose up -d

# View status
docker-compose ps
```

**Production Deployment:**
```bash
# Use production configuration
docker-compose -f docker-compose.prod.yml up -d
```

## ⚙️ Configuration

### Environment Configuration

Create `.env` file for custom settings:
```bash
# Server Configuration
DOMAIN=your-domain.com
PRODUCTION_MODE=true

# Application Settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200

# Resource Limits
CPU_LIMIT=2.0
MEMORY_LIMIT=4G
```

### Nginx Configuration

**Basic HTTP Setup:**
```nginx
# In nginx.conf, update server_name
server_name your-domain.com;
```

**HTTPS Setup (Production):**
1. Obtain SSL certificates:
   ```bash
   # Using Let's Encrypt (recommended)
   sudo apt install certbot
   sudo certbot certonly --standalone -d your-domain.com
   
   # Copy certificates
   sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
   sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
   sudo chown $USER:$USER ssl/*.pem
   ```

2. Enable HTTPS in nginx.conf:
   ```bash
   # Uncomment HTTPS server block
   sed -i 's/^    # server {/    server {/g' nginx.conf
   sed -i 's/^    # }/    }/g' nginx.conf
   # ... (uncomment other HTTPS lines)
   ```

3. Restart nginx:
   ```bash
   docker-compose restart nginx
   ```

### Resource Optimization

**For High-Traffic Deployments:**
```yaml
# In docker-compose.prod.yml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 8G
    reservations:
      cpus: '2.0'
      memory: 4G
  replicas: 2  # For load balancing
```

**For Large File Processing:**
```nginx
# In nginx.conf
client_max_body_size 500M;
client_body_timeout 300s;
proxy_read_timeout 600s;
```

## 📊 Monitoring & Management

### Service Management
```bash
# View service status
docker-compose ps

# View logs
docker-compose logs -f                    # All services
docker-compose logs -f isic-mapper       # Application only
docker-compose logs -f nginx             # Nginx only

# Restart services
docker-compose restart                    # All services
docker-compose restart isic-mapper       # Application only

# Update application
git pull
docker-compose build --no-cache isic-mapper
docker-compose up -d
```

### Health Monitoring
```bash
# Check application health
curl -f http://localhost/health

# Monitor resource usage
docker stats

# View detailed container info
docker-compose exec isic-mapper top
```

### Log Rotation
```bash
# Configure logrotate for Docker logs
sudo tee /etc/logrotate.d/docker-compose << EOF
/var/lib/docker/containers/*/*.log {
    daily
    rotate 30
    compress
    size 100M
    missingok
    delaycompress
    copytruncate
}
EOF
```

## 🔒 Security Configuration

### Firewall Setup
```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 22      # SSH
sudo ufw allow 80      # HTTP
sudo ufw allow 443     # HTTPS
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --add-service=http --permanent
sudo firewall-cmd --add-service=https --permanent
sudo firewall-cmd --add-service=ssh --permanent
sudo firewall-cmd --reload
```

### Security Headers
Already configured in nginx.conf:
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- X-XSS-Protection: enabled
- Referrer-Policy: strict-origin-when-cross-origin

### Rate Limiting
Configured in nginx.conf:
- API requests: 30 requests/minute per IP
- File uploads: 5 requests/minute per IP
- Connection limit: 10 concurrent connections per IP

## 🔄 Backup & Recovery

### Automated Backup Script
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups/isic-mapper"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup user outputs
tar -czf $BACKUP_DIR/outputs_$DATE.tar.gz outputs/

# Backup configuration
tar -czf $BACKUP_DIR/config_$DATE.tar.gz \
    docker-compose*.yml nginx.conf ssl/ .env

# Backup application data
docker-compose exec -T isic-mapper tar -czf - /app/outputs | \
    gzip > $BACKUP_DIR/container_outputs_$DATE.tar.gz

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

### Recovery Process
```bash
# Stop services
docker-compose down

# Restore outputs
tar -xzf outputs_backup.tar.gz

# Restore configuration
tar -xzf config_backup.tar.gz

# Restart services
docker-compose up -d
```

## 🚨 Troubleshooting

### Common Issues

**1. Application Won't Start**
```bash
# Check logs
docker-compose logs isic-mapper

# Check system resources
free -h
df -h

# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**2. High Memory Usage**
```bash
# Monitor memory usage
docker stats isic-mapper

# Restart application to clear memory
docker-compose restart isic-mapper

# Reduce batch size in application settings
```

**3. File Upload Issues**
```bash
# Check nginx upload limits
docker-compose exec nginx nginx -T | grep client_max_body_size

# Check disk space
df -h outputs/

# Verify permissions
ls -la outputs/
```

**4. SSL Certificate Problems**
```bash
# Verify certificate files
openssl x509 -in ssl/cert.pem -text -noout
openssl rsa -in ssl/key.pem -check

# Test SSL configuration
openssl s_client -connect your-domain.com:443 -servername your-domain.com
```

### Performance Optimization

**1. Database Optimization (for large datasets)**
```bash
# Add volume for persistent vector database
# In docker-compose.yml:
volumes:
  - vector_db:/app/vector_db

# Use external Redis for caching (optional)
```

**2. Load Balancing (high traffic)**
```yaml
# docker-compose.prod.yml
services:
  isic-mapper:
    deploy:
      replicas: 3
  
  nginx:
    # Configure upstream load balancing
    volumes:
      - ./nginx-lb.conf:/etc/nginx/nginx.conf:ro
```

## 📈 Scaling for Production

### Horizontal Scaling
```bash
# Scale application instances
docker-compose up -d --scale isic-mapper=3

# Configure nginx load balancing
# Update upstream block in nginx.conf:
upstream streamlit {
    server isic-mapper_1:8501;
    server isic-mapper_2:8501;
    server isic-mapper_3:8501;
}
```

### Vertical Scaling
```bash
# Increase resource limits
# In docker-compose.prod.yml:
deploy:
  resources:
    limits:
      cpus: '8.0'
      memory: 16G
```

## 📞 Support & Maintenance

### Regular Maintenance Tasks

**Weekly:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Check disk usage
df -h

# Review logs for errors
docker-compose logs --since 7d | grep -i error
```

**Monthly:**
```bash
# Update Docker images
docker-compose pull
docker-compose up -d

# Clean unused Docker resources
docker system prune -f

# Backup configuration and data
./backup.sh
```

**SSL Certificate Renewal (Let's Encrypt):**
```bash
# Renew certificates (automated with cron)
sudo certbot renew
sudo systemctl reload nginx

# Or with Docker:
docker-compose restart nginx
```

### Getting Help

1. **Check application logs**: `docker-compose logs isic-mapper`
2. **Review system resources**: `top`, `free -h`, `df -h`
3. **Test connectivity**: `curl -I http://your-domain.com/health`
4. **Verify configuration**: `docker-compose config`

### Contact Information

- **Technical Issues**: Create an issue in the project repository
- **Security Issues**: Contact the security team privately
- **General Questions**: Check the main README.md file

---

**Last Updated**: September 2024  
**Version**: 2.0  
**Compatibility**: Docker 20.10+, Docker Compose 2.0+