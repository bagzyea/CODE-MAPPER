#!/bin/bash

# ISIC Code Mapper Production Deployment Script
# Compatible with Ubuntu/Debian/CentOS servers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Configuration
DOMAIN="${1:-localhost}"
COMPOSE_FILE="docker-compose.yml"
PRODUCTION_MODE="${2:-false}"

if [ "$PRODUCTION_MODE" = "true" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
fi

log "🚀 Starting ISIC Code Mapper deployment..."
log "📋 Domain: $DOMAIN"
log "📋 Production Mode: $PRODUCTION_MODE"
log "📋 Compose File: $COMPOSE_FILE"

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    warn "Running as root. Consider using a non-root user for security."
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    error "Cannot detect OS version"
fi

log "📋 Detected OS: $OS $VER"

# Update system based on OS
log "📦 Updating system packages..."
if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    sudo apt update && sudo apt upgrade -y
elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]]; then
    sudo yum update -y
    sudo yum install -y epel-release
else
    warn "OS not fully supported, proceeding with caution..."
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    log "🐳 Installing Docker..."
    if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
        sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt update
        sudo apt install -y docker-ce docker-ce-cli containerd.io
    elif [[ "$OS" == *"CentOS"* ]]; then
        sudo yum install -y yum-utils
        sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        sudo yum install -y docker-ce docker-ce-cli containerd.io
        sudo systemctl start docker
        sudo systemctl enable docker
    fi
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    warn "Please log out and back in for Docker permissions to take effect"
    
    # Start Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
else
    log "✅ Docker already installed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    log "🔧 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Create symlink for docker compose (newer syntax)
    sudo ln -sf /usr/local/bin/docker-compose /usr/local/bin/docker
else
    log "✅ Docker Compose already installed"
fi

# Create necessary directories and set permissions
log "📁 Setting up directories..."
mkdir -p outputs ssl logs logs/nginx
chmod 755 outputs ssl logs logs/nginx

# Create logs directory structure
mkdir -p logs/app logs/nginx
touch logs/app/app.log logs/nginx/access.log logs/nginx/error.log

# Generate basic nginx config if domain is specified
if [ "$DOMAIN" != "localhost" ]; then
    log "🔧 Updating nginx configuration for domain: $DOMAIN"
    sed -i "s/server_name localhost;/server_name $DOMAIN;/g" nginx.conf
fi

# Check if required files exist
log "🔍 Checking required files..."
REQUIRED_FILES=("Dockerfile" "requirements.txt" "app.py" "$COMPOSE_FILE")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        error "Required file $file not found!"
    fi
done

# Create .dockerignore if it doesn't exist
if [ ! -f ".dockerignore" ]; then
    log "📝 Creating .dockerignore..."
    cat > .dockerignore << EOF
.git
.gitignore
README.md
Dockerfile
docker-compose*.yml
.dockerignore
logs/*
__pycache__
*.pyc
.env
.venv
EOF
fi

# Build and deploy
log "🏗️  Building and starting services..."
docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
docker-compose -f "$COMPOSE_FILE" build --no-cache

log "🚀 Starting services..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to start and check health
log "⏳ Waiting for services to start..."
sleep 30

# Check service status
log "🔍 Checking service status..."
docker-compose -f "$COMPOSE_FILE" ps

# Health check
log "🩺 Performing health checks..."
if docker-compose -f "$COMPOSE_FILE" exec -T isic-mapper curl -f http://localhost:8501/_stcore/health >/dev/null 2>&1; then
    log "✅ Application health check passed"
else
    warn "Application health check failed, but continuing..."
fi

# Show logs
log "📝 Recent application logs:"
docker-compose -f "$COMPOSE_FILE" logs --tail=10 isic-mapper

# Get server info
SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')
LOCAL_IP=$(hostname -I | awk '{print $1}')

# Display deployment info
log "✅ Deployment complete!"
echo ""
echo "🌐 Access your application at:"
if [ "$PRODUCTION_MODE" = "true" ]; then
    echo "   Production URL:  http://$DOMAIN (or http://$SERVER_IP)"
    echo "   Local access:    http://$LOCAL_IP"
else
    echo "   Direct access:   http://$SERVER_IP:8501"
    echo "   With nginx:      http://$SERVER_IP"
    echo "   Local access:    http://localhost:8501"
fi
echo ""
echo "📊 Useful commands:"
echo "   Monitor logs:    docker-compose -f $COMPOSE_FILE logs -f"
echo "   View status:     docker-compose -f $COMPOSE_FILE ps"
echo "   Restart app:     docker-compose -f $COMPOSE_FILE restart isic-mapper"
echo "   Stop services:   docker-compose -f $COMPOSE_FILE down"
echo "   Update app:      ./deploy.sh $DOMAIN $PRODUCTION_MODE"
echo ""
echo "📁 Data directories:"
echo "   User outputs:    $(pwd)/outputs/"
echo "   Application logs: $(pwd)/logs/"
echo ""

if [ "$PRODUCTION_MODE" = "true" ]; then
    log "🔐 For HTTPS in production:"
    echo "   1. Obtain SSL certificates (Let's Encrypt recommended)"
    echo "   2. Place cert.pem and key.pem in ./ssl/ directory"  
    echo "   3. Uncomment HTTPS section in nginx.conf"
    echo "   4. Restart nginx: docker-compose -f $COMPOSE_FILE restart nginx"
fi

log "🎉 Deployment completed successfully!"