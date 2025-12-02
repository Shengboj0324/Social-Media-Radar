# 🚀 Installation and Deployment Guide

**Social Media Radar - Industrial-Grade Production Deployment**

---

## Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, or Windows 10+ with WSL2
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space
- **CPU**: 4 cores minimum, 8 cores recommended

### Required Software
- **Poetry**: 1.7.1+ (Python dependency management)
- **Docker**: 20.10+ (for PostgreSQL, Redis)
- **Docker Compose**: 2.0+
- **FFmpeg**: 4.4+ (for video processing)
- **Tesseract OCR**: 5.0+ (for image text extraction)
- **Git**: 2.30+

---

## 📦 Step 1: Clone Repository

```bash
git clone https://github.com/your-org/social-media-radar.git
cd social-media-radar
```

---

## 🔧 Step 2: Install System Dependencies

### Ubuntu/Debian
```bash
# Update package list
sudo apt update

# Install FFmpeg
sudo apt install -y ffmpeg

# Install Tesseract OCR
sudo apt install -y tesseract-ocr tesseract-ocr-eng

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"
```

### macOS
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install FFmpeg
brew install ffmpeg

# Install Tesseract OCR
brew install tesseract

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

### Windows (WSL2)
```bash
# Install FFmpeg
sudo apt install -y ffmpeg

# Install Tesseract OCR
sudo apt install -y tesseract-ocr

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
```

---

## 🐍 Step 3: Install Python Dependencies

```bash
# Install all dependencies
poetry install

# Activate virtual environment
poetry shell

# Download spaCy language model
python -m spacy download en_core_web_lg
```

**Expected Output**:
```
Installing dependencies from lock file
...
✔ Download and installation successful
```

---

## 🐳 Step 4: Start Infrastructure Services

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Verify services are running
docker-compose ps
```

**Expected Output**:
```
NAME                    STATUS              PORTS
social-media-radar-postgres-1   Up 10 seconds       0.0.0.0:5432->5432/tcp
social-media-radar-redis-1      Up 10 seconds       0.0.0.0:6379->6379/tcp
```

---

## 🗄️ Step 5: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env  # or vim, code, etc.
```

**Required Environment Variables**:
```bash
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/social_media_radar
SYNC_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/social_media_radar

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here-change-this-in-production
ENCRYPTION_KEY=your-32-byte-base64-encoded-key-here

# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key-here

# Anthropic API (optional)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
```

**Generate Secure Keys**:
```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate ENCRYPTION_KEY
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

---

## 🗃️ Step 6: Initialize Database

```bash
# Run database migrations
poetry run alembic upgrade head

# Create initial user (optional)
poetry run python scripts/create_user.py
```

**Expected Output**:
```
INFO  [alembic.runtime.migration] Running upgrade -> abc123, Initial schema
INFO  [alembic.runtime.migration] Running upgrade abc123 -> def456, Add media tables
✅ Database initialized successfully
```

---

## 🚀 Step 7: Start Application

### Development Mode
```bash
# Terminal 1: Start API server
poetry run uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Celery worker
poetry run celery -A app.ingestion.celery_app worker --loglevel=info

# Terminal 3: Start Celery beat (scheduler)
poetry run celery -A app.ingestion.celery_app beat --loglevel=info
```

### Production Mode
```bash
# Start all services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f api worker
```

---

## ✅ Step 8: Verify Installation

### Health Check
```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"0.1.0"}
```

### API Documentation
Open browser to: http://localhost:8000/docs

### Run Tests
```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage
poetry run pytest --cov=app --cov-report=html tests/

# View coverage report
open htmlcov/index.html
```

---

## 🔐 Step 9: Security Hardening

### 1. Change Default Credentials
```bash
# Update .env file with strong passwords
# Never use default credentials in production
```

### 2. Enable HTTPS
```bash
# Install certbot
sudo apt install certbot

# Get SSL certificate
sudo certbot certonly --standalone -d your-domain.com

# Update nginx configuration
# See infra/nginx/nginx.conf for example
```

### 3. Configure Firewall
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### 4. Set Up Monitoring
```bash
# Start Prometheus and Grafana
docker-compose --profile monitoring up -d

# Access Grafana: http://localhost:3000
# Default credentials: admin/admin (change immediately)
```

---

## 📊 Step 10: Performance Optimization

### 1. Configure Redis Caching
```bash
# Edit .env
REDIS_MAX_CONNECTIONS=50
CACHE_TTL_SECONDS=3600
```

### 2. Optimize Database
```bash
# Create indexes
poetry run python scripts/optimize_db.py

# Configure connection pooling
# Edit .env:
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
```

### 3. Enable CDN (Optional)
```bash
# Configure S3/MinIO for media storage
# Edit .env:
MEDIA_STORAGE_BACKEND=s3
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_S3_BUCKET=your-bucket
```

---

## 🔄 Step 11: Backup and Recovery

### Database Backup
```bash
# Create backup
docker exec social-media-radar-postgres-1 pg_dump -U postgres social_media_radar > backup.sql

# Restore backup
docker exec -i social-media-radar-postgres-1 psql -U postgres social_media_radar < backup.sql
```

### Redis Backup
```bash
# Enable RDB snapshots in docker-compose.yml
# Redis will automatically save to /data/dump.rdb
```

---

## 🐛 Troubleshooting

### Issue: Poetry install fails
```bash
# Clear cache and retry
poetry cache clear pypi --all
poetry install
```

### Issue: Database connection fails
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check connection
psql postgresql://postgres:postgres@localhost:5432/social_media_radar
```

### Issue: Redis connection fails
```bash
# Check Redis is running
docker-compose ps redis

# Test connection
redis-cli ping
```

### Issue: FFmpeg not found
```bash
# Verify installation
ffmpeg -version

# Reinstall if needed
sudo apt install --reinstall ffmpeg
```

---

## 📚 Next Steps

1. **Configure Platform Connectors**: See `docs/PLATFORM_CONNECTORS.md`
2. **Set Up OAuth**: See `docs/USER_GUIDE.md`
3. **Deploy to Production**: See `docs/PRODUCTION_DEPLOYMENT.md`
4. **Monitor Performance**: See `docs/MONITORING.md`
5. **Scale Infrastructure**: See `infra/k8s/README.md`

---

## 🆘 Support

- **Documentation**: `docs/`
- **Issues**: GitHub Issues
- **Security**: security@your-domain.com

---

## ✅ Installation Complete!

Your Social Media Radar installation is complete and ready for production use! 🎉

