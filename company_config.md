# 🏢 Company Configuration Guide

## Customizing the Service for Your Company

This guide helps you configure the Whisper Diarization Service specifically for your company's needs.

## 🎯 Company Use Cases

### 1. **Call Center Operations**
- **Configuration**: High accuracy, real-time processing
- **Model**: `large-v3` for best quality
- **Languages**: English + your target languages
- **Output**: JSON with speaker identification

### 2. **Meeting Recordings**
- **Configuration**: Balanced speed/accuracy
- **Model**: `medium.en` (default)
- **Languages**: Multi-language support
- **Output**: SRT format for video editing

### 3. **Legal Proceedings**
- **Configuration**: Maximum accuracy, compliance
- **Model**: `large-v3` with word timestamps
- **Languages**: Primary language only
- **Output**: Detailed JSON with confidence scores

### 4. **Podcast Production**
- **Configuration**: Good quality, reasonable speed
- **Model**: `small.en` or `medium.en`
- **Languages**: English (or your language)
- **Output**: Multiple formats (JSON, SRT, TXT)

## 🔧 Configuration Customization

### Environment Variables (`config.env`)

```bash
# Company-specific settings
COMPANY_NAME="Your Company Name"
COMPANY_DOMAIN="yourcompany.com"
COMPANY_LOGO_URL="https://yourcompany.com/logo.png"

# API Security
API_KEYS=["company-api-key-1", "company-api-key-2"]
ALLOWED_ORIGINS=["https://yourcompany.com", "https://app.yourcompany.com"]

# Processing Limits
MAX_FILE_SIZE=1048576000  # 1GB for long recordings
MAX_AUDIO_DURATION=7200   # 2 hours max

# Quality Settings
WHISPER_MODEL=large-v3    # Best accuracy for legal/important content
WHISPER_BATCH_SIZE=8      # Smaller batches for stability
```

### API Customization

#### 1. **Add Company Branding**
```python
# In app/main.py
app = FastAPI(
    title=f"{settings.COMPANY_NAME} Transcription Service",
    description="Professional audio transcription with speaker identification",
    version="1.0.0",
    contact={
        "name": f"{settings.COMPANY_NAME} Support",
        "email": "support@{settings.COMPANY_DOMAIN}",
        "url": f"https://{settings.COMPANY_DOMAIN}"
    }
)
```

#### 2. **Custom Response Headers**
```python
# Add company headers to all responses
@app.middleware("http")
async def add_company_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Company"] = settings.COMPANY_NAME
    response.headers["X-Service"] = "Transcription API"
    return response
```

## 🚀 Deployment Options

### Option 1: **Internal Company Server**
```bash
# On your company server
git clone <your-repo>
cd asr_whispher
./setup.sh
docker-compose up -d

# Access at: http://your-server:8000
```

### Option 2: **Cloud Deployment (AWS/GCP/Azure)**
```bash
# Use provided docker-compose.yml
# Deploy to your cloud provider
# Set up load balancer and SSL
```

### Option 3: **Kubernetes Cluster**
```bash
# Use k8s/ directory for deployment
kubectl apply -f k8s/
```

## 🔒 Security & Compliance

### 1. **API Key Management**
```bash
# Generate secure API keys
openssl rand -hex 32

# Set in config.env
API_KEYS=["generated-key-1", "generated-key-2"]
```

### 2. **CORS Configuration**
```bash
# Restrict to company domains only
ALLOWED_ORIGINS=[
    "https://yourcompany.com",
    "https://app.yourcompany.com",
    "https://internal.yourcompany.com"
]
```

### 3. **Data Retention**
```bash
# Set retention policies
DATA_RETENTION_DAYS=90
AUTO_DELETE_ENABLED=true
```

## 📊 Monitoring & Analytics

### 1. **Company Dashboard**
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Custom metrics**: Company-specific KPIs

### 2. **Business Metrics**
```python
# Track company usage
COMPANY_METRICS = {
    "total_transcriptions": 0,
    "total_audio_hours": 0.0,
    "average_processing_time": 0.0,
    "success_rate": 100.0
}
```

### 3. **Alerting**
```yaml
# prometheus.yml
alerts:
  - alert: HighErrorRate
    expr: rate(transcription_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High transcription error rate"
      description: "Error rate is {{ $value }}"
```

## 🔄 Integration with Company Systems

### 1. **Webhook Notifications**
```python
# Send completion notifications
WEBHOOK_URLS = [
    "https://yourcompany.com/api/transcriptions",
    "https://slack.yourcompany.com/webhook"
]
```

### 2. **Database Integration**
```python
# Connect to company database
COMPANY_DB_URL = "postgresql://user:pass@db.yourcompany.com/transcriptions"
COMPANY_DB_SCHEMA = "company_transcriptions"
```

### 3. **File Storage**
```python
# Use company storage
COMPANY_STORAGE = {
    "type": "s3",  # or "gcs", "azure"
    "bucket": "yourcompany-transcriptions",
    "region": "us-east-1"
}
```

## 📋 Company Workflow Examples

### Example 1: **Daily Call Center Processing**
```bash
# 1. Upload call recordings
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "X-API-Key: company-api-key" \
  -F "file=@daily_calls.zip" \
  -F "language=en" \
  -F "enable_diarization=true"

# 2. Monitor progress
curl "http://localhost:8000/api/v1/transcribe/{id}"

# 3. Download results
curl "http://localhost:8000/api/v1/transcribe/{id}/download?format=json"
```

### Example 2: **Batch Meeting Processing**
```bash
# Process multiple files
python cli.py transcribe meeting1.mp3 --output meeting1.json
python cli.py transcribe meeting2.mp3 --output meeting2.json
python cli.py transcribe meeting3.mp3 --output meeting3.json
```

## 🎨 Customization Checklist

- [ ] **Company branding** in API responses
- [ ] **API keys** configured and secured
- [ ] **CORS** restricted to company domains
- [ ] **File limits** adjusted for company needs
- [ ] **Quality settings** optimized for use case
- [ ] **Monitoring** set up with company metrics
- [ ] **Integration** with existing company systems
- [ ] **Documentation** updated for company users
- [ ] **Training** provided to company staff
- [ ] **Support** process established

## 🆘 Company Support

### Internal Support Team
- **Technical Lead**: [Your Name]
- **API Support**: [Support Email]
- **Documentation**: [Wiki/Confluence Link]

### Escalation Process
1. **Level 1**: Check logs and basic troubleshooting
2. **Level 2**: Review configuration and system resources
3. **Level 3**: Contact development team or vendor support

---

**🎯 Ready to deploy!** Your company now has a customized, production-ready transcription service that can handle your specific business needs with proper security, monitoring, and integration capabilities.
