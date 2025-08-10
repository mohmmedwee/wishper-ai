# 🚀 Quick Start Guide - Whisper Diarization Service

## What This Service Does

This is a **production-ready AI service** that combines:
- **Speech Recognition**: Converts audio to text using OpenAI's Whisper
- **Speaker Diarization**: Identifies who is speaking when
- **REST API**: Easy integration with your applications
- **CLI Tools**: Command-line usage for batch processing

## 🎯 Perfect For

- **Call Centers**: Transcribe customer calls with speaker identification
- **Meetings**: Convert meeting recordings to searchable text
- **Podcasts**: Generate transcripts with speaker labels
- **Interviews**: Document interviews with clear speaker attribution
- **Legal**: Transcribe legal proceedings with speaker identification

## ⚡ Quick Setup (Choose One Method)

### Method 1: Automated Setup (Recommended)
```bash
# Run the setup script
./setup.sh

# Copy configuration
cp config.env .env

# Start with Docker
docker-compose up -d
```

### Method 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p uploads outputs models

# Start service
python -m app.main
```

## 🧪 Test Your Setup

### 1. Check Service Health
```bash
curl http://localhost:8000/health
```

### 2. Test Transcription API
```bash
# Upload an audio file
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.mp3" \
  -F "language=en" \
  -F "enable_diarization=true"
```

### 3. Use CLI Tool
```bash
# Activate virtual environment
source venv/bin/activate

# Transcribe audio file
python cli.py transcribe your_audio.mp3 --language en --output result.json
```

## 📁 Project Structure

```
asr_whispher/
├── app/                    # Main application code
│   ├── api/               # REST API endpoints
│   ├── services/          # Core business logic
│   └── models/            # Data models
├── config/                 # Configuration files
├── uploads/                # Audio file uploads
├── outputs/                # Generated transcriptions
├── docker-compose.yml      # Docker setup
├── requirements.txt        # Python dependencies
└── cli.py                 # Command-line interface
```

## 🔧 Configuration

Edit `config.env` (or `.env`) to customize:
- **Whisper Model**: Choose accuracy vs. speed
- **API Keys**: Secure your endpoints
- **File Limits**: Adjust maximum file sizes
- **Languages**: Support multiple languages

## 🐳 Docker Deployment

### Start All Services
```bash
docker-compose up -d
```

### Services Included
- **Whisper Service**: Main transcription service
- **Redis**: Job queue and caching

### Stop Services
```bash
docker-compose down
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/api/v1/transcribe` | POST | Transcribe audio file |
| `/api/v1/transcribe/{id}` | GET | Get transcription status |
| `/api/v1/transcribe/{id}/download` | GET | Download results |

## 🎵 Supported Audio Formats

- **WAV** - Best quality, largest size
- **MP3** - Good balance, widely supported
- **M4A** - Apple devices, good quality
- **FLAC** - Lossless, high quality
- **OGG** - Open source, good compression

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use smaller Whisper model (tiny, base, small)
   - Reduce batch size in config

2. **Audio Format Issues**
   - Install FFmpeg: `brew install ffmpeg`
   - Check file format support

3. **Service Won't Start**
   - Check port availability
   - Verify Python version (3.10+)
   - Check dependency installation

### Get Help
- Check logs: `docker-compose logs whisper-diarization`
- Review configuration: `config.env`
- Check README.md for detailed documentation

## 🔒 Security & Production

### Before Going Live
1. **Set API Keys**: Edit `.env` file
2. **Restrict Origins**: Configure CORS settings
3. **Enable HTTPS**: Use reverse proxy (nginx)
4. **Monitor Resources**: Check service logs and performance
5. **Backup Data**: Regular backups of outputs

### Scaling
- **Horizontal**: Multiple service instances
- **Vertical**: Increase container resources
- **Queue**: Redis-based job processing

## 📈 Next Steps

1. **Customize**: Adjust configuration for your needs
2. **Integrate**: Connect to your existing systems
3. **Monitor**: Set up alerting and dashboards
4. **Scale**: Add more instances as needed
5. **Train**: Fine-tune models for your domain

## 🆘 Support

- **Documentation**: README.md
- **Issues**: Check logs and configuration
- **Community**: GitHub issues and discussions

---

**🎉 You're all set!** Your company now has a powerful AI-powered transcription service that can handle audio files with speaker identification. Start with small files to test, then scale up as needed.
