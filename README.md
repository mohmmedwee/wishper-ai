# 🎯 Enhanced Whisper Diarization Service

A **production-ready AI service** that combines OpenAI's Whisper speech recognition with advanced speaker diarization and enhanced features.

## ✨ Features

### Core Capabilities
- **🎤 Speech Recognition**: High-quality transcription using OpenAI Whisper
- **👥 Speaker Diarization**: Identify who is speaking when using NeMo
- **🌐 RESTful API**: Easy integration with your applications
- **🖥️ CLI Tools**: Command-line interface for batch processing

### Enhanced Features (Production Ready)
- **🎵 Source Separation**: Audio enhancement using Demucs
- **⚡ Parallel Processing**: Faster results with concurrent execution
- **🎯 Enhanced Alignment**: Precise timestamp alignment with CTC forced aligner
- **🌍 Language Detection**: Automatic language identification
- **💬 Multilingual Punctuation**: Smart punctuation for multiple languages
- **🚀 GPU Acceleration**: CUDA support for faster processing

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** (for production with ML models)
- **Python 3.13** (for development/testing - limited features)
- **Docker & Docker Compose** (recommended for production)

### Development Setup (Python 3.13)
```bash
# Clone the repository
git clone https://github.com/mohmmedwee/wishper-ai.git
cd wishper-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install basic dependencies
pip install -r requirements.txt

# Start the service
./start.sh
```

### Production Setup (Python 3.10/3.11)
```bash
# Use production requirements for full ML capabilities
pip install -r requirements.production.txt

# Start with full features
./start.sh
```

### Docker Deployment
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f whisper-diarization
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/docs` | GET | Interactive API documentation |
| `/api/v1/transcribe` | POST | Transcribe audio file |
| `/api/v1/transcribe/batch` | POST | Batch transcription |
| `/api/v1/transcribe/{id}` | GET | Get transcription status |
| `/api/v1/transcribe/{id}/download` | GET | Download results |
| `/api/v1/transcribe/{id}/delete` | DELETE | Delete transcription |
| `/api/v1/features` | GET | Get supported features |

## 🎵 Supported Audio Formats

- **WAV** - Best quality, largest size
- **MP3** - Good balance, widely supported
- **M4A** - Apple devices, good quality
- **FLAC** - Lossless, high quality
- **OGG** - Open source, good compression

## 🛠️ CLI Usage

```bash
# Activate virtual environment
source venv/bin/activate

# Get service information
python cli.py info

# Check feature status
python cli.py features

# Transcribe audio file
python cli.py transcribe audio.mp3 --language en --output result.json

# List available models
python cli.py models
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file based on `config.env`:

```bash
# Server settings
PORT=80
HOST=0.0.0.0

# Whisper settings
WHISPER_MODEL=medium.en
WHISPER_DEVICE=auto

# NeMo settings
NEMO_DEVICE=auto

# Storage settings
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs
MAX_FILE_SIZE=524288000
```

### Feature Flags
Control enhanced features via API parameters:

```python
# Example transcription request
{
    "source_separation": true,      # Enable Demucs source separation
    "parallel_processing": true,    # Enable parallel execution
    "enhanced_alignment": true,     # Enable CTC alignment
    "enable_diarization": true      # Enable speaker diarization
}
```

## 🐳 Docker Configuration

### Services
- **whisper-diarization**: Main transcription service
- **redis**: Job queue and caching (optional)

### GPU Support
The service includes GPU configuration for CUDA-enabled servers:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 📊 Performance

### Model Comparison
| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|---------|----------|
| `tiny` | ⚡⚡⚡ | ⭐⭐ | 💾 | Real-time, low-resource |
| `base` | ⚡⚡ | ⭐⭐⭐ | 💾💾 | Balanced performance |
| `small` | ⚡ | ⭐⭐⭐⭐ | 💾💾💾 | High accuracy |
| `medium` | 🐌 | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | Best quality |
| `large` | 🐌🐌 | ⭐⭐⭐⭐⭐ | 💾💾💾💾💾 | Research/enterprise |

### Scaling Options
- **Horizontal**: Multiple service instances behind a load balancer
- **Vertical**: Increase container resources (CPU, GPU, memory)
- **Queue-based**: Redis-based job processing for high throughput

## 🔒 Security & Production

### Before Going Live
1. **Set API Keys**: Configure authentication in `.env`
2. **Restrict Origins**: Configure CORS settings
3. **Enable HTTPS**: Use reverse proxy (nginx)
4. **Monitor Resources**: Set up logging and alerting
5. **Backup Data**: Regular backups of outputs and configurations

### Security Features
- API key authentication
- File upload validation
- CORS configuration
- Rate limiting (configurable)
- Secure file handling

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use smaller model
   export WHISPER_MODEL=base.en
   
   # Reduce batch size
   export WHISPER_BATCH_SIZE=8
   ```

2. **Audio Format Issues**
   ```bash
   # Install FFmpeg
   brew install ffmpeg  # macOS
   apt-get install ffmpeg  # Ubuntu
   ```

3. **Service Won't Start**
   ```bash
   # Check port availability
   lsof -i :80
   
   # Verify Python version
   python --version
   
   # Check dependencies
   pip list
   ```

### Get Help
- **Logs**: `docker-compose logs whisper-diarization`
- **Health**: `curl http://localhost:80/health`
- **Features**: `curl http://localhost:80/api/v1/features`
- **Documentation**: `http://localhost:80/docs`

## 🔄 Development vs Production

### Development Mode (Python 3.13)
- ✅ Basic service structure
- ✅ API endpoints
- ✅ CLI tools
- ✅ Mock transcription results
- ❌ ML models (not compatible)
- ❌ GPU acceleration

### Production Mode (Python 3.10/3.11)
- ✅ All features enabled
- ✅ Full ML model support
- ✅ GPU acceleration
- ✅ Source separation
- ✅ Enhanced alignment
- ✅ Parallel processing

## 📈 Roadmap

- [ ] **v2.1**: Advanced speaker clustering algorithms
- [ ] [ ] **v2.2**: Real-time streaming transcription
- [ ] **v2.3**: Custom model fine-tuning
- [ ] **v2.4**: Multi-language simultaneous translation
- [ ] **v2.5**: Enterprise SSO integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for Whisper speech recognition
- **NVIDIA NeMo** for speaker diarization
- **Facebook Research** for Demucs source separation
- **Hugging Face** for model hosting and tools

---

**🎉 Ready to deploy!** Your enhanced transcription service is now production-ready with advanced features and GPU acceleration support.
