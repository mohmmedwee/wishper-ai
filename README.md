# Whisper Diarization Service

A production-ready service for automatic speech recognition (ASR) with speaker diarization, built on top of OpenAI's Whisper and NeMo speaker diarization models.

## Features

- **High-Quality Transcription**: Powered by OpenAI Whisper models
- **Speaker Diarization**: Identify and separate different speakers in audio
- **RESTful API**: Easy-to-use HTTP endpoints for transcription
- **Async Processing**: Non-blocking audio processing with background tasks
- **Multiple Formats**: Support for various audio formats (WAV, MP3, M4A, FLAC, OGG)
- **Configurable Models**: Choose from different Whisper model sizes
- **Production Ready**: Docker support, logging, monitoring, and health checks
- **Scalable**: Redis-based job queue and caching support

## Architecture

The service combines several technologies:

1. **Whisper ASR**: For high-quality speech-to-text transcription
2. **NeMo Speaker Diarization**: For identifying different speakers
3. **FastAPI**: Modern, fast web framework for building APIs
4. **Redis**: For job queuing and caching
5. **Docker**: For easy deployment and scaling

## Quick Start

### Prerequisites

- Python 3.10+
- FFmpeg
- Docker and Docker Compose (recommended)
- NVIDIA GPU with CUDA support (optional, for faster processing)

### Using Docker (Recommended)

1. Clone the repository:
```bash
git clone <your-repo-url>
cd asr_whispher
```

2. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the service:
```bash
docker-compose up -d
```

4. The service will be available at `http://localhost:8000`

### Manual Installation

1. Install system dependencies:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg libsndfile1-dev

# macOS
brew install ffmpeg

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the service:
```bash
python -m app.main
```

## API Usage

### Transcribe Audio

```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "language=en" \
  -F "enable_diarization=true"
```

### Get Transcription Status

```bash
curl "http://localhost:8000/api/v1/transcribe/{transcription_id}"
```

### Download Transcription

```bash
curl "http://localhost:8000/api/v1/transcribe/{transcription_id}/download?format=json"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `WHISPER_MODEL` | Whisper model size | `medium.en` |
| `WHISPER_DEVICE` | Processing device | `auto` |
| `MAX_FILE_SIZE` | Maximum file size | `500MB` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Whisper Models

Available models (in order of size and accuracy):
- `tiny` / `tiny.en` - Fastest, least accurate
- `base` / `base.en` - Good balance
- `small` / `small.en` - Better accuracy
- `medium` / `medium.en` - Recommended (default)
- `large` / `large-v2` / `large-v3` - Best accuracy, slowest

## Development

### Project Structure

```
asr_whispher/
├── app/
│   ├── api/           # API routes and endpoints
│   ├── core/          # Configuration and core utilities
│   ├── models/        # Data models and schemas
│   ├── services/      # Business logic services
│   └── utils/         # Utility functions
├── config/            # Configuration files
├── tests/             # Test suite
├── uploads/           # Uploaded audio files
├── outputs/           # Generated transcriptions
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose setup
└── requirements.txt   # Python dependencies
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
```

## Deployment

### Production Considerations

1. **Security**: Use proper API keys and authentication
2. **Scaling**: Use Redis for job queuing and load balancing
3. **Monitoring**: Custom monitoring and alerting support
4. **Storage**: Use persistent volumes for uploads and outputs
5. **Backup**: Implement regular backup strategies

### Kubernetes Deployment

Example deployment YAML files are provided in the `k8s/` directory.

### Cloud Deployment

The service can be deployed on:
- AWS (ECS, EKS)
- Google Cloud (GKE, Cloud Run)
- Azure (AKS, Container Instances)
- DigitalOcean (App Platform)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use smaller Whisper model
2. **Audio Format Issues**: Ensure FFmpeg is properly installed
3. **Model Download Failures**: Check internet connection and Hugging Face access

### Logs

Check logs with:
```bash
docker-compose logs whisper-diarization
```

### Performance Tuning

1. **GPU Acceleration**: Ensure CUDA is properly configured
2. **Batch Processing**: Adjust batch sizes based on available memory
3. **Model Selection**: Choose appropriate Whisper model for your use case

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [NeMo](https://github.com/NVIDIA/NeMo) - Speaker diarization
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
