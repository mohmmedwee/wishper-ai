# Whisper Diarization Service

A comprehensive speaker diarization service based on OpenAI Whisper and NeMo, providing automatic speech recognition with speaker identification.

## 🚀 Features

- **Automatic Speech Recognition (ASR)** using OpenAI Whisper
- **Speaker Diarization** with NeMo MSDD
- **Language Detection** (auto-detect or manual specification)
- **Multiple Output Formats**: SRT, VTT, TXT, JSON, RTTM, CSV
- **Audio Preprocessing** and quality analysis
- **Real-time Processing** with progress tracking
- **Batch Processing** support for multiple files
- **RESTful API** with FastAPI

## 🏗️ Architecture

The service follows the whisper-diarization pipeline:

1. **Audio Input** → Validation and preprocessing
2. **Language Detection** → Auto-detect or use specified language
3. **Whisper Transcription** → Generate text with timestamps
4. **Speaker Diarization** → Identify speakers for each segment
5. **Alignment & Post-processing** → Refine timestamps and format output

## 📋 Requirements

### System Requirements
- **Python**: 3.11 or 3.12 (3.13 not yet supported by ML packages)
- **FFmpeg**: For audio processing
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ for models and cache

### ML Packages (when using Python 3.11/3.12)
- PyTorch & TorchAudio
- Faster Whisper
- NeMo Toolkit
- Demucs (source separation)
- CTC Forced Aligner

## 🚀 Quick Start

### Option 1: Docker (Recommended for ML functionality)

```bash
# Clone the repository
git clone <your-repo-url>
cd asr_whispher

# Run the ML service
./run_ml_service.sh
```

The service will be available at `http://localhost:8000`

### Option 2: Local Development (Python 3.11/3.12)

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Python 3.13 (Limited functionality)

```bash
# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate

# Install basic dependencies
pip install numpy librosa soundfile pydub fastapi uvicorn

# Run with enhanced mock mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
asr_whispher/
├── app/
│   ├── api/                 # API routes and endpoints
│   ├── core/               # Configuration and logging
│   ├── models/             # Data models and schemas
│   ├── services/           # Core business logic
│   │   ├── diarization_service.py    # Main diarization service
│   │   └── batch_processor.py        # Batch processing
│   └── utils/              # Utility functions
│       ├── audio_processor.py        # Audio processing
│       ├── output_formats.py         # Output format conversion
│       └── whisper_utils.py          # Whisper utilities
├── config/                  # NeMo configuration files
├── models/                  # Downloaded ML models
├── uploads/                 # Audio file uploads
├── outputs/                 # Transcription results
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker services
└── run_ml_service.sh        # ML service runner
```

## 🔧 API Usage

### Transcribe Audio

```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@your_audio.wav" \
  -F "language=en" \
  -F "whisper_model=base"
```

### Check Service Status

```bash
curl "http://localhost:8000/api/status"
```

### Health Check

```bash
curl "http://localhost:8000/api/health"
```

## 📊 Output Formats

The service supports multiple output formats:

- **SRT**: SubRip subtitle format
- **VTT**: WebVTT format
- **TXT**: Plain text transcription
- **JSON**: Structured data with metadata
- **RTTM**: Rich Transcription Time Marked
- **CSV**: Comma-separated values

## 🎯 Configuration

### Whisper Models
- `tiny`: 39M parameters, fastest
- `base`: 74M parameters, balanced
- `small`: 244M parameters, good quality
- `medium`: 769M parameters, high quality
- `large`: 1550M parameters, best quality

### Language Support
Supports 100+ languages including:
- English (en), Spanish (es), French (fr)
- German (de), Chinese (zh), Japanese (ja)
- Arabic (ar), Russian (ru), and many more

## 🔍 Troubleshooting

### Python 3.13 Compatibility Issues

If you're using Python 3.13, you'll see this message:
```
ML models not available due to Python 3.13 compatibility
Service will provide basic audio processing and enhanced mock transcription
```

**Solutions:**
1. **Use Docker** (recommended): `./run_ml_service.sh`
2. **Downgrade Python**: Use Python 3.11 or 3.12
3. **Wait for updates**: ML packages will support Python 3.13 in 3-6 months

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller Whisper model
2. **Audio Format Error**: Ensure audio file is in supported format
3. **Model Download Failed**: Check internet connection and try again

## 🚀 Performance Tips

- Use **GPU acceleration** when available
- Choose appropriate **Whisper model size** for your needs
- Enable **batch processing** for multiple files
- Use **source separation** for noisy audio

## 📈 Monitoring

The service provides:
- Real-time processing status
- Audio quality metrics
- Speaker detection confidence
- Processing time statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for ASR
- [NeMo](https://github.com/NVIDIA/NeMo) for speaker diarization
- [Faster Whisper](https://github.com/guillaumekln/faster-whisper) for optimized inference
- [Demucs](https://github.com/facebookresearch/demucs) for source separation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include audio file samples and error logs
