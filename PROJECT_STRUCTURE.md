# Enhanced Whisper Diarization Service - Project Structure

## 📁 Directory Structure

```
asr_whispher/
├── app/                          # Main application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI application entry point
│   ├── api/                      # API routes and endpoints
│   │   ├── __init__.py
│   │   └── routes.py            # REST API endpoints
│   ├── core/                     # Core configuration and settings
│   │   ├── __init__.py
│   │   ├── config.py            # Application configuration
│   │   └── logging.py           # Logging configuration
│   ├── models/                   # Pydantic data models
│   │   └── transcription.py     # Request/response models
│   ├── services/                 # Business logic services
│   │   ├── __init__.py
│   │   ├── diarization_service.py  # Core diarization service
│   │   └── batch_processor.py   # Batch processing service
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── audio_processor.py   # Audio processing utilities
│       ├── diarize_original.py  # Original diarization logic
│       ├── output_formats.py    # Output format converters
│       └── whisper_utils.py     # Whisper-specific utilities
├── config/                       # Configuration files
│   └── nemo_msdd_configs/       # NeMo MSDD configurations
│       ├── msdd_config.yaml     # Standard configuration
│       ├── production_config.yaml # Production configuration
│       ├── development_config.yaml # Development configuration
│       └── README.md            # Configuration documentation
├── scripts/                      # Utility scripts
│   └── download_nemo_models.py  # NeMo model downloader
├── tests/                        # Test suite
│   ├── __init__.py
│   └── test_main.py             # Main test file
├── logs/                         # Application logs
├── cache/                        # Cache directory
├── temp/                         # Temporary files
├── uploads/                      # File upload directory
├── outputs/                      # Processing outputs
├── models/                       # ML model storage
├── venv/                         # Python virtual environment
├── .git/                         # Git repository
├── .gitignore                    # Git ignore rules
├── config.env                    # Environment configuration template
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
├── run.sh                        # Unified setup and run script
├── cli.py                        # Command-line interface
├── README.md                     # Project documentation
└── PROJECT_STRUCTURE.md          # This file
```

## 🚀 Key Components

### **Core Application (`app/`)**
- **FastAPI Application**: RESTful API with automatic documentation
- **Service Layer**: Business logic for audio processing and diarization
- **Data Models**: Pydantic models for request/response validation
- **Utilities**: Audio processing, output formatting, and Whisper integration

### **Configuration (`config/`)**
- **NeMo MSDD Configs**: Advanced speaker diarization configurations
- **Environment Settings**: Application configuration via environment variables
- **Docker Configuration**: Container orchestration and deployment

### **Scripts & Tools (`scripts/`)**
- **Model Downloader**: Automated NeMo model acquisition
- **CLI Interface**: Command-line tool for direct usage
- **Unified Runner**: Single script for setup, start, and Docker operations

### **Infrastructure**
- **Docker Support**: Containerized deployment with GPU support
- **Virtual Environment**: Isolated Python dependencies
- **Logging & Monitoring**: Structured logging and error tracking

## 🔧 Development Workflow

1. **Setup**: `./run.sh setup` - Create virtual environment and install dependencies
2. **Development**: `./run.sh start` - Start service in development mode
3. **Testing**: `./run.sh docker start` - Test with Docker
4. **Production**: `./run.sh docker start` - Deploy with production configuration

## 📊 Configuration Options

- **Development**: Lightweight configuration for testing
- **Standard**: Balanced performance and quality
- **Production**: High-performance GPU-optimized configuration

## 🧹 Cleanup Notes

- Removed duplicate CLI files
- Cleaned up Python cache files
- Organized configuration structure
- Added proper .gitignore rules
- Created organized directory structure
