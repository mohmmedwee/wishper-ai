# NeMo MSDD Configuration Files

This directory contains configuration files for NeMo Multi-Speaker Diarization Detection (MSDD) models, optimized for different use cases.

## 📁 Configuration Files

### 1. `msdd_config.yaml` - Standard Configuration
- **Purpose**: Balanced configuration for general use
- **Features**: Full MSDD pipeline with moderate performance settings
- **Use Case**: Production deployment with balanced speed/accuracy

### 2. `production_config.yaml` - High-Performance Production
- **Purpose**: Optimized for high-throughput server deployment
- **Features**: GPU acceleration, advanced overlap detection, high-quality embeddings
- **Use Case**: Enterprise production environments with GPU resources

### 3. `development_config.yaml` - Development & Testing
- **Purpose**: Lightweight configuration for development and debugging
- **Features**: Simplified pipeline, verbose logging, faster processing
- **Use Case**: Development, testing, and debugging

## 🚀 Key Features

### Voice Activity Detection (VAD)
- **Model**: `vad_multilingual_marblenet`
- **Optimization**: Multi-scale processing with configurable thresholds
- **Languages**: Multilingual support

### Speaker Embedding Extraction
- **Model**: `titanet_large`
- **Features**: Multi-scale window processing
- **Quality**: High-fidelity speaker representations

### Clustering Algorithm
- **Method**: Spectral clustering with enhanced speaker counting
- **Optimization**: Adaptive thresholds and chunk-based processing
- **Scalability**: Handles long-form audio efficiently

### MSDD Model
- **Capabilities**: Multi-scale diarization with overlap detection
- **Performance**: GPU-optimized inference
- **Quality**: Adaptive thresholding for optimal results

## ⚙️ Configuration Options

### Performance Tuning
```yaml
performance:
  max_memory_usage: "16GB"
  enable_mixed_precision: true
  num_parallel_workers: 8
  enable_async_processing: true
```

### Quality Settings
```yaml
msdd_advanced:
  speaker_verification: true
  overlap_detection: true
  temporal_consistency: true
  multiscale: true
```

### Output Formats
```yaml
output:
  formats: ["rttm", "json", "srt", "vtt", "txt"]
  include_confidence: true
  include_speaker_info: true
```

## 🔧 Usage

### 1. Standard Configuration
```python
from app.services.diarization_service import DiarizationService

service = DiarizationService()
service.config_path = "config/nemo_msdd_configs/msdd_config.yaml"
```

### 2. Production Configuration
```python
# For high-performance deployment
service.config_path = "config/nemo_msdd_configs/production_config.yaml"
```

### 3. Development Configuration
```python
# For testing and development
service.config_path = "config/nemo_msdd_configs/development_config.yaml"
```

## 📊 Performance Comparison

| Configuration | Speed | Quality | Memory | GPU Usage |
|---------------|-------|---------|---------|-----------|
| Development   | ⚡⚡⚡ | ⭐⭐⭐ | 💾💾 | 🟡 |
| Standard     | ⚡⚡ | ⭐⭐⭐⭐ | 💾💾💾 | 🟢 |
| Production   | ⚡ | ⭐⭐⭐⭐⭐ | 💾💾💾💾 | 🟢 |

## 🎯 Optimization Tips

### For Speed
- Use development configuration
- Reduce `num_workers` and `batch_size`
- Disable advanced features

### For Quality
- Use production configuration
- Increase `max_num_speakers`
- Enable all advanced features

### For Memory
- Reduce `embeddings_per_chunk`
- Lower `max_memory_usage`
- Use smaller batch sizes

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```yaml
   performance:
     max_memory_usage: "8GB"  # Reduce this value
     embeddings_per_chunk: 5000  # Reduce chunk size
   ```

2. **Slow Processing**
   ```yaml
   performance:
     num_parallel_workers: 4  # Increase workers
     enable_async_processing: true
   ```

3. **Low Quality Results**
   ```yaml
   msdd_advanced:
     speaker_verification: true
     overlap_detection: true
     temporal_consistency: true
   ```

## 📚 References

- [NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/)
- [MSDD Paper](https://arxiv.org/abs/2203.16768)
- [Speaker Diarization Guide](https://github.com/NVIDIA/NeMo/tree/main/examples/speaker_tasks/diarization)

## 🆘 Support

For configuration issues or optimization questions:
1. Check the NeMo documentation
2. Review the configuration parameters
3. Test with development configuration first
4. Monitor system resources during processing
