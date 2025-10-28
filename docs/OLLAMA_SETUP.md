# Ollama Setup Guide for AuraTrade

This guide will help you set up Ollama for local LLM usage with AuraTrade.

## What is Ollama?

Ollama is a tool that allows you to run large language models locally on your machine. This means:
- **No API costs** - Run models completely free
- **Privacy** - All data stays on your machine
- **Speed** - No network latency for LLM calls
- **Reliability** - Works offline

## Installation

### 1. Download and Install Ollama

Visit [https://ollama.ai](https://ollama.ai) and download Ollama for Windows.

### 2. Install Ollama

Run the installer and follow the setup wizard.

### 3. Verify Installation

Open Command Prompt or PowerShell and run:
```bash
ollama --version
```

You should see the Ollama version number.

## Download Models

### Recommended Models for AuraTrade

**For Development/Testing:**
```bash
ollama pull llama3.1:8b
```
- Good balance of performance and speed
- ~4.7GB download
- Suitable for most analysis tasks

**For Production (if you have a powerful machine):**
```bash
ollama pull llama3.1:70b
```
- Highest quality responses
- ~40GB download
- Requires significant RAM (32GB+ recommended)

**Lightweight Option:**
```bash
ollama pull llama3.1:7b
```
- Fastest inference
- ~3.8GB download
- Good for rapid prototyping

### Check Available Models
```bash
ollama list
```

## Configure AuraTrade

### 1. Update .env Configuration

Edit your `.env` file to use Ollama as primary:

```env
# Ollama Configuration (Primary)
AURA_LLM_OLLAMA_ENABLED=true
AURA_LLM_OLLAMA_MODEL=llama3.1:8b
AURA_LLM_OLLAMA_URL=http://localhost:11434
AURA_LLM_OLLAMA_PRIMARY=true

# OpenAI as Fallback (Optional)
AURA_LLM_OPENAI_ENABLED=true
AURA_LLM_OPENAI_PRIMARY=false
OPENAI_API_KEY=your_key_here  # Optional fallback
```

### 2. Start Ollama Service

Ollama usually starts automatically, but you can start it manually:
```bash
ollama serve
```

### 3. Test the Setup

Run AuraTrade in demo mode:
```bash
python main.py --mode demo
```

## Model Selection Guide

| Model | Size | RAM Required | Use Case |
|-------|------|--------------|----------|
| llama3.1:7b | 3.8GB | 8GB+ | Development, testing |
| llama3.1:8b | 4.7GB | 12GB+ | **Recommended for AuraTrade** |
| llama3.1:70b | 40GB | 64GB+ | High-quality production |
| mistral:7b | 4.1GB | 8GB+ | Alternative, good performance |
| codellama:13b | 7.3GB | 16GB+ | Code-focused tasks |

## Performance Tips

### 1. GPU Acceleration (Optional)

If you have an NVIDIA GPU:
- Ollama automatically uses GPU if available
- Significantly faster inference
- Check GPU usage with `nvidia-smi`

### 2. Memory Management

- Close other applications when running large models
- Monitor RAM usage with Task Manager
- Consider using smaller models if memory is limited

### 3. Optimize Model Loading

Models stay loaded in memory after first use:
- First query may be slow (model loading)
- Subsequent queries are much faster
- Restart Ollama to free memory: `ollama stop` then `ollama serve`

## Troubleshooting

### Common Issues

**"Connection refused" errors:**
- Check if Ollama service is running: `ollama list`
- Restart Ollama: `ollama serve`
- Check port 11434 is not blocked

**Slow responses:**
- Ensure sufficient RAM available
- Try a smaller model
- Close unnecessary applications

**Model download fails:**
- Check internet connection
- Ensure sufficient disk space
- Try downloading again: `ollama pull <model>`

**Out of memory errors:**
- Use a smaller model
- Increase virtual memory/page file
- Close other applications

### Getting Help

1. **Ollama Documentation**: [https://github.com/ollama/ollama](https://github.com/ollama/ollama)
2. **AuraTrade Issues**: Check the repository issues page
3. **Community**: Ollama has an active Discord community

## Advanced Configuration

### Custom Model Parameters

You can adjust model parameters in `.env`:

```env
# Increase creativity (0.0-2.0)
AURA_LLM_OLLAMA_TEMPERATURE=0.9

# Longer responses
AURA_LLM_OLLAMA_MAX_TOKENS=3000

# Longer timeout for complex analysis
AURA_LLM_OLLAMA_TIMEOUT=120
```

### Multiple Models

You can have multiple models and switch between them:

```bash
# Download multiple models
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull codellama:13b

# Switch model in .env
AURA_LLM_OLLAMA_MODEL=mistral:7b
```

### API Compatibility

Ollama provides OpenAI-compatible API, so it works seamlessly with AuraTrade's LLM abstraction layer.

## Production Considerations

### 1. Resource Planning
- Allocate sufficient RAM for your chosen model
- Consider GPU requirements for heavy usage
- Plan for concurrent model usage

### 2. Model Selection
- Test different models with your specific use cases
- Balance quality vs. speed vs. resource usage
- Consider fine-tuned models for financial analysis

### 3. Monitoring
- Monitor resource usage during trading hours
- Set up alerts for service failures
- Have fallback LLM providers configured

### 4. Security
- Ollama runs locally - no data leaves your machine
- Consider network access controls if running on servers
- Keep Ollama updated for security patches

## Getting Started Checklist

- [ ] Install Ollama from ollama.ai
- [ ] Download recommended model: `ollama pull llama3.1:8b`
- [ ] Update `.env` file with Ollama configuration
- [ ] Test with: `python main.py --mode demo`
- [ ] Verify LLM responses in demo output
- [ ] Configure fallback providers if needed

With Ollama configured, AuraTrade will use local LLM processing for all analysis tasks, providing cost-effective and private AI-powered trading decisions!