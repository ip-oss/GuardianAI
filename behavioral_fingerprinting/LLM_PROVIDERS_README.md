# Multi-Provider LLM Support

The experiment now supports multiple LLM providers, allowing you to run experiments with different models without changing code. This is essential for scaling experiments cost-effectively.

## Supported Providers

1. **Anthropic (Claude Sonnet 4.5)** - Newest Claude model, high quality but expensive
2. **OpenAI (GPT-4o, GPT-4 Turbo, GPT-3.5)** - Alternative API provider
3. **DeepSeek** - Much cheaper alternative (~14-50x cheaper than Claude!)
4. **Ollama** - Run local models for FREE

## Quick Start

### Option 1: Command Line Arguments (Recommended)

```bash
# Use DeepSeek (cheapest API option)
export DEEPSEEK_API_KEY="your-key-here"
python experiments/exp7_llm_taxonomy.py --provider deepseek --trajectories 20

# Use local Ollama (FREE!)
python experiments/exp7_llm_taxonomy.py --provider ollama --model llama3.1 --trajectories 20

# Use Claude (original)
export ANTHROPIC_API_KEY="your-key-here"
python experiments/exp7_llm_taxonomy.py --provider anthropic --trajectories 20

# Use GPT-4o (newest from OpenAI)
export OPENAI_API_KEY="your-key-here"
python experiments/exp7_llm_taxonomy.py --provider openai --model gpt-4o --trajectories 20
```

### Option 2: Configuration File

Create a config file (or use examples in `configs/`):

```json
{
  "provider": "deepseek",
  "model": "deepseek-chat",
  "api_key": "your-api-key"
}
```

Then run:
```bash
python experiments/exp7_llm_taxonomy.py --config-file configs/deepseek_example.json --trajectories 20
```

## Cost Comparison

For running 20 trajectories per agent type (140 total trajectories):

| Provider | Model | Estimated Cost | Notes |
|----------|-------|----------------|-------|
| **Ollama** | llama3.1 | **$0 (FREE)** | Runs locally, requires ~8GB RAM |
| **DeepSeek** | deepseek-chat | **$1.40 - $2.80** | Cheapest API option! |
| **OpenAI** | gpt-3.5-turbo | $7 - $15 | Good balance |
| **OpenAI** | gpt-4o | $7 - $21 | Latest GPT, faster & cheaper than GPT-4 |
| **OpenAI** | gpt-4-turbo | $14 - $35 | High quality |
| **Anthropic** | claude-sonnet-4.5 | $21 - $42 | Newest Sonnet, high quality |
| **Anthropic** | claude-opus-4 | $105 - $210 | Best quality, most expensive |

## Recommended Approach for Scaling

Based on your analysis showing you need to scale from n=3 to n=20 per type:

### Phase 1: Test with Ollama (FREE)
```bash
# First, install Ollama: https://ollama.ai
ollama pull llama3.1

# Run small test
python experiments/exp7_llm_taxonomy.py --provider ollama --trajectories 3

# If results look reasonable, scale up
python experiments/exp7_llm_taxonomy.py --provider ollama --trajectories 20
```

### Phase 2: Validate with DeepSeek ($1-3)
```bash
# Get API key from: https://platform.deepseek.com
export DEEPSEEK_API_KEY="your-key"

# Run full experiment at ~14-50x cheaper than Claude
python experiments/exp7_llm_taxonomy.py --provider deepseek --trajectories 20
```

### Phase 3: Compare with Claude (if needed)
```bash
# Only run this if DeepSeek results are promising
# This confirms findings generalize across models
python experiments/exp7_llm_taxonomy.py --provider anthropic --trajectories 5
```

## Setting Up Each Provider

### DeepSeek (Recommended for scaling)

1. Get API key: https://platform.deepseek.com
2. Export environment variable:
   ```bash
   export DEEPSEEK_API_KEY="your-key-here"
   ```
3. Install required package (if not already installed):
   ```bash
   pip install openai
   ```
4. Run experiment:
   ```bash
   python experiments/exp7_llm_taxonomy.py --provider deepseek --trajectories 20
   ```

**Why DeepSeek?**
- 14-50x cheaper than Claude ($0.14/1M tokens vs $3-15/1M)
- Good quality (competitive with GPT-3.5)
- Same API interface as OpenAI (uses OpenAI SDK)
- For 20 trajectories: ~$2 instead of $30-50

### Ollama (Best for rapid iteration)

1. Install Ollama:
   ```bash
   # macOS
   brew install ollama

   # Or download from https://ollama.ai
   ```

2. Start Ollama service:
   ```bash
   ollama serve
   ```

3. Pull a model:
   ```bash
   # Recommended: Llama 3.1 (8B parameters, ~4GB RAM)
   ollama pull llama3.1

   # Or larger model for better quality (70B parameters, ~40GB RAM)
   ollama pull llama3.1:70b

   # Or other models
   ollama pull mistral
   ollama pull qwen2.5:7b
   ```

4. Run experiment:
   ```bash
   python experiments/exp7_llm_taxonomy.py --provider ollama --model llama3.1 --trajectories 20
   ```

**Why Ollama?**
- Completely FREE
- Fast iteration (no API rate limits)
- Privacy (data stays local)
- Good for rapid prototyping

### Claude (Original)

```bash
export ANTHROPIC_API_KEY="your-key-here"
python experiments/exp7_llm_taxonomy.py --provider anthropic --trajectories 20
```

### GPT-4o (Alternative high-quality option)

```bash
export OPENAI_API_KEY="your-key-here"
python experiments/exp7_llm_taxonomy.py --provider openai --model gpt-4o --trajectories 20
```

**Note:** GPT-4o is faster and cheaper than GPT-4 Turbo while being equally capable.

## Example: Running Multi-Model Comparison

```bash
# Test if findings hold across different models

# 1. Ollama (FREE baseline)
python experiments/exp7_llm_taxonomy.py --provider ollama --trajectories 10

# 2. DeepSeek (cheap validation)
python experiments/exp7_llm_taxonomy.py --provider deepseek --trajectories 10

# 3. Claude (original, if budget allows)
python experiments/exp7_llm_taxonomy.py --provider anthropic --trajectories 5

# Compare results across providers to see if behavioral signatures are model-agnostic
```

## Troubleshooting

### Ollama Issues

**Error: "Connection refused"**
- Make sure Ollama is running: `ollama serve`
- Check the service is accessible: `curl http://localhost:11434/api/generate`

**Error: "Model not found"**
- Pull the model first: `ollama pull llama3.1`
- List available models: `ollama list`

**Slow performance**
- Use a smaller model (llama3.1 instead of llama3.1:70b)
- Increase timeout in code if needed
- Check RAM usage (models need 4-40GB depending on size)

### DeepSeek Issues

**Error: "Invalid API key"**
- Get key from https://platform.deepseek.com
- Make sure it's exported: `echo $DEEPSEEK_API_KEY`
- Try setting it in config file instead

**Error: "openai module not found"**
- Install: `pip install openai`

### General API Issues

**Rate limiting**
- Add delays between requests
- Use local Ollama for unlimited requests

**Cost concerns**
- Always start with small `--trajectories` value (3-5)
- Check cost estimates printed at start
- Use DeepSeek or Ollama for large-scale experiments

## Performance Comparison

Based on preliminary testing:

| Provider | Quality | Speed | Cost | Best For |
|----------|---------|-------|------|----------|
| Claude Opus | Excellent | Medium | Very High | Final validation only |
| Claude Sonnet 4.5 | Excellent | Fast | High | Original baseline |
| GPT-4o | Excellent | Very Fast | Medium | Cross-validation, good balance |
| GPT-4 Turbo | Very Good | Fast | Medium | Alternative to Claude |
| GPT-3.5 | Good | Fast | Low | Quick tests |
| DeepSeek | Good | Fast | Very Low | **Scaling experiments** |
| Llama 3.1 (Ollama) | Good | Fast | FREE | **Rapid iteration** |

## Next Steps

1. **Test quickly with Ollama** (free, fast iteration)
2. **Scale with DeepSeek** (cheap, good quality)
3. **Validate findings** across multiple models
4. **Compare results** to see if behavioral fingerprints are model-agnostic

This approach lets you:
- Test ideas rapidly (Ollama)
- Scale cheaply (DeepSeek: $2-3 instead of $300-500)
- Validate robustness (compare across providers)
