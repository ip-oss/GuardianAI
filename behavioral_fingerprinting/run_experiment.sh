#!/bin/bash
# Quick start script for running LLM behavioral fingerprinting experiments
# with different providers

set -e

echo "======================================================================"
echo "LLM Behavioral Fingerprinting - Multi-Provider Support"
echo "======================================================================"
echo ""

# Function to display usage
usage() {
    echo "Usage: $0 [provider] [trajectories]"
    echo ""
    echo "Providers:"
    echo "  ollama      - FREE local models (requires Ollama installed)"
    echo "  deepseek    - Cheapest API option (~$2 for 20 trajectories)"
    echo "  anthropic   - Original Claude (~$30-50 for 20 trajectories)"
    echo "  openai      - GPT-4 (~$14-35 for 20 trajectories)"
    echo ""
    echo "Examples:"
    echo "  $0 ollama 20          # Run with local Llama 3.1 (FREE)"
    echo "  $0 deepseek 20        # Run with DeepSeek API (cheap)"
    echo "  $0 anthropic 5        # Run with Claude (expensive)"
    echo ""
    exit 1
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

PROVIDER=$1
TRAJECTORIES=${2:-20}

# Navigate to experiment directory
cd "$(dirname "$0")"

# Run experiment based on provider
case $PROVIDER in
    ollama)
        echo "Provider: Ollama (Local - FREE)"
        echo "Model: llama3.1"
        echo "Trajectories: $TRAJECTORIES"
        echo ""

        # Check if Ollama is running
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "❌ Error: Ollama is not running!"
            echo ""
            echo "Please install and start Ollama:"
            echo "  1. Install: https://ollama.ai"
            echo "  2. Run: ollama serve"
            echo "  3. Pull model: ollama pull llama3.1"
            echo ""
            exit 1
        fi

        # Check if model is available
        if ! ollama list | grep -q "llama3.1"; then
            echo "⚠️  llama3.1 model not found. Pulling it now..."
            ollama pull llama3.1
        fi

        echo "✅ Ollama is ready!"
        echo ""
        python experiments/exp7_llm_taxonomy.py --provider ollama --model llama3.1 --trajectories $TRAJECTORIES
        ;;

    deepseek)
        echo "Provider: DeepSeek API (Cheap)"
        echo "Model: deepseek-chat"
        echo "Trajectories: $TRAJECTORIES"
        echo "Estimated cost: \$$(python -c "print(f'{$TRAJECTORIES * 7 * 0.01:.2f}')" 2>/dev/null || echo "1-3")"
        echo ""

        if [ -z "$DEEPSEEK_API_KEY" ]; then
            echo "❌ Error: DEEPSEEK_API_KEY not set!"
            echo ""
            echo "Get your API key from: https://platform.deepseek.com"
            echo "Then run: export DEEPSEEK_API_KEY='your-key-here'"
            echo ""
            exit 1
        fi

        echo "✅ API key found!"
        echo ""
        python experiments/exp7_llm_taxonomy.py --provider deepseek --trajectories $TRAJECTORIES
        ;;

    anthropic)
        echo "Provider: Anthropic (Claude)"
        echo "Model: claude-sonnet-4-5-20250929 (Sonnet 4.5 - newest)"
        echo "Trajectories: $TRAJECTORIES"
        echo "Estimated cost: \$$(python -c "print(f'{$TRAJECTORIES * 7 * 0.15:.2f}')" 2>/dev/null || echo "21-42")"
        echo ""

        if [ -z "$ANTHROPIC_API_KEY" ]; then
            echo "❌ Error: ANTHROPIC_API_KEY not set!"
            echo ""
            echo "Get your API key from: https://console.anthropic.com"
            echo "Then run: export ANTHROPIC_API_KEY='your-key-here'"
            echo ""
            exit 1
        fi

        echo "✅ API key found!"
        echo ""
        python experiments/exp7_llm_taxonomy.py --provider anthropic --trajectories $TRAJECTORIES
        ;;

    openai)
        echo "Provider: OpenAI"
        echo "Model: gpt-4o (latest GPT model)"
        echo "Trajectories: $TRAJECTORIES"
        echo "Estimated cost: \$$(python -c "print(f'{$TRAJECTORIES * 7 * 0.05:.2f}')" 2>/dev/null || echo "7-21")"
        echo ""

        if [ -z "$OPENAI_API_KEY" ]; then
            echo "❌ Error: OPENAI_API_KEY not set!"
            echo ""
            echo "Get your API key from: https://platform.openai.com"
            echo "Then run: export OPENAI_API_KEY='your-key-here'"
            echo ""
            exit 1
        fi

        echo "✅ API key found!"
        echo ""
        python experiments/exp7_llm_taxonomy.py --provider openai --model gpt-4o --trajectories $TRAJECTORIES
        ;;

    *)
        echo "❌ Unknown provider: $PROVIDER"
        echo ""
        usage
        ;;
esac
