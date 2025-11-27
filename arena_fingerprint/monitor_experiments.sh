#!/bin/bash
# Monitor running experiments

echo "Monitoring Arena Fingerprint Experiments"
echo "=========================================="

while true; do
    clear
    echo "Arena Fingerprint - Experiment Monitor"
    echo "Time: $(date)"
    echo ""

    # Check running processes
    echo "Running Experiments:"
    ps aux | grep -E "(exp1_baseline|exp2_single_defector)" | grep -v grep | grep -v monitor || echo "  No experiments running"

    echo ""
    echo "Latest Results:"

    # Check latest exp1 results
    latest_exp1=$(ls -td results/exp1_baseline_* 2>/dev/null | head -1)
    if [ -n "$latest_exp1" ]; then
        echo "  exp1_baseline: $latest_exp1"
        if [ -f "$latest_exp1/results.json" ]; then
            echo "    ✓ COMPLETED"
            cat "$latest_exp1/results.json" | grep -E "(avg_mean_error|success)" | head -2
        else
            echo "    ⏳ RUNNING..."
        fi
    fi

    # Check latest exp2 results
    latest_exp2=$(ls -td results/exp2_defector_* 2>/dev/null | head -1)
    if [ -n "$latest_exp2" ]; then
        echo "  exp2_defector: $latest_exp2"
        if [ -f "$latest_exp2/results.json" ]; then
            echo "    ✓ COMPLETED"
            cat "$latest_exp2/results.json" | grep -E "(avg_accuracy|success)" | head -2
        else
            echo "    ⏳ RUNNING..."
        fi
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 5
done
