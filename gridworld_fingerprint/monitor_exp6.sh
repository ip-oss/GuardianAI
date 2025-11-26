#!/bin/bash
# Monitor exp6_taxonomy progress

LOG_FILE="/tmp/exp6_taxonomy_20x20.log"

while true; do
    clear
    echo "=========================================="
    echo "    EXP6 TAXONOMY - PROGRESS MONITOR"
    echo "=========================================="
    echo ""

    # Show current agent being trained
    echo "📊 Current Status:"
    tail -20 "$LOG_FILE" | grep -E "(Training|Episode|Collecting|Avg|Completion)" | tail -10

    echo ""
    echo "📈 Training Progress:"
    grep -c "Training.*agent" "$LOG_FILE" | xargs -I {} echo "  Agents trained: {}/8"

    echo ""
    echo "⏱  Last update: $(date '+%H:%M:%S')"
    echo ""
    echo "Press Ctrl+C to stop monitoring..."
    echo "Log file: $LOG_FILE"

    sleep 5
done
