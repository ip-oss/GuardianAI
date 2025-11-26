# Experiments

This directory tracks all experiments, model training runs, and results for the iOS World project.

## Structure

```
experiments/
├── README.md           # This file
└── results/            # Training run results
    ├── run_001_baseline/
    │   ├── config.json       # Hyperparameters and settings
    │   ├── best_model.pt     # Best model checkpoint
    │   ├── metrics.json      # Evaluation metrics
    │   ├── training_log.txt  # Training output
    │   └── samples/          # Sample predictions (images)
    │       ├── sample_001_prediction.png
    │       └── sample_001_ground_truth.png
    └── run_002_clip_encoder/
        └── ...
```

## Experiment Tracking

Each experiment should have:

1. **Unique name**: `run_NNN_description` (e.g., `run_001_baseline`)
2. **Config file**: All hyperparameters and settings
3. **Metrics file**: Final evaluation results
4. **Model checkpoint**: Best performing model
5. **Sample outputs**: Visual examples of predictions

### Config File Format

```json
{
  "experiment_name": "run_001_baseline",
  "created_at": "2025-01-24T15:30:00Z",
  "model": {
    "type": "forward_dynamics",
    "vision_encoder": "resnet50",
    "action_encoder": "simple_embedding",
    "hidden_dim": 512
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 50,
    "optimizer": "adam",
    "early_stopping_patience": 5
  },
  "data": {
    "dataset_path": "../data/processed",
    "train_size": 8195,
    "val_size": 1024,
    "test_size": 1024,
    "image_size": [512, 1024]
  }
}
```

### Metrics File Format

```json
{
  "experiment_name": "run_001_baseline",
  "evaluated_at": "2025-01-25T10:00:00Z",
  "test_metrics": {
    "loss": 0.234,
    "accuracy": 0.756,
    "ssim": 0.823,
    "mse": 0.045
  },
  "per_action_type": {
    "tap": {"accuracy": 0.834, "count": 2876},
    "swipe": {"accuracy": 0.712, "count": 654},
    "type_text": {"accuracy": 0.623, "count": 432}
  },
  "training_history": {
    "epochs": 35,
    "best_epoch": 28,
    "final_train_loss": 0.189,
    "final_val_loss": 0.234
  }
}
```

## Planned Experiments

### Phase 2: Model Training

**Experiment 1: Baseline**
- Simple ResNet-50 encoder
- Forward dynamics only
- Establishes performance floor

**Experiment 2: CLIP Encoder**
- Use pre-trained CLIP vision encoder
- May work better for UI understanding
- Compare to baseline

**Experiment 3: Inverse Dynamics**
- Train on action prediction task
- Validates data quality
- May be easier than forward

**Experiment 4: Joint Training**
- Multi-task: both forward and inverse
- May learn better representations
- Compare to single-task models

**Experiment 5: Larger Model**
- Increase capacity (more layers/dimensions)
- See if we're underfitting
- Check if data is sufficient

**Experiment 6: Different Image Sizes**
- Try 256x512 (faster) vs 768x1536 (more detail)
- Find optimal size/speed tradeoff

### Phase 3: Bug Detection

**Experiment 7: Bug Injection**
- Test on app with known bugs
- Measure detection rate
- Tune confidence threshold

**Experiment 8: Generalization**
- Test on app not in training set
- How well does it transfer?

**Experiment 9: Confidence Calibration**
- Analyze when model is wrong but confident
- Improve uncertainty estimation

## Experiment Log

Keep a running log of insights from each experiment:

### Run 001: Baseline (Planned)
**Goal**: Establish baseline performance
**Status**: Not started (waiting for data)
**Hypothesis**: ResNet-50 should achieve ~60% prediction accuracy
**Result**: TBD

### Run 002: CLIP Encoder (Planned)
**Goal**: Improve on baseline with better vision encoder
**Status**: Not started
**Hypothesis**: CLIP's multimodal training may help with UI understanding
**Result**: TBD

---

*Update this log as experiments are run*

## Analysis Tools

Useful analysis scripts to write:

1. **Compare runs**: Side-by-side metrics comparison
2. **Visualize predictions**: Show model outputs vs ground truth
3. **Error analysis**: What kinds of transitions does the model struggle with?
4. **Learning curves**: Plot train/val loss over time

## Best Practices

1. **Always save config**: Can't reproduce without it
2. **Commit code before run**: Note git hash in config
3. **Save random seeds**: For reproducibility
4. **Log to wandb**: Easy to track many runs
5. **Archive failed runs**: Even failures teach us something

## Next Steps

After experiments are complete:

1. Select best model
2. Write up findings in `docs/learnings.md`
3. Decide: Is this good enough to pursue further?
4. If yes: Move to Phase 4 (paper/patents)
5. If no: Document why and what would be needed

See `PROJECT_DIRECTION.md` for overall project plan.
