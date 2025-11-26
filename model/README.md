# Model Training

Vision-language model for predicting iOS UI transitions and detecting bugs.

> **Note**: This is a Phase 2 task. The model will be implemented after data collection is complete.

## Overview

The iOS World model learns to predict UI transitions in two ways:

1. **Forward Dynamics**: Given a screen and an action, predict the next screen
2. **Inverse Dynamics**: Given two screens, predict what action caused the transition

## Architecture (Planned)

```
┌─────────────────┐
│  Before Screen  │──┐
│   (512x1024)    │  │
└─────────────────┘  │
                     ├──> Vision Encoder ──> Screen Embedding (512d)
┌─────────────────┐  │
│  After Screen   │──┘
│   (512x1024)    │
└─────────────────┘

┌─────────────────┐
│     Action      │───> Action Encoder ──> Action Embedding (128d)
│  (type + params)│
└─────────────────┘

Forward Model:  [Screen, Action] -> Predicted Screen
Inverse Model:  [Before, After]  -> Predicted Action
```

## Model Components

### Vision Encoder

**Options to explore**:
1. Pre-trained ResNet-50 or EfficientNet
2. CLIP vision encoder (best for multimodal)
3. Vision Transformer (ViT)

**Decision criteria**: Speed vs. accuracy tradeoff

### Action Encoder

Embed discrete action type + continuous parameters:
- Action type: One-hot or learned embedding
- Coordinates: Normalized (x, y)
- Text (if applicable): Simple bag-of-words or BERT

### Prediction Head

**Forward model output options**:
- Full screen prediction (harder, more informative)
- Diff prediction (easier, more focused)
- Embedding distance (fastest, less interpretable)

## Training

### Prerequisites

```bash
cd model
pip install -r requirements.txt
```

### Command

```bash
python train.py \
    --data-dir ../data/processed \
    --output-dir ../experiments/results/run_001 \
    --model-type forward \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-4
```

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Batch size | 32 | Adjust based on GPU memory |
| Learning rate | 1e-4 | Adam optimizer |
| Epochs | 50 | Early stopping on validation loss |
| Image size | 512x1024 | Set during preprocessing |

### Training Strategy

1. **Start with inverse dynamics**: Easier task, validates data quality
2. **Then train forward dynamics**: Main task for bug detection
3. **Joint training** (optional): Multi-task learning for both

### Logging

Training uses Weights & Biases (wandb) for tracking:
- Loss curves
- Sample predictions
- Validation metrics
- Hyperparameters

## Evaluation

```bash
python evaluate.py \
    --model-path experiments/results/run_001/best_model.pt \
    --data-dir ../data/processed
```

### Metrics

**For forward dynamics**:
- Image similarity (SSIM, MSE)
- Perceptual loss (LPIPS)
- Classification accuracy (if predicting diff type)

**For inverse dynamics**:
- Action classification accuracy
- Top-k accuracy
- Per-action-type breakdown

**For bug detection**:
- True positive rate (catches real bugs)
- False positive rate (flags valid UIs as bugs)
- Confidence calibration

## Inference

```python
from model import UIWorldModel
import torch
from PIL import Image

# Load model
model = UIWorldModel.load('best_model.pt')
model.eval()

# Load image
before_img = Image.open('screen_before.png')
before_tensor = preprocess(before_img)

# Define action
action = {
    'type': 'tap',
    'coordinates': {'x': 100, 'y': 200}
}

# Predict
with torch.no_grad():
    predicted_screen, confidence = model.predict_next_screen(
        before_tensor,
        action
    )

print(f"Confidence: {confidence:.2f}")
```

## Experiments

Track experiments in `experiments/` directory:

```
experiments/
├── README.md
└── results/
    ├── run_001_baseline/
    │   ├── config.json
    │   ├── best_model.pt
    │   ├── metrics.json
    │   └── samples/
    ├── run_002_larger_model/
    └── run_003_clip_encoder/
```

Each run should include:
- `config.json`: Hyperparameters and settings
- `best_model.pt`: Best checkpoint
- `metrics.json`: Final evaluation results
- `samples/`: Example predictions (images)

## Model Selection Criteria

Questions to answer during Phase 2:

1. **How much data is needed?** Plot learning curves to find minimum
2. **Which architecture works best?** Compare ResNet vs. CLIP vs. ViT
3. **Forward vs. inverse vs. joint?** Which training strategy is most effective?
4. **What image resolution?** Can we go smaller for speed?
5. **Does it generalize?** Test on unseen apps

## Next Steps (Phase 3)

Once the model is trained and evaluated:

1. Implement bug detection logic
2. Test on apps with intentionally injected bugs
3. Measure detection accuracy
4. Document findings

See `PROJECT_DIRECTION.md` for the complete roadmap.
