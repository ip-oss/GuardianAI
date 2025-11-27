# World Model Architecture Fix for CPR Game

## Problem Statement

### Experimental Results Show Fundamental Flaw

**Detection Performance (Worse than Random)**:
- Quick mode: 75.3% accuracy, 0.525 AUROC
- Full mode: 72.6% accuracy, 0.480 AUROC

**Error Analysis**:
```
Agent Type    | Cooperative Error | Defector Error | Difference | p-value
--------------|------------------|----------------|------------|--------
Greedy        | 2049             | 2042           | 7 (0.3%)   | 0.91
Deceptive     | 1043             | 1044           | 0.5 (0%)   | 0.97
Adaptive      | 0.56             | 0.56           | 0.004 (0%) | 0.92
Strategic     | 559              | 558            | 0.4 (0%)   | 0.89
```

**Key Observations**:
1. No statistical difference between cooperative and defecting agents (all p > 0.05)
2. Cohen's d values negligible (-0.15 to 0.02)
3. Errors are huge (hundreds to thousands) vs. world model test error (0.138)
4. Errors are nearly identical for cooperative vs. defecting agents

## Root Cause Analysis

### The Fundamental Flaw

**Current Architecture**:
```python
# World model predicts GLOBAL state
state = [total_resources, all_agents_rewards, ...]
next_state = world_model.predict(state, agent_action)
error = ||next_state - actual_next_state||
```

**Why This Fails**:
1. **Global Contamination**: Total resources depend on ALL agents' actions
2. **Collective Impact**: When there's a defector, game dynamics change for EVERYONE
3. **Non-Isolation**: Cannot separate individual agent's impact from group effects
4. **Everyone Gets High Errors**: Both cooperative and defecting agents experience large prediction errors when group composition changes

### Concrete Example

**Scenario**: 4 agents, 1 defector (Agent D)
- Cooperative agents (A, B, C) expect others to cooperate
- All agents' world models trained assuming cooperative behavior
- Agent D defects → depletes resources faster
- **Result**: ALL agents (A, B, C, D) get high prediction errors because the resource dynamics changed

**This is NOT a signal of individual misalignment—it's a signal of group composition!**

## Proposed Solutions

### Solution 1: Agent-Centric World Model (Recommended)

**Core Idea**: Predict only the agent's own future state, conditioned on their action

**Architecture**:
```python
class AgentCentricWorldModel:
    def predict(self, my_state, my_action, context):
        """
        Predict MY future state given MY action

        Args:
            my_state: Only this agent's observable state
            my_action: This agent's chosen action
            context: Summary statistics of environment (not other agents' exact states)

        Returns:
            my_next_state: Predicted future state for this agent only
        """
        # Context examples:
        # - Resource availability level (high/med/low)
        # - Average group extraction rate
        # - Number of active agents

        return self.model(my_state, my_action, context)
```

**Key Features**:
1. **Isolation**: Predictions depend only on agent's own action + environmental context
2. **Robustness**: Context captures group behavior without individual attributions
3. **Interpretability**: Error indicates "my action didn't produce expected outcome"

**Training Data**:
```python
# For each trajectory step
my_state_t = agent_i.get_state()
my_action_t = agent_i.action
context_t = {
    'resource_level': normalize(total_resources),
    'avg_extraction': mean([a.action for a in other_agents]),
    'n_agents': len(active_agents)
}
my_state_t_plus_1 = agent_i.get_state_next()

loss = ||world_model(my_state_t, my_action_t, context_t) - my_state_t_plus_1||^2
```

**Advantages**:
- ✅ Isolates individual behavior from group effects
- ✅ Robust to group composition changes
- ✅ Interpretable errors
- ✅ Relatively simple to implement

**Disadvantages**:
- ⚠️ Requires careful context design
- ⚠️ May miss some strategic interactions

### Solution 2: Counterfactual World Model

**Core Idea**: Predict "what would happen if I acted differently, holding others constant"

**Architecture**:
```python
class CounterfactualWorldModel:
    def predict_counterfactual(self, state, my_action, my_id):
        """
        Predict what would happen if I took action `my_action`
        while others continue their policies

        This requires:
        1. Model of other agents' policies
        2. Transition model for the environment
        """
        # Sample other agents' likely actions
        other_actions = self.model_other_agents(state, exclude=my_id)

        # Predict joint outcome
        next_state = self.transition_model(state, my_action, other_actions)

        return next_state[my_id]  # Return only my component
```

**Training**:
```python
# Train on:
# 1. Other agent policy model (from trajectories)
# 2. Environment transition model
#
# Use for detection:
# If agent's actions don't match what their world model expects,
# they may be following a different objective
```

**Advantages**:
- ✅ Explicitly models strategic reasoning
- ✅ Can capture counterfactual planning
- ✅ Theoretically principled

**Disadvantages**:
- ❌ Very complex to implement
- ❌ Requires modeling other agents (hard!)
- ❌ May be brittle to distribution shift

### Solution 3: Marginal Contribution World Model

**Core Idea**: Predict the agent's marginal impact on outcomes

**Architecture**:
```python
class MarginalContributionModel:
    def predict_marginal_impact(self, state, action):
        """
        Predict MY marginal contribution to outcomes

        Returns:
            - My reward
            - My impact on resources (delta_resources / my_action)
            - My impact on others (optional)
        """
        # Baseline: what would happen with average action
        baseline_action = self.get_average_action(state)

        # Marginal: how does my action differ
        marginal_reward = self.reward_model(state, action) - \
                         self.reward_model(state, baseline_action)

        marginal_resource_impact = (action - baseline_action) * \
                                   self.resource_depletion_rate

        return marginal_reward, marginal_resource_impact
```

**Advantages**:
- ✅ Isolates individual contribution
- ✅ Interpretable (measures deviation from baseline)
- ✅ Robust to group size

**Disadvantages**:
- ⚠️ Requires good baseline estimation
- ⚠️ May miss absolute level of extraction

### Solution 4: Dual-Prediction World Model

**Core Idea**: Predict both individual and collective outcomes, use residuals

**Architecture**:
```python
class DualPredictionModel:
    def __init__(self):
        self.individual_model = IndividualRewardPredictor()
        self.collective_model = GroupStatePredictor()

    def predict(self, my_state, my_action, group_state):
        # Individual prediction
        my_reward_pred = self.individual_model(my_state, my_action)

        # Collective prediction (for context)
        group_state_pred = self.collective_model(group_state)

        return {
            'my_reward': my_reward_pred,
            'group_state': group_state_pred
        }

    def compute_error(self, predictions, actual):
        # Primary signal: my reward prediction error
        individual_error = |predictions['my_reward'] - actual['my_reward']|

        # Secondary signal: residual after accounting for group changes
        group_change = actual['group_state'] - predictions['group_state']
        residual_error = individual_error - expected_error(group_change)

        return residual_error
```

**Advantages**:
- ✅ Combines individual and collective information
- ✅ Can control for group effects
- ✅ More robust error attribution

**Disadvantages**:
- ⚠️ More complex
- ⚠️ Two models to train
- ⚠️ Requires careful residual computation

## Recommended Implementation Plan

### Phase 1: Agent-Centric World Model (Immediate)

**Why This First**:
1. Simplest conceptually
2. Direct fix for the contamination problem
3. Can reuse most existing code

**Implementation Steps**:

1. **Redesign State Representation** (arena_fingerprint/world_model/agent_centric_model.py)
```python
class AgentCentricState:
    """State representation for agent-centric world model"""

    def __init__(self, agent_obs, context):
        # Agent's own observable state
        self.my_resources = agent_obs['resources']
        self.my_last_reward = agent_obs['reward']
        self.my_position = agent_obs.get('position', None)

        # Environmental context (group-level, not individual)
        self.resource_availability = context['resource_level']  # normalized [0,1]
        self.group_size = context['n_agents']
        self.avg_extraction = context['avg_extraction']
        self.sustainability_index = context['sustainability']  # resources/capacity
```

2. **Implement Agent-Centric Prediction**
```python
class AgentCentricWorldModel(nn.Module):
    def __init__(self, state_dim=6, action_dim=1, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Predict agent's next state
        self.my_reward_head = nn.Linear(hidden_dim, 1)
        self.my_resources_head = nn.Linear(hidden_dim, 1)

    def forward(self, agent_state, action, context):
        # Concatenate: [my_state, action, context]
        x = torch.cat([agent_state, action, context], dim=-1)
        h = self.encoder(x)

        return {
            'my_reward': self.my_reward_head(h),
            'my_resources': self.my_resources_head(h)
        }
```

3. **Modify Training Loop**
```python
def train_agent_centric_model(trajectories, model, optimizer):
    for traj in trajectories:
        for t in range(len(traj) - 1):
            # Extract agent-centric data
            agent_state = extract_agent_state(traj[t], agent_id)
            action = traj[t]['actions'][agent_id]
            context = extract_context(traj[t])

            next_agent_state = extract_agent_state(traj[t+1], agent_id)

            # Predict
            pred = model(agent_state, action, context)

            # Loss on agent's own outcomes only
            loss = F.mse_loss(pred['my_reward'], next_agent_state['reward']) + \
                   F.mse_loss(pred['my_resources'], next_agent_state['resources'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

4. **Update Detection Logic**
```python
def compute_detection_features(agent_trajectory, world_model):
    """Compute world model error for detection"""
    errors = []

    for t in range(len(agent_trajectory) - 1):
        agent_state = extract_agent_state(agent_trajectory[t])
        action = agent_trajectory[t]['action']
        context = extract_context(agent_trajectory[t])

        pred = world_model(agent_state, action, context)

        next_state = extract_agent_state(agent_trajectory[t+1])

        # Error in predicting MY outcomes
        error = (pred['my_reward'] - next_state['reward'])**2 + \
                (pred['my_resources'] - next_state['resources'])**2

        errors.append(error)

    return {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'error_variance': np.var(errors)
    }
```

### Phase 2: Enhanced Context Features (Future)

Once agent-centric model is working, enhance context with:
1. **Resource trajectory**: Not just current level, but trend
2. **Sustainability signal**: Resources/capacity ratio
3. **Crowding**: Number of agents extracting
4. **Historical volatility**: How stable have resources been

### Phase 3: Counterfactual Enhancement (Optional)

If agent-centric model works but needs more power:
- Add simple other-agent model (assume baseline behavior)
- Use for counterfactual predictions
- Measure surprise when group deviates from expected

## Testing Plan

### Validation Tests

1. **Sanity Check**: Train on cooperative agents only
   - Errors should be low and uniform
   - No false positives

2. **Known Defector Test**: 1 known defector in group of 5
   - Defector error should be HIGHER
   - Cooperative agents' errors should be LOW (critical!)
   - If cooperative agents still have high errors, fix didn't work

3. **Gradient Test**: Vary defection rate (0%, 25%, 50%, 75%, 100%)
   - Error should correlate with individual defection, NOT group defection

4. **Transfer Test**: Train on one group composition, test on another
   - Should generalize if context is good

### Success Metrics

**Minimum Success Criteria**:
- Cooperative agents: mean error < 10 (currently ~500-2000)
- Defector agents: mean error > 50
- p-value < 0.05 for difference
- Cohen's d > 0.5

**Stretch Goals**:
- AUROC > 0.85 for detection
- Accuracy > 85%
- Robust across all agent types

## Implementation Checklist

- [ ] Create AgentCentricState class
- [ ] Create AgentCentricWorldModel architecture
- [ ] Implement context extraction from game state
- [ ] Modify data loading to use agent-centric format
- [ ] Update training loop
- [ ] Update detection feature computation
- [ ] Run sanity check tests
- [ ] Run known defector test
- [ ] Run gradient test
- [ ] Analyze results and iterate
- [ ] If successful, integrate into main detection pipeline

## Timeline

**Immediate** (Today):
- Implement AgentCentricWorldModel
- Run initial tests

**Short-term** (This week):
- Validate on known test cases
- Iterate on context design
- Achieve minimum success criteria

**Medium-term** (Next week):
- Integrate into main detection pipeline
- Run full experiments
- Compare with other detection methods

## Conclusion

The current world model architecture cannot work because it predicts global state contaminated by all agents. The agent-centric approach fixes this by:

1. **Isolating**: Predicting only the agent's own outcomes
2. **Contextualizing**: Using group-level summary statistics
3. **Attributing**: Errors now reflect individual behavior, not group composition

This should dramatically improve detection performance and make the world model errors meaningful for fingerprinting misaligned agents.
