# Turn-Level Credit Assignment for Multi-Turn GRPO

基于 MAPO (arxiv 2603.06194) 的混合优势估计思路，对当前 GRPO 算法进行改进，实现 turn 级别的信用分配。

---

## 1. 当前问题

当前 GRPO 实现（`core_algos.py:265-345`）：

```python
# 整条 trajectory 求和得到一个标量 reward
scores = token_level_rewards.sum(dim=-1)           # (bs,)
# group 内 z-score 归一化
advantage = (score - group_mean) / (group_std + eps)  # scalar
# 同一个标量广播给所有 token
token_level_advantages = advantage.unsqueeze(-1) * response_mask
```

**缺陷**：一条 6 轮对话中，第 1 轮 Grounding 正确、第 2 轮 ImageEdit 成功、第 3 轮没反思就 Terminate——所有 token 拿到相同的 advantage，模型无法学到"第 3 轮需要改进"。

---

## 2. 改进目标

- 每轮 (turn) 分配不同的 advantage，让梯度信号精确到轮次
- 好的轮次（正确调用工具、视觉反思）获得正 advantage
- 差的轮次（盲目 Terminate、工具调用失败）获得负 advantage
- 不引入额外 critic 网络，保持 GRPO 的 critic-free 特性

---

## 3. 算法设计

### 3.1 Per-Turn Reward 构造

每条 trajectory 有 T 轮 assistant turn，为每轮分配一个即时 reward `r_t`：

| Turn 类型 | Reward 来源 | 示例值 |
|-----------|------------|--------|
| 工具调用成功 | `tool_rewards[t]` | 0.0 |
| 工具调用失败 | `tool_rewards[t]` | -0.1 |
| 最终轮 (Terminate) | `judge_score` + `reflection_bonus` | 0.0 ~ 1.0 |
| 含视觉反思的轮次 | 规则 bonus | +0.05 |
| 含自我修正的轮次 | 规则 bonus | +0.05 |

**构造逻辑**：

```python
def build_per_turn_rewards(judge_score, tool_rewards, reflection_signals, n_turns):
    """
    Args:
        judge_score: float, 最终 judge 评分 [0, 1]
        tool_rewards: list[float], 每次工具调用的 reward (0.0=success, <0=error)
        reflection_signals: list[dict], 每轮的反思检测结果
            e.g. [{"has_observe": True, "has_self_correct": False}, ...]
        n_turns: int, assistant turn 数
    Returns:
        per_turn_rewards: list[float], 每轮的即时 reward
    """
    per_turn_rewards = []

    for t in range(n_turns):
        r_t = 0.0

        # 1) 工具执行 reward（如果这一轮有工具调用）
        if t < len(tool_rewards):
            r_t += tool_rewards[t]  # 0.0 or negative

        # 2) 反思 bonus
        if t < len(reflection_signals):
            sig = reflection_signals[t]
            if sig.get("has_observe"):
                r_t += 0.05
            if sig.get("has_self_correct"):
                r_t += 0.05

        # 3) 最终轮追加 judge score
        if t == n_turns - 1:
            r_t += judge_score

        per_turn_rewards.append(r_t)

    return per_turn_rewards
```

### 3.2 Monte Carlo Return

从后往前计算折扣累积回报，让每轮都能"看到"后续轮次的好坏：

```
R_T = r_T
R_t = r_t + γ · R_{t+1}     (γ = 0.99)
```

```python
def compute_mc_returns(per_turn_rewards, gamma=0.99):
    T = len(per_turn_rewards)
    returns = [0.0] * T
    returns[-1] = per_turn_rewards[-1]
    for t in range(T - 2, -1, -1):
        returns[t] = per_turn_rewards[t] + gamma * returns[t + 1]
    return returns
```

**示例**（3 轮 trajectory, γ=0.99）：

```
r = [0.0, -0.1, 0.7]   # 第1轮工具成功, 第2轮工具失败, 第3轮judge=0.7
R = [0.0 + 0.99*(−0.1 + 0.99*0.7),  −0.1 + 0.99*0.7,  0.7]
  = [0.587,  0.593,  0.7]
```

第 2 轮虽然工具失败 (-0.1)，但后续 judge 给了高分，所以 R₂ 仍然为正——说明整体 trajectory 还是好的，但第 2 轮的 advantage 会比第 1、3 轮低。

### 3.3 Turn-Level Advantage（长程信号）

在 group 内（同一 prompt 的 n 个 rollout），对同一轮位置的 MC return 做归一化：

```
A_turn(t, i) = (R_t^(i) - μ_t) / (σ_t + ε)
```

其中 `μ_t, σ_t` 是第 t 轮的 MC return 在 group 内的均值和标准差。

**注意**：不同 rollout 的轮数可能不同（有的 3 轮结束，有的 6 轮结束）。只对 `t ≤ T_min`（group 内最短 trajectory 的长度）的轮次计算 turn-level advantage，超出部分设为 0。

```python
def compute_turn_level_advantage(mc_returns_group, epsilon=1e-6):
    """
    Args:
        mc_returns_group: list[list[float]], shape (n_rollouts, variable_turns)
    Returns:
        turn_advantages: list[list[float]], same shape
    """
    n = len(mc_returns_group)
    T_min = min(len(r) for r in mc_returns_group)

    turn_advantages = [[0.0] * len(r) for r in mc_returns_group]

    for t in range(T_min):
        values = [mc_returns_group[i][t] for i in range(n)]
        mu = sum(values) / n
        std = (sum((v - mu) ** 2 for v in values) / n) ** 0.5

        for i in range(n):
            turn_advantages[i][t] = (mc_returns_group[i][t] - mu) / (std + epsilon)

    return turn_advantages
```

### 3.4 Batch-Level Advantage（局部信号）

对整个 batch 内所有轮次的即时 reward 做归一化，提供一个全局的"这一轮的绝对好坏"信号：

```
A_batch(t, i) = (r_t^(i) - μ_batch) / (σ_batch + ε)
```

其中 `μ_batch, σ_batch` 是 batch 内所有 `r_t` 的均值和标准差。

```python
def compute_batch_level_advantage(all_per_turn_rewards, epsilon=1e-6):
    """
    Args:
        all_per_turn_rewards: list[list[float]], all trajectories' per-turn rewards
    Returns:
        batch_advantages: list[list[float]], same shape
    """
    # Flatten all immediate rewards
    all_r = [r for traj in all_per_turn_rewards for r in traj]
    mu = sum(all_r) / len(all_r)
    std = (sum((r - mu) ** 2 for r in all_r) / len(all_r)) ** 0.5

    batch_advantages = []
    for traj in all_per_turn_rewards:
        adv = [(r - mu) / (std + epsilon) for r in traj]
        batch_advantages.append(adv)

    return batch_advantages
```

### 3.5 混合 Advantage

```
A(t, i) = α · A_turn(t, i) + β · A_batch(t, i)
```

推荐 `α = β = 0.5`（论文中的最优配置）。

```python
def compute_mixed_advantage(turn_adv, batch_adv, alpha=0.5):
    mixed = []
    for i in range(len(turn_adv)):
        traj_adv = []
        for t in range(len(turn_adv[i])):
            a = alpha * turn_adv[i][t] + (1 - alpha) * batch_adv[i][t]
            traj_adv.append(a)
        mixed.append(traj_adv)
    return mixed
```

### 3.6 Token-Level 广播

将每轮的 advantage 广播给该轮中所有 token（而非全局广播一个标量）：

```python
def broadcast_turn_advantages_to_tokens(mixed_advantages, turn_boundaries, response_length):
    """
    Args:
        mixed_advantages: list[float], per-turn advantage for one trajectory
        turn_boundaries: list[tuple(start, end)], token 位置区间 for each turn
        response_length: int, total response token length
    Returns:
        token_advantages: tensor of shape (response_length,)
    """
    token_advantages = torch.zeros(response_length)
    for t, (start, end) in enumerate(turn_boundaries):
        if t < len(mixed_advantages):
            token_advantages[start:end] = mixed_advantages[t]
    return token_advantages
```

---

## 4. 数据流变化

### 4.1 reward 函数改动（`multiturn_reward.py`）

`compute_score` 需要额外返回 per-turn 信息：

```python
def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # ... existing logic ...

    return {
        "score": score,                      # 保留原有标量（兼容）
        "judge_score": judge_score,
        "error_penalty": error_penalty,
        "error_count": error_count,
        "reflection_bonus": reflection_bonus,
        "api_failed": api_failed,
        # ---- 新增 ----
        "per_turn_rewards": per_turn_rewards, # list[float], 每轮即时 reward
        "n_turns": n_turns,                   # int, assistant turn 数
    }
```

### 4.2 advantage 计算改动（`core_algos.py`）

新增 `compute_grpo_turn_level_advantage` 函数：

```python
def compute_grpo_turn_level_advantage(
    token_level_rewards: torch.Tensor,    # (bs, response_length)
    response_mask: torch.Tensor,          # (bs, response_length)
    index: np.ndarray,                    # group IDs
    per_turn_rewards: list[list[float]],  # (bs, variable_turns)
    turn_boundaries: list[list[tuple]],   # (bs, variable_turns) token ranges
    gamma: float = 0.99,
    alpha: float = 0.5,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn-level mixed advantage (MAPO-style).

    Returns:
        advantages: (bs, response_length), per-token advantages
        returns: (bs, response_length), per-token returns (for logging)
    """
    bsz, seq_len = token_level_rewards.shape

    # Step 1: MC returns per trajectory
    mc_returns_all = []
    for i in range(bsz):
        mc = compute_mc_returns(per_turn_rewards[i], gamma)
        mc_returns_all.append(mc)

    # Step 2: Group by prompt, compute turn-level advantage
    groups = defaultdict(list)
    for i in range(bsz):
        groups[index[i]].append(i)

    turn_adv = [None] * bsz
    for gid, members in groups.items():
        mc_group = [mc_returns_all[i] for i in members]
        tadv = compute_turn_level_advantage(mc_group, epsilon)
        for j, i in enumerate(members):
            turn_adv[i] = tadv[j]

    # Step 3: Batch-level advantage
    batch_adv = compute_batch_level_advantage(per_turn_rewards, epsilon)

    # Step 4: Mix
    mixed_adv = compute_mixed_advantage(turn_adv, batch_adv, alpha)

    # Step 5: Broadcast to tokens
    advantages = torch.zeros(bsz, seq_len, device=token_level_rewards.device)
    for i in range(bsz):
        advantages[i] = broadcast_turn_advantages_to_tokens(
            mixed_adv[i], turn_boundaries[i], seq_len
        )
    advantages = advantages * response_mask

    return advantages, advantages
```

### 4.3 trainer 调度改动（`ray_trainer.py`）

在 `compute_advantage` 中新增分支：

```python
if config.algorithm.adv_estimator == "grpo_turn_level":
    advantages, returns = compute_grpo_turn_level_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=batch.batch["response_mask"],
        index=batch.non_tensor_batch["uid"],
        per_turn_rewards=batch.non_tensor_batch["per_turn_rewards"],
        turn_boundaries=batch.non_tensor_batch["turn_boundaries"],
        gamma=config.algorithm.get("gamma", 0.99),
        alpha=config.algorithm.get("turn_level_alpha", 0.5),
    )
```

---

## 5. 需要传递的新字段

agent loop / reward manager 需要额外向 batch 中注入：

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `per_turn_rewards` | `list[float]` | reward 函数 | 每轮即时 reward |
| `turn_boundaries` | `list[tuple(int,int)]` | agent loop | 每轮 token 的 (start, end) 位置 |
| `n_turns` | `int` | agent loop | assistant turn 数 |

`turn_boundaries` 在 `tool_agent_loop.py` 中已有类似信息（`response_mask` 的分段），需要显式记录每轮的 token 区间。

---

## 6. 训练脚本配置

```bash
# run_qwen3vl-8b_rl10k_test.sh 新增/修改的参数
algorithm.adv_estimator=grpo_turn_level \
algorithm.gamma=0.99 \
algorithm.turn_level_alpha=0.5 \
```

---

## 7. 预期效果

| 维度 | 当前 GRPO | Turn-Level GRPO |
|------|-----------|-----------------|
| 信用分配粒度 | trajectory 级 | turn 级 |
| 好轮 vs 坏轮 | 同一 advantage | 不同 advantage |
| 反思行为学习 | 弱（奖惩不分轮次） | 强（反思轮获正信号） |
| 工具失败惩罚 | 间接（error_penalty扣总分） | 直接（失败轮获负advantage） |
| 额外开销 | 无 | 需要 per-turn reward + turn boundaries |

### 示例对比

一条 trajectory: `Grounding(成功) → ImageEdit(成功) → 没反思直接Terminate`

**当前 GRPO**: judge 给 0.6 分 → 3 轮所有 token 共享同一个 advantage

**Turn-Level GRPO**:
- Turn 1 (Grounding): r₁=0.0, R₁=0.587 → 正 advantage
- Turn 2 (ImageEdit): r₂=0.0, R₂=0.594 → 正 advantage
- Turn 3 (Terminate): r₃=0.6 (judge), 但没反思所以无 bonus → advantage 相对低

模型学到：前两轮是对的，但第三轮应该先看图反思再决定是否 Terminate。

---

## 8. 实现优先级

1. **第一步**：在 `tool_agent_loop.py` 中记录 `turn_boundaries`
2. **第二步**：修改 `compute_score` 返回 `per_turn_rewards`
3. **第三步**：在 `core_algos.py` 中实现 `compute_grpo_turn_level_advantage`
4. **第四步**：在 `ray_trainer.py` 中接入新的 advantage 计算
5. **第五步**：训练脚本切换 `algorithm.adv_estimator=grpo_turn_level`
