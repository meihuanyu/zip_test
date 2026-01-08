import math
from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class Hypothesis:
    """
    用于 CTC 对齐的单条路径。

    token_index:
        当前对齐到了文章的第几个 Token（可以是 BPE / Char / Phoneme，下标从 0 开始）
    score:
        到当前帧为止的路径累积得分（对数概率，越大越好，通常是负数）
    state:
        CTC 状态机标志
        0 = Blank State  (上一帧是 blank，或者刚发生状态转移)
        1 = Token State  (上一帧是非 blank，且保持在当前 token)
    """

    token_index: int
    score: float
    state: int  # 0 = blank, 1 = token


@dataclass
class AlignConfig:
    """
    Beam Search 对齐的超参数。
    """

    beam_width: int = 20
    prune_threshold: float = 50.0
    skip_penalty: float = -3.0
    window_size: int = 100


def _log_sum_exp(a: float, b: float) -> float:
    """
    对标量 a, b 计算 log(exp(a) + exp(b))，避免数值下溢。
    """
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

class StreamingCTCAligner:
    """
    最简流式 CTC 对齐器。

    使用方式：
      1. 初始化：
           aligner = StreamingCTCAligner(target_tokens, blank_id, config)
      2. 对于每一帧 log_probs_t (形状 (V,) 或 (1, V))：
           idx = aligner.step(log_probs_t)
         其中 idx 为当前估计的 token_index。
    """

    def __init__(
        self,
        target_tokens: torch.Tensor,
        blank_id: int,
        config: Optional[AlignConfig] = None,
    ) -> None:
        if config is None:
            config = AlignConfig()

        assert target_tokens.ndim == 1, "target_tokens 应为一维 (L,)"

        self.target_tokens = target_tokens
        self.blank_id = blank_id
        self.config = config
        self.L = int(target_tokens.numel())

        # beam 初始状态
        self.beam: List[Hypothesis] = [
            Hypothesis(token_index=0, score=0.0, state=0),
        ]
        self.max_token_index: int = 0

    def step(self, log_probs_t: torch.Tensor) -> int:
        """
        处理单帧 CTC log 概率，更新内部 beam，返回当前最优 token_index。

        参数
        ----
        log_probs_t:
            形状为 (V,) 或 (1, V) 的张量，对应当前一帧的 log_softmax 输出。
        """
        if self.L == 0:
            return 0

        if log_probs_t.ndim == 2:
            # 假设形状为 (1, V)
            log_probs_t = log_probs_t[0]

        frame = log_probs_t  # (V,)
        log_p_blank = float(frame[self.blank_id].item())

        token_logp = frame[self.target_tokens]  # (L,)

        next_beam_candidates: List[Hypothesis] = []

        max_token_index = self.max_token_index

        for hyp in self.beam:
            i = hyp.token_index
            base_score = hyp.score

            # 1) stay-blank
            stay_blank_score = base_score + log_p_blank
            next_beam_candidates.append(
                Hypothesis(token_index=i, score=stay_blank_score, state=0)
            )

            # 2) stay-token
            if i < self.L:
                log_p_token_i = float(token_logp[i].item())
                stay_token_score = base_score + log_p_token_i
                next_beam_candidates.append(
                    Hypothesis(token_index=i, score=stay_token_score, state=1)
                )

            # 3) skip 到下一个 token
            next_i = i + 1
            if next_i < self.L and next_i <= max_token_index + self.config.window_size:
                log_p_next = float(token_logp[next_i].item())
                skip_score = base_score + log_p_next + self.config.skip_penalty
                next_beam_candidates.append(
                    Hypothesis(token_index=next_i, score=skip_score, state=1)
                )

        merged = {}
        for hyp in next_beam_candidates:
            key = (hyp.token_index, hyp.state)
            if key in merged:
                merged[key].score = _log_sum_exp(merged[key].score, hyp.score)
            else:
                merged[key] = hyp

        merged_beam = list(merged.values())
        if not merged_beam:
            merged_beam = [Hypothesis(token_index=0, score=0.0, state=0)]

        best_score = max(h.score for h in merged_beam)
        pruned = [
            h for h in merged_beam if h.score >= best_score - self.config.prune_threshold
        ]
        if not pruned:
            pruned = merged_beam

        pruned.sort(key=lambda h: h.score, reverse=True)
        self.beam = pruned[: self.config.beam_width]

        # 更新 max_token_index
        self.max_token_index = max(h.token_index for h in self.beam)

        best_hyp = max(self.beam, key=lambda h: h.score)
        return max(0, min(best_hyp.token_index, self.L - 1))


__all__ = [
    "Hypothesis",
    "AlignConfig",
    "StreamingCTCAligner",
]


