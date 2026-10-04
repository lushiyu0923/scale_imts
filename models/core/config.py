from __future__ import annotations

"""ScaleIMTS 相关的配置和中间数据结构定义。"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ScaleIMTSConfig:
    """双路径模型配置。

    这里同时描述：
    - foundation 路的 backbone 配置；
    - irregular 路的 patch / 图建模配置；
    - 公共的输入输出维度与训练超参数。
    """

    enc_in: int
    c_out: int
    seq_len_max_irr: int
    pred_len_max_irr: int
    features: str = "M"
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.1
    use_time_features: bool = True
    foundation_te_model: str = "gpt"
    foundation_st_model: str = "bert"
    foundation_n_te_layers: int = 2
    foundation_n_st_layers: int = 2
    foundation_semi_freeze: bool = False
    foundation_pretrained_root: str = ""
    patch_len: int = 12
    max_time_steps: int = 200
    irregular_hidden_dim: int = 64
    irregular_time_dim: int = 16
    irregular_node_dim: int = 16
    foundation_reduced_dim: int = 128
    foundation_gate_hidden_dim: int = 256
    fusion_type: str = "film_i2f_var"
    fusion_hidden_dim: int = 128
    film_hidden_dim: int = 256
    foundation_scale_mode: str = "none"
    foundation_scale_num_segments: int = 4
    foundation_scale_num_experts: int = 3
    foundation_scale_router_hidden_dim: int = 256
    foundation_scale_expert_hidden_dim: int = 256
    foundation_scale_temperature: float = 1.0
    foundation_scale_use_future_context: bool = True
    foundation_scale_use_irregular_context: bool = True
    foundation_train_last_n_layers: int = 0
    ablation_variant: str = "full"


@dataclass
class BranchAlignedBatch:
    """某一条分支对齐后的输入。

    foundation 路和 irregular 路都会复用这个容器，但只填自己需要的字段。
    """

    values: Tensor
    time: Tensor
    mask: Tensor
    tokens: Tensor | None = None
    token_mask: Tensor | None = None
    variable_prompts: Tensor | None = None
    patch_values: Tensor | None = None
    patch_times: Tensor | None = None
    patch_masks: Tensor | None = None
    patch_valid: Tensor | None = None


@dataclass
class DualBranchAlignedBatch:
    """两条 aligner 的输出组合。"""

    foundation: BranchAlignedBatch
    irregular: BranchAlignedBatch


@dataclass
class BranchFeatures:
    """某一条分支在融合前输出的表征。"""

    sequence_features: Tensor
    global_features: Tensor
    sequence_mask: Tensor | None = None
    routing_weights: Tensor | None = None


@dataclass
class ScaleRoutingContext:
    """Optional context for scale-aware foundation routing."""

    future_queries: Tensor | None = None
    irregular_global_features: Tensor | None = None
