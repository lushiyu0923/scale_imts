from __future__ import annotations

"""Backbone loading with optional local pretrained weights.

The original experiments used custom ``*_wope`` Transformers classes.  This
repository keeps the experiment reproducible without shipping model weights or
patching the installed Transformers package: custom classes are used when
available, otherwise the public BERT/GPT-2 implementations are used.
"""

from pathlib import Path


def _load_local_or_random(model_cls, config_cls, path: Path, hidden_dim: int, n_layers: int, kind: str):
    if path and (path / "config.json").exists():
        model = model_cls.from_pretrained(str(path), local_files_only=True)
    else:
        if kind == "gpt":
            heads = max(1, min(12, hidden_dim // 64))
            while hidden_dim % heads != 0 and heads > 1:
                heads -= 1
            config = config_cls(
                n_embd=hidden_dim,
                n_layer=max(1, n_layers),
                n_head=heads,
                n_positions=2048,
                n_ctx=2048,
            )
        else:
            heads = max(1, min(12, hidden_dim // 64))
            while hidden_dim % heads != 0 and heads > 1:
                heads -= 1
            config = config_cls(
                hidden_size=hidden_dim,
                num_hidden_layers=max(1, n_layers),
                num_attention_heads=heads,
                intermediate_size=max(4 * hidden_dim, 64),
            )
        model = model_cls(config)
    return model


def build_gpt_backbone(hidden_dim: int, n_layers: int, pretrained_path: Path):
    try:
        from transformers.models.gpt2.modeling_gpt2_wope import GPT2Model_wope as model_cls
        from transformers.models.gpt2.configuration_gpt2 import GPT2Config as config_cls
    except ImportError:
        from transformers import GPT2Config as config_cls
        from transformers import GPT2Model as model_cls

    model = _load_local_or_random(model_cls, config_cls, pretrained_path, hidden_dim, n_layers, "gpt")
    if hasattr(model, "h"):
        model.h = model.h[:max(1, n_layers)]
    actual_dim = getattr(model.config, "n_embd", None)
    if actual_dim != hidden_dim:
        raise ValueError(f"GPT-2 hidden size {actual_dim} does not match hidden_dim={hidden_dim}.")
    return model


def build_bert_backbone(hidden_dim: int, n_layers: int, pretrained_path: Path):
    try:
        from transformers.models.bert.modeling_bert_wope import BertModel_wope as model_cls
        from transformers.models.bert.configuration_bert import BertConfig as config_cls
    except ImportError:
        from transformers import BertConfig as config_cls
        from transformers import BertModel as model_cls

    model = _load_local_or_random(model_cls, config_cls, pretrained_path, hidden_dim, n_layers, "bert")
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        encoder.layer = encoder.layer[:max(1, n_layers)]
    actual_dim = getattr(model.config, "hidden_size", None)
    if actual_dim != hidden_dim:
        raise ValueError(f"BERT hidden size {actual_dim} does not match hidden_dim={hidden_dim}.")
    return model
