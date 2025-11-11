from src.utils import QFilters
from transformers.cache_utils import Cache
from typing import Any, Dict, List, Optional, Tuple, Union
import torch



class KNormCache(Cache):
    """
    An implementation of KNorm filtering in transformers' KV cache framework.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the short context window.
        max_length (`int`):
            The maximum cache length.
    """

    is_sliding = False

    def __init__(self, max_length:int, window_length: int) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.max_length = max_length
        self.window_length = window_length

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object, in case of SinkCache it is the window length."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]

        if key_cache.shape[2] < self.max_length:
            return key_cache, value_cache

        key_length = key_cache.shape[-2]
        

        comp_key, window_key = key_cache[..., :key_length-self.window_length, :], key_cache[..., key_length-self.window_length:,:]
        comp_value, window_value = value_cache[..., :key_length-self.window_length, :], value_cache[..., key_length-self.window_length:,:]

        proj_key = comp_key.norm(dim=-1)
        key_idx = (-proj_key).topk(self.max_length - self.window_length, -1).indices.sort().values
        key_idx = key_idx[..., None].repeat(1, 1, 1, key_cache.shape[-1])

        comp_key = torch.gather(comp_key, -2, key_idx)
        comp_value = torch.gather(comp_value, -2, key_idx)

        self.key_cache[layer_idx] = torch.cat((comp_key, window_key), -2)
        self.value_cache[layer_idx] = torch.cat((comp_value, window_value), -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]



class QFiltersCache(Cache):
    """
    An implementation of Q-Filters in transformers' KV cache framework.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the short context window.
        max_length (`int`):
            The maximum cache length.
    """

    is_sliding = False

    def __init__(self, max_length:int, window_length: int, model_name: str) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.max_length = max_length
        self.window_length = window_length

        model_suffix = model_name.split("/")[-1]
        self.q_filters = QFilters.from_pretrained(f"nthngdy/{model_suffix}_qfilt").q_filters

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object, in case of SinkCache it is the window length."""
        return self.window_length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The following arguments can be used in `SinkCache`: `sin`,
                `cos` and `partial_rotation_size`. These arguments are used with models using RoPE, to recompute the
                rotation as the tokens are shifted.

        Return:
            A tuple containing the updated key and value states.
        """
        # [bsz, num_heads, seq_len, head_dim]
        if len(self.key_cache) <= layer_idx:
            # Empty cache
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]

        if key_cache.shape[2] < self.max_length:
            return key_cache, value_cache
        
        current_filter = self.q_filters[layer_idx].to(key_cache.device)

        key_length = key_cache.shape[-2]
        

        comp_key, window_key = key_cache[..., :key_length-self.window_length, :], key_cache[..., key_length-self.window_length:,:]
        comp_value, window_value = value_cache[..., :key_length-self.window_length, :], value_cache[..., key_length-self.window_length:,:]

        proj_key = (comp_key * current_filter[None, :, None]).sum(-1)
        raise ValueError(f"PROJ KEY {proj_key.shape}, {proj_key}")
        key_idx = (-proj_key).topk(self.max_length - self.window_length, -1).indices.sort().values
        key_idx = key_idx[..., None].repeat(1, 1, 1, key_cache.shape[-1])

        comp_key = torch.gather(comp_key, -2, key_idx)
        comp_value = torch.gather(comp_value, -2, key_idx)

        self.key_cache[layer_idx] = torch.cat((comp_key, window_key), -2)
        self.value_cache[layer_idx] = torch.cat((comp_value, window_value), -2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
