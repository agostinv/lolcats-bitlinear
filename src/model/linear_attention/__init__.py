"""
Linear and linear attention + sliding window classes
"""

from .linear_attention import LolcatsLinearAttention, LinearAttentionState
from .linear_window_attention_tk import (
    LolcatsTKWindowAttention,
    LinearAttentionTKWindowCache,
)

# from .linear_window_attention_sw import (
#     LolcatsSlidingWindowAttention, LinearAttentionSlidingWindowCache
# )
# Experimental chunk linear attentions
from .linear_window_attention_tk_long import (
    LolcatsTKWindowLongAttention,
)

# from .linear_window_attention_sw_long import (
#     LolcatsSlidingWindowLongAttention,
# )
from .linear_window_attention_tk_gen import (
    LolcatsWindowAttentionTKGen,
    LinearAttentionTKWindowGenerationCache,
)
from .linear_window_attention_topk_linear import (
    LolcatsLinearSlidingWindowTopk,
    LinearAttentionSlidingWindowTopkCache,
)
from .linear_window_attention_topk_blockwise import (
    LolcatsLinearSlidingWindowTopkBlockwise,
    LinearAttentionSlidingWindowTopkBlockwiseCache,
)
from .linear_window_attention_random_mask import (
    LolcatsLinearSlidingWindowRandomMask,
    LinearAttentionSlidingWindowRandomMaskCache,
)
from .linear_window_attention_topk_random import (
    LolcatsLinearSlidingWindowTopkRandom,
    LinearAttentionSlidingWindowTopkRandomCache,
)
from .linear_window_attention_random_mask_no_window_factor_or_max import (
    LolcatsLinearSlidingWindowRandomMaskNoWindowFactorOrMax,
    LinearAttentionSlidingWindowRandomMaskNoWindowFactorOrMaxCache,
)
from .linear_hybrid_sparse_eviction_attention import (
    LolcatsLinearHybridSparseEviction,
    LinearAttentionHybridSparseEvictionCache,
)
