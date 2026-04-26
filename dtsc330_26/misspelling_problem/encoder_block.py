from typing import Any

import keras


class EncoderBlock(keras.layers.Layer):
    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1
    ):
        """Creates a new layer for the transformer encoder block. 
        Split into definition (__init__) and calling mechanism (call)
        
        Args:
            embed_dim (int): number of embeddings
            num_heads (int): number of parallel heads
            ff_dim (int): dimensions of the feedforward layers
            dropout (flot): fraction of neurons to turn off during 
                           each training run. Default: 0.1
        """
        super().__init__()
        
        self.attn = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )

        # This is the multilayer perceptron / fully
        # connected component of the transformer block.
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )

        # Layer normalization to mean of 0 and stdev of 1
        self.norm1 = keras.layers.LayerNormalization()
        self.norm2 = keras.layers.LayerNormalization()

        # Dropout allows for better learning
        self.drop1 = keras.layers.Dropout(dropout)
        self.drop2 = keras.layers.Dropout(dropout)

    def call(self, x: Any, training: bool = False, mask: Any | None = None) -> Any:
        """Call method defined by keras. This is used to define the forward pass

        Args: 
            x: previous layer
            training (bool):
            mask:

        Returns: 
            Any: the processing from the transformer.
        """
        attn_out = self.attn(x, x, attention_mask=mask)
        attn_out = self.drop1(attn_out, training=training)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.norm2(x + ffn_out)