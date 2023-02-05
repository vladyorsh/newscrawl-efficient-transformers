# H-Transformer-1D Model
This repository contains an implementation of the H-Transformer-1D. Based on [this](https://github.com/lucidrains/h-transformer-1d) implementation, it additionally provides some bugfixes and an implementation of the Cross-Attention mechanism through the rectangular blocks to handle the inequality of query and key lengths. Note that this requires both a spatial correspondence between queries and keys (e.g. summarization or translation tasks) and an output length known beforehand.
## Usage
The model is intended to be pretrained on the News Crawl dataset as a whole, but if you want to use the H-Attention module you can import it from the `modeling.modules` file and instantiate as the following:

    att = HAttention1D(
	    config.hidden_dim, config.qkv_dim, config.num_heads,
	    causal=not self.is_encoder, block_size = config.Nr, eps=config.eps
    )
