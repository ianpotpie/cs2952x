1. Create encoder-decoder model. In the latent space, fix some dimensions to represent transformation parameters: scale, rotation, shift. Force the other dimensions to be invariant to transformations (by taking the difference of these features between transformations of the same image and performing gradient desent)

> I don't think this is very different from what we saw in class.

2. Take the utop transformation of some data and use it as the target for a repression model. This should create a smooth representation of the utop transformation. The key assumption here is examples whose raw feature values are similar (have a close distance) will also be close in the embedding space. We hope that an embedding space that preserves these properties will encode useful information about the examples.
