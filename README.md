# npneural
* This repo is not actively maintained, and may have bugs!
* This was an attempt to create modular, object-oriented neural networks using numpy.
* Run standalone.ipynb for the condensed ideas, or
* For a quick glimpse, compare example1.py and example2.py
* Check tests for import handling and quick implementations of network models to predict:
 >* XOR circuit
 >* MNIST handwritten digits
 >* 2D multiclass spiral data classification
 >* 2D multiclass vertical data classification

## Dependencies
* **numpy** - Our workhorse, needed for the base implementation.
* **matplotlib** - Plot and analyze predictions.
* **seaborn** - Beautify and simplify plotting.
* **scikit-learn** - Used for testing and pre-processing.

## References
* Intuition & Basic Math
 >* https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
 >* https://www.youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3
 >* https://www.youtube.com/watch?v=pauPCy_s0Ok
 >* https://www.pinecone.io/learn/softmax-activation/
 >* https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
* Softmax Cross Entropy Explosion
 >* https://datascience.stackexchange.com/a/58770
 >* https://stackoverflow.com/q/48600374/
 >* https://stackoverflow.com/q/49016723/
* CS231n - Stanford (data and plotting)
 >* https://cs231n.github.io/
 >* https://cs231n.github.io/neural-networks-case-study/
 >* https://cs.stanford.edu/people/karpathy/cs231nfiles/minimal_net.html
* Optimizations
 >* https://github.com/pseeth/autoclip
 >* https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
 >* https://datascience.stackexchange.com/q/20139

## Debugging Notes
* Method: Handling numpy warnings as errors
 >* olderr = numpy.seterr(all='raise')
 >* exception handling
 >* olderr = numpy.seterr(\*\*olderr)
* Checks:
 >* Check if XOR circuit works well, after each tweak
 >* Overflow, Underflow in softmax exponentiation
 >* Vanishing, Exploding of gradients and predictions
* Optimizations:
 >* Scaling or TanH to solve explosion
 >* Weights are initialized by Glorot's method
 >* ReLU, TanH handle vanishing gradients efficiently
 >* AutoClip gradient clipping for explosions (algorithm only)
* MSE was optimized to work only with 2D arrays. Results:
 >* Accuracy with ReLU, TanH activations almost doubled
 >* Accuracy with (Sigmoid→Softmax) activation is horrible
 >* Cross Entropy implementation has been merged with MSE
 >* MSE has been set as the default loss function in Network
* Suggestions for my implementation
 >* To disable gradient clipping, pass clip_p=100 in Network.fit()
 >* Best Practice → Scale Outputs → Sigmoid → SoftMax
 >* Replacing (Sigmoid→SoftMax) with (TanH→ReLU) boosts accuracy
 >* Redundant (StandardScaler→TanH) gives mild boost with complex data
