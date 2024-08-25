*Work done during an internship at MILES, Paris Dauphine University, under the supervision of Yann Chevaleyre and Muni Sreenivas Pydi.*

This is a codebase to replicate the results of the [blog-post](https://www.lesswrong.com/collaborateOnPost?postId=mu2iJxZWszkKSGgxf&key=ebde77264286ccaadec55ba649f7df). It allows us to visualize small attention-only Transformers with embedding dimension 3. 
* models.py contains the architecture of the Attention only Transformer,
* train.py is used to train the Transformer normally or using head-boosting (training heads individually),
* interp.py contains the main functions to visualize the Transformer,

Use the notebook to see how to use the codebase, and replicate the results. You can read the blog-post below. 

# Visualizing small Attention-only Transformers

Research has indicated that in large Transformers, facts are primarily stored in the MLP layer rather than the attention layer. However, it's worth exploring whether the attention layer also plays a role in memorizing some part of the data. **Can an attention layer memorize information, and if so, how?**

In this blog-post, we define the memorization task as predicting the correct next token for a pair of input tokens. Our goal was to determine if the Transformer exhibits any structure that supports memorization for this task. For instance, it is conceivable that attention heads could distribute the workload, with each head remembering specific associations.

Ultimately, we were unable to identify any clear structure within the Transformer that would allow us to propose a definitive algorithm for how memorization occurs. The internal workings are too complex, and the basis of the attention heads does not reveal any discernible structure. While this could be due to the inherent lack of structure in the data itself [1], it might also suggest that the optimal configuration of the Transformer is inherently "messy". Therefore, we aim to present this complexity and how it can be broken down into smaller, more understandable components. Even if the entire system remains opaque, some sub-steps can be comprehended relatively well.

## The Transformer architecture

We will use a standard architecture for the Transformer, but in a very simplified form. The dictionary size is *N*, and the sequence length is *S*. Each token is embedded in a *d*-dimensional vector space and added to its positional embedding. These vectors are then passed through a layer normalization followed by an attention module with *H* heads, each of dimension *d_h*. The outputs of the heads are summed back into the residual stream and projected to logits through an unembedding matrix [2].

Here is the equation for a one-layer Transformer predicting only the last token:

$$
T(t_{1:S}) = W_U\left(\sum_{h=1}^HA_h(t_{1:S}) + e(t_S)+pos_S\right)
$$

$$
A^h(t_{1:S}) = W_{OV}^h\sum_{s=1}^Sa_s^h(t_{1:S})LN(e(t_s)+pos_s)

$$

$$
a^h(t_{1:S}) = softmax(LN(e(t_S)+pos_S)^TW_{QK}^hLN(e(t_s)+pos_s), s=1:S)
$$

## Simplifications and Setup

Now to visualize this Transformer, we want to use some assumptions which will give nicer results without changing too much the spirit of the architecture. We take *S=2, N=5* and *d=3*. Additionally, we move the layer-norm right after the word embedding. This adjustment reduces the dimensionality of the word embedding from the [3D space to a circle](https://www.geogebra.org/classic/u4twbjkg). Therefore, the Transformer with continuous word embedding takes two points on a circle, which is equivalent to a torus!

![Layer norm visualization](./Images/Capture%20d’ecran%202024-06-20%20à%2012.31.27.png)

Using this approach, we can represent a Transformer with an inner dimension of 3, any number of heads or tokens, using only 2D plots. In this context, plots on a torus imply that opposite sides of the square image are connected.

The training setup is as follows: we have a distribution $\pi$ on tokens 1, 2, and 3. The prior distribution is uniform, and the conditional distribution is a Dirac at the next token. Thus, we can measure memorization as the accuracy of our Transformer. Once the Transformer is trained, we can start visualizing each of its subcomponents. We will visualize a Transformer with 3 heads, each of dimension 3.

## Visualizations

Let us start by visualizing the whole network from top to bottom. There are 5 different classes to be predicted, each one will be represented by a color. The Transformer can thus be understood as a map from the torus to the predicted class. In fact, we plot below the probability of the most likely class, in the color of that class, where intensity encodes probability.

![Transformer clustering.png](./Images/Transformer%20clustering(1).png)

On the above graph we also plotted colored dots corresponding to the pairs of input tokens, colored by their true classes. They form a grid on the torus which comes from the fact that the word embedding is the same at both position. Thus, conditional on the rest of the network, **the word embedding is a grid on the torus chosen to maximize accuracy**.

What is striking on the plot is how intricate the frontier of each class is. We will see that this is mainly due to the attention pattern’s expressivity.

The above plot is a combination of two maps: the attention layer and the unembedding. Let’s examine the unembedding. This map transforms the 3D space into *N* classes. Since we use the argmax of probabilities to determine the class, the temperature won’t influence the predicted class. Therefore, we can visualize the unembedding on a 2D sphere.

$$
W_U:\mathbb{S}^2\rightarrow [N]
$$

In the plot below, the images of the input pairs after the attention layer are shown in black, normalized on the sphere. The *centers* of the normalized columns of the unembedding are shown in white. We can observe that the unembedding creates a partition of the sphere into **weighted Voronoi** cells:

$$
V_{cells}(c) = \{x: w_U(c)\cdot x \geq w_U(c')\cdot x\}
$$

![sphereplot.png](./Images/sphereplot(1).png)

Thus, conditioned on the Attention layer, **the unembedding matrix is chosen to create Voronoi cells that maximize accuracy**. Note that on this plot, the black points, which are indexed by the token pairs, aren’t a grid anymore since we changed the view from the torus to the sphere.

### Visualizing the attention layer

We would like to now visualize the attention layer, which is a map from the torus to the 3D space. We will start by visualizing one head, and then show how they combine. Recall the fomula for one head in our simplified setup.

$$
A(e_1, e_2) = W_{OV}((1-a_1(e_1, e_2))(e_2+pos_2)+a_1(e_1, e_2)(e_1+pos_1))

$$

$$
a_1(e_1, e_2) = \frac{1}{1+e^{(e_2+pos_2)^TW_{QK}(e_2+pos_2-e_1-pos_1)}}
$$

We can break the process into three parts: the attention pattern, combining the word and positional embedding using this pattern, and multiplying by the value-output matrix.

To visualize the attention pattern, we simply plot the probability to attend to the 1st token on the torus. 

![Attention pattern.png](./Images/Attention%20pattern.png)

Further analysis into the formula let us see that the raw attention pattern is in fact a trigonometric polynomial. For head dimension 3, it has 11 harmonics and for 9 free parameters (2 of the harmonics are tied to others), and for head dimension 1 we have 7 harmonics with 5 free parameters. Thus, examining the zeros of this polynomial, we see it possesses significant expressivity, which is challenging to reduce to a simple mathematical object. We can still plot the ([quite mesmerizing](https://www.geogebra.org/classic/cxfk3tra)) moving frontier when these free parameter vary, giving an idea of the expressivity of attention patterns.

Below is the formula of the trigonometric polynomial corresponding to one-dimensional heads. Then we plotted the level line of the polynomial when varying parameters.

$$
raw(\theta_1, \theta_2) = c_0+ c_1(\cos(\theta_1) - \cos(\theta_2 - \phi)) + c_2 \cos(\theta_2 + \phi) + c_3 (\cos(\theta_1 + \theta_2 + \phi) - \cos(\theta_2 - \theta_1 + \phi) - \cos(2\theta_2))
$$

![GIF](./Images/Enregistrementdelecran2024-06-20a14.55.24-ezgif.com-video-to-gif-converter.gif)

Let’s move to the second step of combining the attention pattern with our positional and word embeddings. To simplify, we’ll start by replacing the softmax with an argmax. Since the plots are in 3D, we can visualize them using colors. We clip the vectors into *[0,1]* to make them plottable, ensuring that the colored maps can be summed to obtain the sum of the maps:

$$
RGB(Im_1)+RGB(Im_2)=RGB(Im_1+Im_2)
$$

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/hardmax%20positional.png" width="100%" />
    Position with hardmax
  </div>
  <div style="text-align: center; width: 45%; float: left;">
    <img src="./Images/hardmax%20word.png" width="100%" />
    Word with hardmax
  </div>
</div>

On the left, we have the term depending only on the position $ a_2(e_1, e_2)pos_2+a_1(e_1, e_2)pos_1 $, thus we see a constant color with edges defined by the hardmax of the attention pattern. On the right, $ a_2(e_1, e_2)pos_2+a_1(e_1, e_2)pos_1 $, the plot has the same edges but the colors vary periodically. Note that the colors used for both angles are the same since the vectors themselves are the same. Replacing the hardmax with the softmax will just make the transition smoother at the edges. We finish by adding together the positional and word maps.

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/softmax%20positional.png" width="100%" />
    Position with softmax
  </div>
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/softmax%20word.png" width="100%" />
    Word with softmax
  </div>
  <div style="text-align: center; width: 45%; float: left;">
    <img src="./Images/softmax%20both.png" width="100%" />
    Sum with softmax
  </div>
</div>

Now the last step is to multiply by the value-output matrix of the head. Intuitively, this matrix should change the colorscale as well as the intensity of the colors. However, it is hard to make sense of this transform in itself, especially when there are other heads in parallel since only their sum should be meaningfull. 

![head 1.png](./Images/head%201(1).png)

By summing up each head, we get a new map which is hard to analyse in term of the colors, but in terms of the shapes, we see that each edges of the sum of head is simply the sum of the edges of each head, which are then directly attributed to the shape of the attention pattern. Thus, **the query-key matrices are encoding for the shapes of the edges of the attention layer**. 

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/head%201(1).png" width="100%" />
    Head 1
  </div>
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/head%202.png" width="100%" />
    Head 2
  </div>
  <div style="text-align: center; width: 45%; float: left;">
    <img src="./Images/head%203.png" width="100%" />
    Head 3
  </div>
</div>

![full attention layer.png](./Images/full%20attention%20layer.png)

However, when composing this map with the unembedding to get the Transformer’s clustering, we observe that neither edges nor colors easily correlate with the way the clustering is done. This is partly because we don’t have a method to color the 3D space in a manner that is both linear (where adding two maps gives the map of their addition) and semantically meaningful.

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/Transformer%20clustering(1).png" width="100%" />
  </div>
  <div style="text-align: center; width: 45%; float: left;">
    <img src="./Images/full%20attention%20layer.png" width="100%" />
  </div>
</div>

Another possibility is to view the output of the attention layer as a continuous deformation of the 3D torus, which will nicely compose with our representation of the unembedding matrix on the unit sphere. Below are such representations (normalized by axis to make the shapes more visible): the color intensity increases with the angles, with red representing the first angle and blue the second.

![3d attention.png](./Images/3d%20attention.png)

![3d attention blue.png](./Images/3d%20attention%20blue.png)

This concludes our tour of the visualition of a small attention-only Transformer. 

# Are Attention heads a meaningful basis ?

The main question I wanted to answer with this visualization is the following: **for the memorization task, do we observe distributed learning or not ?** 

This phenomenon is often observed in large language models, where components of the network can be pruned without a loss in their predictive capacity. In our case, this would mean that each head is a little circuit computing its own next-token. Since they have limited memory capacity, they could share the work, making each head focus on a subset of token, while together they obtain 100% accuracy.

Naively, we can look at the probablity mapping of each individual head. Hopefully, they should predict correctly different sets of points, and the sum of their accuracy should be 1. However, this is not what happens in practice, as shown below.

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/Head 1 alone.png" width="100%" />
  </div>
  <div style="text-align: center; width: 45%; float: left;">
    <img src="./Images/Head 2 alone.png.png" width="100%" />
  </div>
</div>

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/Head 3 alone.png.png" width="100%" />
  </div>
  <div style="text-align: center; width: 45%; float: left;">
    <img src="./Images/All heads.png" width="100%" />
  </div>
</div>

By removing all but one head of the AoT, we might be putting the model out-of-distribution: since no head was trained to predict alone, one cannot expect them to produce good output. To mitigate that effect, we propose two experiments:
* First, instead of removing completely all but one head, we can plot the accuracy, which is the metric of interest here, when the contribution of each head varies. To this end, we plot the accuracy of each attention head on a simplex: $Acc(c_0head_0+c_1head_1+c_2head_2)$, where $c_i\geq 0$ and $c_0+c_1+c_2=1$.
* Second, we train the head in a boosting fashion: we train alone head one, then freeze it and train head 2, etc. So if there exist a learning strategy of the sort "each head focuses on a subset of example", we could find it this way. There are again problems with this method, but the general idea is to see whether or not with this training method the AoT can obtain the same accuracya the unconstrained AoT.

Let us start with the Simplexial plots: we produce triangle shaped plot of the accuracy where the proportion of each head varies, for the whole dataset, one input, or for all input that have the same output.

<div style="display: flex; justify-content: space-around; align-items: center;">
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/All Simplex.png" width="100%" />
  </div>
  <div style="text-align: center; width: 45%; float: left; margin-right: 10px;">
    <img src="./Images/Input Simplex.png" width="100%" />
  </div>
  <div style="text-align: center; width: 45%; float: left;">
    <img src="./Images/Output Simplex.png" width="100%" />
  </div>
</div>

As expected, the accuracy over all the dataset is maximal when at the center, which correspond to the training procedure. Looking at the Accuracy on all input with a fixed output, it is difficult to know whether one head is responsible for learning all these examples. By playing with the seeds, one can see very different behaviors:
* The accuracy is greater in one corner than in the others, meaning that this head is mostly responsible for learning these examples,
* The accuracy is large in two corners, meaning that one head is useless, 
* The accuracy is large in between two heads, meaning that both head are remembering together these examples,
* One cannot identify any pattern.

In absence of a clearer metric for "understanding the mechanism" it is hard to conclude from that experiments that the heads are indeed performing some distributed learning.

Now on the second experiment, we train an AoT with the same number of heads as before, but heads are trained one after the other. In particular, we train the first head for longer: we do so because experimentally, we observe that the hypothesis stated above is not true, ie that the boosting AoT is not as expressive as the normal AoT. So to obtain a good loss (and reasonable training time), we decrease the training time of the second and further heads.


![Training Dynamic](./Images/Training%20dynamic.png)

The above plot shows that asking the model to do boosting is less efficient than training the AoT without constraint. This means that making the heads share the memorization work is not optimal, and so it won't be learned during training.

Beyond the empirical analysis, if an AoT has enough attention heads ($\frac{N^2}{d}$), it can make each head remember a different set of associations. If the boosting strategy couldn't remember as well as the normal strategy, it is because the network was under-parametrized. But even with enough parameters, the structure doesn't emerge by default: why would the network make specialized heads if this has no benefits ? It looks that modularity of the heads is not an implicit bias of SGD here. 

So, if one wants the structure of the memorization in an AoT to be "understandable", one should create a penalization or any other metric to induce that behavior. In *Fact Finding*, Neel Nanda concludes that it is hard to understand how MLP store information. This look just as hard for memorization in an Attention Layer if we don't induce somehow understandability.