# Notes

Self Supervised learning: Trying to learn a taks or representation that makes the model useful for other tasks. Example: Learning to classify shapes encodes the information to predict orientation, size, location, etc.

## Lecture 2

Self Supervised Learning Through Pretext Tasks

- Example: (paper name?)

  - feed the model augmentations of the original images (namely, rotations)
  - then, predict the rotation amount

- Example: Discriminative Unsupervised Feature Learning with Confolutional Neural Networks

  - sample N patches (locations and scales)
  - for each patch, produce K augmentations
  - all augmentations will be labeled with their patch. They should predict which patch they came from.
  - Results:
    - too many patches (big K): supervised loss will try to separate patches (patches from the same area are labeled differently), but if they are the same, this is harmful
    - more samples per class: accuracy generally improves
    - many hyperparameters are dataset-specific

- Example: Learning to Count
  - motivation: The model should count the number of objects in an image. Peforming the tast on subsections of an image, and then adding the results should give the result of performing the task on the entire image. (the results should be additive). Additionally, we two images are unlikely to have the same counts of 'things'.
  - We can train an _unsupervized_!! model that learns to 'count things' without labels

AlexNet vs ResNet (standard past 2017)

what is an ad-hoc task?

- Example: (paper name?)

  - take two patches from the same image and learn to predict the relative location (usually just by quadrants)

- Example: (paper name?)
  - predict the patch that was removed from an image
  - an example of 'in-painting'

in-painting can be applied to other domains like language (predict missing words)

- Example: (paper name?)
  - take black and white image and learn to predict colors

## Lecture 3

Class Notes:

- Group youselves for the sep 18 and first group presentation on the 23rd

Examples of Pretext Tasks from the class:

- jigsaw puzzle shuffle of an image
- predict the noise level applied to image
- mask most of the image and ask the model to recreate the image
- combine two images and ask the model to disambiguate them

There is no clear method to derive a 'good' pretext task. Just trail and error.

### Deriving a Pretext task from first principles

Suppose we just want to learn the approximation:

f(x) = log(p^\hat_x(x))

Given a finite dataset: X = \{x_1, ..., x_N\}

Samples M noise samples V = \{v_1, ..., v_M\} unifomrly from supp(p_X)

Solve the binary classification task X|V with f(x)

In summary, we can prove taht for large enough N and M, we can show that the model will learn the true f(x) = log(p_X(x)) value.

Paper: Noise-contrastive estimation: A new estimation principle for unnormalized statistical models.

Curse of Dimensionality: The number of samples needed to train a D dimensional dataset, the number of samples needed grows exponentially with D.

Smarter Approaches:

- Data vs Independent noise: Noise-contrastive estimation: A new estimation principle for unnormalized statistical models

- Data vs Data + small noise: A Scalable Approach to Density and Score Estimation.

good density estimator = good SSL representation

Side Challenges:

- find all the functions that belong to a space that give a good approximation of log(p_X(x))

Where pretext tasks fail:

The rotation prediction task: A "shotgun approach" might jsut be to learn to detect eyes and mouth and then place the eyes above the mouth. It doesn't need to learn anything else about the context of the scene (animal type, outside vs inside).

To solve ethe rotation prediction task, we need to be invariant to animals features and be equivariant to rotation. Even just knowing this, we can see we may get an undesirable outcode --- to encode animal featurees, we don't want to be invariant across featuers.

- Self-Supervised Learning of Pretext-Invariant Representations

New Approach: Take a sample and a transformed version of itself. We want the result to be as similar as possible. To avoid a constant output, we add a term to the loss function.

In General: The things that the model becomes invariant to are the tasks it will no longer be good at.

## Lecture 4 - 9/11/2024

### Manifold Learning Motivation

- Suppose thereexists an underlying generative process for your data from coordinates to observations

- Define by $\theta$ the intrinsic coordinate of the data and bv $x(\theta)$ its observation. i.e. $x(\theta)$ applies a transformatoin $\theta$ to an image $x$

- What could we do if we were given $x^{-1}$? (This would remove the image data and just retain the position/scale/orientation data)

### Why "Manifold" Learning?

- A **manifold** is a topological space that locally resembles Euclidean space near each point.

- Numerous methods rely on that assumption: $x(\theta + v) \sim x(\theta) + J_x(\theta)^T v$

### How to Learn the Manifold

- Suppose we are given a set of D-dimensional samples $X := \{x_1, ..., x_N\}$

- Assume the samples lie on a K-dimensional affine subspace

- Solve the least square problem $\min\limits_{W \in \R^{D\times K}} = \frac{1}{N} \sum_{n=1}^N || x_n - WW^Tx_n ||^2_2$

- Do we recover our goal $x^{-1}(x(\theta)) = W^Tx(\theta)$

$$


$$

- In general\* recovery of $\theta$ is always up to some transformations e.g. rotation.

\*could be partial observations of $\theta$ or controlled data sampling

### Does "up to some transformations" Matter?

- Suppose we recover some tranformed coordinates $W^T x(\theta) = R^T\theta$

- Does it matter if you use k-NN classificier for your downstream task?

  - No, rotation (depending on distance metric) doesn't depend on rotation.

- Does it matter if you use linear classifier for you downstream task?
  - No, you can just rotate the decision boundary with the data.

TLDR it does not really matter unless you care about "disentanglement" (What is that?)

Datasets for coordinate Recovery: Sprites, 3dshapes, and celeba

### Deep Autoencoders

- Replace the linear mapping $W$ with the nonlinear Deep Network $f$.

- We keeep the same observe/motivations, just without the affine: $\min\limits_{W \in \R^{D\times K}} = \frac{1}{N} \sum_{n=1}^N || x_n - g(f(x_n)) ||^2_2$

### Denoising Autoencoders

- We keeep the same objserve/motivations + add some noise to the input: $\min\limits_{W \in \R^{D\times K}} = \frac{1}{N} \sum_{n=1}^N || x_n - g(f(x_n)) ||^2_2$

- Noise acts as a regularizer, but there are other ways to think about it:

- We are trying to become invariant to the noice distribution. A key question here is which noise distribution to use? What noise do we want to become invariant to?

- Learning this task can also result in having a model for denoising images!

### Masked Autoencoders

- A different type of noise specifically designed for ViTs.

### Variational Autoencoders

A more probability viewpoint

- The encodier's output $f(x)$ is ow predicting a distribution's parameters e.g. mean and std for Gaussian.
