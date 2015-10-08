# [ECE 6504 Deep Learning for Perception][1]

## Homework 2

In this homework, we continue learning [Caffe][2], and implement dropout and
data augmentation in our earlier ConvNet. We then fine-tune a pre-trained model,
AlexNet, for style classification on the WikiArt dataset. Finally, we visualize
data gradients and learn to generate images to fool a pre-trained ConvNet.

Download the starter code [here](https://github.com/batra-mlp-lab/VT-F15-ECE6504-HW2/archive/1.0.zip).

### Q1: Dropout and Data Augmentation (15 points)

In this exercise, we'll be working with the same two-layer ConvNet
we trained on the CIFAR-10 dataset in the previous assignment and
implementing two ways to reduce overfitting - dropout and data
augmentation, using Caffe.

Go through the specification of the [`DropoutLayer`][6] and
read network prototxt files of [AlexNet][9] & [CaffeNet][10] to see
how dropout layers are implemented in Caffe.

There is in-built support for simple data augmentations such
as random crops and mirroring in Caffe. This is defined by the
`transform_param` parameter inside a `DataLayer` definition.

```
layer {
  name: "data"
  type: "Data"
  [...]
  transform_param {
    scale: 0.1
    mean_file_size: mean.binaryproto
    # for images in particular horizontal mirroring and random cropping
    # can be done as simple data augmentations.
    mirror: 1  # 1 = on, 0 = off
    # crop a `crop_size` x `crop_size` patch:
    # - at random during training
    # - from the center during testing
    crop_size: 227
  }
}
```

- Use a smaller training set, so that the network overfits (high training accuracy, low validation accuracy)
- Define a dropout layer
- Add data augmentation parameters to the Data layer
- Train the network again on the smaller set. You should see higher validation accuracy

**Optional**: Other common data augmentation techniques used to
improve accuracy are rotations, shearing & perspective
wrapping. Take a look at the [ChenglongChen/caffe-rta][8]
repository to see how the author has implemented these.

**Deliverables**

- Network prototxt with dropout and data augmentation (5 points)
- `Validation Loss v/s Iterations` plot with and without dropout (10 points)


### Q2: Fine-tuning AlexNet for Style classification on WikiArt data (20 points)

Given the WikiArt dataset, which consists of 10000 images of paintings
of arbitrary sizes from 10 different styles - Baroque, Realism,
Expressionism, etc., the goal is to fine-tune a pretrained model, AlexNet, to
predict painting style with reasonable performance and minimal training time.

#### Obtaining the dataset

The dataset consists of 10000 images in total from 10 different styles
of painting - 1000 images each. Use the `download_wikiart.py` script
to download a subset of the data and split it into training and
validation sets.

```bash
% python download_wikiart.py -h
usage: download_wikiart.py [-h] [-s SEED] [-i IMAGES] [-w WORKERS]

Download a subset of the WikiArt style dataset to a directory.

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  random seed
  -i IMAGES, --images IMAGES
                        number of images to use (-1 for all [default])
  -w WORKERS, --workers WORKERS
                        num workers used to download images. -x uses (all - x)
                        cores [-1 default].

% python download_wikiart.py -i 2000 -s 761218
Downloading 2000 images with 7 workers...
Writing train/val for 1996 successfully downloaded images.
```

#### Setting up the AlexNet prototxt files

Copy the AlexNet prototxt files, `solver.prototxt` and `train_val.prototxt` from
`$CAFFE_ROOT/models/bvlc_alexnet` to the working directory.

```bash
cp $CAFFE_ROOT/models/bvlc_alexnet/solver.prototxt ./
cp $CAFFE_ROOT/models/bvlc_alexnet/train_val.prototxt ./
```

Since you'll be fine-tuning a network pretrained on the ImageNet dataset,
you will also need the ImageNet mean file. Note that if you train a network
from scratch, then you should instead compute the mean over your own training
data. Run `$CAFFE_ROOT/data/ilsvrc12/get_ilsvrc_aux.sh` to obtain this. You
will also need the AlexNet pretrained model.

```bash
python $CAFFE_ROOT/scripts/download_model_binary.py $CAFFE_ROOT/models/bvlc_alexnet
```

#### Transfer Learning

There are two main transfer learning scenarios:

- **ConvNet as a fixed feature extractor**: We take a ConvNet pretrained
on the ImageNet dataset, remove the final fully-connected layer and treat
the rest of the ConvNet as a fixed feature extractor for the new dataset.
We can train a linear classifier (linear SVM or SoftMax classifier) on these
extracted features (4096-D vectors for every image in case of AlexNet) for
the new dataset. In Caffe, this is achieved by setting the learning rates
of the intermediate layers (`blobs_lr`) to 0.

- **Finetuning the ConvNet**: The second strategy is to not only replace
and retrain the classifier on top of the ConvNet on the new dataset,
but to also fine-tune the weights of the pretrained network by continuing
the backpropagation.

#### Fine-tuning

Look at `train_val.prototxt` and `solver.prototxt` closely. To fine-tune on the
WikiArt dataset, we'll start with the weights of the pretrained model for
all layers. Since our dataset consists of 10 classes instead of 1000
(for ImageNet), we'll modify the last layer. Note that in Caffe when we
start training with a pretrained model, weights of layers with the same
name are retained and new layers are initialized with random weights.

From the Caffe example on [fine-tuning CaffeNet for style recognition on
Flickr style data][3]:

*We will also decrease the overall learning rate `base_lr` in the solver prototxt,
but boost the `blobs_lr` on the newly introduced layer. The idea is to have the
rest of the model change very slowly with new data, but let the new layer learn fast.
Additionally, we set `stepsize` in the solver to a lower value than if we were training
from scratch, since we’re virtually far along in training and therefore want the
learning rate to go down faster. Note that we could also entirely prevent fine-tuning
of all layers other than `fc8_flickr` by setting their `blobs_lr` to 0.*

- Change the data layer
- Change last layer
- Modify hyperparameters

Now you can start training.

```bash
$CAFFE_ROOT/build/tools/caffe train -solver solver.txt -weights $CAFFE_ROOT/models/bvlc_alexnet/bvlc_alexnet.caffemodel

```

**Deliverables**

- Prototxt files (`train_val`,`solver`,`deploy`) (10 points)
- `Training Loss v/s Iteration` plot (5 points)
- [Kaggle contest][15] (5 points + up to 10 extra points for beating TA entry and top performers)

### Q3: Visualizing and Breaking ConvNets (15 points)

In this exercise, we'll work with the Python interface for Caffe and learn to
visualize data gradients and generate images to fool ConvNets.

#### Class Model Visualizations

We'll be using the method outlined in the paper "Deep Inside Convolutional
Networks: Visualising Image Classification Models and Saliency Maps" \[[3][11]\]
to visualize a class model learnt by a convolutional neural network.

In order to generate the class model visualization, we need to optimize the
unnormalized class score with respect to the image.

$$
\mathop{\arg\,\max}\limits\_I S\_c(I) - \lambda \lVert I \rVert^2
$$

This is done by standard backpropagation as done during the training phase
of the network with the difference that instead of updating the network
parameters, we'll be updating the image to maximise the score, a method
known as **gradient ascent**. Also note that we'll drop the final layer
of the network and maximize the unnormalized class score instead of the
probability as outlined in the paper.

Copy the AlexNet `deploy.prototxt` into the working directory and edit it.

```bash
cp $CAFFE_ROOT/models/bvlc_alexnet/deploy.prototxt 3_visualizing-breaking-convnets/
```

- Delete the final layer
- Add "force_backward: true", to propagate the gradients back to the data layer
in the backward pass
- Change the number of input dimensions to 1

Open the IPython notebook `class-model-visualizations.ipynb` and complete
the missing code to generate the class model visualizations.

#### Image-Specific Class Saliency Visualisation

Section 3 of the paper \[[3][11]\] describes a method to understand which part
of an image is important for classification by visualizing the gradient
of the correct class score with respect to the input image. The core idea
behing this is to find the pixels which need to be changed the least.

Open the IPython notebook `saliency-maps.ipynb` and complete the missing code
to extract and visualize image-specific saliency maps.

#### Generating Fooling Images to Break ConvNets

Several papers \[[4][12],[5][13],[6][14]\] have suggested ways to perform optimization over the
input image to construct images that break a trained ConvNet. These papers showed
that given a trained ConvNet, an input image, and a desired label, that we can
add a small amount of noise to the input image to force the ConvNet to classify
it as having the desired label.

We will create a fooling image by solving the following optimization problem:

$$
x\_f = \mathop{\arg\,\min}\limits\_x (L (x,y,m) + \frac{\lambda}{2} \lVert x - x_0 \rVert ^2)
$$

Open the IPython notebook `breaking-convnets.ipynb` and complete the missing code
to generate fooling images that break pretrained ConvNets.

**Deliverables**

- Completed IPython notebooks `class-model-visualizations.ipynb`,
`saliency-maps.ipynb` & `breaking-convnets.ipynb` (5 points x 3)

References:

1. [Assignment 3, CS231n, Stanford][5]
2. [Fine-tuning CaffeNet for Style Recognition on “Flickr Style” Data][3]
3. [Simonyan et al., "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps", ICLR 2014][11]
4. [Nguyen et al., "Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images", CVPR 2015][12]
5. [Szegedy et al., "Intriguing properties of neural networks"][13]
6. [Goodfellow et al., "Explaining and Harnessing Adversarial Examples", ICLR 2015][14]

[1]: https://computing.ece.vt.edu/~f15ece6504/
[2]: http://caffe.berkeleyvision.org/
[3]: http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html
[4]: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
[5]: http://cs231n.github.io/assignment3/
[6]: http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DropoutLayer.html
[7]: http://caffe.berkeleyvision.org/tutorial/data.html
[8]: https://github.com/ChenglongChen/caffe-windows
[9]: https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt
[10]: https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/deploy.prototxt
[11]: http://arxiv.org/abs/1312.6034
[12]: http://arxiv.org/abs/1412.1897
[13]: http://arxiv.org/abs/1312.6199
[14]: http://arxiv.org/abs/1412.6572
[15]: https://inclass.kaggle.com/c/2015-fall-vt-ece-deep-learning-homework-2

---

&#169; 2015 Virginia Tech
