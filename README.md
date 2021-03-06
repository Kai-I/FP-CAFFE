# Caffe

## Update for fixed point support

We already modified the caffe to support fixed point operations

The fixed point flow can be three steps:
### fix the network

The command is `caffe fix` with three new parameters extra to the original `model` and `weights` parameters

* fixwidth: data width for fixed point, default value is 8
* fixinfo: filename to save fixed point information for all layers
* fixweights: new filename to save the weights after this fix step

For example, you can run as
```
caffe fix -model vgg16.prototxt -weights vgg16.caffemodel -fixwidth 8 -fixinfo vgg16_fix.txt -fixweights vgg16_fix.caffemodel
```

### run fixed point forward

The command is `caffe fixtest` with one new parameter `fixinfo`, and please notice that
**we should be specify the new fixed weights file for `fixweights` parameter.**

For example, you can run as
```
caffe fixtest -model vgg16.prototxt -fixweights vgg16_fix.caffemodel -fixinfo vgg16_fix.txt
```

### finetune fixed point network

After `fix` and `fixtest`, you may find the network performance become worse, you can use `fixtune` to finetune the network.
We should not forget that
* use the fixed point weights file for `fixweights` parameter;
* also specify `fixinfo` file;
* specify the learning rates of all convolutional layers to 0

For example, you can run as
```
caffe fixtune -solve vgg16_solver.prototxt -fixweights vgg16_fix.caffemodel -fixinfo vgg16_fix.txt
```

---

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
