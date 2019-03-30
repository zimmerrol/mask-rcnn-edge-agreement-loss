# Faster Training of Mask R-CNN by Focusing on Instance Boundaries

This is an implementation of the improved training scheme [Faster Training of Mask R-CNN by Focusing on Instance Boundaries](https://arxiv.org/abs/1809.07069) on Python 3, Keras, and TensorFlow. The code is an extension of the existing implementation of [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN). The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

If you like this work and want to use this in your work or research, please cite:

    @article{DBLP:journals/corr/abs-1809-07069,
    author    = {Roland S. Zimmermann and
                Julien N. Siems},
    title     = {Faster Training of Mask {R-CNN} by Focusing on Instance Boundaries},
    journal   = {CoRR},
    volume    = {abs/1809.07069},
    year      = {2018},
    url       = {http://arxiv.org/abs/1809.07069},
    archivePrefix = {arXiv},
    eprint    = {1809.07069},
    timestamp = {Fri, 05 Oct 2018 11:34:52 +0200},
    biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1809-07069},
    bibsource = {dblp computer science bibliography, https://dblp.org}
    }