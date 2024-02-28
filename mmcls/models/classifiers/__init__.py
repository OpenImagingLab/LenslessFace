# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .image import AffineFaceImageClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', 'AffineFaceImageClassifier']
