# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from mmcls.models import build_classifier
from mmcls.models import build_optical
from mmpose.models import build_posenet
MODELS = Registry('models', parent=MMCV_MODELS)

ALGORITHMS = MODELS
MUTABLES = MODELS
DISTILLERS = MODELS
LOSSES = MODELS
OPS = MODELS
PRUNERS = MODELS
QUANTIZERS = MODELS
ARCHITECTURES = MODELS
MUTATORS = MODELS
HYBRIDS = MODELS
SIMULATOR = MODELS


def build_simulator(cfg):
    """Build posenet."""
    return SIMULATOR.build(cfg)


def build_algorithm(cfg):
    """Build compressor."""
    return ALGORITHMS.build(cfg)


def build_architecture(cfg):
    """Build architecture."""
    return ARCHITECTURES.build(cfg)


def build_mutator(cfg):
    """Build mutator."""
    return MUTATORS.build(cfg)


def build_distiller(cfg):
    """Build distiller."""
    return DISTILLERS.build(cfg)


def build_pruner(cfg):
    """Build pruner."""
    return PRUNERS.build(cfg)


def build_mutable(cfg):
    """Build mutable."""
    return MUTABLES.build(cfg)


def build_op(cfg):
    """Build op."""
    return OPS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)

def build_hybrid(cfg):
    """Build posenet."""
    return HYBRIDS.build(cfg)