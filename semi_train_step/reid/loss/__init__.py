from __future__ import absolute_import

from .oim import oim, OIM, OIMLoss
from .triplet import TripletLoss,FocalLoss
from .weight_cross_entropy import WeightCE
from .center_loss import CenterLoss

__all__ = [
    'oim',
    'OIM',
    'OIMLoss',
    'TripletLoss',
    'WeightCE',
    'CenterLoss'
]
