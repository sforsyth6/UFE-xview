from .folder import ImageFolderInstance
from .cifar import CIFAR10Instance, CIFAR100Instance
from .mnist import MNISTInstance
from .xviewchips import ComposeWithArg, CropPad, CenterCropTensor, XviewChips

__all__ = ('ImageFolderInstance', 'MNISTInstance', 'CIFAR10Instance', 'CIFAR100Instance')

