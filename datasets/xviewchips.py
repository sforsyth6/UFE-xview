import torch
from inspect import signature
from PIL import Image
from . import ImageFolderInstance

class ComposeWithArg(object):
    """Composes several transforms together, passing optional arg to each transform if
       required.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

        arg: (any type) argument for one or more transforms

    Example:
        >>> transforms.ComposeWithArgs([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.XviewUFLChips(percent)
        >>>     transforms.ToTensor(),
        >>> ])

        With XviewChips, it'll expect a path argument, this'll be passed automatically.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, arg=None):
        for t in self.transforms:

            if len(signature(t.__call__).parameters) > 1:
                img = t(img, arg)
            else:
                img = t(img)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class CropPad(object):
    """Zero pads pixels in a Tensor outside of a given percent of a contained bounding
       box. Tensor size is unchanged. Example: percent = 1.0 will return just the bounding
       box pixels, all others will be set to 0. If desired percent exceeds the image
       boundary, the original image will be returned. Expects channels in the first dimension.

       Must follow transforms.ToTensor()!

    Args:
        percent (float): percent of bounding box pixels to retain. Others will be zero filled.
    """

    def __init__(self, percent):
        self.percent = float(percent)

    def __call__(self, img, path):
        """
        Args:
            img (torch.Tensor): Tensor to be cropped & padded.
            path (str)        : Formated path containing the bounding box of the object.

        Returns:
            torch.Tensor: Cropped image tensor.
        """
        bbox = tuple(int(x) for x in path.split('_')[-1].replace('.png', '').split(','))

        padded = torch.zeros_like(img)

        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]

        y1 = int(bbox[1] + h / 2 - (h * self.percent) / 2)
        if y1 < 0:
            y1 = 0
        y2 = int(bbox[3] - h / 2 + (h * self.percent) / 2)

        x1 = int(bbox[0] + w / 2 - (w * self.percent) / 2)
        if x1 < 0:
            x1 = 0
        x2 = int(bbox[2] - w / 2 + (w * self.percent) / 2)

        padded[:, y1:y2, x1:x2] = img[:, y1:y2, x1:x2]

        return padded

    def __repr__(self):
        return self.__class__.__name__ + '(percent={0})'.format(self.percent)

class CenterCropTensor(object):    
    """Square crops the given Tensor from the center.

    Args:
        size (int or list(int, int)): Desired output size of the crop. If int, 
        it'll square crop to size x size. Otherwise, it'll crop to size[0] x size[1]
    """
    
    def __init__(self, size):
        if type(size) == int:
            self.size = [size, size]
        else:
            self.size = size

    def __call__(self, tensor):
        """
        Args:
            img (Tensor): Tensor image to be cropped.

        Returns:
            torch.Tensor: Cropped image tensor.
        """
        
        # If desired crop is bigger than tensor, don't do anything:
        if self.size[0] > tensor.shape[1]:
            crop_dim1 = (0,tensor.shape[1])
        else:
            crop_dim1 = ((tensor.shape[1] - self.size[0]) // 2, \
                              self.size[0] + (tensor.shape[1] - self.size[0]) // 2)
            
        if self.size[1] > tensor.shape[2]:
            crop_dim2 = (0,tensor.shape[2])
        else:    
            crop_dim2 = ((tensor.shape[2] - self.size[1]) // 2, \
                              self.size[1] + (tensor.shape[2] - self.size[1]) // 2)
            
        return tensor[:,  crop_dim1[0] : crop_dim1[1], crop_dim2[0] : crop_dim2[1]]

    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class XviewChips(ImageFolderInstance):
    """: ImageFolderInstance which provides the local coordinates of annotations in chips
         to transforms.

         Transforms must be wrapped in a ComposeWithArg instance.
    """

    #     def __init__(self):
    #         super(XviewChips, self).__init__(root, loader, IMG_EXTENSIONS,
    #                                             transform=transform,)

    def loader(self, path):
        # Assumes a PIL image_backend
        with open(path, 'rb') as f:
            img = Image.open(f)
            bbox_coords = tuple(int(x) for x in path.split('_')[-1].replace('.png', '').split(','))
            return img.convert('RGB'), bbox_coords

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img, path)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


