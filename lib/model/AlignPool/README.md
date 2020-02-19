# RoI Utilities for PyTorch 1.0
RoIAlign, RoIPool and Non-Max Suppression implementation for PyTorch 1.0.

Code in this repository is a part of [FAIR's official implementation of Mask R-CNN Benchmark](https://github.com/facebookresearch/maskrcnn-benchmark),
and support for PyTorch 1.0 and higher is [patched](https://gist.github.com/ruotianluo/582c0e9b46ea202ed96eaddf08f80034) 
by [ruotianluo](https://github.com/ruotianluo).

## Requirements
* pytorch >= 1.0.0
* torchvision>=0.2.0
* cython >= 0.29.2
* matplotlib
* numpy
* scipy

\[NOTE\] CUDA support is highly recommended because RoI pooling is not implemented for CPU yet.

## Installation
```
python setup.py install
```

## Usage
Import this library in your code:
```python
from roi_util import ROIAlign, ROIPool, nms
```
See [notebook/RoI_Util_API.ipynb](https://github.com/linkinpark213/RoIAlign_PyTorch/blob/master/notebook/RoI_Util_API.ipynb) for examples.

## Documentation
### ROIAlign
```python
ROIAlign(output_size: tuple, spatial_scale: float, sampling_ratio: int)
```
Parameters:
* __output_size__ - A tuple of 2 integers: expected size of the output feature map of the RoI.
* __spatial_scale__ - A floating point number: relative size of the input feature map to the original input image. 
Equal to `feature_map_width / original_image_width`.
* __sampling_ratio__ - An integer: the sampling ratio for RoI alignment.

Inputs: input, rois
* __input__ - A `torch.Tensor` of shape `(batch, num_channels, height, width)`: a batch of feature maps.
* __rois__ - A `torch.Tensor` of shape `(total_num_rois, 5)`: the batch indices and coordinates of all RoIs.
Each line of this tensor is an RoI with data (batch_index, x1, y1, x2, y2), 
since one feature map could correspond to several RoIs.
`x1, y1, x2, y2` denotes the coordinates of the top-left corner and the bottom-right corner of each RoI in the original image.
Values of `x1` and `x2` should be between `0` and `original_image_width`, 
while values of `y1` and `y2` should be between `0` and `original_image_height`.
If their values exceeds the range of original image size, the exceeded part would be padded with 0.

Outputs: output
* __output__ - A `torch.Tensor` of shape (total_num_rois, num_channels, output_size[0], output_size[1]).


### ROIPool
```python
ROIPool(output_size: tuple, spatial_scale: float)
```
Parameters:
* __output_size__ - A tuple of 2 integers: expected size of the output feature map of the RoI.
* __spatial_scale__ - A floating point number: relative size of the input feature map to the original input image. 
Equal to `feature_map_width / original_image_width`.

Inputs: input, rois
* __input__ - A `torch.Tensor` of shape `(batch, num_channels, height, width)`: a batch of feature maps.
* __rois__ - A `torch.Tensor` of shape `(total_num_rois, 5)`: the batch indices and coordinates of all RoIs.
Each line of this tensor is an RoI with data (batch_index, x1, y1, x2, y2), 
since one feature map could correspond to several RoIs.
`x1, y1, x2, y2` denotes the coordinates of the top-left corner and the bottom-right corner of each RoI in the original image.
Values of `x1` and `x2` should be between `0` and `original_image_width`, 
while values of `y1` and `y2` should be between `0` and `original_image_height`.
If their values exceeds the range of original image size, the exceeded part would be padded with 0.

Outputs: output
* __output__ - A `torch.Tensor` of shape `(total_num_rois, num_channels, output_size[0], output_size[1])`.

### nms
```python
nms(dets: torch.Tensor, scores: torch.Tensor, overlap_threshold: float) -> torch.Tensor
```
Parameters:
* __dets__ - A `torch.Tensor` of shape `(num_detection, 4)`: top-left and bottom-right coordinates of all detected boxes.
* __scores__ - A `torch.Tensor` of shape `(num_detection)`: detection scores of all the boxes.
* __overlap_threshold__ - A floating point number: the overlapping threshold. If two boxes have a higher IoU than the threshold, the box with lower score will be removed.

Returns:
* __indices__ - A `torch.Tensor` of shape `(num_filtered_detection)`: the indices of remaining boxes after filtering by non-max-suppression.

## Issues
* Segmentation faults occurs on certain circumstances. The root cause is yet to be found.