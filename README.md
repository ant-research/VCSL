# VCSL Benchmark

This is the code for benchmarking segment-level video copy detection approaches
with the Video Copy Segment Localization (VCSL) dataset.

## Update in Jan, 2022
- We release extracted frame features (RMAC, ViSiL, ViT, DINO) of all videos in VCSL (9207 in total).
- We provide SPD pretrained models on RMAC, ViSiL, ViT, DINO features. 
- The above features and models download links are contained in file `data/vcsl_features_spd_models.zip`.
The unzipped password is the submission paper ID of VCSL.
- We implement SPD code based on Yolov5 from this [page](https://github.com/ultralytics/yolov5). 
 A clean version will alo be provided and merged into this repo after the paper is accepted.
- The solution of small part of invalid video links is referred to [here](#solution-to-invalid-links).

## Installation

Requirements
 - python 3.6+
 - pytorch 1.7+
 
In order to install PyTorch, follow the official PyTorch guidelines.
Other required packages can be installed by:
```bash
pip install -r requirements.txt
``` 

## Pipeline

### Video Download
We provide the original video urls in file `data/videos_url_uuid.csv`, you can use tools such as youtube-dl or ykdl to download
the videos. 

### Frame Sampling
We use FFmpeg to extract the frames from a video.
A command-line example:
```bash
ffmpeg -nostdin -y -vf fps=1 -start_number 0 -q 0 ${video_id}/%05d.jpg
``` 
Frames will be sampled at 1FPS and will be saved as JPEG images 
with name `%05d.jpg` (00000.jpg for example) under directory `${video_id}`

Together with the extracted frames for all videos, a csv file should also be
generated containing the following info:

| Header | Description |
| ----------- | ----------- |
| uuid | video id |
| path | relative path of the directory of frames |
| frame_count | number of extracted frames of a video |

An example of the csv file looks like:
```
uuid,path,frame_count
13fbffb8ccf94766b9066225ca226ca2,13fbffb8ccf94766b9066225ca226ca2,193
7f7a837c5dd548eca00389dc6b870a11,7f7a837c5dd548eca00389dc6b870a11,79
```


### Feature Extraction and Similarity Map Calculation

We provide some handy torch-like `Dataset` classes to read the previously extracted frames.
```python
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from vcsl import VideoFramesDataset
from torch.utils.data import DataLoader

# the video_frames.csv must contain columns described in Frame Sampling
df = pd.read_csv("video_frames.csv")
data_list = df[['uuid', 'path', 'frame_count']].values.tolist()

data_transforms = [
    lambda x: x.convert('RGB'),
    transforms.Resize((224, 224)),
    lambda x: np.array(x)[:, :, ::-1]
]

dataset = VideoFramesDataset(data_list,
                             id_to_key_fn=VideoFramesDataset.build_image_key,
                             transforms=data_transforms,
                             root=args.input_root,
                             store_type="local")

loader = DataLoader(dataset, collate_fn=lambda x: x,
                    batch_size=args.batch_size,
                    num_workers=args.data_workers)

for batch_data in loader:
    # batch data: List[Tuple[str, int, np.ndarray]]
    video_ids, frame_ids, images = zip(*batch_data)
    
    # for torch models, transform the images to a Tensor
    images = torch.Tensor(images)  # shape(B, H, W, C)
    # preprocess the images if necessary and use your model to do the inference

```
Features of the same video should be concatenated in temporal order, i.e., a `numpy.ndarray`
or a `torch.Tensor` with shape `(N, D)` where `N` is the frame number and `D` is the feature dimension.
We recommend to put the features together as follows:
```bash
├── OUTPUT_ROOT
    ├── 0000ab50f69044d898ffd71a3d215a81.npy
    └── 6445fe9aa1564a3783141ee9f8f56d3c.npy
    
``` 
where each file named as ${video_id}.npy represents a video feature.

To compute the similarity map between videos, run
```bash
bash scripts/test_video_sim.sh
```
Please remember to correctly set the `--input-root` parameter 
which is usually the `--output-root` from the previous step. 

### Temporal Alignment
We re-implement some Video Temporal Alignment (VTA) algorithms with some modifications to
make them suitable for detecting more than one copied segments between two videos.

For fair comparison, we tune the hyper params of each method with a given feature on
the validation set, run the script to start the tuning process and find the best hyper params
```bash
bash scripts/test_video_vta_tune.sh
```
The script first runs `run_video_vta_tune.py` to tune the hyper params on the valid set data in a grid search manner
 and to evaluate on the val set to find the best hyper params.
Then it runs `run_video_vta.py` to run the VTA method with the best hyper params on the test set.

Please refer to the code and comments for tunable parameters of different VTA algorithms. 

### Evaluation
After the optional parameter tuning step, we evaluate the method on the test set,
run the script as:
```bash
bash scripts/test_video_vta.sh
```
 then you can see the final metric on the test set in the terminal outputs.
 
### Solution to invalid links
- VCSL is originally constructed in mid-2021 and up to now (January 2022) around 8% urls are removed by video websites. 
Although the available raw data are still far larger than other existing VCD datasets, we promise to adding labelled video 
data to VCSL continuously (every six months or every year) to maintain its large scale. We hope this will make the dataset 
valuable and usable for the community in the long term.
- As to result reproducibility and fair comparison across methods, we release the features of all the videos including the 
invalid ones and other necessary experiment details. Thus, one can reproduce the results reported in the paper. 
We encourage researchers who use VCSL do what we have done. In this way, following works can compare fairly different 
approaches by filtering out invalid data at that time.
- For researchers whose interests are in video temporal alignment and other fields that do not depend on raw video data, 
they can leverage the released features and annotations to develop new techniques on which the invalid videos will have no effect.
 
## License
The code is released under MIT license

```bash
MIT License

Copyright (c) 2022 Alipay

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``` 

