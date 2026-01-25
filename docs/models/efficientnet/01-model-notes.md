# EfficientNet — Model Notes

## What the model does (1–3 paragraphs)

- Problem it solves: This model is a type of CNN which composed of blocks of mobile inverted bottleneck convolution layer. This model can be used for both deepfake detection and general AI video detection. 
However, certain level of transfer learning and fine tuning are required. 
- Input → output:
B0: 224 x 224 -> Real/Fake
B1: 240 x 240 -> Real/Fake
B2: 260 x 260 -> Real/Fake
B3: 300 x 300 -> Real/Fake
B4: 380 x 380 -> Real/Fake
B5: 456 x 456 -> Real/Fake
B6: 528 x 528 -> Real/Fake
B7: 600 x 600 -> Real/Fake

- Why it's relevant for AI video detection / deepfake detection:
Those pretrained model has been trained to recognize various objects from over thousands of categories. Hence, it is pretty useful for specific face swapped detection and overall AI traces in the video.

## Paper / reference

- Paper title: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- Authors / year: Mingxing Tan, Quoc V. Le, 2019
- Link: https://arxiv.org/abs/1905.11946
- Key ideas (bullet points): 
  1. It describes a new way to scale up a type of CNN which are built with MB convolution layer. Previous papers only concern about scaling up CNN in one dimension(width, depth, resolution), this paper investigate the effect of scaling up all three dimensions based on a certain ratio. The paper call this method compound scaling. 
  2. Through different application of compound scaling, the paper come up with different size of CNN.
  From small to large, the paper gives 8 different scales, calling them B0, B1, ... B7 respectively.  
  3. As the size of CNN grow, the amount of computation double. B7 requires approximately ~(2^7) times computation more than B0. Their input resolution is therefore different.
  4. The baseline, B0, is created using neural architecture search. Which let computer to do computation 
  to find the most efficient(min computation + maximum accuracy) architecture. 

- Architecture summary (high-level):
Take B0 as example, it consists of multiple blocks of MBConv. 
The MBConv block performs what is so called "inverted bottleneck" that first uses a 1 x 1 convolution to expand channel count, creating a high-dimensional space for features. It then performs depthwise separable convolutions to process each channel individually, effectively reducing the amount of parameters compared to standard CNN. Then, the squeeze and excitation layer weights channels based on their importance to the specific image. Lastly, the data is projected back to its original width through 1 x 1 convolution and combined with the initial input through residual connection to improve learning.

## What I learned (bullet points)

- Important details: Structure of CNN, how research modify CNN to explore ways to increase accuracy and lower computation cost.
- Gotchas / assumptions: Assuming increasing all three dimensions of the CNN can outperform those that only expand in one dimension
- Strengths: The model has several pretrained version, which is convenient as we can select based on our need and computation power.
- Weaknesses: Need to spend time to learn transfer learning and fine tuning

## How it should be used in our project

- Expected preprocessing: Crop video into individual frame and maybe compress the video so it has lower resolution
- Expected input format: depend on the version of efficient net model, range from 224 x 224 to 600 x 600
- Metrics typically reported: 
1. Top-1 and Top-5 Accuracy
2. number of parameters
3. FLOPS
4. inference latency
## Screenshots / diagrams (optional)

Put images in ./assets and reference them like:
![description](assets/<filename.png>)

## Open questions

- Questions to ask in weekly meeting:
- Things to verify:
