# XceptionNet — Model Notes

## What the model does (1–3 paragraphs)

- Problem it solves:
This model solves the problem of identifying deepfakes in images and videos. 

- Input → output:
This model takes as input a cropped face image, and outputs a probability of it being real and a probability of it being fake. 

- Why it's relevant for AI video detection / deepfake detection:
XceptionNet is excellent in capturing intricate visual patterns and signs of deepfake manipulation. Thus, it is very useful in deepfake detection pipelines. 


## Paper / reference

- Paper title: Xception: Deep Learning with Depthwise Separable Convolutions
- Authors / year: François Chollet, 2017
- Link: https://arxiv.org/abs/1610.02357
- Key ideas (bullet points): 
    • Instead of mapping cross-channel correlations and spatial correlations together, it would be much simpler and more effective to map them separately. 
    • While the Inception architecture partially separates the cross-channel correlations and spatial correlations, Xception (Extreme Inception) architecture fully separates them. 
    • Xception shows better results compared to Inception V3 on ImageNet, JFT without fully-connected layers, and JFT with fully-connected layers. 

- Architecture summary (high-level):
XceptionNet is a CNN that consists of 36 depthwise separable convolution layers, divided into 14 modules, which are connected using linear residual connections except for the first and the last module. In the XceptionNet architecture, data first goes through the entry flow, then through a middle flow and repeated 8 times, and finally through an exit flow. 

## What I learned (bullet points)

- Important details: 
    • XceptionNet shows that modern CNNs do not have to strictly map cross-channel correlations and spatial correlations together and still achieve amazing results. 
- Gotchas / assumptions:
    • The models assume large datasets, otherwise they perform poorly. 
- Strengths:
    • XceptionNet performs better than Inception V3 with same number of parameters. 
    • The Xception architecture is very easy to define and modify as it only takes 30 to 40 lines of code using high level libraries. 
- Weaknesses:
    • Performance is heavily impacted by the size of datasets. 

## How it should be used in our project

- Expected preprocessing: First extract frames from videos, then detect and crop faces from those frames. 
- Expected input format: single face image in jpg
- Metrics typically reported: Top-1 accuracy, Top-5 accuracy, and MAP (Mean Average Precision). 

## Screenshots / diagrams (optional)

Put images in ./assets and reference them like:
![description](assets/<filename.png>)

## Open questions

- Questions to ask in weekly meeting:
- Things to verify:
