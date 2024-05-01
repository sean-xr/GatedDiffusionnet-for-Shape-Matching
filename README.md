# GatedDiffusionnet-for-Shape-Matching
TUM Pracical Course Project: Praktikum on 3D Computer Vision

Code based on the implementation of Diffusionnet(https://github.com/nmwsharp/diffusion-net) and Diff-FMAPS(https://github.com/riccardomarin/Diff-FMaps)

Main Contribution of this project:

1. Extend the Diff-FMAP framework to be using Diffusionnet as a backbone for linearly-invariant embedding learning.

2. I implemented GatedDiffusionnet, showing that diffusionnet with gating mechanism could greatly improves the supervised mathcing performance. (we use ZoomOut to compare all the methods)
   
3. Maintain decent level of robustness under the remeshed setting, when comparing with axiomatic basis such as the elastic basis (https://github.com/flrneha/ElasticBasisForSpectralMatching)

For the detailed presentation, please check out our final presentation slides: https://docs.google.com/presentation/d/1reJIi8HVSSRQO4o1XVgrDbntdGRi3nXowF2u0I0MYHY/edit?usp=sharing


