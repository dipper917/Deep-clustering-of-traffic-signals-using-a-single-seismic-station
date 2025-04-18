# Deep-clustering-of-traffic-signals-using-a-single-seismic-station
We present an unsupervised deep clustering framework based on Deep Embedded Clustering (DEC) to identify traffic-induced signal segments from high-frequency seismic noise recorded by a single seismic station. This method enables cost-effective event extraction from ambient noise without labeled data or dense sensors, offering a alternative for urban traffic monitoring, particularly in areas with limited data or few monitoring devices.

The code provided here demonstrates the model training using data from Site 1, part of our study on analyzing traffic-induced seismic signals at urban scales. The implementation is adapted and modified from previous works(J. Xie et al., 2016; Snover et al., 2021; Mousavi et al., 2019).

# Training dataset:
The dataset used in this study can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.15229794). Please ensure that you download the appropriate dataset before running the code.

# Requirements-Package Versions:
•Python == 3.9.19  

•TensorFlow == 2.11.0  

•numpy == 1.26.4  

•scikit-learn == 1.5.2  

•pandas == 2.2.3  

•matplotlib == 3.9.2  

•h5py == 3.11.0

# References:
Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised deep embedding for clustering analysis. In Proceedings of the 33rd International Conference on Machine Learning (pp. 478–487). PMLR. Retrieved from https://proceedings.mlr.press/v48/xieb16.html  

Snover, D., Johnson, C. W., Bianco, M. J., & Gerstoft, P. (2021). Deep Clustering to Identify Sources of Urban Seismic Noise in Long Beach, California. Seismological Research Letters, 92(2A), 1011–1022. https://doi.org/10.1785/0220200164  

Mousavi, S. M., Zhu, W., Ellsworth, W., & Beroza, G. (2019). Unsupervised Clustering of Seismic Signals Using Deep Convolutional Autoencoders. IEEE Geoscience and Remote Sensing Letters, 16(11), 1693–1697. https://doi.org/10.1109/LGRS.2019.2909218  

# Related Code Repositorys:
https://github.com/dsnover/Unsupervised_Machine_Learning_for_Urban_Seismic_Noise  

https://github.com/smousavi05/Unsupervised_Deep_Learning  

https://github.com/Tony607/Keras_Deep_Clustering
