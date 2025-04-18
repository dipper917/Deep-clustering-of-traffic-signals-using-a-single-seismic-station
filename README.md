# Deep-clustering-of-traffic-signals-using-a-single-seismic-station
We present an unsupervised deep clustering framework based on Deep Embedded Clustering (DEC) to identify traffic-induced signal segments from high-frequency seismic noise recorded by a single seismic station. This method enables cost-effective event extraction from ambient noise without labeled data or dense sensors, offering a alternative for urban traffic monitoring, particularly in areas with limited data or few monitoring devices.

The code provided here demonstrates the model training using data from Site 1. The implementation is adapted and modified from previous works(J. Xie et al., 2016; Snover et al., 2021; Mousavi et al., 2019).  

# Simulation and Field Data
The Traffic-induced Signals Simulation folder includes Python code for simulating traffic-induced signals. Key functions are:  
• col2row: Ensures correct vector orientation for matrix operations in seismic simulations.  
• SeismicCurve: Plots seismic waveforms as time-offset sections.  

Field data can be read and segmented using the provided MATLAB code. 
The field dataset used in this study can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.15239872). Please ensure that you download the appropriate dataset before running the code.  

The Model folder contains the trained DEC model files for the three sites.  
The label folder provides the labels of signal segments based on video annotations.

# Requirements-Package Versions:
• Python == 3.9.19  
• TensorFlow == 2.11.0  
• numpy == 1.26.4  
• scikit-learn == 1.5.2  
• pandas == 2.2.3  
• matplotlib == 3.9.2  
• h5py == 3.11.0

# References:
[1]Xie, J., Girshick, R., & Farhadi, A. (2016). Unsupervised deep embedding for clustering analysis. In Proceedings of the 33rd International Conference on Machine Learning (pp. 478–487). PMLR. Retrieved from https://proceedings.mlr.press/v48/xieb16.html  
[2]Snover, D., Johnson, C. W., Bianco, M. J., & Gerstoft, P. (2021). Deep Clustering to Identify Sources of Urban Seismic Noise in Long Beach, California. Seismological Research Letters, 92(2A), 1011–1022. https://doi.org/10.1785/0220200164  
[3]Mousavi, S. M., Zhu, W., Ellsworth, W., & Beroza, G. (2019). Unsupervised Clustering of Seismic Signals Using Deep Convolutional Autoencoders. IEEE Geoscience and Remote Sensing Letters, 16(11), 1693–1697. https://doi.org/10.1109/LGRS.2019.2909218  

## Related Code Repositorys:
https://github.com/dsnover/Unsupervised_Machine_Learning_for_Urban_Seismic_Noise  
https://github.com/smousavi05/Unsupervised_Deep_Learning  
https://github.com/Tony607/Keras_Deep_Clustering
