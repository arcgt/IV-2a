# Fast and Accurate Multiclass Inference for MI-BCIs Using Large Multiscale Temporal and Spectral Features

This is the code of a conference paper for EUSIPCO 2018. Feel free to check our results. 

## Getting Started

First, download the source code.
Then, download the dataset "Four class motor imagery (001-2014)" of the BCI competition IV-2a from: http://bnci-horizon-2020.eu/database/data-sets
Put all files of the dataset (A01T.mat-A09E.mat) into a subfolder within the project called 'dataset' or change self.data_path in main_csp and main_riemannian. 

### Prerequisites

- python3
- numpy
- sklearn
- pyriemann
- scipy


### Recreate results

For the recreation of the CSP results run main_csp.py. 
Change self.svm_kernel for testing different kernels:
- self.svm_kernel='linear'  -> self.svm_c = 0.05
- self.svm_kernel='rbf'     -> self.svm_c = 20
- self.svm_kernel='poly'    -> self.svm_c = 0.1

```
python3 main_csp.py
```
For the recreation of the Riemannian results run main_riemannian.py. 
Change self.svm_kernel for testing different kernels:
- self.svm_kernel='linear'  -> self.svm_c = 0.1
- self.svm_kernel='rbf'     -> self.svm_c = 20

Change self.settings for testing different means (0 -> Riemannian mean, 1 -> euclid mean, 2 -> identity matix):
- self.settings=0
- self.settings=1 
- self.settings=2

```
python3 main_riemannian.py
```

## Authors

* **Michael Hersche** - *Initial work* - [MHersche](https://github.com/MHersche)
* **Tino Rellstab** - *Initial work* - [tinorellstab](https://github.com/tinorellstab)
