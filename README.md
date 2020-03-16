# CoarseHash
Benchmark datasets used in ICRA 2020 paper: Fast, Compact and Highly Scalable Visual Place Recognition through Sequence-based Matching of Overloaded Representations

## Generating Pseudo Localization Datasets

### 1. Download Ingredient Datasets

#### Deep1B 
The dataset is available [here](https://yadi.sk/d/11eDCm7Dsn9GA) for download. A [download script](https://github.com/arbabenko/GNOIMI) is provided by the original authors.

We only use the descriptors within the base/base00 file, so you may only want to download that single file.


#### FAS100K
The dataset is described in the paper and its NetVLAD descriptors for both the reference and the query set are available for download [here](https://data.mendeley.com/datasets/zh5g5wbcj9/1).

The downloaded `.npz` comprises 7 ndarrays named `arr_0` to `arr_6`, comprising respectively reference descriptors, query descriptors, ground truth match indices for query data, reference poses (xyz), query poses (xyz), ignore, ignore. 

### 2. Generate 20K, 1M, and 10M

#### Prerequisites
```
numpy
scikit_learn
```

See `requirements.txt`, generated using `pipreqs==0.4.10` and `python3.5.6`


#### Run
Set the `path` and `dataset` variable ("20K", "1M" or "10M") and run `python preProcData.py` to generate the localization dataset. The "10M" dataset can take around 25 GB of RAM when performing PCA. A low-memory alternative would be to use [Incremental PCA](https://scikit-learn.org/stable/auto_examples/decomposition/plot_incremental_pca.html).


### License
The code is released under MIT License. FAS100K license is as specified on the download link. For Deep1B, refer to its original sources as mentioned above.


*If you find this repository useful or use these datasets, cite:*

Garg, Sourav, and Michael Milford. "Fast, Compact and Highly Scalable Visual Place Recognition through Sequence-based Matching of Overloaded Representations." In 2020 International Conference on Robotics and Automation (ICRA). IEEE, 2019.

bibtex:
```
@inproceedings{garg2020fast,
  title={Fast, Compact and Highly Scalable Visual Place Recognition through Sequence-based Matching of Overloaded Representations},
  author={Garg, Sourav and Milford, Michael},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2020}
}
```


