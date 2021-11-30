A Decomposition based on Constrained Clustering Algorithm for Job-shop Sccheduling
==============================

This folder contains the implementation of the paper "Decomposition-based job-shop scheduling with constrained clustering".

It contains the K-means constrained clustering algorithm we have developed to solve the Job-Shop Scheduling problem by decomposition approach.

Getting Started
------------

This work uses Ray's RLLib, Tensorflow and Wandb.

Make sure you have `git`, `cmake`, `zlib1g`, and, on Linux, `zlib1g-dev` installed.

You also need to have a Weight and Bias account to log your metrics. 
Otherwise, just remove all occurrence of wandb and log the metrics in another way.

```shell
git clone https://github.com/prosysscience/JSS
cd JSS
pip install -r requirements.txt
```

### Important: Your instance must follow [Taillard's specification](http://jobshop.jjvh.nl/explanation.php#taillard_def). 

Project Organization
------------


--------

## License

MIT License
