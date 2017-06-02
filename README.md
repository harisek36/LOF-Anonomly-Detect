# Optimized LOF Unsupervised Anomoly Detection

This repository includes four different implementations of the local outlier factor (LOF) unsupervised annomoly detection algorithm.  I found that using a KD-tree to perform k nearest neighbor search performed much better that creating a distance matrix to do this same thing.  I also optimized LOF computations using numpy.  The quickest implementation of the algorithm is included in KD-Optimized_Lof.py.  

## Authors

* **Dylan Slack** 

## References

I include a full list of references in the attached paper.  However please not that naive_lof.py is taken from https://github.com/damjankuznar/pylof/blob/master/lof.py.  All credit goes to him for that implementation.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
