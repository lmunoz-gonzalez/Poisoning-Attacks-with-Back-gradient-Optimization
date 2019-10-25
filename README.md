# Poisoning Attacks with Back-gradient Optimization
Matlab code with an example of the poisoning attack described in the paper [**"Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization."**](https://dl.acm.org/citation.cfm?id=3140451) The code includes the attack against Adaline, Logistic Regression and a small MultiLayer Perceptron for MNIST dataset (using digits 1 and 7). 

## Use

To generate the random training/validation splits, first run the script *createSplits.m* in the "MNIST_splits" folder. Then, the scripts to run the attacks against Adaline, logistic regression and the MLP are *testAttackAdalineMNIST.m*, *testAttackLRmnist.m* and *testAttackMLPmnist.m* respectively.

## Citation

Please cite this paper if you use the code in this repository as part of a published research project.

```
@inproceedings{munoz2017towards,
  title={{Towards Poisoning of Deep Learning Algorithms with Back-gradient Optimization}},
  author={Mu{\~n}oz-Gonz{\'a}lez, Luis and Biggio, Battista and Demontis, Ambra and Paudice, Andrea and Wongrassamee, Vasin and Lupu, Emil C and Roli, Fabio},
  booktitle={Proceedings of the 10th ACM Workshop on Artificial Intelligence and Security},
  pages={27--38},
  year={2017}
}
```

## Related papers

You may also be interested some of our related papers on data poisoning: 
- [**"Poisoning Attacks with Generative Adversarial Nets."**](https://arxiv.org/pdf/1906.07773.pdf) L. Muñoz-González, B. Pfitzner, M. Russo, J. Carnerero-Cano, E.C. Lupu. ArXiv preprint arXiv:1906.07773, 2019 (*code available soon*).
- [**"Label Sanitization against Label Flipping Poisoning Attacks."**](http://www.research.ibm.com/labs/ireland/nemesis2018/pdf/paper1.pdf) A. Paudice, L. Muñoz-González, E.C. Lupu. Nemesis Workshop on Adversarial Machine Learning. Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 5-15, 2018.
- [**"Detection of Adversarial Training Examples in Poisoning Attacks through Anomaly Detection."**](https://arxiv.org/pdf/1802.03041.pdf) A. Paudice, L. Muñoz-González, A. Gyorgy, E.C. Lupu. ArXiv preprint: arXiv:1802.03041, 2018.


## About the authors

This research work has been a collaboration between the [Resilient Information Systems Security (RISS) group](rissgroup.org) at [Imperial College London](https://www.imperial.ac.uk/) and the [Pattern Recognition and Applications (PRA) Lab](https://pralab.diee.unica.it/en) at the [University of Cagliari](https://www.unica.it/unica/en/homepage.page). 
