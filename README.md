# Verifiable Noise Reduction for Privacy-Preserving and Robust Federated Learning

This is the official repository for the paper "Verifiable Noise Reduction for Privacy-Preserving and Robust Federated Learning." The code is built using the [Flower federated learning framework](https://flower.ai/), version 1.31.0. The other dependencies are automatically installed by Flower. The script ``tests.py`` is configured to run the accuracy and robustness tests detailed in the paper. By enabling the ``write-time`` and ``measure-messages`` flags in ``pyproject.toml``, you can have the program write end-to-end training time or average message size to an external file. The implementations of centralized learning are in the ``centralizedLearningTests`` directory. The implementations of FedAvg, FedDMC, and FedDMC with standard DP (for efficiency tests) are in the ``timingTests/OneServer`` directory. To run the efficiency tests for those three frameworks, replace the ``client_app.py``, ``nrfl_strategy.py``, and ``server_app.py`` files in the ``src`` directory with the ones in ``timingTests/OneServer``. To enable FedAvg, in ``pyproject.toml``, set:
* ``use-dp = false``
* ``noise-reduction = false``
* ``use-feddmc = false``

Similarly, to enable FedDMC, use the above settings with ``use-feddmc = true``, and for FedDMC with standard DP, use ``use-feddmc = true`` and ``use-dp = true``. Finally, to run the efficiency tests for the two-server protocol, replace the ``client_app.py`` and ``nrfl_strategy.py`` files in the ``src`` directory with the ones in ``timingTests/TwoServer``.
## Suplementary Materials

Detailed experimental results are contained in the ``results`` directory. Tables detailing results of robustness tests against the five attacks evaluated in the FedDMC paper are in ``results/supplementary_tables.pdf``. The file ``results/runinfo.csv`` lists experiment IDs and the parameters used in each experiment; for each experiment, the model loss/accuracy, as well as (when applicable) detection accuracy/precision/recall and attack success rate at each round of training in the csv named with the experiment's ID number.
## Citation

TODO: add citation, make sure scripts and pyproject.toml are correct
