# Byzantine-tolerant distributed learning of finite mixture models

> ## Provenance — please read first
>
> **This is not the original repository.** It is a student reproduction, prepared as
> part of a graduate coursework essay on statistics in computer science.
>
> **Original work.** The algorithms, the `CTDGMR` package, `pmle.py`, `global_local.py`,
> `simulation.py`, `nist.py` and the population generators are the work of
> **Zhang, Tan & Chen** and come from the public code released with their paper (cited
> below). All intellectual credit for the method (COAT, DFMR, CTD-based mixture
> reduction) belongs to them. This repository carries no licence of its own precisely
> because most of the code is not mine to license — see the original release for terms.
>
> **What I changed or added**, and nothing else:
>
> | File | Change |
> | --- | --- |
> | `simulation/CTDGMR/utils.py`, `real_data/NIST/CTDGMR/utils.py` | `generate_random_covariance`: restored the `d⁻²` factor on the covariance-failure noise. The paper specifies `d⁻² Σᵢ ξᵢξᵢᵀ`; the published code omits it. Kept byte-identical across both copies. See the docstring for the evidence. |
> | `simulation/simulation.py` | Weight attack draws Dirichlet parameters from the **closed** interval `[10, 20]`; `np.arange(10, 20)` never draws 20, so it is now `np.arange(10, 21)`. |
> | `simulation/postprocess/aggregate.py` | Mine. Collapses stage-2 pickles into one tidy long-format CSV. |
> | `simulation/postprocess/essay_figures.py` | Mine. Produces the summary figure and methods table used in the essay. |
> | `simulation/generated_pop/generate_all.R` | Made portable — reads the optional `R_MIXSIM_LIB` instead of a hardcoded library path. |
>
> The two `utils.py` fixes change the numbers: under covariance failure the
> non-robust baseline goes from `1.39 → 4.09` (published code) to `0.085 → 0.152`
> (with `d⁻²`), the latter matching Figure 3 of the paper. Anyone comparing this
> repository against the original release should expect that difference.

The folder contains the code for both the simulated and real data analysis in the paper entitled **Byzantine-tolerant distributed learning of finite mixture models**. 
There are two subdirectories named real_data and simulation are the code for reproducing the simulation and the real data experiment in the paper respectively. 

## Reference

> Zhang, Q., Tan, Y. S., & Chen, J. *Byzantine-tolerant distributed learning of
> finite mixture models*. arXiv:2407.13980.
> <https://arxiv.org/abs/2407.13980> — doi:10.48550/arXiv.2407.13980

The paper itself is not redistributed in this repository; please obtain it from
the link above.

## Requirements
The code runs on Python 3.10. Package dependencies are listed in requirements.txt. To install the packages, run


```
pip install -r requirements.txt
```

## Simulation

To run the simulation, you first need to install the ``MixSim`` package in R and run the following code.

```
cd simulation/generated_pop
Rscript generate_all.R
```

``generate_all.R`` generates every ``(seed, K, d, overlap)`` configuration in a
single R session and skips configurations whose output files already exist. If
``MixSim`` is installed in a user library that is not on the default
``.libPaths()``, point ``R_MIXSIM_LIB`` at that directory:

```
R_MIXSIM_LIB=~/R/win-library/4.6 Rscript generate_all.R
```

This will produce the parameter values for the 300 repetitions in our experiment. The parameter values are stored in ``txt`` files under the ``generated_pop/true_param`` folder.


Then you can run the simulations by first fit local by running `global_local.py` file and then `simulation.py' file.
The following is the demo code when $n=5000$, $m=20$, MaxOmega=0.1, and under weight attack.

```
cd simulation
python global_local.py --ss 100000 --seed 1 --overlap 0.1 --n_split 20 
python simulation.py --ss 100000 --seed 1 --attack_mode 3 --overlap 0.1 --n_split 20
```

The output will be stored in a pickle file under output/save_data directory.
Then you can load the pickle file to post-process the simulation results.



## Real Data
The ``real_data`` folder contains the check point of the pretrained NN, you can also train it from scratch using `nn_feature_extractor.py` by following the instructions of the README file under real_data.


The NIST folder contains the code for our experiment.
To run an experiment, you can simply run 
```
python nist.py --local_ss 5000 --seed 1
```
