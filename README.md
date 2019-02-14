# Convmaps
Generate HEALPix convergence maps from cosmological N-body simulations. 
The convergence field is directly estimated from the simulated density field using the Born approximation.

The implementation follows previous work (e.g. [Teyssier et al. (2009)](https://arxiv.org/abs/0807.3651)). Additionally 
there is also an optional replication procedure implemented that allows the construction of large scale volumes from 
small simulated sub-volumes. 


## Installing

Install by cloning the repository:

```
git clone https://github.com/ajefweiss/convmaps.git
```

At this point it is highly recommended to use a python virtual environment.
Make sure you have the latest version of `virtualenv` installed and create a new virtual environment.

```
cd convmaps
virtualenv -p /usr/bin/python3.7 venv
```

Enter the virtual environment and install all required packages:


```
source venv/bin/activate
pip install -r requirements.txt
```

Note that if you want to use MPI parallelization you will need to additionally install the `mpi4py`.

```
pip install mpi4py>=2.1.0
```

## Basic Usage

Given a set of snapshots from a N-body simulation we can then easily generate the convergence map 
(using default parameters).

```
python convmaps/mapgen.py -p SNAPSHOT_FOLDER -o output/healpixmap.npy 
```

Note that the snapshot format, volume and particle count must be given as command line arguments. 

The code also works
with a single snapshot. In this case the shell boundaries must be given in redshift. The resulting maps from multiple 
snapshots must then be added manually to construct the full convergence map spanning the entire lightcone.

```
python convmaps/mapgen.py -p $SNAPSHOT -o output/healpixmap_part.npy --lc-shell-range $Z1 $Z2
```

## Features

One can run `python convmaps/convgen.py --help` to list all command line arguments.

Currently only the TIPSY format is implemented (PKDGRAV3 output).

Different lightcone modes are available.
* `--lc-mode fullsky`: the simulation volume is replicated in order to generate a full sky convergence map
* `--lc-mode smallsky`: the simulation volume is only replicated along one axis, survey area depends on the original 
simulation volume and the number of required replications to reach the maximum redshift
* `--lc-mode simple`: no replication (this will lead to incorrect results if the lightcone reaches further than half 
the simulation volume)

During the replication process the replicated boxes can be "randomized". Each replicated box has its axes 
rotated/inversed randomly. Then its particles are shifted by a random vector (the box boundaries are periodic). By
choosing different random seeds for this process multiple maps can be generated from the same N-body simulation. This
feature can be enabled using:

```
--randomize 
--seed 42
```

Note that the randomization process, in its current implementation, requires significantly more computing time.
