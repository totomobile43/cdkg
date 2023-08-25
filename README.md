# Computational Design of Passive and Active Kinesthetic Garments
This is the official repository for the EuroGraphics '22 and UIST '22 papers

Given a set of motions and a compliant garment with electrostatic clutches our method automatically generates efficient connecting structures. 
[teaser](./figures/teaser.png)


## Install & Configure
### Create and setup environment
- Create an environment using your favourite package manager.
- Installing Pytorch with GPU/CUDA support is recommended for faster simulation.
- Install aitviewer as ut contains most of the dependancies required in this package.
- Install pytorch3d. It can be installed on most systems, but may require to be compiled on Windows. 
``` 
conda create --name cdkg python=3.9  
pip install pytorch
[pip install aitviewer] .
[pip install pytorch3d]
```

Clone repo and optionally install it as a package: 
```
git clone https://github.com/totomobile43/cdkg
cd cdkg
[pip install -e .]
```

### Configure
#### Model Paths
We use the STAR body model based on the SMPL family of models. To install STAR see the
[STAR github repo](https://github.com/ahmedosman/STAR)

There is a `cdkg.yml` file that you can specify paths to data and models in. Make sure to set the STAR model path there.

#### Data Paths
By default, the `datasets.kg` variable will point to the `/data` folder, which should work out of the box. 
Some motions may require parts of the AMASS dataset, which you can also setup the path for here. 

#### Other settings
This settings file also includes settings for the BESO algorithm to control the target material ratio and the number of iterations.
You can override settings of the aitviewer here as well, for example, the width and height of the visualizer, and the floating point precision.

## Data
We include a number of useful garment designs and optimization results. 

`> garments` contains base garments that can be used as a starting point for simulation or TPO.

`> garments_opt_eg22` contains already optimized garments (results) from the EuroGraphics '22 Paper

`> garments_opt_uist_22` contains already optimized garments (results) from the UIST '22 Paper

`> clutches` contains clutches which can be attached to garments. Clutches will only attach to a particular garment (TODO: association needs to be labeled). 
For example, `clutches_3` only attaches to the `shirt` design. These are based on barycentric coordinates and face_ids which differ based on different garment topologies.

`> eso` contains progress states ( *animations* ) of the BESO algorithm working (~1.6 GB). They need to be downloaded separately at: [https://drive.google.com/file/d/1sf1RnZ1aveV2nwSqPyc148j1R1SrbXsm/view?usp=sharing](https://drive.google.com/file/d/1sf1RnZ1aveV2nwSqPyc148j1R1SrbXsm/view?usp=sharing)

## Apps
In order to load, visualize, simulate, and optimize garments, we include a number of useful applications:

`> design.py` can be used to load existing full or optimized garments. You can also edit existing garments to add or remove clutches, and edit attachments. 

`> sim.py` can be used to load and then simulate existing garments. 

`> tpo_run.py` will run BESO topology optimization with the specified parameters. Results will be saved directly to disk as they are very memory intensive to keep alongside visualization. 

`> tpo_load.py` can load existing tpo progress animations (see `eso` dir above). 

Note: There are no command line switches, just edit the code to load/run what you need.

## Code
The code consists of 2 main parts - `renderables` and `models`. 

Renderables describe the geometry (and parametric construction of geometry) for the inputs (garments, clutches, bodies)

Models are fairly generic formulations to compute energies used in the simulation and in the implementation of the BESO TPO algorithm.

Note: `cs_tri` is used to compute energies for constant strain triangles, while `cs_tet` is for tetrehedral elements (which is not part of the paper, but I am including the code here).


## Cite

Please cite our papers if you find them useful, or if you use the code in this repository:

```
@inproceedings{vechev2022computational,
  title={Computational Design of Kinesthetic Garments},
  author={Vechev, Velko and Zarate, Juan and Thomaszewski, Bernhard and Hilliges, Otmar},
  booktitle={Computer Graphics Forum},
  volume={41},
  number={2},
  pages={535--546},
  year={2022},
  organization={Wiley Online Library}
}

@inproceedings{vechev2022computational,
  title={Computational Design of Active Kinesthetic Garments},
  author={Vechev, Velko and Hinchet, Ronan and Coros, Stelian and Thomaszewski, Bernhard and Hilliges, Otmar},
  booktitle={Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology},
  pages={1--11},
  year={2022}
}
```

## Acknowledgements
This work was supported in part by grants from the Hasler Foundation (Switzerland) and funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme grant agreement No 717054.
We would like to thank Manuel Kaufmann and Dario Mylonopoulos for their help with visualization support. We also want to thank Thomas Langerak for assistance during the deadlines.