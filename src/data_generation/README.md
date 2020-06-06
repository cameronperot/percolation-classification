# Data Generation

The data is generated using the `Lattice2D` type and associated methods found in `Lattice2D.jl`.
This implements the Hoshen-Kopelman cluster labeling algorithm and then checks the labeled clusters to see if any of them span across the lattice in one of the dimensions.
More information on the HK algorithm can be found [here](https://www.ocf.berkeley.edu/~fricke/projects/hoshenkopelman/hoshenkopelman.html).

To generate the data run:

```bash
julia data_generation.jl
```

The data will be saved as `.npz` files to the `../../data` directory.
