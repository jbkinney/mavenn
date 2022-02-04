"""
2022.02.04
----------
model2.py: Defines the Model() class, which represents all MAVE-NN2 models.
Unlike version, this model class contains new features such as custom
measurement processes, new measurement processes such as tite-seq,
multi-latent phenotype models, and more.

Pseudocode showing some of the updated workflow and new features

# Define GP map (sets dimensionality of latent phenotype phi)
gpmap = ThermodynamicGPMap(...)

# Define measurement processes (specify dimensions of phi and form of y)
mp_ge = GEMeasurementProcess(...)
mp_mpa = MPAMeasurementProcess(...)

# Define model
model = Model(gpmap = gpmap,
              mplist = [mp_ge, mp_mpa])

# Set data
model.set_data(x = x,
               y_list = [y_ge, y_mpa],
               validation_flags = validation_flags)

# Fit model
model.fit(...)
"""

# TODO: init will have to import gpmap so user can instantiate them (create notebook that does gpmap == gpmap.additive())
# TODO: init will also have to import MP class to users can instantiate them and pass them to model


class Model:

    """
     Represents a MAVE-NN (version 2) model, which includes a genotype-phenotype (G-P) map
     as well as a list of measurement processes.

     Parameters
     ----------
     gpmap: (MAVE-NN gpmap)
         MAVE-NN's Genotype-phenotype object.

     mp_list: (list)
        List of measurement processes. 

    """


    def __init__(self,
                 gpmap,
                 mp_list):

        self.gpmap = gpmap
        self.mp_list = mp_list

        pass