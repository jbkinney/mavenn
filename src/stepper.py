import pymc
import scipy as sp
'''Script for updating our MCMC'''

class MatColumnMetropolis(pymc.Metropolis):
    def __init__(self,stochastic):
        pymc.Metropolis.__init__(self,stochastic)

    def propose(self):
        sigma = 0.5*self.adaptive_scale_factor
        j_p = sp.random.randint(self.stochastic.value.shape[1]) # column to perturb
        emat_temp = self.stochastic.value.copy()
        emat_temp[:,j_p] = emat_temp[:,j_p] + sigma*sp.randn(4)
   
        self.stochastic.value = emat_temp

class GaugePreservingStepper(pymc.Metropolis):
    """Perform monte carlo steps that preserve the following choise of gauge:
    sum of elements in each column = 0, overall matrix norm = 1."""
    def __init__(self,stochastic):
        pymc.Metropolis.__init__(self,stochastic)

    def propose(self):
        # to avoid overwriting weirdness
        emat_temp = self.stochastic.value.copy()
        # number of columns in matrix
        num_col = emat_temp.shape[1]
        # first choice a random 4L dimensional vector
        r = sp.random.standard_normal(emat_temp.shape)
        # dot product of proposed direction with current vector
        lambda_0 = sp.sum(emat_temp*r)
        # dot products of each column with basis vectors
        lambda_vec = 0.5*sp.sum(r,axis=0)

        s = sp.zeros_like(r)
        # s is a vector based on r, but along the surface of the hypersphere
        for j in range(emat_temp.shape[1]):
            s[:,j] = r[:,j] - lambda_0*emat_temp[:,j] - lambda_vec[j]*(0.5*sp.ones(emat_temp.shape[0]))
        dx = self.adaptive_scale_factor*s/sp.sqrt(sp.sum(s*s))

        self.stochastic.value = (emat_temp+dx)/sp.sqrt(sp.sum((emat_temp+dx)**2))
