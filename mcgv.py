import numpy             as np
import matplotlib.pyplot as plt

from astropy                 import constants as const
from astropy                 import units as u
from astropy.visualization   import astropy_mpl_style, quantity_support
from scipy.interpolate       import griddata

import ntdisk

######################################################
class mcgv:
    ######################################################
    def __init__(self, mydisk, ntheta): # Input is an ntdisk class
        self.mydisk      = mydisk
        mdotedd          = 4 * np.pi * const.G.cgs * self.mydisk.mbh / (0.1 * const.c.cgs * const.sigma_T.cgs)
        self.Gamma       = (self.mydisk.mdot / mdotedd).decompose()

        # Create grids in r and theta (spherical coordinates)
        # For r, use the same grid as the cylindrical r coordinate in myspec
        self.nr          = self.mydisk.nr
        self.rlo         = self.mydisk.rlo
        self.rhi         = self.mydisk.rhi
        self.rstar       = self.mydisk.rstar
        self.zt1         = self.mydisk.zt1

        self.ntheta      = ntheta
        self.theta       = np.linspace(0,2.0*np.pi,num=self.ntheta)

        self.vr          = np.empty([self.nr,self.ntheta])
        self.vtheta      = np.empty([self.nr,self.ntheta])
        self.vphi        = np.empty([self.nr,self.ntheta])
        self.density     = np.empty([self.nr,self.ntheta])
        self.temperature = np.empty([self.nr,self.ntheta])
        self.pgas        = np.empty([self.nr,self.ntheta])

        for rdx in range(self.nr):
            for tdx in range(self.ntheta):
                # self.mydisk.verticaldensity_onepoint and self.mydisk.verticaltemperature_onepoint take the cylindrical r and z coordinates
                # to compute the denisty and temperature
                z    = self.rstar[rdx] * np.sin(self.theta[tdx])
                rcyl = self.rstar[rdx] * np.cos(self.theta[tdx])
                self.density[rdx,tdx] = self.mydisk.verticaldensity_onepoint(self.rstar[rdx] * np.cos(self.theta[tdx]),
                                                                             self.rstar[rdx] * np.sin(self.theta[tdx]))
                self.temperature[rdx,tdx] = self.mydisk.verticaltemperature_onepoint(self.rstar[rdx] * np.cos(self.theta[tdx]),
                                                                                     self.rstar[rdx] * np.sin(self.theta[tdx]))
                # Use the ideal gas law to get the pressure
                self.pgas[rdx,tdx] = const.k_B.cgs * self.density[rdx,tdx] * self.temperature[rdx,tdx] / const.u.cgs

                # The domain of the calculation should be determined by where the tau=1 surface from mydisk is (using mydisk.zt1, zt1 is in units of rg)...
                rcdx = np.min(np.extract(rcyl > self.mydisk.rstar, range(self.nr)))
                if z < mydisk.zt1[rcdx]: # Then we are inside the disk
                    self.vr[i,j]     = 0 * u.cm / u.s
                    self.vtheta[i,j] = 0 * u.cm / u.s
                    self.vphi[i,j]   = np.sqrt(const.G.cgs * self.mydisk.mbh / (self.mydisk.rstar[rcdx] * self.mydisk.rg)) # The circular velocity

        ######################################################
        # These tables are formatted to be easy to parse (see the python example below),
        # and a C++ interface to simulation codes is provided. The first entry in each
        # table is N_xi, the number of photoionization parameter values. The remainder 
        # of the first row contains the N_xi values of log10(xi). The remainder of the
        # first column is all the log10 values of the optical depth parameter, t. The
        # entries corresponding to a given (t,xi) pair are the values of log10(M), where
        # M is the force multiplier.
        #
        # fmult[0,:] = log10(xi)    xi = 4 np.pi Fx / nH     Fx = 0.1-1000 Ryd integrated flux
        # fmult[:,0] = log10(t)
        fmultfile = "D:/ganguly/AALSynth2/DPKW19_tables/DPKW19_tables/AGN1_Fmult.dat"
        self.fmultarray = np.genfromtxt(fmultfile)

        # Call as fmultfunc((lgt,lgxi))
        self.fmultgridfunc = RegularGridInterpolator((self.fmultarray[1:,0],self.fmultarray[0,1:]), self.fmultarray[1:,1:], bounds_error=False, fill_value=0) 


    ######################################################
    def calcstreamline(self, rf):
        # Treat rf as the cylyndrical r coordinate. Find the corresponding z coordinate.
        z0 = (np.interp(rf,self.rstar,self.zt1))[0]

        # Then the vertical (z) component of the velocity
        # Vertical - using the sound speed for a gamma=5/3 gas, but maybe that is not quite right... cgs units cm/s
        vz0 = ((np.sqrt(5 * const.k_B.cgs * self.tempt1[r] / (3 * const.u.cgs))).decompose(bases=u.cgs.bases)) 

        # Transform vz0 into spherical r and theta components
        r      = np.sqrt(rf * rf + z0 * z0)
        theta  = np.atan(rf/z0)
        vr     = np.array([vz0 * np.sin(theta)])
        vtheta = np.array([vz0 * np.cos(theta)])
        
        done = 0
        dt   = 0.1
        i    = -1
        while done == 0:
            r_ip1     = r[i] + (vr[i]/self.rg) * dt
            theta_ip1 = theta[i] + vtheta[i] * dt / (self.rg * r[i])

            # Grab the frce multiplier from self.mydisk.fmltfunc (which should have been defined by a previous call to self.mydisk.makedisk
            Mr,Mtheta = self.fmultfunc(r[i],theta[i])


            if (vr.size > 2):
                dr = r_ip1 - r[i-1]
                vr_im1 = vr[i-1]

            else:
                dr = 2 * (r_ip1 - r[i])
                vr_im1 = vr[i]

            vr_ip1  = vr_im1
            vr_ip1 -= (dr * self.mydisk.rg / vr[i]) * (const.G.cgs * self.mydisk.mbh / np.square(r[i] * self.mydisk.rg)) * ((1-self.Gamma)*(1-r[0]/r[i]) + self.Gamma * Mr)
            vr_ip1 -= dPr / (vr[i] * rho)

            r     = np.append(    r,     r_ip1)
            theta = np.append(theta, theta_ip1)
            vr    = np.append(   vr,    vr_ip1)
            
    ######################################################
    def fmultfunc(self, r, theta):

