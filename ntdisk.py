import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from scipy import special
from scipy import interpolate
from astropy.modeling.models import BlackBody

######################################################

def surfdens(z, z0, rho0):
    s   = np.empty(z.size)
    udx = np.extract(z <= z0, np.arange(0,s.size,1))
    vdx = np.extract(z > z0,  np.arange(0,s.size,1))
    if udx.size > 0:
        s[udx] = np.sqrt(np.pi)*special.erf(z[udx]/z0) / 2.0
    if vdx.size > 0:
        s[vdx] = (np.sqrt(np.pi)*special.erf(1) / 2.0 + np.exp(-2) * (1 - np.exp(1-z[vdx]/z0)))
    return s * z0 * rho0

######################################################

def verticaldensity(z, z0, rho0):
    if z.size > 1:
        rho  = rho0 * np.zeros(z.size)
        udx = np.extract(z <= z0, np.arange(0,rho.size,1))
        vdx = np.extract(z > z0,  np.arange(0,rho.size,1))
        if udx.size > 0:
            rho[udx] = rho0 * np.exp(-np.multiply(z[udx]/z0,z[udx]/z0))
        if vdx.size > 0:
            rho[vdx] = rho0 * np.exp(-z[vdx]/z0)
    else:
        if z <= z0:
            rho = rho0 * np.exp(-(z/z0)*(z/z0))
        else:
            rho = rho0 * np.exp(-z/z0)
    return rho

######################################################

def verticaltemperature(z, z0, rho0, t0):
    zp = np.append(z,100.0*z0)
    sd = surfdens(zp,z0,rho0)
    sd0 = 2.0 * np.squeeze(sd[sd.size-1])
    sdc = np.copy(sd[0:sd.size-1])
    sdr = 1 - 4*np.power(sdc/sd0,2)
    return t0 * np.power(sdr, 0.25)


######################################################

class ntdisk:
    def __init__(self, sbh, mbh, mdot, alpha, inclination, robs, nr, rlo, rhi):
        self.sbh         = sbh
        self.mbh         = mbh
        self.mdot        = mdot
        self.alpha       = alpha
        self.inclination = inclination
        self.robs        = robs
        self.zobs        = robs / np.tan(self.inclination)
        self.thetaobs    = 0.0
        self.nr          = nr
        self.rlo         = rlo
        self.rhi         = rhi
        self.iref        = 0
        self.diskheight  = np.empty(self.nr)
        self.zt1         = np.empty(self.nr)
        self.dzdr        = np.empty(self.nr)
        self.temperature = np.empty(self.nr)
        self.tempt1      = np.empty(self.nr)
        self.density     = np.empty(self.nr)
        self.denst1      = np.empty(self.nr)
        z1    = np.power(1.0+self.sbh, 1./3.) + np.power(1.0-self.sbh,1./3.)
        z1    = 1.0 + np.power(1.0-self.sbh*self.sbh,1./3.) * ( z1 )
        z2    = np.sqrt(3.0 * self.sbh * self.sbh + z1*z1)
        self.rms   = 3.0 + z2 - np.sqrt((3.0-z1)*(3+z1+2*z2))
        y0    = np.sqrt(self.rms)
        self.rstar = np.squeeze(np.logspace(np.log10(self.rms * 1.01), np.log10(self.rms * self.rhi), num = self.nr))
        ######################################################
        # Compute the annuli thicknesses
        # rstar = np.squeeze(np.logspace(np.log10(rms*1.01),6.0,num=3000))
        # 0.5*(rstar[i+1]-rstar[i-1]) = rstar[i]*0.5*(f-1/f)
        self.deltar = np.empty(self.nr)
        self.ntheta = np.empty(self.nr, dtype=np.int8)
        for i in range(1,self.nr-1):
            self.deltar[i] = self.rstar[i] * (np.sqrt(self.rstar[i+1] / self.rstar[i-1])-1)
        self.deltar[0]  = self.deltar[1]
        self.deltar[-1] = self.deltar[-2]
        self.ntheta = np.ceil(2.0 * np.pi * np.multiply(self.rstar, 0.5 / self.deltar))

    def makedisk(self):
        rg       = const.G.cgs * self.mbh * const.M_sun.cgs / (const.c.cgs * const.c.cgs)
        mstar    = self.mbh / 3.0
        mdotstar = self.mdot * 1.998e+33 / (365.2422 * 24 * 3600) / 1.0e+17
        ######################################################
        y     = np.sqrt(self.rstar)
        ######################################################
        a = 1.0 + self.sbh * self.sbh * y * y * y * y * (1.0 + 2.0/(y * y))
        b = 1.0 + self.sbh / (y * y * y)
        c = 1.0 - 3.0 / (y * y) + 2.0 * self.sbh * self.sbh / (y * y * y)
        d = 1.0 - 2 / (y * y) + self.sbh * self.sbh / (y * y * y * y)
        e = 1.0 + self.sbh * self.sbh * (4.0 - 4.0 / (y * y) + 3.0 / (y * y * y * y)) / (y * y * y * y)
        f = 1.0 - 2.0 * self.sbh / (y * y * y) + self.sbh * self.sbh / (y * y * y * y)
        g = 1.0 - 2.0 / (y * y) + self.sbh / (y * y * y)
        r = f * f / c - self.sbh * self.sbh * (g / np.sqrt(c) - 1) / (y * y)
        s = a * a * c * r / (b * b * d)
        ######################################################
        y0 = np.sqrt(self.rms)
        y1 = 2.0 * np.cos((np.arccos(self.sbh)-np.pi)/3.0)
        y2 = 2.0 * np.cos((np.arccos(self.sbh)+np.pi)/3.0)
        y3 = -2.0 * np.cos(np.arccos(self.sbh)/3.0)
        ######################################################
        q0 = b / (y * np.sqrt(c))
        q1 = y - y0
        q2 = -1.5 * self.sbh * np.log(y / y0)
        q3 = -3 * (y1 - self.sbh) * (y1 - self.sbh) * np.log((y - y1) / (y0 - y1)) / (y1 * (y1 - y2) * (y1 - y3))
        q4 = -3 * (y2 - self.sbh) * (y2 - self.sbh) * np.log((y - y2) / (y0 - y2)) / (y2 * (y2 - y1) * (y2 - y3))
        q5 = -3 * (y3 - self.sbh) * (y3 - self.sbh) * np.log((y - y3) / (y0 - y3)) / (y3 * (y3 - y1) * (y3 - y2))
        q = q0 * (q1 + q2 + q3 + q4 + q5)
        ######################################################
        diskheightinner   = (1.0e+5 * u.cm / rg)     * np.power(self.alpha,0)     * np.power(mstar, 0)     *          mdotstar        * np.power(r,0)   * np.power(y,0)      * np.power(a,2)      * np.power(b,-3)     * np.power(c,1/2) * np.power(d,-1)     * np.power(s,-1)     *          q
        diskheightmiddle  = (3.0e+3 * u.cm / rg)     * np.power(self.alpha,-1/10) * np.power(mstar, 9/10)  * np.power(mdotstar,2/10)  * np.power(r,0)   * np.power(y,21/20)  *          a         * np.power(b,-6/5)   * np.power(c,1/2) * np.power(d,-3/5)   * np.power(s,-1/2)   * np.power(q,1/5)
        diskheightouter   = ( 900.0 * u.cm / rg)     * np.power(self.alpha,-1/10) * np.power(mstar, 9/10)  * np.power(mdotstar,3/10)  * np.power(r,9/8) * np.power(y,0)      * np.power(a,19/20)  * np.power(b,-11/10) * np.power(c,1/2) * np.power(d,-23/40) * np.power(s,-19/40) * np.power(q,3/40)
        temperatureinner  = (4.0e+7 * u.Kelvin)      * np.power(self.alpha,-1/4)  * np.power(mstar, -1/4)  * np.power(mdotstar,0)     * np.power(r,0)   * np.power(y,-3/4)   * np.power(a,-1/2)   * np.power(b,1/2)    * np.power(c,0)   * np.power(d,0)      * np.power(s,1/4)    * np.power(q,0)
        temperaturemiddle = (3.0e+8 * u.Kelvin)      * np.power(self.alpha,-1/5)  * np.power(mstar, -3/5)  * np.power(mdotstar,2/5)   * np.power(r,0)   * np.power(y,-9/5)   * np.power(a,0)      * np.power(b,-2/5)   * np.power(c,0)   * np.power(d,-1/5)   * np.power(s,0)      * np.power(q,2/5)
        temperatureouter  = (8.0e+7 * u.Kelvin)      * np.power(self.alpha,-1/5)  * np.power(mstar, -1/2)  * np.power(mdotstar,3/10)  * np.power(r,0)   * np.power(y,-3/2)   * np.power(a,-1/10)  * np.power(b,-1/5)   * np.power(c,0)   * np.power(d,-3/20)  * np.power(s,1/20)   * np.power(q,3/10)
        densityinner      = (1.0e-4 * u.g / u.cm**3) * np.power(self.alpha,-1)    *          mstar         * np.power(mdotstar,-2)    * np.power(r,0)   * np.power(y,3)      * np.power(a,-4)     * np.power(b,6)      * np.power(c,0)   *          d         * np.power(s,2)      * np.power(q,-2)    / const.u.cgs
        densitymiddle     = (10.0   * u.g / u.cm**3) * np.power(self.alpha,-7/10) * np.power(mstar,-11/10) * np.power(mdotstar,2/5)   * np.power(r,0)   * np.power(y,-33/10) * np.power(a,-1)     * np.power(b,3/5)    * np.power(c,0)   * np.power(d,-1/5)   * np.power(s,1/2)    * np.power(q,2/5)   / const.u.cgs
        densityouter      = (80.0   * u.g / u.cm**3) * np.power(self.alpha,-7/10) * np.power(mstar,-5/4)   * np.power(mdotstar,11/20) * np.power(r,0)   * np.power(y,-15/4)  * np.power(a,-17/20) * np.power(b,3/10)   * np.power(c,0)   * np.power(d,-11/40) * np.power(s,17/40)  * np.power(q,11/20) / const.u.cgs
        pratio            = (5e-5)                   * np.power(self.alpha,-1/4)  * np.power(mstar, 7/4)   * np.power(mdotstar,-2)    * np.power(r,0)   * np.power(y,21/4)   * np.power(a,-5/2)   * np.power(b,9/2)    * np.power(c,0)   *          d         * np.power(s,5/4)    * np.power(q,-2)
        tratio            = (6e-6)                                                *          mstar         * np.power(mdotstar,-1)    * np.power(r,0)   * np.power(y,3)      * np.power(a,-1)     * np.power(b,2)      * np.power(c,0)   * np.power(d,1/2)    * np.power(s,1/2)    * np.power(q,-1)
        ######################################################
        pdx    = np.extract(pratio > 1, np.arange(0,pratio.size,1))
        tdx    = np.extract(tratio > 1, np.arange(0,tratio.size,1))
        while self.rstar[pdx[0]] < 6:
            pdx = np.delete(pdx,0)
        while self.rstar[tdx[0]] < 6:
            tdx = np.delete(tdx,0)
        ######################################################
        dpdx = np.copy(pdx)
        dpdx[1:pdx.size-2] = 0.5*(pdx[2:pdx.size-1]-pdx[0:pdx.size-3])
        pdx = np.extract(dpdx == 1, pdx)
        ######################################################
        dtdx = np.copy(tdx)
        dtdx[1:tdx.size-2] = 0.5*(tdx[2:tdx.size-1]-tdx[0:tdx.size-3])
        tdx = np.extract(dtdx == 1, tdx)
        ######################################################
        self.diskheight       = np.copy(diskheightinner)
        self.temperature      = np.copy(temperatureinner)
        self.density          = np.copy(densityinner)
        self.diskheight[pdx]  = np.copy(diskheightmiddle[pdx])  * self.diskheight[pdx[0]]  / diskheightmiddle[pdx[0]]
        self.temperature[pdx] = np.copy(temperaturemiddle[pdx]) * self.temperature[pdx[0]] / temperaturemiddle[pdx[0]]
        self.density[pdx]     = np.copy(densitymiddle[pdx])     * self.density[pdx[0]]     / densitymiddle[pdx[0]]
        self.diskheight[tdx]  = np.copy(diskheightouter[tdx])   * self.diskheight[tdx[0]]  / diskheightouter[tdx[0]]
        self.temperature[tdx] = np.copy(temperatureouter[tdx])  * self.temperature[tdx[0]] / temperatureouter[tdx[0]]
        self.density[tdx]     = np.copy(densityouter[tdx])      * self.density[tdx[0]]     / densityouter[tdx[0]]
        ######################################################
        x1 = self.rstar[pdx[0]]
        x2 = self.rstar[tdx[0]]
        print('Pressure change at ',x1,' Rg (',x1*rg,')  Optical depth change at ',x2,' Rg (',x2*rg,')')
        ######################################################

    def photosphere(self):
        self.zt1 = np.empty(0)
        filename = "Sbh{}-MBH{:.2f}-Mdot{}-alpha{}"+".dat"
        if os.path.isfile(filename.format(self.sbh, np.log10(self.mbh), self.mdot, self.alpha)):
            print("Reading "+filename.format(self.sbh, np.log10(self.mbh), self.mdot, self.alpha))
            f = open(filename.format(self.sbh, np.log10(self.mbh), self.mdot, self.alpha),"r")
            for x in f:
                y = x.split()
                if self.zt1.size == 0:
                    self.zt1    = np.array([float(y[2])])
                    self.tempt1 = np.array([float(y[6])]) 
                    self.denst1 = np.array([float(y[8])])
                else:
                    self.zt1    = np.append(self.zt1,    float(y[2]))
                    self.tempt1 = np.append(self.tempt1, float(y[6]))
                    self.denst1 = np.append(self.denst1, float(y[8]))
            self.tempt1 *= u.Kelvin
            self.denst1 /= u.cm**3
        else:
            print("Writing "+filename.format(self.sbh, np.log10(self.mbh), self.mdot, self.alpha))
            self.zt1  = np.zeros(self.rstar.size)
            self.dzdr = np.empty(self.rstar.size)
            self.tempt1 = np.empty(self.rstar.size) * u.Kelvin
            self.denst1 = np.empty(self.rstar.size) / u.cm**3
            nit = 0
            zlo = 1.0e-4
            zhi = 1.0e+2
            for i in range(self.rstar.size):
                done = 0
                nit = 0
                while done == 0:
                    z          = np.squeeze(np.logspace(np.log10(zlo),np.log10(zhi),num=np.power(10,4+nit)))
                    rhoz       = verticaldensity(z * rg, diskheight[i] * rg, density[i]) * const.u.cgs
                    rdx        = np.extract(rhoz * u.cm**3 / u.g <= 1.0e-5 * const.u.cgs / u.g, np.arange(0,rhoz.size,1))
                    rhoz[rdx]  = 1.0e-5 * const.u.cgs / u.cm**3
                    zp         = np.append(z,100.0*self.diskheight[i])
                    sd         = surfdens(zp * rg, self.diskheight[i] * rg, self.density[i])
                    sd0        = 2.0 * np.squeeze(sd[sd.size-1])
                    sdc        = np.copy(sd[0:sd.size-1])
                    sdr        = 1 - 4*np.power(sdc/sd0,2)
                    tempz      = self.temperature[i] * np.power(sdr, 0.25)
                    tmpdx      = np.extract(tempz / u.Kelvin <= 1.0e-5, np.arange(0,tempz.size,1))
                    if tmpdx.size > 0:
                        tempz[tmpdx] = tempz[np.min(tmpdx)-1] + 1.0e-5 * u.Kelvin
                    ffopacity  = (0.64e+23 * u.cm**2 / u.g) * np.multiply((rhoz * u.cm**3 / u.g),np.power(tempz / u.Kelvin, -7/2))
                    opacity    = (ffopacity + 0.4* u.cm**2 / u.g) * const.u.cgs
                    sd = surfdens(z * rg, self.diskheight[i] * rg, self.density[i])
                    dsd        = 0.5*(sd[2:sd.size]-sd[0:sd.size-2])
                    dsd        = np.insert(dsd,0,dsd[0])
                    dsd        = np.append(dsd,dsd[-1])
                    dtau       = np.multiply(opacity,dsd)
                    opticaldepth = np.zeros(dtau.size)
                    tracko = 0
                    for o in range(dtau.size):
                        opticaldepth[o] = np.sum(dtau[o:])
                        if opticaldepth[o] > 1:
                            tracko = o
                    done = 1
                    if (np.max(opticaldepth) < 1) or (np.min(opticaldepth) > 1) or (opticaldepth[tracko]/opticaldepth[tracko+1]-1 > 1.0e-2):
                        done = 0
                        zlo *= 0.99
                        zhi *= 1.01
                    if done == 0:
                        nit += 1
                self.zt1[i]    = z[tracko] + (z[tracko+1]-z[tracko])*(opticaldepth[tracko+1]-1)/(opticaldepth[tracko+1]-opticaldepth[tracko])
                self.tempt1[i] = verticaltemperature(zt1[i] * rg, diskheight[i] * rg, self.density[i], self.temperature[i])
                if self.zt1[i] <= self.diskheight[i]:
                    self.denst1[i] = self.density[i] * np.exp(-np.power(self.zt1[i]/self.diskheight[i], 2))
                else:
                    self.denst1[i] = self.density[i] * np.exp(-self.zt1[i]/self.diskheight[i])
            ##print(txt.format(i,rstar[i],zt1[i],zt1[i]/diskheight[i],opticaldepth[tracko],opticaldepth[tracko+1],tempt1[i],denst1[i],nit))
            f = open(filename.format(self.sbh, np.log10(self.mbh), self.mdot, self.alpha),"w")
            for i in range(self.rstar.size):
                f.write(txt.format(i,self.rstar[i],self.zt1[i],self.zt1[i]/self.diskheight[i],opticaldepth[tracko],opticaldepth[tracko+1],self.tempt1[i],self.denst1[i],nit)+"\n")
        f.close()
        zt1cs = interpolate.splrep(self.rstar,self.zt1,s=len(self.rstar))
        self.dzdr  = interpolate.splev(self.rstar,zt1cs,der=1)

    def fnudiskpath(self,frequency,theta):
        Rz = self.zt1[self.iref] - self.zobs # rg
        Rr = self.robs * np.sqrt(1.0 + np.power(self.rstar[self.iref]/self.robs,2) - 2.0 * (self.rstar[self.iref]/self.robs) * np.cos(theta-self.thetaobs)) # rg
        costhetap  = self.rstar[self.iref] * np.cos(theta - self.thetaobs) - self.robs # rg
        sinthetap  = self.rstar[self.iref] * np.sin(theta - self.thetaobs) #rg
        thetap     = np.arctan2(sinthetap, costhetap)
        Rtheta     = np.mod(self.thetaobs + thetap, 2.0*np.pi)
        Rmag       = np.sqrt(Rr*Rr + Rz*Rz)

        gradzz     = 1.0
        gradzr     = -2.0 * self.dzdr[self.iref] / np.sin(2.0 * theta)
        gradztheta = np.arctan2(-np.cos(theta),-np.sin(theta))
        gradzmag   = np.sqrt(gradzr*gradzr + gradzz*gradzz)
        cosbeta    = (gradzr * Rr * np.cos(gradztheta - Rtheta) + gradzz * Rz)/(Rmag * gradzmag)
        if cosbeta > 0.0:
            # Blackbody eitted from the tau=1 surface
            bb     = BlackBody(temperature=self.tempt1[self.iref])
            # Doppler shift:
            # Relativistic beaming:
            beta     = np.sqrt(1.0/self.rstar[self.iref])
            gamma    = 1.0/np.sqrt(1-beta*beta)
            costheta = (beta * self.robs * np.cos(theta+np.pi/2.0-thetap))/(beta * np.sqrt(self.robs * self.robs + self.zobs * self.zobs))
            D        = 1./(gamma * (1. - beta * costheta))
            #Absorption * exp(-optdepth)
            return D * D * D * bb(frequency) * cosbeta * self.rstar[self.iref] * self.deltar[self.iref] * dtheta/(Rmag*Rmag)
        else:
            return np.zeros(frequency.size)



######################################################
sbh         = 0.99
mbh         = 1.0e+8 # Solar masses
mdot        = 1.0 # solar mass per year
alpha       = 0.1
inclination = 60.0 * (np.pi / 180.0)
robs        = 1.0e+20 # rg
thetaobs    = 0.0
nr          = 3000
rlo         = 1.01 # rms
rhi         = 1.0e+6 # rms

mydisk = ntdisk(sbh, mbh, mdot, alpha, inclination, robs, nr, rlo, rhi)
mydisk.makedisk()
mydisk.photosphere()

#diskheightall  = np.squeeze([diskheight,diskheightinner,diskheightmiddle,diskheightouter])
#temperatureall = np.squeeze([temperature,temperatureinner,temperaturemiddle,temperatureouter])
#densityall     = np.squeeze([density,densityinner,densitymiddle,densityouter])


######################################################

txt = "{} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6e} {:10.6e} {}"

plt.ion()

fu = u.erg / (u.Hz * u.s * u.sr * u.cm * u.cm)

mydisk.robs = mydisk.rstar[5]
mydisk.zobs = 5.0*mydisk.zt1[5]

# Determine flux-weighted average radius as a function of wavelength
#Also want the 90% range...
frequency = np.logspace(13,17,num=3000) * u.Hz
fluxr     = np.zeros(frequency.size)
flux      = np.zeros(frequency.size) * fu
rave      = np.zeros(frequency.size) * fu
for i in range(mydisk.rstar.size):
    dtheta = 2.0 * np.pi / mydisk.ntheta[i]
    nti = int(mydisk.ntheta[i])
    theta = np.linspace(0,2.0*np.pi,num=nti)
    mydisk.iref = i
    for t in range(nti):
        fluxr = mydisk.fnudiskpath(frequency,theta[t])
        flux += np.copy(fluxr)
        rave += np.copy(fluxr) * mydisk.rstar[i]
    if np.max(flux) > 0.0:
        plt.clf()
        plt.loglog(frequency,np.multiply(flux,frequency))
        plt.title("r/rg = "+str(round(mydisk.rstar[i],2))+" ntheta = "+str(nti))
        plt.ylabel('nuFnu (erg/s/cm2)')
        plt.xlabel('Frequency (Hz)')
        plt.pause(0.001)
rave = rave/flux

########################################################
##plt.subplot(111,frameon=False)
##plt.axis('off')
##plt.title(r'$S_{bh}$ = '+str(sbh)+ r' ---  log $M_{bh}/M_\odot$ = '+str(np.log10(mbh))+r' ---  $\dot{M}$ = '+str(mdot)+r' $M_\odot$/yr')
##
########################################################
##plt.subplot(222)
##plt.axis('on')
##plt.loglog(frequency,np.multiply(flux,frequency))
##txt2 = "Flux (erg/s/cm2/sr) at r/rg= {:10.2e} z/rg= {:10.2e}"
##plt.ylabel(txt2.format(robs,zobs))
##plt.subplot(224)
##plt.loglog(frequency,rave)
##plt.ylabel(r'<r>/$r_g$')
##plt.xlabel("Frequency (Hz)")
##plt.grid()
##
##
########################################################
##
##plt.subplot(321)
####plt.plot(rstar,diskheightinner, 'r')
####plt.plot(rstar,diskheightmiddle * diskheight[pdx[0]]/diskheightmiddle[pdx[0]], 'g')
####plt.plot(rstar,diskheightouter  * diskheight[tdx[0]]/diskheightouter[tdx[0]], 'b')
##for i in range(1):
##    plt.plot(rstar,diskheight*(i+1), 'k--')
##plt.plot(rstar[:zt1.size],zt1,'b--')
##plt.plot(rstar[:zt1.size],dzdr,'r--')
####plt.fill_between(np.linspace(0.1,2,num=100),np.sqrt(4-np.power(np.linspace(0.1,2,num=100),2)),color='k')
####plt.plot([x1,x1],[np.min(diskheightall),np.max(diskheightall)], 'm--')
####plt.plot([x2,x2],[np.min(diskheightall),np.max(diskheightall)], 'm--')
####plt.plot([np.min(rstar),np.max(diskheightall)],[np.min(rstar),np.max(diskheightall)], 'k')
####for i in range(4):
####    plt.plot(rstar,tempcontour(np.power(10,3+0.5*i)*u.Kelvin))
##plt.ylabel(r'Disk height/$r_g$')
##plt.yscale("log")
##plt.xscale("log")
####plt.title(r'$S_{bh}$ = '+str(sbh)+ r' ---  log $M_{bh}/M_\odot$ = '+str(np.log10(mbh))+r' ---  $\dot{M}$ = '+str(mdot)+r' $M_\odot$/yr')
##
########################################################
##
##plt.subplot(323)
####plt.plot(rstar,temperatureinner, 'r')
####plt.plot(rstar,temperaturemiddle * temperature[pdx[0]]/temperaturemiddle[pdx[0]], 'g')
####plt.plot(rstar,temperatureouter  * temperature[tdx[0]]/temperatureouter[tdx[0]], 'b')
##plt.plot(rstar,temperature, 'k--')
##plt.plot(rstar[:zt1.size],tempt1,'b--')
##plt.plot([x1,x1],[np.min(temperatureall),np.max(temperatureall)], 'm--')
##plt.plot([x2,x2],[np.min(temperatureall),np.max(temperatureall)], 'm--')
##plt.ylabel("Disk temperature (K)")
##plt.yscale("log")
##plt.xscale("log")
##
########################################################
##
##plt.subplot(325)
###plt.plot(rstar,densityinner, 'r')
###plt.plot(rstar,densitymiddle * density[pdx[0]]/densitymiddle[pdx[0]], 'g')
###plt.plot(rstar,densityouter  * density[tdx[0]]/densityouter[tdx[0]], 'b')
##plt.plot(rstar,density, 'k--')
##plt.plot(rstar[:zt1.size],denst1,'b--')
##plt.plot([x1,x1],[np.min(densityall),np.max(densityall)], 'm--')
##plt.plot([x2,x2],[np.min(densityall),np.max(densityall)], 'm--')
##plt.ylabel(r'Disk density (atoms/cm$^3$)')
##plt.xlabel(r'Radial distance $r/r_g$  ($r_g = GM/c^2 = $'+f'{rg:.2e}'+')')
##plt.yscale("log")
##plt.xscale("log")
##plt.show()


##rdx = np.extract(rhoz <= 1.0/np.power(3.0e+18,3), np.arange(0,rhoz.size,1))
##sdx = np.extract(rhoz >  1.0/np.power(3.0e+18,3), np.arange(0,rhoz.size,1))
##mfp = np.empty(rhoz.size)
##mfp[sdx] = np.power(rhoz[sdx],-1/3)
##mfp[rdx] = 3.0e+18

##plt.subplot(311)
##plt.plot(z,rhoz)
##plt.plot([diskheight[4],diskheight[4]],[np.min(rhoz * u.cm**3 / u.g),np.max(rhoz * u.cm**3 / u.g)], 'm--')
##plt.ylabel(r'Density (g cm$^{-3}$)')
##plt.yscale("log")
##plt.xscale("log")
##plt.subplot(312)
##plt.plot(z,tempz)
##plt.plot([diskheight[4],diskheight[4]],[np.min(tempz/u.Kelvin),np.max(tempz/u.Kelvin)], 'm--')
##plt.ylabel(r'Temperature (K)')
##plt.yscale("log")
##plt.xscale("log")
##plt.subplot(313)
##plt.plot(z,opticaldepth)
##plt.plot([diskheight[4],diskheight[4]],[np.min(np.multiply(rhoz,tempz) * u.cm**3 / u.Kelvin),np.max(np.multiply(rhoz,tempz) * u.cm**3 / u.Kelvin)], 'm--')
##plt.ylabel(r'Optical Depth')
##plt.xlabel(r'Vertical distance $z/r_g$  ($r_g = GM/c^2$)')
##plt.yscale("log")
##plt.xscale("log")




########################################################
##
##def V(y,sbh,y0):
##    yFoverG = y0*F(y0,sbh)/G(y0,sbh)
##    #print('Inside V: y0 F0 / G0 = ',yFoverG)
##    v1 = np.power(y, -4) * (sbh * sbh - yFoverG * yFoverG)
##    v2 = np.power(y, -6) * (sbh - yFoverG)
##    return (1.0 + v1 + 2.0*v2)/D(y,sbh)
##
########################################################
##
##def Phi(y,sbh):
##    n1 = 0.02 * np.power(alpha * y0,9/8) * np.power(mstar,-3/8)
##    n2 = n1 * np.power(mdotstar,1/4) * B(y,sbh) * np.power(C(y0,sbh),-5/8)
##    print("Inside Phi:")
##    print(y0,sbh,V(y0,sbh,y0),D(y0,sbh),sbh,y0,F(y0,sbh),G(y0,sbh))
##    numerator = n2 * np.sqrt(V(y0,sbh,y0))
##    return Q(y,sbh) + numerator / (y * np.sqrt(C(y,sbh)))
##
########################################################
##
##def getVzero(y0,sbh):
##    x = rms
##    scale = 0.1
##    direc = 1
##    while np.fabs(V(np.sqrt(x),sbh,y0)) > 1.0e-14 or V(np.sqrt(x),sbh,y0) < 0.0:
##        olddirec = direc
##        if V(np.sqrt(x),sbh,y0) < 0:
##            direc = 1
##            if direc == olddirec:
##                scale = 1.1*scale
##            else:
##                scale = 0.9*scale
##            x = x*(1.0+scale)
##        else:
##            direc = -1
##            if direc == olddirec:
##                scale = 1.1*scale
##            else:
##                scale = 0.9*scale
##            x = x*(1.0-scale)
##    return x
##
