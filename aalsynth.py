import numpy             as np
import matplotlib.pyplot as plt

from astropy                 import constants as const
from astropy                 import units as u
from astropy.visualization   import astropy_mpl_style, quantity_support
from ntdisk                  import ntdisk

######################################################
plt.style.use(astropy_mpl_style)
plt.ion()
quantity_support()
plt.clf()
plt.pause(3)
######################################################
#sbh         = 0.01
#mbh         = 1.0e+8 # Solar masses
#mdot        = 1.8 # solar mass per year
sbh         = 0.99
mbh         = 1.0e+8 # Solar masses
mdot        = 1.0 # solar mass per year
alpha       = 0.1
inclination = 30.0 * (np.pi / 180.0)
robs        = 1.0e+1 # rg
thetaobs    = 0.0
nr          = 300
#nr          = 3000
rlo         = 1.01 # rms
rhi         = 1.0e+5 # rms

mydisk = ntdisk(sbh, mbh, mdot, alpha, inclination, robs, nr, rlo, rhi)
mydisk.makedisk()
mydisk.photosphere()
mydisk.pltdisk()
mydisk.genforcemultgrid(20,clobber=False)



######################################################
if False:
    mydisk.pltdisk()
    plt.subplot(224)
    plt.plot(mydisk.rstar,mydisk.diskheight,'k')
    plt.plot(mydisk.rstar[:mydisk.zt1.size],mydisk.zt1,'b--')
    plt.ylabel(r'z/$r_g$')
    plt.xlabel(r'Radial distance $r/r_g$') #  ($r_g = GM/c^2 = $'+f'{mydisk.rg:.2e}'+')')
    plt.yscale("log")
    plt.xscale("log")
    plt.subplot(222)
    plt.xlabel(r'$x/r_g$') #  ($r_g = GM/c^2 = $'+f'{mydisk.rg:.2e}'+')')
    plt.ylabel(r'$y/r_g$')
    plt.yscale("log")
    plt.xscale("log")
    plt.pause(0.001)



######################################################
if False:
    for r in np.random.randint(0,high=nr-1,size=nr):
        print(f"Computing test particle trajectory lanched from ({mydisk.rstar[r]}, {mydisk.zt1[r]})")
        solution = mydisk.traceparticle(r,0,np.linspace(0,3.1e+7,365*24*60))
#        x_of_t  = solution.y[:,0] * u.cm / mydisk.rg
#        vx_of_t = solution.y[:,1] * (u.cm).to(u.kilometer) / u.s
#        y_of_t  = solution.y[:,2] * u.cm / mydisk.rg
#        vy_of_t = solution.y[:,3] * (u.cm).to(u.kilometer) / u.s
#        z_of_t  = solution.y[:,4] * u.cm / mydisk.rg
#        vz_of_t = solution.y[:,5] * (u.cm).to(u.kilometer) / u.s
        x_of_t  = solution[:,0] * u.cm / mydisk.rg
        vx_of_t = solution[:,1] * (u.cm).to(u.kilometer) / u.s
        y_of_t  = solution[:,2] * u.cm / mydisk.rg
        vy_of_t = solution[:,3] * (u.cm).to(u.kilometer) / u.s
        z_of_t  = solution[:,4] * u.cm / mydisk.rg
        vz_of_t = solution[:,5] * (u.cm).to(u.kilometer) / u.s

        r_of_t = np.sqrt(x_of_t * x_of_t + y_of_t * y_of_t)
        theta_of_t = np.rad2deg(np.arctan2(y_of_t,x_of_t))

        plt.subplot(224)
        plt.scatter(r_of_t, z_of_t, c=theta_of_t.value, lw=0, s=8, cmap="viridis")
        plt.subplot(222)
        plt.scatter(x_of_t, y_of_t, c=theta_of_t.value, lw=0, s=8, cmap="viridis")
        m = np.max([x_of_t,y_of_t])
        plt.xlim([1,m])
        plt.ylim([1,m])
        plt.pause(0.001)




######################################################
if False:
    txt = "{} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6e} {:10.6e} {}"

    plt.ion()

    fu = u.erg / (u.s * u.cm * u.cm * u.sr * u.Hz)

    ngrid     = 2
    lgrend    = np.log10(np.max(0.9*mydisk.rstar))
    lgzend    = 3 
    tol       = 1.0e-7
    frequency = np.logspace(13,19,num=3000) * u.Hz
    fx        = np.zeros((ngrid, ngrid)) * u.cm / u.s**2
    fy        = np.zeros((ngrid, ngrid)) * u.cm / u.s**2
    fz        = np.zeros((ngrid, ngrid)) * u.cm / u.s**2
    rgrid     = mydisk.rms * np.power(10.0, lgrend * (np.arange(ngrid)/ngrid)) / np.sqrt(2)
    zgrid     = mydisk.rms * np.power(10.0, lgzend * (np.arange(ngrid)/ngrid)) / np.sqrt(2)

    # Observer is in the x-z plane -> yobs=0, xobs = robs
    for j in range(ngrid):
        mydisk.robs = rgrid[j]
        for k in range(ngrid):
            mydisk.zobs  = zgrid[k]
            if np.max(mydisk.rstar) > mydisk.robs:
                if (mydisk.zobs > (np.extract(mydisk.rstar > mydisk.robs, mydisk.zt1))[0]):
                    fluxrt    = np.zeros(frequency.size) * fu
                    flux      = np.zeros(frequency.size) * fu
                    rave      = np.zeros(frequency.size) * fu

                    # Gravity from the black hole
                    # dynes per gram (cm/s^2)
                    fx[j,k]   = -((const.G.cgs * mydisk.mbh / (mydisk.rg * mydisk.rg) ) * (mydisk.robs / np.power(mydisk.robs * mydisk.robs + mydisk.zobs * mydisk.zobs, 3/2)))
                    fz[j,k]   = -((const.G.cgs * mydisk.mbh / (mydisk.rg * mydisk.rg) ) * (mydisk.zobs / np.power(mydisk.robs * mydisk.robs + mydisk.zobs * mydisk.zobs, 3/2)))

                    # Gravity from the disk
                    # dynes per gram (cm/s^2)
                    [fdx,fdz] = mydisk.diskgravity(rgrid[j]*mydisk.rg,zgrid[k]*mydisk.rg)
                    fx[j,k] += fdx
                    fz[j,k] += fdz

                    for r in range(mydisk.rstar.size):
                        dtheta      = 2.0 * np.pi / mydisk.ntheta[r]
                        ntr         = int(mydisk.ntheta[r])
                        theta       = np.linspace(0,2.0*np.pi,num=ntr)
                        mydisk.rref = r
                        fluxr       = np.zeros(frequency.size) * fu
                        oldfx       = fx[j,k] # Keep a copy of the force/mass due to just BH + disk gravity
                        oldfy       = fy[j,k]
                        oldfz       = fz[j,k]


                        (Rx, Ry, Rz) = (mydisk.robs * np.cos(mydisk.thetaobs) - mydisk.rstar[mydisk.rref] * np.cos(theta),
                                        mydisk.robs * np.sin(mydisk.thetaobs) - mydisk.rstar[mydisk.rref] * np.sin(theta),
                                        np.broadcast_to(mydisk.zobs - mydisk.zt1[mydisk.rref], theta.shape)) # rg
                        Rr = np.sqrt(Rx*Rx + Ry*Ry)
                        Rmag       = np.sqrt(Rr*Rr + Rz*Rz)
                        Rhat = np.array([Rx,Ry,Rz])/Rmag

                        (gradzx,gradzy,gradzz) = (-mydisk.dzdr[mydisk.rref] * np.cos(theta),
                                                  -mydisk.dzdr[mydisk.rref] * np.sin(theta),
                                                  np.ones(theta.size))
                        gradzr     = np.sqrt(gradzx*gradzx + gradzy*gradzy)
                        gradzmag   = np.sqrt(gradzr*gradzr + gradzz*gradzz)

                        cosbeta    = np.sum(np.array([Rx,Ry,Rz])*np.array([gradzx,gradzy,gradzz]), axis=0) / (Rmag * gradzmag)

                        cbdx = np.extract(cosbeta > 0, np.arange(cosbeta.size))
                        for t in cbdx:
                            fluxrt = mydisk.fnudiskpatch(frequency, theta[t])
                            fluxr += np.copy(fluxrt)
                            flux  += np.copy(fluxrt)
                            rave  += np.copy(fluxrt) * mydisk.rstar[r]

                            # magnitude of radiation force per unit area: fluxrt/const.c.cgs
                            # [fx] = erg / (sr * cm^3) = dyne / (sr cm^2)
                            # direction of radiation force per unit area:

                            dOmega = (mydisk.rstar[r] * mydisk.deltar[r] * dtheta * cosbeta[t] / (Rmag[t] * Rmag[t])) * u.sr

                            # Components of the force per unit mass - This is just electron scattering. Need force multiplier for lines/edges
                            nufnurt  = np.sum(np.multiply(fluxrt,frequency)) * dOmega/(const.c.cgs * const.u.cgs)
                            fx[j,k] += nufnurt * (const.sigma_T.cgs) * Rx[t] / Rmag[t]
                            fy[j,k] += nufnurt * (const.sigma_T.cgs) * Ry[t] / Rmag[t]
                            fz[j,k] += nufnurt * (const.sigma_T.cgs) * Rz[t] / Rmag[t]

                        if np.max(flux.value) > 0.0:
                            csflux = CubicSpline(frequency.value,flux.value)
                            csiflux = csflux.integrate((0.1 * ((u.Ry)/const.h).to(u.Hz)).value, (1000 * ((u.Ry)/const.h).to(u.Hz)).value) * fu * u.Hz
                            lgxi = np.log10((4 * np.pi * csiflux / mydisk.verticaldensity(mydisk.zobs * mydisk.rg, r)).value)
                            lSob = (vth * vth / np.sqrt(fx[j,k]*fx[j,k] + fy[j,k]*fy[j,k] + fz[j,k]*fz[j,k])).decompose()
                            lgt  = np.log10((const.sigma_T * mydisk.verticaldensity(mydisk.zobs * mydisk.rg,r) * lSob).decompose())
                            fm = np.power(10.0, mydisk.fmultfunc((lgt,lgxi)))

                            nit = 5
                            while nit > 0:
                                nit -= 1
                                fx[j,k] = oldfx
                                fy[j,k] = oldfy
                                fz[j,k] = oldfz
                                for t in cbdx:
                                    fluxrt = mydisk.fnudiskpatch(frequency, theta[t])

                                    # magnitude of radiation force per unit area: fluxrt/const.c.cgs
                                    # [fx] = erg / (sr * cm^3) = dyne / (sr cm^2)
                                    # direction of radiation force per unit area:

                                    dOmega = (mydisk.rstar[r] * mydisk.deltar[r] * dtheta * cosbeta[t] / (Rmag[t] * Rmag[t])) * u.sr

                                    # Components of the force per unit mass - This is just electron scattering. Need force multiplier for lines/edges
                                    nufnurt  = np.sum(np.multiply(fluxrt,frequency)) * dOmega/(const.c.cgs * const.u.cgs)
                                    fx[j,k] += fm * nufnurt * (const.sigma_T.cgs) * Rx[t] / Rmag[t]
                                    fy[j,k] += fm * nufnurt * (const.sigma_T.cgs) * Ry[t] / Rmag[t]
                                    fz[j,k] += fm * nufnurt * (const.sigma_T.cgs) * Rz[t] / Rmag[t]

                                lSob = (vth * vth / np.sqrt(fx[j,k]*fx[j,k] + fy[j,k]*fy[j,k] + fz[j,k]*fz[j,k])).decompose()
                                lgt  = np.log10((const.sigma_T * mydisk.verticaldensity(mydisk.zobs * mydisk.rg,r) * lSob).decompose())
                                fm = np.power(10.0, fmultfunc((lgt,lgxi)))

                            dr = np.multiply(fx.value, 1.0/np.sqrt(np.square(fx.value) + np.square(fy.value) + np.square(fz.value)+tol))
                            dz = np.multiply(fz.value, 1.0/np.sqrt(np.square(fx.value) + np.square(fy.value) + np.square(fz.value)+tol))

                            plt.clf()
                            mydisk.pltdisk(r)

                            plt.subplot(322)
                            plt.axis('on')
                            nufnu = np.multiply(flux,frequency)
                            plt.loglog(frequency,nufnu)
                            plt.loglog(frequency,np.multiply(fluxr,frequency))
                            ax = plt.gca()
                            ax.set_ylim([np.max([0.9*np.min(nufnu/(fu * u.Hz)),
                                                 1.0e-10 * np.max(np.multiply(fluxr,frequency)/(fu * u.Hz))]),
                                         1.5*np.max(nufnu/(fu * u.Hz))])
                            plt.ylabel("nuFnu (erg/s/cm2)")
                            plt.xlabel('Frequency (Hz)')
                            plt.text(1.1*np.min(frequency / u.Hz),30.0*np.min(nufnu/(fu * u.Hz)), r'$dF_x$ = '+'{:0.5g}'.format(np.fabs((fx[j,k].value-oldfx.value)/(oldfx.value+tol)))+r'  $dF_y$ = '+'{:0.5g}'.format(np.fabs((fy[j,k].value-oldfy.value)/(oldfy.value+tol)))+r'  $dF_z$ = '+'{:0.5g}'.format(np.fabs((fz[j,k].value-oldfz.value)/(oldfz.value+tol))))
                            plt.text(1.1*np.min(frequency / u.Hz),1.5*np.min(nufnu/(fu * u.Hz)), r'$F_r$ = $F_x$ = '+'{:0.5g}'.format(fx[j,k])+r'  $F_y$ = '+'{:0.5g}'.format(fy[j,k])+r'  $F_z$ = '+'{:0.5g}'.format(fz[j,k]))

                            plt.subplot(324)
                            plt.loglog(frequency,rave/flux)
                            plt.ylabel(r'<r>/$r_g$')
                            plt.xlabel("Frequency (Hz)")
                            plt.grid()

                            plt.subplot(326)
                            plt.plot(mydisk.rstar,mydisk.diskheight,'k')
                            plt.plot(mydisk.rstar[:mydisk.zt1.size],mydisk.zt1,'b--')
                            for l in range(ngrid):
                                rarr = rgrid[l]
                                for m in range(ngrid):
                                    zarr = zgrid[m]
                                    if np.sqrt(dr[l,m] * dr[l,m] + dz[l,m] * dz[l,m]) > 0:
                                        plt.arrow(rarr, zarr, dr[l,m], dz[l,m])
                            plt.yscale("log")
                            plt.xscale("log")

                            plt.pause(0.001)

                        if (np.fabs((fx[j,k].value-oldfx.value)/(oldfx.value+tol)) < tol) and (np.fabs((fy[j,k].value-oldfy.value)/(oldfy.value+tol)) < tol) and (np.fabs((fz[j,k].value-oldfz.value)/(oldfz.value+tol)) < tol) and (np.max(flux.value) > 0.0) and (r > int(0.1*mydisk.rstar.size)):
                            break
                    rave = rave/flux
            print(f"{rgrid[j]:.2e} {zgrid[k]:.2e} {fx[j,k]:.2e} {fz[j,k]:.2e} {fdx:.2e} {fdz:.2e} {fx[j,k]:.2e} {fz[j,k]:.2e} {lSob:.2e} {lgxi:.2e} {lgt:.2e}")


    filename = "Sbh{mydisk.sbh}-MBH{np.log10(mydisk.mbh):.2f}-Mdot{mydisk.mdot}-alpha{mydisk.alpha}-rad.fits"
    fxmaster = np.zeros((rgrid.size+1,zgrid.size+1))
    fxmaster[1:,0] = np.copy(rgrid)
    fxmaster[0,1:] = np.copy(zgrid)
    fxmaster[1:,1:] = np.copy(fx)
    fymaster = np.zeros((rgrid.size+1,zgrid.size+1))
    fymaster[1:,0] = np.copy(rgrid)
    fymaster[0,1:] = np.copy(zgrid)
    fymaster[1:,1:] = np.copy(fy)
    fzmaster = np.zeros((rgrid.size+1,zgrid.size+1))
    fzmaster[1:,0] = np.copy(rgrid)
    fzmaster[0,1:] = np.copy(zgrid)
    fzmaster[1:,1:] = np.copy(fz)
    data = Table(data=[fxmaster,fymaster,fzmaster],
                         names=["fx","fy","fz"])
    data.write(filename, format="fits", overwrite=True)
    
