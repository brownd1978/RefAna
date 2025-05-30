#
# fit reflected momentum different response to a convolved function.
#
import uproot
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special
import SurfaceIds as SID
import MyHist
import h5py

def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

def fxn_conv_expGauss(x, amp, mu, sigma, lamb):
    step =0.010 # 10 KeV step
    lowmom = -10.0
    himom = 5.0
    xvals = np.arange(lowmom,himom,step)
    pars = np.array([amp, mu, sigma, lamb]) # initial parameters
    yvals = list(map(lambda x: fxn_expGauss(x,*pars),xvals))
    conv = np.convolve(yvals,yvals,mode="same")
    ibin = np.floor((x-lowmom)/step).astype(np.int64)
    return conv[ibin]

class FitReflections(object):
    def __init__(self,savefile):
        self.HDeltaNoMatMom = MyHist.MyHist(name="DeltaMom",label="No Mat",file=savefile)
        self.HDeltaTgtMom = MyHist.MyHist(name="DeltaMom",label="Target",file=savefile)

    def Test(self):
        step =0.1
        lowmom = -10.0
        himom = 5.0
        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        amp_0 = dmomsum*step # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        xvals = np.arange(lowmom,himom,step)
        edges = np.arange(lowmom-0.4999*step,himom+0.4999*step,step)
        noconvvals = list(map(lambda x: fxn_expGauss(x,*p0),xvals))
        convvals = list(map(lambda x: fxn_conv_expGauss(x,*p0),xvals))
        fig, (noconv,conv) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        noconvplt = noconv.stairs(noconvvals,edges,label="noconv")
        convplt = conv.stairs(convvals,edges,label="convolved")


    def Fit(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = self.HDeltaNoMatMom.binErrors()
        dmommid = self.HDeltaNoMatMom.binCenters()
        dmomsum = self.HDeltaNoMatMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaNoMatMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaNoMatMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaNoMatMom.edges[1]- self.HDeltaNoMatMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_conv_expGauss, dmommid, self.HDeltaNoMatMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        self.HDeltaNoMatMom.plotErrors(delmom)
        delmom.plot(dmommid, fxn_conv_expGauss(dmommid, *popt), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

        dmomerr = self.HDeltaTgtMom.binErrors()
        dmommid = self.HDeltaTgtMom.binCenters()
        dmomsum = self.HDeltaTgtMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaTgtMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaTgtMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaTgtMom.edges[1]- self.HDeltaTgtMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_conv_expGauss, dmommid, self.HDeltaTgtMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        self.HDeltaTgtMom.plotErrors(delselmom)
        delselmom.plot(dmommid, fxn_conv_expGauss(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")


