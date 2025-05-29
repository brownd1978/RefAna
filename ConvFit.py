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

class ConvFit(object):
    def __init__(self,momrange,pdg,sid):
        # PDG cods of signal and background particles
        self.PDG = pdg
        PDGNames = {-13:"$\\mu^+$",-11:"$e^+$",11:"$e^-$",13:"$\\mu^-$"}
        self.PDGName = PDGNames[self.PDG]
        # setup cuts; these should be overrideable FIXME
        self.MinNHits = 20
        self.MinFitCon = 1.0e-5
        self.MaxDeltaT = 5.0 # nsec
        self.MomRange = momrange
        # Surface Ids
        self.SID = sid
        self.CompName = SID.SurfaceName(sid)
        # intersection histograms
        nNMatBins = 20
        NMatRange = [-0.5,19.5]
        self.HNST = MyHist.MyHist(bins=nNMatBins,range=NMatRange,label="All ST",xlabel="N Intersections",title="Material Intersections")
        self.HNIPA = MyHist.MyHist(bins=nNMatBins,range=NMatRange,label="All IPA",xlabel="N Intersections")
        self.HNSTSel = MyHist.MyHist(bins=nNMatBins,range=NMatRange,label="Selected ST",xlabel="N Intersections",title="Selected Material Intersections")
        self.HNIPASel = MyHist.MyHist(bins=nNMatBins,range=NMatRange,label="Selected IPA",xlabel="N Intersections")
        # Momentum histograms
        nMomBins = 100
        momrange=(40.0,200.0)
        nDeltaMomBins = 200
        deltamomrange=(-10,5)
        self.HDnMom = MyHist.MyHist(label="All "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title="Downstream Momentum at "+self.CompName)
        self.HDnSelMom = MyHist.MyHist(label="Selected "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)")
        self.HUpMom = MyHist.MyHist(label="All "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title="Upstream Momentum at "+self.CompName)
        self.HUpSelMom = MyHist.MyHist(label="Selected "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)")
        # Momentum comparison histograms
        self.HDeltaMom = MyHist.MyHist(label="All", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title ="$\\Delta$ Momentum at "+self.CompName)

        self.HDeltaSelMom = MyHist.MyHist(label="Selected", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)")

    def Print(self):
        print("Convolution Fit, nhits =",self.MinNHits,"Mom Range",self.MomRange,"Comparison at",self.CompName,"PDG",self.PDGName)

    def Loop(self,files,treename):
        # global counts
        NReflect = 0
        NRecoReflect = 0
        NsigPartReflect = 0
        # append tree to files for uproot
        Files = [None]*len(files)
        for i in range(0,len(files)):
            Files[i] = files[i]+":"+treename
        ibatch = 0
        print("Processing batch ",end=' ')
        for batch,rep in uproot.iterate(Files,filter_name="/trk|trksegs|trkmcsim|gtrksegsmc/i",report=True):
            print(ibatch,end=' ')
            ibatch = ibatch+1
            segs = batch['trksegs'] # track fit samples
            nhits = batch['trk.nactive']  # track N hits
            fitcon = batch['trk.fitcon']  # track fit consistency
            fitpdg = batch['trk.pdg']  # track fit consistency
            # compress out unneeded dimensions
            upSegs = segs[:,0] # upstream track fits
            dnSegs = segs[:,1] # downstream track fits
            # basic consistency test
            assert((len(upSegs) == len(dnSegs)) )
            upFitPDG = fitpdg[:,0]
            dnFitPDG = fitpdg[:,1]
            upFitCon = fitcon[:,0]
            dnFitCon = fitcon[:,1]
            upNhits = nhits[:,0]
            dnNhits = nhits[:,1]
            # select fits that match PDG code
            upSigPart = (upFitPDG == self.PDG)
            dnSigPart = (dnFitPDG == self.PDG)
            sigPartFit = upSigPart & dnSigPart
            # select based on fit quality
            upGoodFit = (upNhits >= self.MinNHits) & (upFitCon > self.MinFitCon)
            dnGoodFit = (dnNhits >= self.MinNHits) & (dnFitCon > self.MinFitCon)
            goodFit = upGoodFit & dnGoodFit
            goodReco = sigPartFit & goodFit
            NReflect = NReflect + len(goodReco)
            NRecoReflect = NRecoReflect + sum(goodReco)
            # select based on time difference at tracker entrance
            upEntTime = upSegs[(upSegs.sid==SID.TT_Front()) & (upSegs.mom.z() > 0.0) ].time
            dnEntTime = dnSegs[(dnSegs.sid==SID.TT_Front()) & (dnSegs.mom.z() > 0.0) ].time
            deltaEntTime = dnEntTime-upEntTime
            goodDeltaT = abs(deltaEntTime) < self.MaxDeltaT
            # good fits
            goodFit = goodReco & goodDeltaT
            goodFit = ak.ravel(goodFit)
            NsigPartReflect = NsigPartReflect + sum(goodFit)
            # total momentum of upstream and downstream fits at the comparison point
            upMom = np.array(ak.flatten(upSegs[(upSegs.sid == self.SID) & (upSegs.mom.Z() < 0.0)].mom.magnitude(),axis=1))
            dnMom = np.array(ak.flatten(dnSegs[(dnSegs.sid == self.SID) & (dnSegs.mom.Z() > 0.0)].mom.magnitude(),axis=1))
            if len(upMom) != len(dnMom):
                print()
                print("Upstream and Downstream fits don't match!",len(upMom),len(dnMom))
                continue
            self.HUpMom.fill(upMom[goodFit])
            self.HDnMom.fill(dnMom[goodFit])
            deltaMom = dnMom - upMom
            self.HDeltaMom.fill(deltaMom[goodFit])
            # count IPA and target intersections
            nfoil = np.array(ak.count_nonzero(upSegs.sid==SID.ST_Foils(),axis=1))
            self.HNST.fill(nfoil)
            nipa = np.array(ak.count_nonzero(upSegs.sid==SID.IPA(),axis=1))
            self.HNIPA.fill(nipa)
            # select fits
            selected = (dnMom > self.MomRange[0]) & (dnMom < self.MomRange[1]) & (upMom > self.MomRange[0]) & (upMom < self.MomRange[1]) & (nfoil>0)
            goodSel = selected & goodFit
            nfoilsel = nfoil[goodSel]
            self.HNSTSel.fill(nfoilsel)
            nipasel = nipa[goodSel]
            self.HNIPASel.fill(nipasel)

            upSelMom = upMom[goodSel]
            dnSelMom = dnMom[goodSel]
            self.HUpSelMom.fill(upSelMom)
            self.HDnSelMom.fill(dnSelMom)
            # reflection momentum difference
            deltaSelMom = dnSelMom - upSelMom
            self.HDeltaSelMom.fill(deltaSelMom)
        print()
        print("From", NReflect,"total reflections found", NRecoReflect,"with good quality reco,", NsigPartReflect, "confirmed",self.PDGName,"and",self.HUpSelMom.integral(), "Selected",self.MomRange)

    def PlotIntersections(self):
        fig, (cmat,cselmat) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
        nipa = self.HNIPA.plot(cmat)
        nst = self.HNST.plot(cmat)
        cmat.legend(loc="upper right")
        nipasel = self.HNIPASel.plot(cselmat)
        nstsel = self.HNSTSel.plot(cselmat)
        cselmat.legend(loc="upper right")

    def PlotMomentum(self):
        fig, (upMom, dnMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upmom = self.HUpMom.plot(upMom)
        upselmom = self.HUpSelMom.plot(upMom)
        upMom.legend(loc="upper right")
        dnmom = self.HDnMom.plot(dnMom)
        dnselmom = self.HDnSelMom.plot(dnMom)
        dnMom.legend(loc="upper right")
        delmom = self.HDeltaMom.plot(deltaMom)
        delselmom = self.HDeltaSelMom.plot(deltaMom)
        deltaMom.legend(loc="upper right")

    def Fit(self):
        fig, (delmom,delselmom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))

        dmomerr = self.HDeltaMom.binErrors()
        dmommid = self.HDeltaMom.binCenters()
        dmomsum = self.HDeltaMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaMom.edges[1]- self.HDeltaMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_expGauss, dmommid, self.HDeltaMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)

        self.HDeltaMom.plotErrors(delmom)
        delmom.plot(dmommid, fxn_expGauss(dmommid, *popt), 'r-',label="Fit")
        delmom.legend(loc="upper right")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

        dmomerr = self.HDeltaSelMom.binErrors()
        dmommid = self.HDeltaSelMom.binCenters()
        dmomsum = self.HDeltaSelMom.integral()
        # initialize the fit parameters
        mu_0 = np.mean(dmommid*self.HDeltaSelMom.data/dmomsum) # initial mean
        var = np.sum(((dmommid**2)*self.HDeltaSelMom.data)/dmomsum) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = self.HDeltaSelMom.edges[1]- self.HDeltaSelMom.edges[0]
        amp_0 = dmomsum*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_expGauss, dmommid, self.HDeltaSelMom.data, p0, sigma=dmomerr)
        print("All fit parameters",popt)
        print("All fit covariance",pcov)
        self.HDeltaSelMom.plotErrors(delselmom)
        delselmom.plot(dmommid, fxn_expGauss(dmommid, *popt), 'r-',label="Fit")
        delselmom.legend(loc="upper right")
        fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

    def Write(self,savefile):
        with h5py.File(savefile, 'w') as hdf5file:
            self.HDeltaMom.save(hdf5file)
            self.HDeltaSelMom.save(hdf5file)

    def Read(self,savefile):
        self.HDeltaMom = MyHist.MyHist(label="All",file=savefile)
        self.HDeltaSelMom = MyHist.MyHist(label="Selected",file=savefile)

