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


def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

class MyHist(object):
    def __init__(self,bins,range,label,title="",xlabel=""):
        data = []
        self.data, self.edges = np.histogram(data, bins=bins, range=range)
        self.label = label
        self.title = title
        self.xlabel = xlabel
    def fill(self,data):
        newdata , newedges = np.histogram(data, bins=self.edges)
        self.data += newdata
    def plot(self,axis):
        plt = axis.stairs(self.data,self.edges,label=self.label)
        if(self.title != ""):
            axis.set_title(self.title)
        if(self.xlabel != ""):
            axis.set_xlabel(self.xlabel)
        return plt
    def integral(self):
        return np.sum(self.data)

class ConvFit(object):
    def __init__(self,momrange,sigpdg,trksid):
        # PDG cods of signal and background particles
        self.PDG = sigpdg
        PDGNames = {-13:"$\\mu^+$",-11:"$e^+$",11:"$e^-$",13:"$\\mu^-$"}
        self.PDGName = PDGNames[self.PDG]
        # setup cuts; these should be overrideable FIXME
        self.MinNHits = 20
        self.MinFitCon = 1.0e-5
        self.MaxDeltaT = 5.0 # nsec
        self.MomRange = momrange
        # Surface Ids
        self.TrkEntSID = SID.TT_Front()
        self.TrkCompSID = trksid
        nNMatBins = 20
        NMatRange = [-0.5,19.5]
        self.HNST = MyHist(bins=nNMatBins,range=NMatRange,label="All ST",xlabel="N Intersections",title="Material Intersections")
        self.HNIPA = MyHist(bins=nNMatBins,range=NMatRange,label="All IPA",xlabel="N Intersections")
        self.HNSTSel = MyHist(bins=nNMatBins,range=NMatRange,label="Selected ST",xlabel="N Intersections",title="Selected Material Intersections")
        self.HNIPASel = MyHist(bins=nNMatBins,range=NMatRange,label="Selected IPA",xlabel="N Intersections")

        nMomBins = 100
        momrange=(40.0,200.0)
        nDeltaMomBins = 200
        deltamomrange=(-10,5)

        self.HDnCompMom = MyHist(label="All "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)", title="Downstream Momentum at Tracker Mid")
        self.HDnSignalCompMom = MyHist(label="Selected "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)")
        self.HUpCompMom = MyHist(label="All "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)",title="Upstream Momentum at Tracker Mid")
        self.HUpSignalCompMom = MyHist(label="Selected "+self.PDGName, bins=nMomBins, range=momrange, xlabel="Fit Momentum (MeV)")
        self.HDeltaCompMom = MyHist(label="Fit $\\Delta$ P", bins=nDeltaMomBins, range=deltamomrange, xlabel="Downstream - Upstream Momentum (MeV)",title ="$\\Delta$ Momentum")


    def Print(self):
        print("Convolution Fit, nhits =",self.MinNHits,"Mom Range",self.MomRange,"Comparison SID=",self.TrkCompSID,"PDG",self.PDGName)

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
            # select fits that match 'signal' PDG
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
            upEntTime = upSegs[(upSegs.sid==self.TrkEntSID) & (upSegs.mom.z() > 0.0) ].time
            dnEntTime = dnSegs[(dnSegs.sid==self.TrkEntSID) & (dnSegs.mom.z() > 0.0) ].time
            deltaEntTime = dnEntTime-upEntTime
            goodDeltaT = abs(deltaEntTime) < self.MaxDeltaT
            # good PID
            goodsigPart = goodReco & goodDeltaT
            goodsigPart = ak.ravel(goodsigPart)
            NsigPartReflect = NsigPartReflect + sum(goodsigPart)
            # total momentum of upstream and downstream fits at the comparison point
            upCompMom = np.array(ak.flatten(upSegs[(upSegs.sid == self.TrkCompSID) & (upSegs.mom.Z() < 0.0)].mom.magnitude(),axis=1))
            dnCompMom = np.array(ak.flatten(dnSegs[(dnSegs.sid == self.TrkCompSID) & (dnSegs.mom.Z() > 0.0)].mom.magnitude(),axis=1))
            self.HUpCompMom.fill(upCompMom)
            self.HDnCompMom.fill(dnCompMom)
            # count IPA and target intersections
            nfoil = np.array(ak.count_nonzero(upSegs.sid==SID.ST_Foils(),axis=1))
            self.HNST.fill(nfoil)
            nipa = np.array(ak.count_nonzero(upSegs.sid==SID.IPA(),axis=1))
            self.HNIPA.fill(nipa)
            # select fits that look like signal electrons, including target foil intersection
            signal =  (dnCompMom > self.MomRange[0]) & (dnCompMom < self.MomRange[1]) & (upCompMom > self.MomRange[0]) & (upCompMom < self.MomRange[1]) & (nfoil>0)
            goodSignalsigPart = signal & goodsigPart

            nfoilsel = nfoil[goodSignalsigPart]
            self.HNSTSel.fill(nfoilsel)
            nipasel = nipa[goodSignalsigPart]
            self.HNIPASel.fill(nipasel)

            upSignalCompMom = upCompMom[goodSignalsigPart]
            dnSignalCompMom = dnCompMom[goodSignalsigPart]
            self.HUpSignalCompMom.fill(upSignalCompMom)
            self.HDnSignalCompMom.fill(dnSignalCompMom)
            # reflection momentum difference of signal-like particles
            deltaCompMom = dnSignalCompMom - upSignalCompMom
            self.HDeltaCompMom.fill(deltaCompMom)
        print()
        print("From", NReflect,"total reflections found", NRecoReflect,"with good quality reco,", NsigPartReflect, "confirmed",self.PDGName,"and",self.HUpSignalCompMom.integral(), "Selected",self.MomRange)
        fig, (cmat,cselmat) = plt.subplots(1,2,layout='constrained', figsize=(15,5))
        nipa = self.HNIPA.plot(cmat)
        nst = self.HNST.plot(cmat)
        cmat.legend(loc="upper right")
        nipasel = self.HNIPASel.plot(cselmat)
        nstsel = self.HNSTSel.plot(cselmat)
        cselmat.legend(loc="upper right")
        #
        fig, (upMom, dnMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(15,5))
        upmom = self.HUpCompMom.plot(upMom)
        upmomsig = self.HUpSignalCompMom.plot(upMom)
        upMom.legend(loc="upper right")
        dnmom = self.HDnCompMom.plot(dnMom)
        dnmomsig = self.HDnSignalCompMom.plot(dnMom)
        dnMom.legend(loc="upper right")

        delmomhist = self.HDeltaCompMom.plot(deltaMom)
        deltaMom.legend(loc="upper right")
        # fit
        DeltaMomHistErrors = np.zeros(len(delmomhist[1])-1)
        delmomhistBinMid =np.zeros(len(delmomhist[1])-1)
        for ibin in range(len(delmomhistErrors)):
            delmomhistBinMid[ibin] = 0.5*(delmomhist[1][ibin] + delmomhist[1][ibin+1])
            delmomhistErrors[ibin] = max(1.0,math.sqrt(delmomhist[0][ibin]))
        #print(delmomhistBinErrors)
        delmomhistIntegral = np.sum(delmomhist[0])
        # initialize the fit parameters
        mu_0 = np.mean(delmomhistBinMid*delmomhist[0]/delmomhistIntegral) # initial mean
        var = np.sum(((delmomhistBinMid**2)*delmomhist[0])/delmomhistIntegral) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = delmomhist[1][1]-delmomhist[1][0]
        amp_0 = delmomhistIntegral*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_expGauss, delmomhistBinMid, delmomhist[0], p0, sigma=delmomhistErrors)
        print("Trk fit parameters",popt)
        print("Trk fit covariance",pcov)
        fig, (Trk) = plt.subplots(1,1,layout='constrained', figsize=(10,5))
        Trk.stairs(edges=delmomhist[1],values=delmomhist[0],label="Track $\\Delta$ P")
        Trk.plot(delmomhistBinMid, fxn_expGauss(delmomhistBinMid, *popt), 'r-',label="EMG Fit")
        Trk.legend(loc="upper right")
        Trk.set_title('EMG Fit to Track '+self.PDGName+' $\\Delta$ P')
        Trk.set_xlabel("Downstream - Upstream Momentum (MeV)")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")
