#
# class to analyze reflection momentum response
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
import SurfaceIDs as SID


def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

class RefResp(object):
    def __init__(self,momrange,sigpdg,bkgpdg):
        # PDG cods of signal and background particles
        self.SigPDG = sigpdg
        self.BkgPDG = bkgpdg
        PDGNames = {-13:"$\\mu^+$",-11:"$e^+$",11:"$e^-$",13:"$\\mu^-$"}
        self.SigPDGName = PDGNames[self.SigPDG]
        self.BkgPDGName = PDGNames[self.BkgPDG]
        # setup cuts; these should be overrideable FIXME
        self.MinNHits = 20
        self.MinFitCon = 1.0e-5
        self.MaxDeltaT = 5.0 # nsec
        self.MomRange = momrange
        # Surface Ids
        self.TrkEntSID = SID.TT_Front()
        self.TrkMidSID = SID.TT_Mid()

    def Print(self):
        print("ReflectionResponse object, nhits =",self.MinNHits)

    def Loop(self,files,treename):
        # arrays for histogramming
        DeltaEntTime = []
        DeltaEntTimeSigMC =[]
        DeltaEntTimeBkgMC =[]
        DeltaEntTimeDkMC =[]
        SelectedDeltaEntTime = []
        SelectedDeltaEntTimeSigMC =[]
        SelectedDeltaEntTimeBkgMC =[]
        SelectedDeltaEntTimeDkMC =[]
        UpMidMom = []
        UpSignalMidMom = []
        DnMidMom = []
        DnSignalMidMom = []
        DeltaMidMom = []
        DeltaMidMomMC = []
        NIPACyl = []
        SelectedNIPACyl = []
        NSTOtCyl = []
        NSTFoil = []
        SelectedNSTFoil = []
       # counts
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
            trkMC = batch['trkmcsim']  # MC genealogy of particles
            segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
            upSegs = segs[:,0] # upstream track fits
            dnSegs = segs[:,1] # downstream track fits
            upSegsMC = segsMC[:,0] # upstream track MC truth
            dnSegsMC = segsMC[:,1] # downstream track MC truth
            upTrkMC = trkMC[:,0] # upstream fit associated true particles
            dnTrkMC = trkMC[:,1] # downstream fit associated true particles
            # basic consistency test
            assert((len(upSegs) == len(dnSegs)) & (len(upSegsMC) == len(dnSegsMC)) & (len(upSegs) == len(upSegsMC))& (len(upTrkMC) == len(dnTrkMC)) & (len(upSegs) == len(upTrkMC)))
            upFitPDG = fitpdg[:,0]
            dnFitPDG = fitpdg[:,1]
            upFitCon = fitcon[:,0]
            dnFitCon = fitcon[:,1]
            upNhits = nhits[:,0]
            dnNhits = nhits[:,1]
            # select fits that match 'signal' PDG
            upSigPart = (upFitPDG == self.SigPDG)
            dnSigPart = (dnFitPDG == self.SigPDG)
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
            DeltaEntTime.extend(ak.flatten(deltaEntTime[goodReco]))
            # good electron
            goodsigPart = goodReco & goodDeltaT
            goodsigPart = ak.ravel(goodsigPart)
            NsigPartReflect = NsigPartReflect + sum(goodsigPart)
            # total momentum at tracker mid, upstream and downstream fits
            upMidMom = upSegs[(upSegs.sid == self.TrkMidSID)].mom.magnitude()
            dnMidMom = dnSegs[(dnSegs.sid == self.TrkMidSID)].mom.magnitude()
            DnMidMom.extend(ak.flatten(dnMidMom,axis=1))
            UpMidMom.extend(ak.flatten(upMidMom,axis=1))
            # count IPA and target intersections
            NIPACyl.extend(ak.count_nonzero(upSegs.sid==SID.IPA(),axis=1))
            NSTOtCyl.extend(ak.count_nonzero(upSegs.sid==SID.ST_Outer(),axis=1))
            NSTFoil.extend(ak.count_nonzero(upSegs.sid==SID.ST_Foils(),axis=1))
            nfoil = ak.count_nonzero(upSegs.sid==SID.ST_Foils(),axis=1)
            nipa = ak.count_nonzero(upSegs.sid==SID.IPA(),axis=1)
            # select fits that look like signal electrons: this needs to include a target constraint TODO
            # protect against missing intersections
            signalMomRange = [False]*len(upMidMom)
            for i in range(0,len(upMidMom)):
                if (len(upMidMom[i]) ==1) & (len(dnMidMom[i]) == 1):
                    signalMomRange[i] = (dnMidMom[i][0] > self.MomRange[0]) & (dnMidMom[i][0] < self.MomRange[1]) & (upMidMom[i][0] > self.MomRange[0]) & (upMidMom[i][0] < self.MomRange[1]) & (nfoil[i]>0)
            goodSignalsigPart = signalMomRange & goodsigPart
            for i in range(0,len(goodSignalsigPart)):
                if(goodSignalsigPart[i]):
                    SelectedNIPACyl.append(nipa[i])
                    SelectedNSTFoil.append(nfoil[i])
            SelectedDeltaEntTime.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange]))
            upSignalMidMom = upMidMom[goodSignalsigPart]
            dnSignalMidMom = dnMidMom[goodSignalsigPart]
            UpSignalMidMom.extend(ak.flatten(upSignalMidMom,axis=1))
            DnSignalMidMom.extend(ak.flatten(dnSignalMidMom,axis=1))
            # reflection momentum difference of signal-like electrons
            deltaMidMom = dnSignalMidMom - upSignalMidMom
            DeltaMidMom.extend(ak.flatten(deltaMidMom,axis=1))
            # Process MC truth
            # first select the most closesly-related MC particle
            upTrkMC = upTrkMC[(upTrkMC.trkrel._rel == 0)]
            dnTrkMC = dnTrkMC[(dnTrkMC.trkrel._rel == 0)]
            # selections based on particle species
            upSigMC = (upTrkMC.pdg == self.SigPDG)
            dnSigMC = (dnTrkMC.pdg == self.SigPDG)
            upBkgMC = (upTrkMC.pdg == self.BkgPDG)
            dnBkgMC = (dnTrkMC.pdg == self.BkgPDG)
            sigMC = upSigMC & dnSigMC
            bkgMC = upBkgMC & dnBkgMC
            dkMC = upBkgMC & dnSigMC # decays in flight
            # select MC truth of entrance times
            DeltaEntTimeSigMC.extend(ak.flatten(deltaEntTime[goodReco & sigMC]))
            DeltaEntTimeBkgMC.extend(ak.flatten(deltaEntTime[goodReco & bkgMC]))
            DeltaEntTimeDkMC.extend(ak.flatten(deltaEntTime[goodReco & dkMC]))

            SelectedDeltaEntTimeSigMC.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange & sigMC]))
            SelectedDeltaEntTimeBkgMC.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange & bkgMC]))
            SelectedDeltaEntTimeDkMC.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange & dkMC]))

            upMidSegsMC = upSegsMC[upSegsMC.sid== self.TrkMidSID]
            dnMidSegsMC = dnSegsMC[dnSegsMC.sid== self.TrkMidSID]

            upMidMomMC = upMidSegsMC.mom.magnitude()
            dnMidMomMC = dnMidSegsMC.mom.magnitude()
            # select correct direction
            upMidMomMC = upMidMomMC[upMidSegsMC.mom.z()<0]
            dnMidMomMC = dnMidMomMC[dnMidSegsMC.mom.z()>0]
            hasUpMidMomMC = ak.count_nonzero(upMidMomMC,axis=1,keepdims=True)==1
            hasDnMidMomMC = ak.count_nonzero(dnMidMomMC,axis=1,keepdims=True)==1
            hasUpMidMomMC = ak.flatten(hasUpMidMomMC)
            hasDnMidMomMC = ak.flatten(hasDnMidMomMC)
            goodMC = goodSignalsigPart & sigMC & hasUpMidMomMC & hasDnMidMomMC
            goodRes = goodsigPart & goodMC  # resolution plot requires a good MC match
            upResMom = upMidMom[goodRes]
            dnResMom = dnMidMom[goodRes]
            upResMomMC = upMidMomMC[goodRes]
            dnResMomMC = dnMidMomMC[goodRes]
            dnResMom = ak.ravel(dnResMom)
            upResMom = ak.ravel(upResMom)
            dnResMomMC = ak.ravel(dnResMomMC)
            upResMomMC = ak.ravel(upResMomMC)
            upMidMomMC = ak.ravel(upMidMomMC)
            dnMidMomMC = ak.ravel(dnMidMomMC)
            deltaMidMomMC = dnResMomMC-upResMomMC
            DeltaMidMomMC.extend(deltaMidMomMC)
        print()
        print("From", NReflect,"total reflections found", NRecoReflect,"with good quality reco,", NsigPartReflect, "confirmed",self.SigPDGName,"and",len(DeltaMidMom), "in momentum range",self.MomRange)
        # compute Delta-T PID performance metrics
        goodDT = np.abs(np.array(DeltaEntTime)) < self.MaxDeltaT
        Ngood = sum( goodDT )
        goodElDT = np.abs(np.array(DeltaEntTimeSigMC)) < self.MaxDeltaT
        NgoodEl = sum( goodElDT)
        NEl = len(DeltaEntTimeSigMC)
        eff = NgoodEl/NEl
        pur = NgoodEl/Ngood
        print("For All |Delta T| < ", self.MaxDeltaT , " efficiency = ",eff," purity = ",pur)
        # plot DeltaT
        fig, (deltat, seldeltat) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        nDeltaTBins = 100
        trange=(-20,20)
        dt =     deltat.hist(DeltaEntTime,label="All", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        dtSigMC = deltat.hist(DeltaEntTimeSigMC,label="True "+self.SigPDGName, bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        dtBkgMC = deltat.hist(DeltaEntTimeBkgMC,label="True "+self.BkgPDGName, bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        dtDkMC = deltat.hist(DeltaEntTimeDkMC,label="Decays", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        deltat.set_title("$\\Delta$ Fit Time at Tracker Entrance")
        deltat.set_xlabel("Downstream - Upstream Time (nSec)")
        deltat.legend(loc="upper right")
        fig.text(0.1, 0.5, f"|$\\Delta$ T| < {self.MaxDeltaT:.2f}")
        fig.text(0.1, 0.4, f"signal purity = {pur:.3f}")
        fig.text(0.1, 0.3,  f"signal efficiency = {eff:.3f}")
        seldt =     seldeltat.hist(SelectedDeltaEntTime,label="All", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        seldtSigMC = seldeltat.hist(SelectedDeltaEntTimeSigMC,label="True "+self.SigPDGName, bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        seldtBkgMC = seldeltat.hist(SelectedDeltaEntTimeBkgMC,label="True "+self.BkgPDGName, bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        seldtDkMC = seldeltat.hist(SelectedDeltaEntTimeDkMC,label="Decays", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
        seldeltat.set_title("Signal-like $\\Delta$ Fit Time at Tracker Entrance")
        seldeltat.set_xlabel("Downstream - Upstream Time (nSec)")
        seldeltat.legend(loc="upper right")
        #
        fig, (cmat,cselmat,cst) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
        nipa = cmat.hist(NIPACyl,label="All IPA", bins=20, range=[-0.5,19.5], histtype='step', stacked=True)
        nst = cmat.hist(NSTFoil,label="All ST", bins=20, range=[-0.5,19.5], histtype='step', stacked=True)
        cmat.set_title("Material Intersections")
        cmat.set_xlabel("N Intersections")
        cmat.legend(loc="upper right")
        nsipa = cselmat.hist(SelectedNIPACyl,label="Selected IPA", bins=20, range=[-0.5,19.5], histtype='step', stacked=True)
        nsst = cselmat.hist(SelectedNSTFoil,label="Selected ST", bins=20, range=[-0.5,19.5], histtype='step', stacked=True)
        cselmat.set_title("Material Intersections")
        cselmat.set_xlabel("N Intersections")
        cselmat.legend(loc="upper right")
        nst = cst.hist2d(NSTOtCyl,NSTFoil,label="ST",bins=[10,10],range=[[-0.5,9.5],[-0.5,9.5]],density=True,norm="log")
        cst.set_title("ST Intersections")
        cst.set_xlabel("N Cyl Intersections")
        cst.set_ylabel("N Foil Intersections")
        #
        nDeltaMomBins = 200
        nMomBins = 100
        momrange=(40.0,200.0)
        momresorange=(-2.5,2.5)
        deltamomrange=(-10,5)
        fig, (upMom, dnMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
        dnMom.hist(DnMidMom,label="All "+self.SigPDGName, bins=nMomBins, range=momrange, histtype='step')
        dnMom.hist(DnSignalMidMom,label="Selected "+self.SigPDGName, bins=nMomBins, range=momrange, histtype='step')
        dnMom.set_title("Downstream Momentum at Tracker Mid")
        dnMom.set_xlabel("Fit Momentum (MeV)")
        dnMom.legend(loc="upper right")
        #
        upMom.hist(UpMidMom,label="All "+self.SigPDGName, bins=nMomBins, range=momrange, histtype='step')
        upMom.hist(UpSignalMidMom,label="Selected "+self.SigPDGName, bins=nMomBins, range=momrange, histtype='step')
        upMom.set_title("Upstream Momentum at Tracker Mid")
        upMom.set_xlabel("Fit Momentum (MeV)")
        upMom.legend(loc="upper right")
        #
        DeltaMomHist = deltaMom.hist(DeltaMidMom,label="Fit $\\Delta$ P", bins=nDeltaMomBins, range=deltamomrange, histtype='step')
        DeltaMomHistMC = deltaMom.hist(DeltaMidMomMC,label="MC $\\Delta$ P", bins=nDeltaMomBins, range=deltamomrange, histtype='step')
        deltaMom.set_xlabel("Downstream - Upstream Momentum (MeV)")
        deltaMom.set_title("$\\Delta$ Momentum at Tracker Middle")
        deltaMom.legend(loc="upper right")
        # fit
        DeltaMomHistErrors = np.zeros(len(DeltaMomHist[1])-1)
        DeltaMomHistBinMid =np.zeros(len(DeltaMomHist[1])-1)
        for ibin in range(len(DeltaMomHistErrors)):
            DeltaMomHistBinMid[ibin] = 0.5*(DeltaMomHist[1][ibin] + DeltaMomHist[1][ibin+1])
            DeltaMomHistErrors[ibin] = max(1.0,math.sqrt(DeltaMomHist[0][ibin]))
        #print(DeltaMomHistBinErrors)
        DeltaMomHistIntegral = np.sum(DeltaMomHist[0])
        # initialize the fit parameters
        mu_0 = np.mean(DeltaMomHistBinMid*DeltaMomHist[0]/DeltaMomHistIntegral) # initial mean
        var = np.sum(((DeltaMomHistBinMid**2)*DeltaMomHist[0])/DeltaMomHistIntegral) - mu_0**2
        sigma_0 = np.sqrt(var) # initial sigma
        lamb_0 = sigma_0 # initial exponential (guess)
        binsize = DeltaMomHist[1][1]-DeltaMomHist[1][0]
        amp_0 = DeltaMomHistIntegral*binsize # initial amplitude
        p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
        # fit, returing optimum parameters and covariance
        popt, pcov = curve_fit(fxn_expGauss, DeltaMomHistBinMid, DeltaMomHist[0], p0, sigma=DeltaMomHistErrors)
        print("Trk fit parameters",popt)
        print("Trk fit covariance",pcov)
        fig, (Trk,MC) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
        Trk.stairs(edges=DeltaMomHist[1],values=DeltaMomHist[0],label="Track $\\Delta$ P")
        Trk.plot(DeltaMomHistBinMid, fxn_expGauss(DeltaMomHistBinMid, *popt), 'r-',label="EMG Fit")
        Trk.legend(loc="upper right")
        Trk.set_title('EMG Fit to Track '+self.SigPDGName+' $\\Delta$ P')
        Trk.set_xlabel("Downstream - Upstream Momentum (MeV)")
        fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")
        # now fit MC for comparison
        for ibin in range(len(DeltaMomHistMC[1])-1):
            DeltaMomHistBinMid[ibin] = 0.5*(DeltaMomHistMC[1][ibin] + DeltaMomHistMC[1][ibin+1])
            DeltaMomHistErrors[ibin] = max(1.0,math.sqrt(DeltaMomHistMC[0][ibin]))
        popt, pcov = curve_fit(fxn_expGauss, DeltaMomHistBinMid, DeltaMomHistMC[0], p0, sigma=DeltaMomHistErrors)
        print("MC Fit parameters",popt)
        print("MC Fit covariance",pcov)
        MC.stairs(edges=DeltaMomHistMC[1],values=DeltaMomHistMC[0],label="MC $\\Delta$ P")
        MC.plot(DeltaMomHistBinMid, fxn_expGauss(DeltaMomHistBinMid, *popt), 'r-',label="EMG Fit")
        MC.legend(loc="upper right")
        MC.set_title('EMG Fit to MC '+self.SigPDGName+' $\\Delta$ P')
        MC.set_xlabel("Downstream - Upstream Momentum (MeV)")
        fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
        fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
        fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")
