# Compare upstream and downstream resolution at tracker mid for e- reflections
#
import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot
import numpy as np
from scipy.optimize import curve_fit
import math
from scipy import special

def fxn_expGauss(x, amp, mu, sigma, lamb):
    z = (mu + lamb*(sigma**2) + x)/(np.sqrt(2)*sigma)
    comp_err_func = special.erfc(z)
    val = amp*(lamb/2)*((math.e)**((lamb/2)*(2*mu+lamb*(sigma**2)+2*x)))*comp_err_func
    return val

#file = '/home/online1/ejc/public/brownd/dts.mu2e.CeEndpoint.MDC2020r.001210_00000000.art.digi.art.ntuple.root'
#file = '/data/HD5/users/brownd/ntp.brownd.Reflections.v4.root'
fl03file = [ "/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple" ]
mbfile = [ "/Users/brownd/data/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple" ]
fl03files = [
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000010.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000024.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000053.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000085.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00004963.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00005432.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00010057.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00010872.root:TAReM/ntuple",
"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00015026.root:TAReM/ntuple" ]
mbfiles = [
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000007.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000000.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000004.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000005.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000003.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000001.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000006.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000009.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000002.root" ]
offfiles = [
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00000010.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00000518.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00001032.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00001547.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00002057.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00002561.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00003067.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00003570.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00004074.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00004577.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00005081.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00005616.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00006156.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00006706.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00007246.root:TARe/ntuple",
#"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00007764.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00008324.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00008894.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00009535.root:TARe/ntuple",
"/data/HD5/users/brownd/61803286/nts.brownd.TAReflect.CRYAllOffSpill.001202_00013483.root:TARe/ntuple"
]
DeltaEntTime = []
DeltaEntTimeElMC =[]
DeltaEntTimeMuMC =[]
DeltaEntTimeDkMC =[]

SelectedDeltaEntTime = []
SelectedDeltaEntTimeElMC =[]
SelectedDeltaEntTimeMuMC =[]
SelectedDeltaEntTimeDkMC =[]
UpMidMom = []
UpSignalMidMom = []
DnMidMom = []
DnSignalMidMom = []
DeltaMidMom = []
DeltaMidMomMC = []
# counts
NReflect = 0
NRecoReflect = 0
NeMinusReflect = 0
# setup general constants
minNHits = 20
minFitCon = 1.0e-5
maxDeltaT = 5.0 # nsec
ibatch=0
ElPDG = 11
MuPDG = 13
# momentum range around a conversion electron
cemom = 104
dmom = 20
minMom = cemom - dmom
maxMom = cemom + dmom
# Surface Ids
trkEntSID = 0
trkMidSID = 1
for batch,rep in uproot.iterate(offfiles,filter_name="/trk|trksegs|trkmcsim|gtrksegsmc/i",report=True):
    print("Processing batch ",ibatch)
    ibatch = ibatch+1
    segs = batch['trksegs'] # track fit samples
    nhits = batch['trk.nactive']  # track N hits
    fitcon = batch['trk.fitcon']  # track fit consistency
    fitpdg = batch['trk.pdg']  # track fit consistency
    trkMC = batch['trkmcsim']  # MC genealogy of particles
    segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
#    ak.type(segs).show()
#    print("segs axis 0: ",ak.num(segs,axis=0))
#    print("segs axis 1: ",ak.num(segs,axis=1))
#    print("segs axis 2: ",ak.num(segs,axis=2))
    upSegs = segs[:,0] # upstream track fits
    dnSegs = segs[:,1] # downstream track fits
    upSegsMC = segsMC[:,0] # upstream track MC truth
    dnSegsMC = segsMC[:,1] # downstream track MC truth
    upTrkMC = trkMC[:,0] # upstream fit associated true particles
    dnTrkMC = trkMC[:,1] # downstream fit associated true particles
    # basic consistency test
    assert((len(upSegs) == len(dnSegs)) & (len(upSegsMC) == len(dnSegsMC)) & (len(upSegs) == len(upSegsMC))& (len(upTrkMC) == len(dnTrkMC)) & (len(upSegs) == len(upTrkMC)))
#    print(upSegsMC)
#    print(len(fitcon), fitcon)
    upFitPDG = fitpdg[:,0]
    dnFitPDG = fitpdg[:,1]
    upFitCon = fitcon[:,0]
    dnFitCon = fitcon[:,1]
    upNhits = nhits[:,0]
    dnNhits = nhits[:,1]
#    print(len(upFitCon),upFitCon)
#    print(len(dnFitCon),dnFitCon)
#    print(len(upNhits),upNhits)
#    print(len(dnNhits),dnNhits)

# select electron fits
    upEMinus = (upFitPDG == ElPDG)
    dnEMinus = (dnFitPDG == ElPDG)
    eMinusFit = upEMinus & dnEMinus
# select based on fit quality
    upGoodFit = (upNhits >= minNHits) & (upFitCon > minFitCon)
    dnGoodFit = (dnNhits >= minNHits) & (dnFitCon > minFitCon)
    goodFit = upGoodFit & dnGoodFit
    goodReco = eMinusFit & goodFit
    NReflect = NReflect + len(goodReco)
    NRecoReflect = NRecoReflect + sum(goodReco)
# select based on time difference at tracker entrance
    upEntTime = upSegs[(upSegs.sid==trkEntSID) & (upSegs.mom.z() > 0.0) ].time
    dnEntTime = dnSegs[(dnSegs.sid==trkEntSID) & (dnSegs.mom.z() > 0.0) ].time
    deltaEntTime = dnEntTime-upEntTime
    goodDeltaT = abs(deltaEntTime) < maxDeltaT
    DeltaEntTime.extend(ak.flatten(deltaEntTime[goodReco]))
# good electron
    goodeMinus = goodReco & goodDeltaT
    goodeMinus = ak.ravel(goodeMinus)
    NeMinusReflect = NeMinusReflect + sum(goodeMinus)
# total momentum at tracker mid, upstream and downstream fits
    upMidMom = upSegs[(upSegs.sid == trkMidSID)].mom.magnitude()
    dnMidMom = dnSegs[(dnSegs.sid == trkMidSID)].mom.magnitude()
    DnMidMom.extend(ak.flatten(dnMidMom,axis=1))
    UpMidMom.extend(ak.flatten(upMidMom,axis=1))
# select fits that look like signal electrons: this needs to include a target constraint TODO
    print("upmid",upMidMom,len(upMidMom),ak.count(upMidMom,axis=None))
    print("dnmid",dnMidMom,len(dnMidMom),ak.count(upMidMom,axis=None))
    testsame = ak.num(upMidMom,axis=1) == ak.num(dnMidMom,axis=1)
    print(ak.all(testsame))
    signalMomRange = (((dnMidMom > minMom) & (dnMidMom < maxMom)) | ((upMidMom > minMom) & (upMidMom < maxMom)))
    signalMomRange = ak.ravel(signalMomRange)
    goodSignaleMinus = signalMomRange & goodeMinus
    SelectedDeltaEntTime.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange]))

    upSignalMidMom = upMidMom[goodSignaleMinus]
    dnSignalMidMom = dnMidMom[goodSignaleMinus]
    UpSignalMidMom.extend(ak.flatten(upSignalMidMom,axis=1))
    DnSignalMidMom.extend(ak.flatten(dnSignalMidMom,axis=1))
# reflection momentum difference of signal-like electrons
    deltaMidMom = dnSignalMidMom - upSignalMidMom
    DeltaMidMom.extend(ak.flatten(deltaMidMom,axis=1))

# Process MC truth
# first select the most closesly-related MC particle
    upTrkMC = upTrkMC[(upTrkMC.trkrel._rel == 0)]
    dnTrkMC = dnTrkMC[(dnTrkMC.trkrel._rel == 0)]
#    upTrkMC = ak.flatten(upTrkMC,axis=1) # project out the struct
#    dnTrkMC = ak.flatten(dnTrkMC,axis=1)

# selections based on particle species
    upElMC = (upTrkMC.pdg == ElPDG)
    dnElMC = (dnTrkMC.pdg == ElPDG)
    upMuMC = (upTrkMC.pdg == MuPDG)
    dnMuMC = (dnTrkMC.pdg == MuPDG)
    elMC = upElMC & dnElMC
    muMC = upMuMC & dnMuMC
    dkMC = upMuMC & dnElMC # decays in flight
# select MC truth of entrance times
    DeltaEntTimeElMC.extend(ak.flatten(deltaEntTime[goodReco & elMC]))
    DeltaEntTimeMuMC.extend(ak.flatten(deltaEntTime[goodReco & muMC]))
    DeltaEntTimeDkMC.extend(ak.flatten(deltaEntTime[goodReco & dkMC]))

    SelectedDeltaEntTimeElMC.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange & elMC]))
    SelectedDeltaEntTimeMuMC.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange & muMC]))
    SelectedDeltaEntTimeDkMC.extend(ak.flatten(deltaEntTime[goodReco & signalMomRange & dkMC]))

    upMidSegsMC = upSegsMC[upSegsMC.sid== trkMidSID]
    dnMidSegsMC = dnSegsMC[dnSegsMC.sid== trkMidSID]

    upMidMomMC = upMidSegsMC.mom.magnitude()
    dnMidMomMC = dnMidSegsMC.mom.magnitude()
# select correct direction
    upMidMomMC = upMidMomMC[upMidSegsMC.mom.z()<0]
    dnMidMomMC = dnMidMomMC[dnMidSegsMC.mom.z()>0]
#
    hasUpMidMomMC = ak.count_nonzero(upMidMomMC,axis=1,keepdims=True)==1
    hasDnMidMomMC = ak.count_nonzero(dnMidMomMC,axis=1,keepdims=True)==1
    hasUpMidMomMC = ak.flatten(hasUpMidMomMC)
    hasDnMidMomMC = ak.flatten(hasDnMidMomMC)
#    print("hasMC",len(hasUpMidMomMC),len(hasDnMidMomMC))
#    print(hasUpMidMomMC)
    goodMC = goodSignaleMinus & elMC & hasUpMidMomMC & hasDnMidMomMC
#    print("goodeMinus",goodeMinus)
    goodRes = goodeMinus & goodMC  # resolution plot requires a good MC match
#    print("goodres",goodRes)
    upResMom = upMidMom[goodRes]
    dnResMom = dnMidMom[goodRes]
    upResMomMC = upMidMomMC[goodRes]
    dnResMomMC = dnMidMomMC[goodRes]
#    print("resmom before flatten",len(upResMom),len(dnResMom),len(upResMomMC),len(dnResMomMC))
    dnResMom = ak.ravel(dnResMom)
    upResMom = ak.ravel(upResMom)
    dnResMomMC = ak.ravel(dnResMomMC)
    upResMomMC = ak.ravel(upResMomMC)
#    print("resmom after flatten",len(upResMom),len(dnResMom),len(upResMomMC),len(dnResMomMC))

    upMidMomMC = ak.ravel(upMidMomMC)
    dnMidMomMC = ak.ravel(dnMidMomMC)
 #   print(upMidMom[0:10],upMidMomMC[0:10])

    deltaMidMomMC = dnResMomMC-upResMomMC
    DeltaMidMomMC.extend(deltaMidMomMC)

print("From ", NReflect," total reflections", NRecoReflect," good quality reco with ", NeMinusReflect, " confirmed eminus and ", len(DeltaMidMom), "signal-like reflections for resolution")

# compute Delta-T PID performance metrics
goodDT = np.abs(np.array(DeltaEntTime)) < maxDeltaT
Ngood = sum( goodDT )
goodElDT = np.abs(np.array(DeltaEntTimeElMC)) < maxDeltaT
NgoodEl = sum( goodElDT)
NEl = len(DeltaEntTimeElMC)
eff = NgoodEl/NEl
pur = NgoodEl/Ngood
print("For |Delta T| < ", maxDeltaT , " efficiency = ",eff," purity = ",pur)
# plot DeltaT
fig, (deltat, seldeltat) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
nDeltaTBins = 100
trange=(-20,20)
dt =     deltat.hist(DeltaEntTime,label="All", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
dtElMC = deltat.hist(DeltaEntTimeElMC,label="True Electron", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
dtMuMC = deltat.hist(DeltaEntTimeMuMC,label="True Muon", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
dtDkMC = deltat.hist(DeltaEntTimeDkMC,label="Muon Decays", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
deltat.set_title("$\\Delta$ Fit Time at Tracker Entrance")
deltat.set_xlabel("Downstream - Upstream Time (nSec)")
deltat.legend()
fig.text(0.1, 0.5, f"|$\\Delta$ T| < {maxDeltaT:.2f}")
fig.text(0.1, 0.4, f"$e^-$ purity = {pur:.3f}")
fig.text(0.1, 0.3,  f"$e^-$ efficiency = {eff:.3f}")
seldt =     seldeltat.hist(SelectedDeltaEntTime,label="All", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
seldtElMC = seldeltat.hist(SelectedDeltaEntTimeElMC,label="True Electron", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
seldtMuMC = seldeltat.hist(SelectedDeltaEntTimeMuMC,label="True Muon", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
seldtDkMC = seldeltat.hist(SelectedDeltaEntTimeDkMC,label="Muon Decays", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
seldeltat.set_title("Signal-like $\\Delta$ Fit Time at Tracker Entrance")
seldeltat.set_xlabel("Downstream - Upstream Time (nSec)")
seldeltat.legend()


nDeltaMomBins = 200
nMomBins = 100
momrange=(50.0,180.0)
momresorange=(-2.5,2.5)
deltamomrange=(-10,5)
fig, (upMom, dnMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
dnMom.hist(DnMidMom,label="All Downstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
dnMom.hist(DnSignalMidMom,label="Selected Downstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
dnMom.set_title("Downstream Momentum at Tracker Mid")
dnMom.set_xlabel("Fit Momentum (MeV)")
dnMom.legend()
#
upMom.hist(UpMidMom,label="All Upstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
upMom.hist(UpSignalMidMom,label="Selected Upstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
upMom.set_title("Upstream Momentum at Tracker Mid")
upMom.set_xlabel("Fit Momentum (MeV)")
upMom.legend()
#
DeltaMomHist = deltaMom.hist(DeltaMidMom,label="Fit $\\Delta$ P", bins=nDeltaMomBins, range=deltamomrange, histtype='step')
DeltaMomHistMC = deltaMom.hist(DeltaMidMomMC,label="MC $\\Delta$ P", bins=nDeltaMomBins, range=deltamomrange, histtype='step')
deltaMom.set_xlabel("Downstream - Upstream Momentum (MeV)")
deltaMom.set_title("$\\Delta$ Momentum at Tracker Middle")
deltaMom.legend()
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
Trk.legend()
Trk.set_title('EMG fit to Track $\\Delta$ P')
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
MC.legend()
MC.set_title('EMG fit to MC $\\Delta$ P')
MC.set_xlabel("Downstream - Upstream Momentum (MeV)")
fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")

