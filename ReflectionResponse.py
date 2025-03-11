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
#file = [ "/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple" ]
file = [ "/Users/brownd/data/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple" ]
#files = [
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000000.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000010.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000024.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000053.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000085.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00004963.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00005432.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00010057.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00010872.root:TAReM/ntuple",
#"/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00015026.root:TAReM/ntuple" ]
files = [
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000007.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000000.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000004.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000005.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000003.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000001.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000006.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000009.root",
        "nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000002.root" ]
DeltaEntTime = []
DeltaEntTimeElMC =[]
DeltaEntTimeMuMC =[]
DeltaEntTimeDeMC =[]
UpMidMom = []
UpGoodMidMom = []
DnMidMom = []
DnGoodMidMom = []
DeltaMidMom = []
DeltaMidMomMC = []
UpMomRes = []
DnMomRes = []
# setup general constants
minNHits = 20
minFitCon = 1.0e-5
maxDeltaT = 5.0 # nsec
ibatch=0
elPDG = 11
muPDG = 13
# momentum range around a conversion electron
cemom = 104
dmom = 20
minMom = cemom - dmom
maxMom = cemom + dmom
# Surface Ids
trkEntSID = 0
trkMidSID = 1
# counts for purity and efficiency
Ngood = 0
NgoodEl = 0
NEl = 0
for batch,rep in uproot.iterate(file,filter_name="/trk|trksegs|trkmcsim|gtrksegsmc/i",report=True):
    print("Processing batch ",ibatch)
    ibatch = ibatch+1
    segs = batch['trksegs'] # track fit samples
    nhits = batch['trk.nactive']  # track N hits
    fitcon = batch['trk.fitcon']  # track fit consistency
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
    upFitCon = fitcon[:,0]
    dnFitCon = fitcon[:,1]
    upNhits = nhits[:,0]
    dnNhits = nhits[:,1]
#    print(len(upFitCon),upFitCon)
#    print(len(dnFitCon),dnFitCon)
#    print(len(upNhits),upNhits)
#    print(len(dnNhits),dnNhits)

# select based on time difference at tracker entrance
    upEntTime = upSegs[(upSegs.sid==trkEntSID) & (upSegs.mom.z() > 0.0) ].time
    dnEntTime = dnSegs[(dnSegs.sid==trkEntSID) & (dnSegs.mom.z() > 0.0) ].time
    deltaEntTime = dnEntTime-upEntTime
    DeltaEntTime.extend(ak.flatten(deltaEntTime))
# select by MC truth
    upTrkMC = upTrkMC[upTrkMC.trkrel._rel == 0] # select the true particle most associated with the track
    dnTrkMC = dnTrkMC[dnTrkMC.trkrel._rel == 0]
    upTrkMC = ak.flatten(upTrkMC,axis=1) # project out the struct
    dnTrkMC = ak.flatten(dnTrkMC,axis=1)

#    print( len(upTrkMC))

    upElMC = upTrkMC.pdg == elPDG
    dnElMC = dnTrkMC.pdg == elPDG
    upMuMC = upTrkMC.pdg == muPDG
    dnMuMC = dnTrkMC.pdg == muPDG
    elMC = upElMC & dnElMC
    muMC = upMuMC & dnMuMC
    deMC = upMuMC & dnElMC
    deltaEntTimeElMC = deltaEntTime[elMC]
    deltaEntTimeMuMC = deltaEntTime[muMC]
    deltaEntTimeDeMC = deltaEntTime[deMC]
    DeltaEntTimeElMC.extend(ak.flatten(deltaEntTimeElMC))
    DeltaEntTimeMuMC.extend(ak.flatten(deltaEntTimeMuMC))
    DeltaEntTimeDeMC.extend(ak.flatten(deltaEntTimeDeMC))

#    print(deltaEntTimeElMC)
#    print(deltaEntTimeMuMC)

# select good electron fits based on time difference at tracker entrance

    goodDeltaT = abs(deltaEntTime) < maxDeltaT
    goodDeltaT = ak.flatten(goodDeltaT)


#    print(goodDeltaT,len(goodDeltaT))
# select based on fit quality
    upGoodFit = (upNhits >= minNHits) & (upFitCon > minFitCon)
    dnGoodFit = (dnNhits >= minNHits) & (dnFitCon > minFitCon)
    goodFit = upGoodFit & dnGoodFit
#    print(goodFit,len(goodFit))


    # sample the fits at middle of traacker
    upMidSegs = upSegs[upSegs.sid== trkMidSID]
    dnMidSegs = dnSegs[dnSegs.sid== trkMidSID]
    upMidSegsMC = upSegsMC[upSegsMC.sid== trkMidSID]
    dnMidSegsMC = dnSegsMC[dnSegsMC.sid== trkMidSID]
#    print("Mid seg counts ",len(upMidSegs),len(dnMidSegs),len(upMidSegsMC),len(dnMidSegsMC),len(elMC),len(goodFit),len(goodDeltaT))
#    print(upMidSegs[0:10])
#    print(upMidSegsMC[0:10])

    # total momentum at tracker mid
    upMidMom = upMidSegs.mom.magnitude()
    dnMidMom = dnMidSegs.mom.magnitude()
    upMidMomMC = upMidSegsMC.mom.magnitude()
    dnMidMomMC = dnMidSegsMC.mom.magnitude()
    # select correct direction
    upMidMomMC = upMidMomMC[upMidSegsMC.mom.z()<0]
    dnMidMomMC = dnMidMomMC[dnMidSegsMC.mom.z()>0]
#    print("midMomMC",len(upMidMomMC),len(dnMidMomMC))
#    print("before flatten",len(upMidMom),len(dnMidMom),len(upMidMomMC),len(dnMidMomMC))
#    print(upMidMomMC[0:10])
#    print(dnMidMomMC[0:10])
    # flatten
    upMidMom = ak.flatten(upMidMom,axis=1)
    dnMidMom = ak.flatten(dnMidMom,axis=1)
#    print("midmom ",len(upMidMom),len(dnMidMom))
    # select 'signal-like' electrons. For now, just the momentum, later maybe add
    # consistency with the target and pitch cuts
    signalLike = ((dnMidMom > minMom) & (dnMidMom < maxMom)) | ((upMidMom > minMom) & (upMidMom < maxMom))
#    print(len(signalLike))

    hasUpMidMomMC = ak.count_nonzero(upMidMomMC,axis=1,keepdims=True)==1
    hasDnMidMomMC = ak.count_nonzero(dnMidMomMC,axis=1,keepdims=True)==1
    hasUpMidMomMC = ak.flatten(hasUpMidMomMC)
    hasDnMidMomMC = ak.flatten(hasDnMidMomMC)
#    print("hasMC",len(hasUpMidMomMC),len(hasDnMidMomMC))
#    print(hasUpMidMomMC)
    goodReco = goodFit & goodDeltaT & signalLike
    goodMC = elMC & hasUpMidMomMC & hasDnMidMomMC
    goodRes = goodReco & goodMC  # resolution plot requires a good MC match
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
    upMomRes = upResMom-upResMomMC
    UpMomRes.extend(upMomRes)
    dnMomRes = dnResMom-dnResMomMC
    DnMomRes.extend(dnMomRes)

    upMidMomMC = ak.ravel(upMidMomMC)
    dnMidMomMC = ak.ravel(dnMidMomMC)
 #   print(upMidMom[0:10],upMidMomMC[0:10])

    # select based on reco
    upGoodMidMom = upMidMom[goodReco]
    dnGoodMidMom = dnMidMom[goodReco]
    DnMidMom.extend(dnMidMom)
    DnGoodMidMom.extend(dnGoodMidMom)
    UpMidMom.extend(upMidMom)
    UpGoodMidMom.extend(upGoodMidMom)
    # reflection momentum difference
    deltaMidMom = dnGoodMidMom - upGoodMidMom
    DeltaMidMom.extend(deltaMidMom)
    deltaMidMomMC = dnResMomMC-upResMomMC
    DeltaMidMomMC.extend(deltaMidMomMC)

# compute purity sums
    Ngood = Ngood + len(deltaEntTime[goodDeltaT])
    NEl = NEl + len(deltaEntTime[elMC])
    NgoodEl = NgoodEl + len(deltaEntTime[elMC & goodDeltaT])

print("From ", len(DnMidMom)," total, selected ",len(DeltaMidMom)," good quality signal-like reflections and ",len(DnMomRes)," reflections for resolution")

# compute Delta-T PID performance metrics
eff = NgoodEl/NEl
pur = NgoodEl/Ngood
print("For |Delta T| < ", maxDeltaT , " efficiency = ",eff," purity = ",pur)
# plot DeltaT
fig, deltat = plt.subplots(1,1,layout='constrained', figsize=(5,5))
nDeltaTBins = 100
trange=(-20,20)
dt =     deltat.hist(DeltaEntTime,label="All", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
dtElMC = deltat.hist(DeltaEntTimeElMC,label="True Electron", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
dtMuMC = deltat.hist(DeltaEntTimeMuMC,label="True Muon", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
dtDeMC = deltat.hist(DeltaEntTimeDeMC,label="Muon Decays", bins=nDeltaTBins, range=trange, histtype='bar', stacked=True)
deltat.set_title("$\\Delta$ Fit Time at Tracker Entrance")
deltat.set_xlabel("Downstream - Upstream Time (nSec)")
deltat.legend()
# plot Momentum
nDeltaMomBins = 200
nMomBins = 100
momrange=(70.0,150.0)
deltamomrange=(-15,5)
fig, (upMom, dnMom, deltaMom) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
dnMom.hist(DnMidMom,label="All Downstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
dnMom.hist(DnGoodMidMom,label="Selected Downstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
dnMom.set_title("Downstream Momentum at Tracker Mid")
dnMom.set_xlabel("Fit Momentum (MeV)")
dnMom.legend()
#
upMom.hist(UpMidMom,label="All Upstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
upMom.hist(UpGoodMidMom,label="Selected Upstream $e^-$", bins=nMomBins, range=momrange, histtype='step')
upMom.set_title("Upstream Momentum at Tracker Mid")
upMom.set_xlabel("Fit Momentum (MeV)")
upMom.legend()
#
DeltaMomHist = deltaMom.hist(DeltaMidMom,label="Fit $\\Delta$ P", bins=nDeltaMomBins, range=deltamomrange, histtype='step')
DeltaMomHistMC = deltaMom.hist(DeltaMidMomMC,label="MC $\\Delta$ P", bins=nDeltaMomBins, range=deltamomrange, histtype='step')
deltaMom.set_xlabel("Downstream - Upstream Momentum (MeV)")
deltaMom.set_title("$\\Delta$ Fit Momentum at Tracker Middle")
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
# plot momentum resolution
nMomResBins = 200
fig, (upMomRes, dnMomRes)= plt.subplots(1,2,layout='constrained', figsize=(10,5))
upMomRes.hist(UpMomRes,label="Upstream",bins=nMomResBins, range=(-2.5,2.5), histtype='bar')
upMomRes.set_title("Upstream Momentum Resolution at Tracker Mid")
upMomRes.set_xlabel("Reco - True Momentum (MeV)")
dnMomRes.hist(DnMomRes,label="Downstream",bins=nMomResBins, range=(-2.5,2.5), histtype='bar')
dnMomRes.set_title("Downstream Momentum Resolution at Tracker Mid")
dnMomRes.set_xlabel("Reco - True Momentum (MeV)")
plt.show()

# plot dnstream and downstream momentum  Resolution
