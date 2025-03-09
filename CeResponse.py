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

file = [ "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/50/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000776.root" ]
files = [
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/50/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000776.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/5b/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000238.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/99/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000849.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/00/eb/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000721.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/0a/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000715.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/46/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000007.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/68/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000510.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/cd/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000289.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/01/dd/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000669.root",
        "/data/HD5/mu2e/MDC2020/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00/02/03/nts.mu2e.DIOtail_95OnSpillTriggered.MDC2020aq_best_v1_3_v06_03_00.001210_00000333.root"
        ]

# setup general constants
minNHits = 20
minFitCon = 1.0e-5
ibatch=0
elPDG = 11
# momentum range around a conversion electron
cemom = 104
dmom = 20
minMom = cemom - dmom
maxMom = cemom + dmom
# Surface Ids
TrkSIDs = [None]*3
TrkSIDs[0] = 0 # entrance
TrkSIDs[1] = 1 # mid
TrkSIDs[2] = 2 # exit
# Momentum and response
Mom = []
MomReso = []
MomResp = []
for isid in range(len(TrkSIDs)) :
    Mom.append([])
    MomReso.append([])
    MomResp.append([])

for batch,rep in uproot.iterate(files,filter_name="/trk|trksegs|trkmcsim|trksegsmc/i",report=True):
    print("Processing batch ",ibatch)
    ibatch = ibatch+1
    segs = batch['trksegs'] # track fit samples
    nhits = batch['trk.nactive']  # track N hits
    fitcon = batch['trk.fitcon']  # track fit consistency
    trkMC = batch['trkmcsim']  # MC genealogy of particles
    segsMC = batch['trksegsmc'] # SurfaceStep infor for true primary particle
    Segs = segs[:,0] # assume the 1st track is the Ce
    SegsMC = segsMC[:,0] # upstream track MC truth
    TrkMC = trkMC[:,0] # upstream fit associated true particles
    FitCon = fitcon[:,0]
    Nhits = nhits[:,0]
    # basic consistency test
    assert((len(Segs) == len(SegsMC)) & (len(Segs) == len(TrkMC)) & (len(Nhits) == len(Segs)))
    # sample the fits at 3 tracker locations
    for isid in range(len(TrkSIDs)) :
        sid = TrkSIDs[isid]
        segs = Segs[Segs.sid == sid]
        mom = segs.mom.magnitude()
        hasmom= ak.count_nonzero(mom,axis=1,keepdims=True)==1
        hasmom =ak.ravel(hasmom)
        signalLike = ((mom > minMom) & (mom < maxMom))
        segsMC = SegsMC[(SegsMC.sid == sid) & (SegsMC.mom.z() > 0.0) ]
        momMC = segsMC.mom.magnitude()
        hasMC = ak.count_nonzero(momMC,axis=1,keepdims=True)==1
        hasMC = ak.ravel(hasMC)
        mom = mom[goodFit & hasmom & hasMC]
        momMC = momMC[goodFit & hasmom & hasMC]
        mom = ak.ravel(mom)
        momMC = ak.ravel(momMC)
        assert(len(mom) == len(momMC))
        Mom[isid].extend(mom)
        momres = mom - momMC
#        print(momres[0:10])
        MomReso[isid].extend(momres)
    # plot Momentum
nDeltaMomBins = 200
nMomBins = 100
momrange=(80.0,110.0)
deltamomrange=(-5,5)
#print(len(Mom[0]),Mom[0:10][0])
#print(len(MomReso[0]),MomReso[0:10][0])
fig, (entMom, midMom, extMom) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
entMom.hist(Mom[0],label="Tracker Entrance", bins=nMomBins, range=momrange, histtype='step')
midMom.hist(Mom[1],label="Tracker Middle", bins=nMomBins, range=momrange, histtype='step')
extMom.hist(Mom[2],label="Tracker Exit", bins=nMomBins, range=momrange, histtype='step')
entMom.set_xlabel("Fit Momentum (MeV)")
midMom.set_xlabel("Fit Momentum (MeV)")
extMom.set_xlabel("Fit Momentum (MeV)")

fig, (entMomReso, midMomReso, extMomReso) = plt.subplots(1,3,layout='constrained', figsize=(10,5))
entMomReso.hist(MomReso[0],label="Tracker Entrance", bins=nMomBins, range=deltamomrange, histtype='step')
midMomReso.hist(MomReso[1],label="Tracker Middle", bins=nMomBins, range=deltamomrange, histtype='step')
extMomReso.hist(MomReso[2],label="Tracker Exit", bins=nMomBins, range=deltamomrange, histtype='step')
entMomReso.set_xlabel("Fit - MC Momentum (MeV)")
midMomReso.set_xlabel("Fit - MC Momentum (MeV)")
extMomReso.set_xlabel("Fit - MC Momentum (MeV)")
plt.show()
#
# fit
#DeltaMomHistErrors = np.zeros(len(DeltaMomHist[1])-1)
#DeltaMomHistBinMid =np.zeros(len(DeltaMomHist[1])-1)
#for ibin in range(len(DeltaMomHist[1])-1):
#    DeltaMomHistBinMid[ibin] = 0.5*(DeltaMomHist[1][ibin] + DeltaMomHist[1][ibin+1])
#    DeltaMomHistErrors[ibin] = max(1.0,math.sqrt(DeltaMomHist[0][ibin]))
##print(DeltaMomHistBinErrors)
#DeltaMomHistIntegral = np.sum(DeltaMomHist[0])
## initialize the fit parameters
#mu_0 = np.mean(DeltaMomHistBinMid*DeltaMomHist[0]/DeltaMomHistIntegral) # initial mean
#var = np.sum(((DeltaMomHistBinMid**2)*DeltaMomHist[0])/DeltaMomHistIntegral) - mu_0**2
#sigma_0 = np.sqrt(var) # initial sigma
#lamb_0 = sigma_0 # initial exponential (guess)
#binsize = DeltaMomHist[1][1]-DeltaMomHist[1][0]
#amp_0 = DeltaMomHistIntegral*binsize # initial amplitude
#p0 = np.array([amp_0, mu_0, sigma_0, lamb_0]) # initial parameters
## fit, returing optimum parameters and covariance
#popt, pcov = curve_fit(fxn_expGauss, DeltaMomHistBinMid, DeltaMomHist[0], p0, sigma=DeltaMomHistErrors)
#print("Trk fit parameters",popt)
#print("Trk fit covariance",pcov)
#fig, (Trk,MC) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
#Trk.stairs(edges=DeltaMomHist[1],values=DeltaMomHist[0],label="Track $\\Delta$ P")
#Trk.plot(DeltaMomHistBinMid, fxn_expGauss(DeltaMomHistBinMid, *popt), 'r-',label="EMG Fit")
#Trk.legend()
#Trk.set_title('EMG fit to Track $\\Delta$ P')
#Trk.set_xlabel("Downstream - Upstream Momentum (MeV)")
#fig.text(0.1, 0.5, f"$\\mu$ = {popt[1]:.3f}")
#fig.text(0.1, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
#fig.text(0.1, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")
## now fit MC for comparison
#for ibin in range(len(DeltaMomHistMC[1])-1):
#    DeltaMomHistBinMid[ibin] = 0.5*(DeltaMomHistMC[1][ibin] + DeltaMomHistMC[1][ibin+1])
#    DeltaMomHistErrors[ibin] = max(1.0,math.sqrt(DeltaMomHistMC[0][ibin]))
#popt, pcov = curve_fit(fxn_expGauss, DeltaMomHistBinMid, DeltaMomHistMC[0], p0, sigma=DeltaMomHistErrors)
#print("MC Fit parameters",popt)
#print("MC Fit covariance",pcov)
#MC.stairs(edges=DeltaMomHistMC[1],values=DeltaMomHistMC[0],label="MC $\\Delta$ P")
#MC.plot(DeltaMomHistBinMid, fxn_expGauss(DeltaMomHistBinMid, *popt), 'r-',label="EMG Fit")
#MC.legend()
#MC.set_title('EMG fit to MC $\\Delta$ P')
#MC.set_xlabel("Downstream - Upstream Momentum (MeV)")
#fig.text(0.6, 0.5, f"$\\mu$ = {popt[1]:.3f}")
#fig.text(0.6, 0.4, f"$\\sigma$ = {popt[2]:.3f}")
#fig.text(0.6, 0.3,  f"$\\lambda$ = {popt[3]:.3f}")
## plot momentum resolution
#nMomResoBins = 200
#fig, (upMomReso, dnMomReso)= plt.subplots(1,2,layout='constrained', figsize=(10,5))
#upMomReso.hist(UpMomReso,label="Upstream",bins=nMomResoBins, range=(-2.5,2.5), histtype='bar')
#upMomReso.set_title("Upstream Momentum Resoolution at Tracker Mid")
#upMomReso.set_xlabel("Reco - True Momentum (MeV)")
#dnMomReso.hist(DnMomReso,label="Downstream",bins=nMomResoBins, range=(-2.5,2.5), histtype='bar')
#dnMomReso.set_title("Downstream Momentum Resoolution at Tracker Mid")
#dnMomReso.set_xlabel("Reco - True Momentum (MeV)")
#plt.show()

# plot dnstream and downstream momentum  Resoolution
