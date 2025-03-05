# Compare upstream and downstream resolution at tracker mid for e- reflections
#

import awkward as ak
import behaviors
from matplotlib import pyplot as plt
import uproot

#file = '/home/online1/ejc/public/brownd/dts.mu2e.CeEndpoint.MDC2020r.001210_00000000.art.digi.art.ntuple.root'
#file = '/data/HD5/users/brownd/ntp.brownd.Reflections.v4.root'
file = "/data/HD5/users/brownd/71077187/nts.brownd.TAReflect.TARef.001202_00000000.root"
files = [
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



with uproot.open(file) as f:
#for batch,rep in uproot.iterate(files,filter_name=\"/trk|trksegs|trkmcsim/i\",report=True):\n",
    tree = f['TAReM']['ntuple']
    segs = tree['trksegs'].array(library='ak') # track fit samples
    nhits = tree['trk.nactive'].array(library='ak') # track
    fitcon = tree['trk.fitcon'].array(library='ak') # track fit
    trkMC = tree['trkmcsim'].array(library='ak') # MC genealogy of particles
    trkSegMC = tree['trksegsmc'].array(library='ak') # SurfaceStep infor for true primary particle
#    ak.type(segs).show()
#    print("segs axis 0: ",ak.num(segs,axis=0))
#    print("segs axis 1: ",ak.num(segs,axis=1))
#    print("segs axis 2: ",ak.num(segs,axis=2))
    upSegs = segs[:,0] # upstream track fits
    dnSegs = segs[:,1] # downstream track fits
#    print(len(fitcon), fitcon)
    upFitCon = fitcon[:,0]
    dnFitCon = fitcon[:,1]
    upNhits = nhits[:,0]
    dnNhits = nhits[:,1]
    print(len(upFitCon),upFitCon)
    print(len(dnFitCon),dnFitCon)
    print(len(upNhits),upNhits)
    print(len(dnNhits),dnNhits)


# select based on time difference at tracker entrance
    upEntTime = upSegs[(upSegs.sid==0) & (upSegs.mom.z() > 0.0) ].time
    dnEntTime = dnSegs[(dnSegs.sid==0) & (dnSegs.mom.z() > 0.0) ].time
    deltaEntTime = upEntTime-dnEntTime
# select by MC truth
    upTrkMC = trkMC[:,0] # upstream fit associated true particles
    dnTrkMC = trkMC[:,1] # downstream fit associated true particles
    upTrkMC = upTrkMC[upTrkMC.trkrel._rel == 0] # select the true particle most associated with the track
    dnTrkMC = dnTrkMC[dnTrkMC.trkrel._rel == 0]
    upTrkMC = ak.flatten(upTrkMC,axis=1) # project out the struct
    dnTrkMC = ak.flatten(dnTrkMC,axis=1)

#    print( len(upTrkMC))

    upElMC = upTrkMC.pdg == 11
    dnElMC = dnTrkMC.pdg == 11
    upMuMC = upTrkMC.pdg == 13
    dnMuMC = dnTrkMC.pdg == 13
    elMC = upElMC & dnElMC
    muMC = upMuMC & dnMuMC
    deMC = upMuMC & dnElMC
    deltaEntTimeElMC = deltaEntTime[elMC]
    deltaEntTimeMuMC = deltaEntTime[muMC]
    deltaEntTimeDeMC = deltaEntTime[deMC]
#    print(deltaEntTimeElMC)
#    print(deltaEntTimeMuMC)

# select good electron fits based on time difference at tracker entrance
    goodele = abs(deltaEntTime) < 5.0
    goodele = ak.flatten(goodele)
#    print(goodele,len(goodele))
# select based on fit quality
    minNHits = 20
    minFitCon = 1.0e-5
    upGoodFit = (upNhits >= minNHits) & (upFitCon > minFitCon)
    dnGoodFit = (dnNhits >= minNHits) & (dnFitCon > minFitCon)
    goodfit = upGoodFit & dnGoodFit
    print(goodfit,len(goodfit))

    # sample the fits at middle of traacker
    upMidSegs = upSegs[upSegs.sid==1]
    dnMidSegs = dnSegs[dnSegs.sid==1]

    # total momentum at tracker mid
    upMidMom = upMidSegs.mom.magnitude()
    dnMidMom = dnMidSegs.mom.magnitude()
    # flatten
    upMidMom = ak.ravel(upMidMom)
    dnMidMom = ak.ravel(dnMidMom)

    # select: first PID and fit quality
    upMidMom = upMidMom[goodele & goodfit]
    dnMidMom = dnMidMom[goodele & goodfit]

    # momentum range around a conversion electron
    cemom = 104
    dmom = 20
    minmom = cemom - dmom
    maxmom = cemom + dmom
    signalLike = (dnMidMom > minmom) & (dnMidMom < maxmom)
    # good quality tracks

#    upGoodFit = (nhits[0] > 19) & (fitcon[0] > 1e-4)
#    print(upGoodFit)

    deltaMidMom = upMidMom - dnMidMom
    deltaMidMom = deltaMidMom[signalLike]
    print("Selected ", len(upMidMom)," total and ",len(deltaMidMom)," signal-like tracks")

fig, deltat = plt.subplots(1,1,layout='constrained', figsize=(5,5))
nbins = 100
trange=(-20,20)
dt =     deltat.hist(deltaEntTime,label="All", bins=nbins, range=trange, histtype='bar', stacked=True)
dtElMC = deltat.hist(deltaEntTimeElMC,label="True Electron", bins=nbins, range=trange, histtype='bar', stacked=True)
dtMuMC = deltat.hist(deltaEntTimeMuMC,label="True Muon", bins=nbins, range=trange, histtype='bar', stacked=True)
dtDeMC = deltat.hist(deltaEntTimeDeMC,label="Muon Decays", bins=nbins, range=trange, histtype='bar', stacked=True)
deltat.set_title("$\\Delta$ Fit Time at Tracker Entrance")
deltat.set_xlabel("Upstream time - Downstreamtime (nSec)")
deltat.legend()
plt.show()

fig, (upMom, deltaMom) = plt.subplots(1,2,layout='constrained', figsize=(10,5))
upMom.hist(upMidMom,label="Upstream Mid Tracker P", bins=100, range=(70.0,150.0), histtype='step')
deltaMom.hist(deltaMidMom,label="Upstream - Downstream Mid Tracker P", bins=100, range=(-10,10), histtype='step')
upMom.set_xlabel("Fit Momentum (MeV)")
deltaMom.set_xlabel("$\\Delta$ Fit Momentum (MeV)")
plt.show()
