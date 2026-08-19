import numpy as np
import h5py
import math
from scipy import stats as stats
from scipy.optimize import curve_fit


class MyHist:
    def __init__(self,name,label,title="",xlabel="",bins=100,range=[],file="",verbose=False):
        self.name = name
        self.label = label
        if(file != ""):
            #locate this histogram as a group with the given label
            grp =self.groupname()
            with h5py.File(file, 'r') as hdf5file: # closes on exit
                self.data = hdf5file[grp+"/data"][:]
                self.edges = hdf5file[grp+"/edges"][:]
                self.title = hdf5file.get(grp+"/title").asstr()[0]
                self.xlabel = hdf5file.get(grp+"/xlabel").asstr()[0]
                if(verbose):
                    print("Read ",end=' ')
                    self.print()
        else:
            data = []
            self.data, self.edges = np.histogram(data, bins=bins, range=range)
            self.title = title
            self.xlabel = xlabel
    def groupname(self):
        return "/"+self.name+"/"+self.label

    def print(self):
        print("MyHist",self.groupname(),"with",len(self.data),"bins and",self.integral(),"entries")

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

    def plotErrors(self,axis):
        plt =self.plot(axis)
        errors = self.binErrors()
        centers = self.binCenters()
        axis.errorbar(x=centers,y=self.data,yerr=errors)
        return plt

    def integral(self):
        return np.sum(self.data)

    def maxVal(self):
        return np.max(self.data)

    def maxX(self):
        maxindex = np.argmax(self.data)
        return 0.5*(self.edges[maxindex]+self.edges[maxindex+1])

    # bin range for bins with values above the given value
    def binRangeAboveValue(self,minval):
        istart=0
        while((istart < len(self.data)) & (self.data[istart]<minval)):
            istart += 1
        iend=len(self.data)-1
        while((iend > 0) & (self.data[iend]<minval)):
            iend -= 1
        iend += 1
        return [istart,iend]

    def binCenter(self, ibin):
        if (ibin > 0 & ibin < len(self.data)):
            return  0.5*(self.edges[ibin] + self.edges[ibin+1])
        elif ibin < 0:
            return self.edges[0]
        else:
            return self.edges[len(self.data)]

    def binCenters(self):
        midbin = np.zeros(len(self.data))
        for ibin in range(len(self.data)):
            midbin[ibin] = 0.5*(self.edges[ibin] + self.edges[ibin+1]) # edges have 1 more entry than data
        return midbin

    def inBin(self,ibin,xval):
        if (ibin >= 0 & ibin < len(self.data)):
            return xval >= self.edges[ibin] & xval < self.edges[ibin+1]
        elif ibin < 0:
            return xval < self.edges[0]
        else:
            return xval > self.edges[-1]

    def binIndex(self, xval):
        if (xval >= self.edges[0] & xval < self.edges[-1]) :
            ibin = int((xval + 0.5 -self.edges[0])/self.binWidth())
            if self.inBin(ibin,xval):
                return ibin
            else:
                # manual scan
                for ibin in range(len(self.data)):
                    if(self.inBin(xval)):
                        return ibin
                return -1
        elif xval > self.edges[-1]:
            return len(self.edges)
        else:
            return -1

    def binWidth(self,ibin=0):
        return self.edges[ibin+1]-self.edges[ibin]

    def mean(self):
        return np.average(self.binCenters(),weights=self.data)

    def variance(self):
        mean = self.mean()
        return np.average(np.square(self.binCenters()-mean),weights=self.data)

    def RMS(self):
        return math.sqrt(self.variance())

    def FWHM(self):
        halfmax = 0.5*self.maxVal()
        [ilow,ihigh] = self.binRangeAboveValue(halfmax)
        # this is intentionally one higher than ilow, as bin edges are at the lower edge
        fwhm = self.edges[ihigh]-self.edges[ilow]
        return fwhm

    def binErrors(self):
        # assume unweighted bins, Poisson statis
        errors = np.sqrt(self.data)
        ones = np.ones(len(self.data))
        errors = np.maximum(ones,errors)
        return errors

    def binRange(self,xrange):
        if (xrange[1] < xrange[0]):
            return[0,len(self.data)-1]
        istart=0
        while((istart < len(self.data)) & (self.data[istart]<minval)):
            istart += 1
        iend=len(self.data)-1
        while((iend > 0) & (self.data[iend]<minval)):
            iend -= 1
        iend += 1
        return [istart,iend]

    def fitArrays(self,xrange=[0,-1]):
        br = self.binRange(xrange)
        binmid = self.binCenters()[br[0]:br[1]]
        binval = self.data[br[0]:br[1]]
        binerr = self.binErrors()[br[0]:br[1]]
        return binmid, binval, binerr

    def save(self,hdf5file,verbose=False):
        grp = hdf5file.create_group(self.groupname())
        grp.create_dataset("data",data=self.data)
        grp.create_dataset("edges",data=self.edges)
        dst = grp.create_dataset("title", shape=1, dtype=h5py.string_dtype())
        dst[:] = self.title
        dsx = grp.create_dataset("xlabel", shape=1, dtype=h5py.string_dtype())
        dsx[:] = self.xlabel
        if(verbose):
            print("Saved",self.groupname(),"to",hdf5file.filename)

    def Fit(self,fitobj,xrange=[0,-1],verbose=False,subplot=None):
        params,pcov,binmid = fitobj.fit(self,verbose=verbose,xrange=xrange)
        if subplot != None:
            fval = fitobj.fxn(binmid, *params)
            subplot.plot(binmid, fval, 'r-',label="Fit")
            ymax = 0.8
            subplot.text(self.edges[-1], ymax, f"{fitobj.name()}",transform=subplot.get_xaxis_transform(),ha="right" )
            for ipar in range(len(params)):
                yval = ymax - 0.05*(1+ipar)
                subplot.text(self.edges[-1], yval, f"{fitobj.pname[ipar]} = {params[ipar]:.3f}",transform=subplot.get_xaxis_transform(),ha="right" )
        return params, pcov
