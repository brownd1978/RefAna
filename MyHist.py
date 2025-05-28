import numpy as np
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

    def plotErrors(self,axis):
        plt =self.plot(axis)
        errors = self.binErrors()
        centers = self.binCenters()
        axis.errorbar(x=centers,y=self.data,yerr=errors)
        return plt

    def integral(self):
        return np.sum(self.data)

    def binCenters(self):
        midbin = np.zeros(len(self.data))
        for ibin in range(len(self.data)):
            midbin[ibin] = 0.5*(self.edges[ibin] + self.edges[ibin+1]) # edges have 1 more entry than data
        return midbin

    def binErrors(self):
        # assume unweighted bins
        errors = np.sqrt(self.data)
        ones = np.ones(len(self.data))
        errors = np.maximum(ones,errors)
        return errors
