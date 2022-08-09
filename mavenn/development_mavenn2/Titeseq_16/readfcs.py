import fcsextract
import pandas

def readfcs(filename):
    [meta, data] =fcsextract.fcsextract(filename)
    num_channels = 0
    maxed = False
    make_key = lambda x:'$P'+str(x+1)+'N'
    while not maxed:
        if make_key(num_channels) in meta:
            num_channels += 1
            maxed = False
        else:
            maxed = True
    out = dict()
    for ii in range(num_channels):
        out[meta[make_key(ii)]] = [flow_tuple[ii] for flow_tuple in data]
    out = pandas.DataFrame(data=out)
    return out

