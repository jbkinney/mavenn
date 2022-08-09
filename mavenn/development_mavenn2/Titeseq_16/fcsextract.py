import sys
#from StringIO import StringIO
from io import BytesIO
import struct
import os

def fcsextract(filename):
    """
    Attempts to parse an FCS (flow cytometry standard) file

    Parameters: filename
        filename: path to the FCS file

    Returns: (vars,events)
    	vars: a dictionary with the KEY/VALUE pairs found in the HEADER
    	this includes the standard '$ABC' style FCS variable as well as any 
    	custom variables added to the header by the machine or operator
	
    	events: an [N x D] matrix of the data (as a Python list of lists)
    	i.e. events[99][2] would be the value at the 3rd dimension
    	of the 100th event
    """
    fcs_file_name = filename

    fcs = open(fcs_file_name,'rb')
    header = fcs.read(58)
    version = header[0:6].strip()
    text_start = int(header[10:18].strip())
    text_end = int(header[18:26].strip())
    data_start = int(header[26:34].strip())
    data_end = int(header[34:42].strip())
    analysis_start = int(header[42:50].strip())
    analysis_end = int(header[50:58].strip())

    #print "Parsing TEXT segment"
    # read TEXT portion
    fcs.seek(text_start)
    delimeter = fcs.read(1)
    # First byte of the text portion defines the delimeter
    #print "delimeter:",delimeter
    text = fcs.read(text_end-text_start+1)

    #Variables in TEXT poriton are stored "key/value/key/value/key/value"
    keyvalarray = text.split(delimeter)
    fcs_vars = {}
    fcs_var_list = []
    # Iterate over every 2 consecutive elements of the array
    for k,v in zip(keyvalarray[::2],keyvalarray[1::2]):
        fcs_vars[k] = v
        fcs_var_list.append((k,v)) # Keep a list around so we can print them in order

    #from pprint import pprint; pprint(fcs_var_list)
    if data_start == 0 and data_end == 0:
        data_start = int(fcs_vars[b'$DATASTART'])
        data_end = int(fcs_vars[b'$DATAEND'])

    num_dims = int(fcs_vars[b'$PAR'])
    #print "Number of dimensions:",num_dims

    num_events = int(fcs_vars[b'$TOT'])
    #print "Number of events:",num_events

    # Read DATA portion
    fcs.seek(data_start)
    #print "# of Data bytes",data_end-data_start+1
    data = fcs.read(data_end-data_start+1)

    # Determine data format
    datatype = fcs_vars[b'$DATATYPE']
    if datatype == b'F':
        datatype = 'f' # set proper data mode for struct module
        #print "Data stored as single-precision (32-bit) floating point numbers"
    elif datatype == 'D':
        datatype = 'd' # set proper data mode for struct module
        #print "Data stored as double-precision (64-bit) floating point numbers"
    else:
        assert False,"Error: Unrecognized $DATATYPE '%s'" % datatype
    
    # Determine endianess
    endian = fcs_vars[b'$BYTEORD']
    endian = ">"

    # Put data in StringIO so we can read bytes like a file    
    data = BytesIO(data)

    #print "Parsing DATA segment"
    # Create format string based on endianeness and the specified data type
    format = endian + str(num_dims) + datatype
    datasize = struct.calcsize(format)
    #print "Data format:",format
    #print "Data size:",datasize
    events = []
    # Read and unpack all the events from the data
    for e in range(num_events):
        event = struct.unpack(format,data.read(datasize))
        events.append(event)
    
    fcs.close()
    return fcs_vars,events
    
def writefcs(fcs_vars,events,fcs_file_name,delimiter=","):
    """
    Outputs FCS variables and data to files
    
    fcs_vars: the dictionary of key/value pairs from HEADER
    events: [N x D] matrix (list of lists) of event data in row-major form
    fcs_file_name: prefix for the output files
    delimiter: specifies separator between values in ASCII file output
        Generates a binary file if None
        
    Creates 3 files
    a) HEADER: fcs_file_name.txt
        the HEADER key/value pairs
    b) DATA: fcs_file_name.csv (or .bin for binary file)
        the raw data, one event per line
    c) INFO: fcs_file_name.info
        list of the dimension names and long-names ($PkN and $PkS)
    """
    num_dims = len(events[0])
    num_events = len(events)

    if delimiter is None:
        # Creates a binary file
        # First 4 bytes are an integer with the number of events
        # Next 4 bytes are an integer with the number of dimensions
        # Rest of the file is consecutive 32-bit floating point numbers
        # Data is stored such that consecutive floats are from the same event 
        # (i.e. an N x D matrix in row-major format)
        bin_file_name = fcs_file_name[:-4]+".bin"
        bin_file = open(bin_file_name,"wb")
        print("Writing DATA output file:",bin_file_name)
        bin_file.write(struct.pack("i",num_events))
        bin_file.write(struct.pack("i",num_dims))
        format = "%df" % num_dims
        for row in events:
            data = [float(x) for x in row]
            bin_file.write(struct.pack(format,*data))
    else:
        csv_file_name = fcs_file_name[:-4]+".csv"
        csv_file = open(csv_file_name,'w')
        print("Writing DATA output file:",csv_file_name)
        format = delimiter.join(["%0.2f"]*num_dims)
        for event in events:
            csv_file.write(format % event + "\n")
        csv_file.close()

    txt_file_name = fcs_file_name[:-4]+".txt"
    txt_file = open(txt_file_name,'w')
    print("Writing TEXT output file:",txt_file_name)
    for k,v in fcs_vars.items():
        txt_file.write("%s,%s\n" % (k,v))
    txt_file.close()

    info_file_name = fcs_file_name[:-4]+".info"
    print("Writing INFO output file:",info_file_name)
    info_file = open(info_file_name,'w')
    for i in range(num_dims):
        info_file.write("%s\t%s\n" % (fcs_vars["$P%dN"%(i+1)],fcs_vars.get("$P%dS"%(i+1),fcs_vars["$P%dN"%(i+1)])))     

if __name__ == '__main__':
    if len(sys.argv) == 2:
        fcs_file_name = sys.argv[1]
        delimiter = ","
    elif len(sys.argv) == 3:
        fcs_file_name = sys.argv[1]
        delimiter = {"0":",","1":" ","2":"\t","3":None}.get(sys.argv[2])
    else:
        print("Usage: python %s path [delimiter]\n" % sys.argv[0])
        print("    path: path to the fcs file or a directory")
        print("    if path is a directory, recursivel extracts all .fcs files\n")
        print("    delimiter: 0 = comma separated (default)")
        print("    delimiter: 1 = space separated")
        print("    delimiter: 2 = tab separated")
        print("    delimiter: 3 = generate binary file")
        sys.exit(1)
 
    files = [fcs_file_name]
    while(files):
        path = files.pop()
        if os.path.isdir(path):
            print("Directory:",path)
            new_files = [os.path.join(path,p) for p in os.listdir(path) if p.lower().endswith("fcs") or os.path.isdir(os.path.join(path,p))]
            files += new_files
            print("Adding files:",new_files)
        # It it's a file, process the file
        elif os.path.isfile(path):
            fcs_file_name = path
            print("\n\nProcessing:",fcs_file_name)
            fcs_vars,events = fcsextract(fcs_file_name)
            writefcs(fcs_vars,events,fcs_file_name,delimiter)
