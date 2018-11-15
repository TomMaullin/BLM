# This is the file containing all inputs for distOLS on a cluster.
# Please edit the variables in this file to specify your inputs.
# ================================================================
# Author: Tom Maullin (14/11/2018)

def main():

    # Maximum Memory for computing in bytes - current default of 2**29 
    # bytes matches that of SPM.
    MAXMEM = 2**29

    # Enter a list of Y files. (This is a temporary way of entering 
    # this data and will change once we know more about the data).
    Y_files = '/users/nichols/inf852/Yfiles.txt'

    # Design matrix and number of parameters.
    X = '/users/nichols/inf852/DistOLS-py/DistOLS/test/data/DesMat.csv'

    # Set this to true if you want a spatially varying analysis and 
    # false otherwise.
    SVFlag = False;

    # There should be an output directory here:
    outdir = ''

    return(MAXMEM, Y_files, X, SVFlag, outdir)


if __name__ == "__main__":
    main()
