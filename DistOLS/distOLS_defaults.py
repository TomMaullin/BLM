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
    Y_files = glob.glob("/well/nichols/users/kfh142/data/IMAGEN/spmstatsintra/*/SessionB/EPI_short_MID/swea/con_0010.nii")

    # Design matrix and number of parameters.
    X = ''

    # Set this to true if you want a spatially varying analysis and 
    # false otherwise.
    SVFlag = False;

    return(MAXMEM, Y_files, X, SVFlag)


if __name__ == "__main__":
    main()
