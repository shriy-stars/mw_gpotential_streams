from astropy import astropy.coordinates

'''
This file provides some help
'''

def read_in_data(data_file, dist_col_label = None, distance_tracers = None):
    '''
    Function takes in:=
    
    data_file: pandas data frame object
    dis_col_label: string of a column header for distances of stream members if the data_file has those included
    distance_tracers: a separate pandas data frame object that contains an array of distances of stream cmembers

    Function returns:=

    df: data frame of the star candidates to construct a dictionary out of in the notebook
    distance_fit: 1 degree polynomial fit for distances
    prog_pars: an inital guess for the 6D coordinates of the stream progenitor's present-day position (in phi1, phi2 frame)
    prog_pars_icrs: 6D present-day coordinate of the stream progenitor in the ICRS frame
    
    '''
    if dist_col_label is None:
        
        return

    else:
        df = data_file
    
        select_phi1 = (df['phi1'] > -1.0) & (df['phi1'] < 5.0)
        
        select_distance = (df[dist_col_label].notna())
        dist_phi1 = df.loc[select_distance, 'phi1']
        dist = df.loc[select_distance, dist_col_label]
    
    
        coefficients = np.polyfit(dist_phi1, dist, deg=1)
    
        distance_fit = np.poly1d(coefficients)
    
        phi1med = np.median(df.loc[select_phi1,'phi1'])
        phi2med = np.median(df.loc[select_phi1,'phi2'])
        distmed = np.median(df.loc[select_phi1 & select_distance, dist_col_label])
        ramed = np.median(df.loc[select_phi1,'ra'])
        decmed = np.median(df.loc[select_phi1,'dec'])
        pmramed = np.median(df.loc[select_phi1, 'pmra'])
        pmdecmed = np.median(df.loc[select_phi1, 'pmdec'])
        rvmed = np.median(df.loc[select_phi1, 'vel_calib'])
    
        prog_pars = [float(phi1med), float(phi2med), float(distmed), float(pmramed), float(pmdecmed), float(rvmed)]
        prog_pars_icrs = [float(ramed), float(decmed), float(distmed), float(pmramed), float(pmdecmed), float(rvmed)]
    
        return df, distance_fit, prog_pars, prog_pars_icrs
    
    
    return data_file, distance_tracers
    