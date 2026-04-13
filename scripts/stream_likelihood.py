from scipy import stats
import numpy as np
from scipy.interpolate import make_splrep, interp1d
from generate_sim_stream import create_stream_particle_spray, generate_stream_coords
from scipy.stats import binned_statistic
from astropy.coordinates import Galactocentric, ICRS
import astropy.units as u
import time as time

from astropy import units as u
import astropy.coordinates as coord
from astropy.coordinates import Galactocentric, ICRS, CartesianDifferential, CartesianRepresentation
from astropy import table
import pandas as pd

from coordinate_utils import get_rotation_matrix, icrs_to_sf, sf_to_icrs


    
def make_spline(x, y, binsize = 0.3):
    """
    Compute a 1D spline interpolation of binned data.

    This function sorts the input data by `x`, bins the data using a fixed `binsize`,
    computes the mean of `y` in each bin, and returns a spline function that
    interpolates these binned means.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    y : array-like
        Dependent variable values corresponding to `x`.
    binsize : float, optional
        Width of bins used to group the data (default is 0.1).

    Returns
    -------
    spline : function
        A 1D interpolating function (`scipy.interpolate.interp1d`) that maps
        x-values to the mean binned y-values. Returns NaN for values outside the domain.
    
    Notes
    -----
    - Uses `scipy.stats.binned_statistic` to bin and average `y` values.
    - The resulting spline uses linear interpolation and does not extrapolate beyond data range.
    """
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices] 
    
    #binned statistic function, arrange with bin size (0.3 in phi1), make a bunch of bins and statistics
    bins = np.arange(x_sorted.min(), x_sorted.max()+binsize, binsize)
    bin_centers = (bins[:-1] + bins[1:])/2
    m = len(bin_centers)
    
    y_median, bin_edges, binnumber = binned_statistic(x_sorted, y_sorted, statistic='mean', bins=bins)
    
    mask = np.isfinite(y_median)
    if mask.sum() < m / 2:
        return None
    
    spline = interp1d(bin_centers[mask],y_median[mask],bounds_error=False,fill_value= np.nan)
    return spline

def log_likelihood(
        prog_pars, data_dict,
        pot, prog_mass, 
        prog_scaleradius, 
        Age_Stream_in_Gyr,
        num_particles, rotation_matrix, 
        phi1_range=[-20,20],
        seed_num=69420,
):
    """
    Compute the log likelihood of the data given the model parameters.
    prog_pars: list of parameters for the stream progenitor
    phi1_obs: observed phi1 values
    phi2_obs: observed phi2 values
    rv_obs: observed radial velocities
    rv_obs_errors: errors on the radial velocities
    pmra_cosdec_obs: observed pmra values
    pmra_cosdec_obs_errors: errors on the pmra values
    pmdec_obs: observed pmdec values
    pmdec_obs_errors: errors on the pmdec values
    pot: potential object
    phi1_range: range of phi1 values to consider
    seed_num: seed number for the random number generator

    """
    ra, dec = sf_to_icrs(0, prog_pars[0], rotation_matrix)
    #print(ra)
    ra, dec = ra.item(), dec.item()
    
    dist, pmra, pmdec, rv = prog_pars[1:]
    
    stream_c = coord.SkyCoord(
        ra=ra*u.degree, dec=dec*u.degree, distance=dist*u.kpc, 
        pm_ra_cosdec=pmra*u.mas/u.yr,
        pm_dec=pmdec*u.mas/u.yr,
        radial_velocity=rv*u.km/u.s
    )
    
    rep = stream_c.transform_to(coord.Galactocentric) # units here are kpc, km/s
    
    prog_wtoday = np.array(
        [rep.x.value, rep.y.value, rep.z.value,
         rep.v_x.value, rep.v_y.value, rep.v_z.value]
    ) # units here are kpc, km/s
    
    # simulate a stream
    stream_unperturb = create_stream_particle_spray(pot_host=pot, 
    initmass=prog_mass, 
    scaleradius=prog_scaleradius, 
    prog_pot_kind='Plummer', 
    sat_cen_present=prog_wtoday, 
    num_particles=num_particles,
    time_end=0.0, 
    time_total=Age_stream_inGyr, save_rate=1,)
    #print("Stream created")

    xv_model = stream_unperturb['part_xv']
    
    # Create Astropy Galactocentric coordinates
    galcen_model = coord.Galactocentric(
        x=xv_model[:,0] * u.kpc,
        y=xv_model[:,1] * u.kpc,
        z=xv_model[:,2] * u.kpc,
        v_x=xv_model[:,3] * u.km/u.s,
        v_y=xv_model[:,4] * u.km/u.s,
        v_z=xv_model[:,5] * u.km/u.s,
        representation_type='cartesian',
        differential_type='cartesian'
    )
    
    # Transform to ICRS
    icrs_model = galcen_model.transform_to(coord.ICRS())
    
    # Extract observable quantities
    #print(icrs_model.ra)
    ra_model = icrs_model.ra.value # in degrees
    dec_model = icrs_model.dec.value # in degrees
    dist_model = icrs_model.distance.value  # in kpc
    rv_model = icrs_model.radial_velocity.value  # in km/s
    pmra_cosdec_model = icrs_model.pm_ra_cosdec.value  # in mas/yr
    pmdec_model = icrs_model.pm_dec.value  # in mas/yr

    phi1_model, phi2_model = icrs_to_sf(ra_model, dec_model, rotation_matrix)
    
    # select only points in the phi1 range
    phi1_model_sel = (phi1_model > phi1_range[0]) & (phi1_model < phi1_range[1]) 
    phi1_obs_sel = (phi1_obs > phi1_range[0]) & (phi1_obs < phi1_range[1]) 
    
    if phi1_model_sel.sum() == 0:
        import pdb; pdb.set_trace()
        
    if phi1_model.min() > phi1_range[0] or phi1_model.max() < phi1_range[1]:
        #print(phi1_model.min())
        #print(phi1_model.max())
        print('phi1_range issues')
        return -np.inf

    #### on-sky track
    # generate a track spline
    phi2_spline = make_spline(phi1_model[phi1_model_sel], phi2_model[phi1_model_sel])
    if phi2_spline is None:
        return -np.inf
    phi2_std = np.nanstd(phi2_model[phi1_model_sel] - phi2_spline(phi1_model[phi1_model_sel]))
    
    phi2_vals = phi2_spline(phi1_obs[phi1_obs_sel])
    lnlk_spatial = stats.norm.logpdf(phi2_obs[phi1_obs_sel], loc = phi2_vals, scale = phi2_std)
    #print("lnlk_spatial:", lnlk_spatial)
    
    ## dist track
    dist_spline = make_spline(phi1_model[phi1_model_sel],dist_model[phi1_model_sel])
    if dist_spline is None:
        print('distance spline issues')
        return -np.inf
    
    dist_mask = (phi1_obs_sel & np.isfinite(dist_obs) & np.isfinite(dist_obs_errors))
    
    if dist_mask.sum() > 0:
        dist_scale = dist_obs_errors[dist_mask]
        dist_vals = dist_spline(phi1_obs[dist_mask])
        lnlk_dist = stats.norm.logpdf(dist_obs[dist_mask], loc = dist_vals, scale = dist_scale)
        #print("lnlk_dist:", lnlk_dist)
        lnlk_dist = np.sum(lnlk_dist)

    else:
        print("distance likelihood reject")
        lnlk_dist = 0.0
       
    #### velocity track
    rv_spline = make_spline(phi1_model[phi1_model_sel], rv_model[phi1_model_sel])
    if rv_spline is None:
        return -np.inf
    rv_vals = rv_spline(phi1_obs[phi1_obs_sel])
    lnlk_velocity = stats.norm.logpdf(rv_obs[phi1_obs_sel], loc = rv_vals, scale = rv_obs_errors[phi1_obs_sel])
    #print("lnlk_rv:", lnlk_velocity)

    ### pmra track
    pmra_cosdec_spline = make_spline(phi1_model[phi1_model_sel], pmra_cosdec_model[phi1_model_sel])
    if pmra_cosdec_spline is None:
        return -np.inf
    pmra_cosdec_vals = pmra_cosdec_spline(phi1_obs[phi1_obs_sel])
    lnlk_pmra_cosdec = stats.norm.logpdf(pmra_cosdec_obs[phi1_obs_sel],loc = pmra_cosdec_vals,scale= pmra_cosdec_obs_errors[phi1_obs_sel])
    #print("lnlk_pmra_cosdec:", lnlk_pmra_cosdec)
    
    ### pmdec track
    pmdec_spline = make_spline(phi1_model[phi1_model_sel], pmdec_model[phi1_model_sel])
    if pmdec_spline is None:
        return -np.inf
    pmdec_vals = pmdec_spline(phi1_obs[phi1_obs_sel])
    lnlk_pmdec = stats.norm.logpdf(pmdec_obs[phi1_obs_sel], loc = pmdec_vals, scale = pmdec_obs_errors[phi1_obs_sel])
    #print("lnlk_pmdec:", lnlk_pmdec)
    
    lnlk_total = lnlk_spatial + lnlk_velocity + lnlk_pmra_cosdec + lnlk_pmdec + lnlk_dist
    
    return np.sum(lnlk_total)

def log_prior(prog_pars): #specify some reasonable bounds with a prior function; that way we don't have to brute force a bunch of stuff
    phi2, dist, pm_ra, pm_dec,rv = prog_pars
    if -2.5 < phi2 < 2.0 and 0.5 <dist < 40 and -3.0 < pm_ra < 3.0 and -3.50 < pm_dec < 0.0 and 250.0 < rv < 281.0:
        return 0.0
    return -np.inf

def log_probability(prog_pars, data_dict, pot, prog_mass, 
        prog_scaleradius, 
        Age_Stream_in_Gyr,
        num_particles, 
        rotation_matrix):
    
    lp = log_prior(prog_pars)
    if not np.isfinite(lp):
        print("Prior rejection:", prog_pars)
        return -np.inf

    ll = log_likelihood(prog_pars, **data_dict, pot=pot)

    if not np.isfinite(ll):
        print("Likelihood rejection")
        return -np.inf
    return lp + ll