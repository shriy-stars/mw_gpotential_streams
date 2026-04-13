import galstreams
import numpy as np

# Set up galstreams and stream object
def get_rotation_matrix(keyword_arg, mws=None):   
    #function that takes in a string, keyword stream to identify 
    if mws is None:
        mws = galstreams.MWStreams(verbose=False)
    
    return mws[keyword_arg].stream_frame._R

def icrs_to_sf(ra_deg, dec_deg, rot_matrix):
    
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    
    R = rot_matrix.value
    
    icrs_vec = np.vstack(
        [
            np.cos(ra_rad) * np.cos(dec_rad),
            np.sin(ra_rad) * np.cos(dec_rad),
            np.sin(dec_rad),
        ]
    ).T

    stream_frame_vec = np.einsum("ij,kj->ki", R, icrs_vec)

    phi1 = np.arctan2(stream_frame_vec[:, 1], stream_frame_vec[:, 0]) * 180 / np.pi
    phi2 = np.arcsin(stream_frame_vec[:, 2]) * 180 / np.pi

    return phi1, phi2

def sf_to_icrs(phi1, phi2, rot_matrix):
    R = rot_matrix.value

    # Convert phi1, phi2 to radians
    phi1_rad = phi1 * np.pi / 180
    phi2_rad = phi2 * np.pi / 180

    # Stream frame vector
    stream_frame_vec = np.vstack(
        [
            np.cos(phi2_rad) * np.cos(phi1_rad),
            np.cos(phi2_rad) * np.sin(phi1_rad),
            np.sin(phi2_rad),
        ]
    ).T

    # Transform back to ICRS frame using the inverse of R
    icrs_vec = np.einsum("ij,kj->ki", R.T, stream_frame_vec)

    # Compute ra and dec in radians
    ra_rad = np.arctan2(icrs_vec[:, 1], icrs_vec[:, 0])
    dec_rad = np.arcsin(icrs_vec[:, 2])

    ra_deg = np.degrees(ra_rad)
    dec_deg = np.degrees(dec_rad)

    return ra_deg, dec_deg

