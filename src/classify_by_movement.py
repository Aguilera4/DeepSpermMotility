import numpy as np
from calculate_features import *

def is_progressive(trajectory, fps, velocity_threshold=10, straightness_threshold=0.8):
    """
    Determine if a sperm is progressive or non-progressive.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
        velocity_threshold (float): Minimum velocity to be considered progressive (pixels/second).
        straightness_threshold (float): Minimum straightness ratio (0 to 1) to be considered progressive.
    
    Returns:
        bool: True if progressive, False if non-progressive.
    """
    # Calculate average velocity
    velocity = calculate_VCL(trajectory, fps)
    
    # Calculate straightness
    straightness_ratio = calculate_STR(trajectory,fps)
    
    # Classify based on thresholds
    if velocity >= velocity_threshold and straightness_ratio >= straightness_threshold:
        return 1  # Progressive
    else:
        return 0  # Non-progressive
    
    
    
def classification_3_classes(trajectory, fps, straightness_threshold=0.8):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
        velocity_threshold (float): Minimum velocity to be considered progressive (pixels/second).
        straightness_threshold (float): Minimum straightness ratio (0 to 1) to be considered progressive.
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    # Calculate average velocity
    velocity = calculate_VCL(trajectory, fps)
    
    # Calculate straightness
    straightness_ratio = calculate_STR(trajectory,fps)
    
    # Classify based on thresholds
    if velocity >= 25 and straightness_ratio >= straightness_threshold:
        return 0  # Linear mean swim
    elif velocity >= 5 and velocity < 25 and straightness_ratio >= straightness_threshold:
        return 1 # Circular swim
    else:
        return 2  # Inmotile - Hiperactive
       

def classification_4_classes(trajectory, fps, straightness_threshold=0.8):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
        velocity_threshold (float): Minimum velocity to be considered progressive (pixels/second).
        straightness_threshold (float): Minimum straightness ratio (0 to 1) to be considered progressive.
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    # Calculate features
    vcl = calculate_VCL(trajectory, fps)
    str = calculate_STR(trajectory,fps)
    
    # Classify based on thresholds
    if vcl >= 25 and str >= straightness_threshold:
        return 0  # Linear mean swim
    elif vcl >= 5 and vcl < 25 and str >= straightness_threshold:
        return 1 # Circular swim
    elif vcl < 5 and str >= straightness_threshold:
        return 2 # Hyperactive
    else:
        return 3  # Inmotile


def classification_4_classes_v2(trajectory, fps):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
        velocity_threshold (float): Minimum velocity to be considered progressive (pixels/second).
        straightness_threshold (float): Minimum straightness ratio (0 to 1) to be considered progressive.
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    # Calculate average velocity
    vcl = calculate_VCL(trajectory, fps)
    vsl = calculate_VSL(trajectory, fps)
    vap = calculate_VAP(trajectory,fps)
    wob = calculate_WOB(trajectory,fps)
    str = calculate_STR(trajectory,fps)
    lin = calculate_linearity(trajectory,fps)
    alh = calculate_ALH(trajectory)
    curvature = calculate_curvature(trajectory)
    '''    # Classify based on thresholds
    if vcl >= 25 and str >= 0.2 and vap >= 20:
        return 0  # Linear mean swim
    elif vcl >= 5 and vcl <= 25 and str >= 0.1 and curvature > 0.3:
        return 1 # Circular swim
    elif vap >= 10 and vsl < 10 and alh > 5:
        return 2 # Hyperactive
    else:
        return 3  # Inmotile
    '''
    # Classify based on thresholds
    if vsl >= 20 and vcl >= 25 and str >= 0.2:
        return 0
    elif 5 <= vcl < 25 and 5 <= vsl < 40:
        return 1
    elif vsl < 5 and vcl > 0:
        return 2
    else:
        return 3


def classification_4_classes_v3(trajectory, fps, straightness_threshold=0.8):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
        velocity_threshold (float): Minimum velocity to be considered progressive (pixels/second).
        straightness_threshold (float): Minimum straightness ratio (0 to 1) to be considered progressive.
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    # Calculate average velocity
    vcl = calculate_VCL(trajectory, fps)
    vsl = calculate_VSL(trajectory, fps)
    lin = calculate_linearity(trajectory,fps)
    alh = calculate_ALH(trajectory)
    curvature = calculate_curvature(trajectory)
    vap = calculate_VAP(trajectory,fps)
    bcf = calculate_BCF(trajectory,fps)
    
    # Classify based on thresholds
    if vap <= 146.9 and vap >= 31.5 and vsl <= 119.5 and vsl >= 29.5 and vcl <= 279.6 and vcl >= 59.1 and alh <= 16.7 and alh >= 4.5 and bcf <= 25.4 and bcf >= 5.2:
        return 0  # Linear mean swim
    if vap <= 183.3 and vap >= 31.1 and vsl <= 140.4 and vsl >= 28.5 and vcl <= 406.3 and vcl >= 61.5 and alh <= 23.7 and alh >= 4.1 and bcf <= 21.1 and bcf >= 4.1:
        return 1 # Circular swim
    if vap <= 171.1 and vap >= 37.9 and vsl <= 73.3 and vsl >= 36.1 and vcl <= 373.6 and vcl >= 78.4 and alh <= 22.7 and alh >= 5.4 and bcf <= 29.1 and bcf >= 11.8:
        return 2 # Hyperactive
    if vap <= 56.2 and vap >= 16.5 and vsl <= 13.6 and vsl >= 6.8 and vcl <= 127.2 and vcl >= 35.4 and alh <= 9.9 and alh >= 4.0 and bcf <= 45.3 and bcf >= 12.6:
        return 3  # Inmotile
    return 3


def classification_4_classes_v4(trajectory, fps, straightness_threshold=0.8):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
        velocity_threshold (float): Minimum velocity to be considered progressive (pixels/second).
        straightness_threshold (float): Minimum straightness ratio (0 to 1) to be considered progressive.
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    # Calculate features
    vcl = calculate_VCL(trajectory, fps)
    vsl = calculate_VSL(trajectory, fps)
    vap = calculate_VAP(trajectory,fps)
    wob = calculate_WOB(trajectory,fps)
    str = calculate_STR(trajectory,fps)
    lin = calculate_linearity(trajectory,fps)
    alh = calculate_ALH(trajectory)
    curvature = calculate_curvature(trajectory)
    
    # Classify based on thresholds
    if vcl >= 25 and lin >= 0.5 and curvature < 0.3:
        return 0  # Linear mean swim
    elif vcl >= 5 and vcl < 25 and lin < 0.5 and curvature >= 0.3:
        return 1 # Circular swim
    elif vcl < 5 and vcl > 0:
        return 2 # Hyperactive
    else:
        return 3  # Inmotile