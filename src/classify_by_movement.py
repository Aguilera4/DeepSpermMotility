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
    
    # Calculate straightness (displacement / total path length)
    start_x, start_y = trajectory[0] # initial coordinates of the centroid
    end_x, end_y = trajectory[-1] # final centroid coordinates
    displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) # calculate displacement
    total_path_length = sum(np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i][1] - trajectory[i-1][1])**2) for i in range(1, len(trajectory)))
    straightness_ratio = displacement / total_path_length if total_path_length > 0 else 0
    
    # Classify based on thresholds
    if velocity >= velocity_threshold and straightness_ratio >= straightness_threshold:
        return 1  # Progressive
    else:
        return 0  # Non-progressive
       

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
    # Calculate average velocity
    velocity = calculate_VCL(trajectory, fps)
    
    # Calculate straightness (displacement / total path length)
    start_x, start_y = trajectory[0] # initial coordinates of the centroid
    end_x, end_y = trajectory[-1] # final centroid coordinates
    displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) # calculate displacement
    total_path_length = sum(np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i][1] - trajectory[i-1][1])**2) for i in range(1, len(trajectory)))
    straightness_ratio = displacement / total_path_length if total_path_length > 0 else 0
    
    # Classify based on thresholds
    if velocity >= 25 and straightness_ratio >= straightness_threshold:
        return 0  # Linear mean swim
    elif velocity >= 5 and velocity < 25 and straightness_ratio >= straightness_threshold:
        return 1 # Circular swim
    elif velocity < 5 and straightness_ratio >= straightness_threshold:
        return 2 # Hyperactive
    else:
        return 3  # Inmotile