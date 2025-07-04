import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import CubicSpline


microns_pixel = 100/285

############### Basic measures ###############

def calculate_time_elapsed(trajectory, fps):
    """
    Calculate the total time elapsed for the trajectory.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: Time elapsed in seconds.
    """
    return (len(trajectory) - 1) / fps


def calculate_displacement(trajectory):
    """
    Calculate the displacement of the sperm.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Displacement in pixels.
    """
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    return np.round(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2),2)


def calculate_total_distance(trajectory):
    """
    Calculate the total distance traveled by the sperm.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Total distance in pixels.
    """
    deltas = np.diff(trajectory, axis=0) # Calculate the difference between each consecutive pair of points
    distances = np.linalg.norm(deltas, axis=1) # Calculate the euclidian distance
    total_distance = np.sum(distances)
    
    return np.round(total_distance,2) 


############### Standard measures ###############

def calculate_VCL(trajectory, fps):
    """
    Calculate the average velocity of the sperm (Curvilinear Velocity).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: Average velocity in pixels/second.
    """
    total_distance = calculate_total_distance(trajectory)
    time_elapsed = calculate_time_elapsed(trajectory,fps)
    return  np.round(np.divide(total_distance, time_elapsed, where=(time_elapsed != 0))*microns_pixel,2)

def calculate_VSL(trajectory, fps):
    """
    VSL is the straight-line distance between the first and last points divided by the total time (Straight Line Velocity).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: Straight line velocity in pixels/second.
    """
    displacement = calculate_displacement(trajectory)
    time_elapsed = calculate_time_elapsed(trajectory,fps)
    return  np.round(np.divide(displacement, time_elapsed, where=(time_elapsed != 0))*microns_pixel,2)


def calculate_VAP(trajectory,fps):
    """
    VAP is the average velocity along the average path (Average Path Velocity).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: Average path velocity in pixels/second.
    """
    # Check if there is more than one point in the trajectory
    if len(trajectory) < 2: 
        return 0
    
    x = [x[0] for x in trajectory]
    y = [x[1] for x in trajectory]
    
    time = np.arange(len(trajectory)) / fps
    
    # Create cubic splines for both x and y coordinates
    cs_x = CubicSpline(time, x)
    cs_y = CubicSpline(time, y)
    
    # Derivative of the spline to calculate velocity
    velocity_x = cs_x.derivative()
    velocity_y = cs_y.derivative()
    
    # Calculate velocity at each time point
    velocities = np.sqrt(velocity_x(time)**2 + velocity_y(time)**2)
    
    # Calculate total distance (integrating velocity over time)
    total_distance = np.sum(velocities / fps)  # Distance = velocity * time (small time intervals)
    
    '''x_smooth = uniform_filter1d(x, size=5)
    y_smooth = uniform_filter1d(y, size=5)
    
    distances = np.sqrt(np.diff(x_smooth)**2 + np.diff(y_smooth)**2)
    total_distance = np.sum(distances)'''

    time_elapsed = calculate_time_elapsed(trajectory,fps)

    return  np.round(np.divide(total_distance, time_elapsed, where=(time_elapsed != 0))*microns_pixel,2)


def calculate_ALH(trajectory):
    """
    Calculate the amplitude of lateral head displacement (ALH).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: ALH value.
    """
    # Calculate the average path (straight line from start to end)
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]

    lateral_displacements = []
    for (x, y) in trajectory:
        # Distance perpendicular to straight line
        numerator = np.abs((end_y - start_y) * x - (end_x - start_x) * y + end_x * start_y - end_y * start_x)
        denominator = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2)
        distance = numerator / denominator if denominator != 0 else 0
        lateral_displacements.append(distance)

    return np.round(((np.max(lateral_displacements) - np.min(lateral_displacements)) / 2), 2)

def calculate_MAD(trajectory):
    """
    Calculate the average angular displacement of the sperm trajectory (MAD).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Average angular displacement in radians.
    """
    if len(trajectory) < 2:
        return 0.0

    # Extract coordinates
    x = [p[0] for p in trajectory]
    y = [p[1] for p in trajectory]

    # Calculates angles between successive steps
    angles = np.arctan2(np.diff(y), np.diff(x))

    # Angle of the straight line between the first and last point (ideal)
    ideal_angle = np.arctan2(y[-1] - y[0], x[-1] - x[0])

    # Diferencia angular corregida al rango [-π, π]
    angle_diff = np.angle(np.exp(1j * (angles - ideal_angle)))

    # Calculate the MAD
    mad = np.mean(np.abs(angle_diff))
    return np.round(mad, 2)


############### Commonly measures ###############

def calculate_linearity(trajectory,fps):
    """
    Calculate the linearity of the sperm trajectory using linear regression.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: Linearity (R^2 value).
    """
    VSL = calculate_VSL(trajectory,fps)
    VCL = calculate_VCL(trajectory, fps)
    return np.round(np.divide(VSL, VCL, where=(VCL != 0)),2)


def calculate_WOB(trajectory,fps):
    """
    WOB is a measure of how much the sperm deviates from its path. It's essentially the ratio of VAP to VCL (Wobble).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: Linearity (R^2 value).
    """
    VAP = calculate_VAP(trajectory,fps)
    VCL = calculate_VCL(trajectory, fps)
    return np.round(np.divide(VAP, VCL, where=(VCL != 0)),2)


def calculate_STR(trajectory,fps):
    """
    STR is the ratio of the straight-line distance to the total path distance (Straightness Ratio).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Straightness ratio (0 to 1).
    """
    vsl = calculate_VSL(trajectory,fps)
    vap = calculate_VAP(trajectory,fps)
    return np.round(np.divide(vsl, vap, where=(vap != 0)),2)


def calculate_BCF(trajectory, fps):
    """
    Calculate the beat cross frequency (BCF).
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: BCF value.
    """
    # Calculate the average path (straight line from start to end)
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    crossings = 0
    for i in range(1, len(trajectory)):
        x1, y1 = trajectory[i-1]
        x2, y2 = trajectory[i]
        # Check if the trajectory crosses the average path
        numerator1 = (end_y - start_y) * x1 - (end_x - start_x) * y1 + end_x * start_y - end_y * start_x
        numerator2 = (end_y - start_y) * x2 - (end_x - start_x) * y2 + end_x * start_y - end_y * start_x
        if numerator1 * numerator2 < 0:
            crossings += 1
    time_elapsed = calculate_time_elapsed(trajectory,fps)
    return np.round(np.divide(crossings, time_elapsed, where=(time_elapsed != 0)),2)


def calculate_curvature(trajectory):
    """
    Calculate the average curvature of the sperm trajectory.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Average curvature.
    """
    curvatures = []
    for i in range(1, len(trajectory) - 1):
        dx1 = trajectory[i][0] - trajectory[i-1][0]
        dy1 = trajectory[i][1] - trajectory[i-1][1]
        dx2 = trajectory[i+1][0] - trajectory[i][0]
        dy2 = trajectory[i+1][1] - trajectory[i][1]
        dtheta = np.arctan2(dy2, dx2) - np.arctan2(dy1, dx1)
        ds = np.sqrt((trajectory[i+1][0] - trajectory[i][0])**2 + (trajectory[i+1][1] - trajectory[i][1])**2)
        curvature = np.abs(np.divide(dtheta, ds, where=(ds != 0)))
        curvatures.append(curvature)
    return np.round(np.mean(curvatures),2)
