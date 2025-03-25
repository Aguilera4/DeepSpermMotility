import numpy as np
from sklearn.linear_model import LinearRegression


def calculate_velocity(trajectory, fps):
    """
    Calculate the average velocity of the sperm.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
        fps (int): Frames per second of the video.
    
    Returns:
        float: Average velocity in pixels/second.
    """
    total_distance = sum(np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i][1] - trajectory[i-1][1])**2) for i in range(1, len(trajectory)))
    time_elapsed = (len(trajectory) - 1) / fps
    return  np.divide(total_distance, time_elapsed, where=(time_elapsed != 0))



def calculate_straightness(trajectory):
    """
    Calculate the straightness ratio of the sperm trajectory.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Straightness ratio (0 to 1).
    """
    start_x, start_y = trajectory[0]
    end_x, end_y = trajectory[-1]
    displacement = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    total_distance = sum(np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + (trajectory[i][1] - trajectory[i-1][1])**2) for i in range(1, len(trajectory)))
    return np.divide(displacement, total_distance, where=(total_distance != 0))



def calculate_angular_displacement(trajectory):
    """
    Calculate the average angular displacement of the sperm trajectory.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Average angular displacement in radians.
    """
    angles = []
    for i in range(1, len(trajectory) - 1):
        dx1 = trajectory[i][0] - trajectory[i-1][0]
        dy1 = trajectory[i][1] - trajectory[i-1][1]
        dx2 = trajectory[i+1][0] - trajectory[i][0]
        dy2 = trajectory[i+1][1] - trajectory[i][1]
        angle = np.arctan2(dy2, dx2) - np.arctan2(dy1, dx1)
        angles.append(angle)
    return np.mean(angles)



def calculate_linearity(trajectory):
    """
    Calculate the linearity of the sperm trajectory using linear regression.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Linearity (R^2 value).
    """
    X = np.array([point[0] for point in trajectory]).reshape(-1, 1)
    y = np.array([point[1] for point in trajectory])
    model = LinearRegression()
    model.fit(X, y)
    return model.score(X, y)



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
    return np.mean(curvatures)



def calculate_alh(trajectory):
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
        # Distance from the point to the straight line
        numerator = np.abs((end_y - start_y) * x - (end_x - start_x) * y + end_x * start_y - end_y * start_x)
        denominator = np.sqrt((end_y - start_y)**2 + (end_x - start_x)**2)
        lateral_displacements.append(np.divide(numerator, denominator, where=(denominator != 0)))
    return np.max(lateral_displacements)



def calculate_bcf(trajectory, fps):
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
    time_elapsed = (len(trajectory) - 1) / fps
    return np.divide(crossings, time_elapsed, where=(time_elapsed != 0))



def calculate_total_distance(trajectory):
    """
    Calculate the total distance traveled by the sperm.
    
    Args:
        trajectory (list of tuples): List of (x, y) positions over time.
    
    Returns:
        float: Total distance in pixels.
    """
    return sum(np.sqrt((trajectory[i][0] - trajectory[i-1][0])**2 + 
               (trajectory[i][1] - trajectory[i-1][1])**2) 
              for i in range(1, len(trajectory)))



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
    return np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)




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