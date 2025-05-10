import numpy as np
from functions_features import *

def classification_2_classes(sperm):
    """
    Determine if a sperm is progressive or non-progressive.
    
    Args:
        sperm: features of sperm
    
    Returns:
        bool: True if progressive, False if non-progressive.
    """
    
    # Classify based on thresholds
    if sperm['vcl'] >= 25 and sperm['str'] >= 0.8:
        return 1  # Progressive
    else:
        return 0  # Non-progressive
    
    
    
def classification_3_classes(sperm):
    """
    Determine if a sperm is Progressive motility, Non-pogressive, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Progressive motility, 1 -> Non-pogressive, 2 -> Inmotile
    """
    
    '''# Classify based on thresholds
    if sperm['vcl'] >= 25:
        return 0  # Progressive motility
    elif sperm['vcl'] >= 5 and sperm['vcl'] < 25:
        return 1 # Non-pogressive
    else:
        return 2  # Inmotile'''
    
    '''# Classify based on thresholds
    if sperm['vcl'] >= 25 and (sperm['lin'] >= 0.5 or sperm['alh'] >= 80 or sperm['bcf'] >= 3 or (sperm['vsl'] > 20 and sperm['lin'] >= 0.5 )):
        return 0  # Progressive motility
    elif (sperm['vcl'] <= 15 and sperm['vsl'] <= 5 and sperm['lin'] <= 0.5) or (sperm['vcl'] >= 5 and sperm['vcl'] < 25 and sperm['lin'] < 0.5):
        return 2
    elif (sperm['vcl'] >= 5 and sperm['vcl'] < 25) or sperm['lin'] < 0.5:
        return 1 # Non-pogressive
    else:
        return 2  # Inmotile'''
    
    '''# Classify based on thresholds
    if sperm['vap'] <= 32.98: 
        if sperm['vcl'] <= 46.32: 
            return 2
        else:
            if sperm['mad'] <= 1.59: 
                return 0
            else:
                return 2
    else:
        if sperm['vap'] <= 87.40: 
            if sperm['vsl'] <= 82.38: 
                return 1
            else:
                return 0
        else:
            if sperm['vcl'] <= 107.49:
                return 1 
            else:
                return 0
                '''
    # Classify based on thresholds
    if sperm['vsl'] <= 5.12:
        return 0
    else:
        if sperm['wob'] <= 0.89: 
            return 1
        else:
            return 2
                
def classification_4_classes(sperm):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    
    # Classify based on thresholds
    if sperm['vcl'] >= 25 and sperm['str'] >= 0.8:
        return 0  # Linear mean swim
    elif sperm['vcl'] >= 5 and sperm['vcl'] < 25 and sperm['str'] >= 0.8:
        return 1 # Circular swim
    elif sperm['vcl'] < 5 and sperm['str'] >= 0.8:
        return 2 # Hyperactive
    else:
        return 3  # Inmotile


def classification_4_classes_v2(sperm):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    
    '''    # Classify based on thresholds
    if vcl >= 25 and str >= 0.2 and vap >= 20:
        return 0  # Linear mean swim
    elif vcl >= 5 and vcl <= 25 and str >= 0.1 and curvature > 0.3:
        return 1 # Circular swim
    elif vap >= 10 and vsl < 10 and alh > 5:
        return 2 # Hyperactive
    else:
        return 3  # Inmotile
        
         # Classify based on thresholds
    if vsl >= 20 and vcl >= 25 and str >= 0.6:
        return 0
    elif 5 <= vcl < 25 and 5 <= vsl < 20:
        return 1
    elif vsl < 5 and vcl > 0:
        return 2
    else:
        return 3
        
        # Classify based on thresholds
    if vsl < 1 and vcl < 1:
        return 3
    elif vsl >= 20 and vcl >= 25 and str >= 0.7:
        return 0
    elif 10 <= vcl < 25 and 5 <= vsl < 20:
        return 1
    elif vsl < 5 and vcl > 0:
        return 2
    else:
        return -1
    '''
    
        # Classify based on thresholds
    if sperm['vcl'] > 25 and sperm['vsl'] > 20:
        return 0 # Rapidly progressive
    elif 10 < sperm['vcl'] <= 25 and 5 < sperm['vsl'] <= 20:
        return 1 # Slowly progressive
    elif sperm['vcl'] <= 10 and sperm['vsl'] <= 5 and sperm['lin'] <= 0.3:
        return 2 # Non-progressive
    else:
        return 3 # Inmotile


def classification_4_classes_v3(sperm):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    
    # Classify based on thresholds
    if sperm['vap'] <= 146.9 and sperm['vap'] >= 31.5 and sperm['vsl'] <= 119.5 and sperm['vsl'] >= 29.5 and sperm['vcl'] <= 279.6 and sperm['vcl'] >= 59.1 and sperm['alh'] <= 16.7 and sperm['alh'] >= 4.5 and sperm['bcf'] <= 25.4 and sperm['bcf'] >= 5.2:
        return 0  # Linear mean swim
    if sperm['vap'] <= 183.3 and sperm['vap'] >= 31.1 and sperm['vsl'] <= 140.4 and sperm['vsl'] >= 28.5 and sperm['vcl'] <= 406.3 and sperm['vcl'] >= 61.5 and sperm['alh'] <= 23.7 and sperm['alh'] >= 4.1 and sperm['bcf'] <= 21.1 and sperm['bcf'] >= 4.1:
        return 1 # Circular swim
    if sperm['vap'] <= 171.1 and sperm['vap'] >= 37.9 and sperm['vsl'] <= 73.3 and sperm['vsl'] >= 36.1 and sperm['vcl'] <= 373.6 and sperm['vcl'] >= 78.4 and sperm['alh'] <= 22.7 and sperm['alh'] >= 5.4 and sperm['bcf'] <= 29.1 and sperm['bcf'] >= 11.8:
        return 2 # Hyperactive
    if sperm['vap'] <= 56.2 and sperm['vap'] >= 16.5 and sperm['vsl'] <= 13.6 and sperm['vsl'] >= 6.8 and sperm['vcl'] <= 127.2 and sperm['vcl'] >= 35.4 and sperm['alh'] <= 9.9 and sperm['alh'] >= 4.0 and sperm['bcf'] <= 45.3 and sperm['bcf'] >= 12.6:
        return 3  # Inmotile
    return 3



def classification_4_c_by_clustering(sperm):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    
    # Classify based on thresholds
    if sperm['vap'] <= 146.9 and sperm['vap'] >= 31.5 and sperm['vsl'] <= 119.5 and sperm['vsl'] >= 29.5 and sperm['vcl'] <= 279.6 and sperm['vcl'] >= 59.1 and sperm['alh'] <= 16.7 and sperm['alh'] >= 4.5 and sperm['bcf'] <= 25.4 and sperm['bcf'] >= 5.2:
        return 0  # Linear mean swim
    if sperm['vap'] <= 183.3 and sperm['vap'] >= 31.1 and sperm['vsl'] <= 140.4 and sperm['vsl'] >= 28.5 and sperm['vcl'] <= 406.3 and sperm['vcl'] >= 61.5 and sperm['alh'] <= 23.7 and sperm['alh'] >= 4.1 and sperm['bcf'] <= 21.1 and sperm['bcf'] >= 4.1:
        return 1 # Circular swim
    if sperm['vap'] <= 171.1 and sperm['vap'] >= 37.9 and sperm['vsl'] <= 73.3 and sperm['vsl'] >= 36.1 and sperm['vcl'] <= 373.6 and sperm['vcl'] >= 78.4 and sperm['alh'] <= 22.7 and sperm['alh'] >= 5.4 and sperm['bcf'] <= 29.1 and sperm['bcf'] >= 11.8:
        return 2 # Hyperactive
    if sperm['vap'] <= 56.2 and sperm['vap'] >= 16.5 and sperm['vsl'] <= 13.6 and sperm['vsl'] >= 6.8 and sperm['vcl'] <= 127.2 and sperm['vcl'] >= 35.4 and sperm['alh'] <= 9.9 and sperm['alh'] >= 4.0 and sperm['bcf'] <= 45.3 and sperm['bcf'] >= 12.6:
        return 3  # Inmotile
    return 3


def classification_4_classes_v4(sperm):
    """
    Determine if a sperm is Linear mean swim, Circular swim, Hyperactivated, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Linear mean swim, 1 -> Circular swim, 2 -> Hyperactivated, 3 -> Inmotile
    """
    
    # Classify based on thresholds
    if sperm['vcl'] >= 25 and sperm['lin'] >= 0.5:
        return 0  # Linear mean swim
    elif sperm['vcl'] >= 5 and sperm['vcl'] < 25 and sperm['lin'] < 0.5:
        return 1 # Circular swim
    elif sperm['vcl'] < 5 and sperm['vcl'] > 0:
        return 2 # Hyperactive
    else:
        return 3  # Inmotile