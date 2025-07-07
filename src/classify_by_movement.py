from functions_features import *

def classification_2_classes(sperm):
    """
    Determine if a sperm is progressive or non-progressive.
    
    Args:
        sperm: features of sperm
    
    Returns:
        bool: True if progressive, False if non-progressive.
    """
    vcl = sperm['vcl']
    vsl = sperm['vsl']
    vap = sperm['vap']
    alh = sperm['alh']
    lin = sperm['lin']
    bcf = sperm['bcf']
    str = sperm['str']
    
    # Classify based on thresholds
    if vcl >= 25 and str > 0.4:
        return 0  # Progressive
    else:
        return 1  # Non-progressive
    
    
    
def classification_3_classes(sperm):
    """
    Determine if a sperm is Progressive motility, Non-pogressive, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Progressive motility, 1 -> Non-pogressive, 2 -> Inmotile
    """
    vcl = sperm['vcl']
    lin = sperm['lin']
    
    if vcl > 5 and lin >= 0.3:
        return 0  # Progressive motility
    elif vcl > 5 and lin < 0.3:
        return 1 # Non-pogressive
    else:
        return 2  # Inmotile
                
def classification_4_classes(sperm):
    """
    Determine if a sperm is Rapdly progressive, Slowly progressive, Non progressive, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Rapdly progressive, 1 -> Slowly progressive, 2 -> Non progressive, 3 -> Inmotile
    """
    
    vcl = sperm['vcl']
    lin = sperm['lin']
    
    # Classify based on thresholds
    if vcl > 25 and lin >= 0.6:
        return 0  # Rapdly progressive
    elif vcl >= 5 and lin >= 0.3:
        return 1 # Slowly progressive
    elif vcl > 5 and lin < 0.3:
        return 2 # Non-progressive
    else:
        return 3  # Inmotile


def classification_4_classes_v2(sperm):
    """
    Determine if a sperm is Rapdly progressive, Slowly progressive, Non progressive, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Rapdly progressive, 1 -> Slowly progressive, 2 -> Non progressive, 3 -> Inmotile
    """
    vcl = sperm['vcl']
    vsl = sperm['vsl']
    lin = sperm['lin']
    
    # Classify based on thresholds
    if vcl > 25 and vsl > 20:
        return 0 # Rapidly progressive
    elif 10 < vcl <= 25 and 5 < vsl <= 20:
        return 1 # Slowly progressive
    elif vcl <= 10 and vsl <= 5 and lin <= 0.3:
        return 2 # Non-progressive
    else:
        return 3 # Inmotile


def classification_4_classes_v3(sperm):
    """
    Determine if a sperm is Rapdly progressive, Slowly progressive, Non progressive, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Rapdly progressive, 1 -> Slowly progressive, 2 -> Non progressive, 3 -> Inmotile
    """
    vcl = sperm['vcl']
    vsl = sperm['vsl']
    vap = sperm['vap']
    alh = sperm['alh']
    bcf = sperm['bcf']
    
    # Classify based on thresholds
    if vap <= 146.9 and vap >= 31.5 and vsl <= 119.5 and vsl >= 29.5 and vcl <= 279.6 and vcl >= 59.1 and alh <= 16.7 and alh >= 4.5 and bcf <= 25.4 and bcf >= 5.2:
        return 0  # Rapdly progressive
    if vap <= 183.3 and vap >= 31.1 and vsl <= 140.4 and vsl >= 28.5 and vcl <= 406.3 and vcl >= 61.5 and alh <= 23.7 and alh >= 4.1 and bcf <= 21.1 and bcf >= 4.1:
        return 1 # Slowly progressive
    if vap <= 171.1 and vap >= 37.9 and vsl <= 73.3 and vsl >= 36.1 and vcl <= 373.6 and vcl >= 78.4 and alh <= 22.7 and alh >= 5.4 and bcf <= 29.1 and bcf >= 11.8:
        return 2 # Hyperactive
    if vap <= 56.2 and vap >= 16.5 and vsl <= 13.6 and vsl >= 6.8 and vcl <= 127.2 and vcl >= 35.4 and alh <= 9.9 and alh >= 4.0 and bcf <= 45.3 and bcf >= 12.6:
        return 3  # Inmotile
    return 3


def classification_4_c_by_clustering(sperm):
    """
    Determine if a sperm is Rapdly progressive, Slowly progressive, Non progressive, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Rapdly progressive, 1 -> Slowly progressive, 2 -> Non progressive, 3 -> Inmotile
    """
                
    vcl = sperm['vcl']
    vsl = sperm['vsl']
    vap = sperm['vap']
    alh = sperm['alh']
    bcf = sperm['bcf']
    
    # Classify based on thresholds
    if vap <= 146.9 and vap >= 31.5 and vsl <= 119.5 and vsl >= 29.5 and vcl <= 279.6 and vcl >= 59.1 and alh <= 16.7 and alh >= 4.5 and bcf <= 25.4 and bcf >= 5.2:
        return 0  # Rapdly progressive
    if vap <= 183.3 and vap >= 31.1 and vsl <= 140.4 and vsl >= 28.5 and vcl <= 406.3 and vcl >= 61.5 and alh <= 23.7 and alh >= 4.1 and bcf <= 21.1 and bcf >= 4.1:
        return 1 # Slowly progressive
    if vap <= 171.1 and vap >= 37.9 and vsl <= 73.3 and vsl >= 36.1 and vcl <= 373.6 and vcl >= 78.4 and alh <= 22.7 and alh >= 5.4 and bcf <= 29.1 and bcf >= 11.8:
        return 2 # Hyperactive
    if vap <= 56.2 and vap >= 16.5 and vsl <= 13.6 and vsl >= 6.8 and vcl <= 127.2 and vcl >= 35.4 and alh <= 9.9 and alh >= 4.0 and bcf <= 45.3 and bcf >= 12.6:
        return 3  # Inmotile
    return 3


def classification_4_classes_v4(sperm):
    """
    Determine if a sperm is Rapdly progressive, Slowly progressive, Non progressive, Inmotile.
    
    Args:
        sperm: features of sperm
    
    Returns:
        int: 0 -> Rapdly progressive, 1 -> Slowly progressive, 2 -> Non progressive, 3 -> Inmotile
    """
                
    vcl = sperm['vcl']
    vsl = sperm['vsl']
    
    # Classify based on thresholds
    if vsl >= 55 and vcl > 25:
        return 0
    elif vsl <= 10 and vcl < 15:
        return 3
    elif vsl <= 30 and vcl > 10:
        return 2
    elif vsl <= 90 and vcl > 5:
        return 1
    else:
        return -1