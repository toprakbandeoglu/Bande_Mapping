"""
TSPLIB instance loader.

Parses standard TSPLIB-format .tsp files into numpy coordinate arrays.
Only EUC_2D type is supported (the most common).
"""

import numpy as np


def load_tsplib(filepath):
    """
    Load a TSPLIB .tsp file with EUC_2D coordinates.

    Parameters
    ----------
    filepath : str
        Path to a .tsp file in standard TSPLIB format.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Coordinate array.

    Raises
    ------
    ValueError
        If the file is not EUC_2D type or cannot be parsed.

    Examples
    --------
    >>> points = load_tsplib("berlin52.tsp")
    >>> points.shape
    (52, 2)
    """
    coords = []
    reading_coords = False
    edge_weight_type = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[-1].strip()
            elif line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            elif line in ("EOF", "TOUR_SECTION", "DISPLAY_DATA_SECTION"):
                break
            elif reading_coords and line:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append([float(parts[1]), float(parts[2])])

    if edge_weight_type not in ("EUC_2D", "ATT", "GEO", None):
        raise ValueError(
            f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}. "
            "Only EUC_2D is fully supported."
        )

    if not coords:
        raise ValueError(f"No coordinates found in {filepath}")

    return np.array(coords, dtype=float)


# Embedded benchmark instances for quick examples
EMBEDDED_INSTANCES = {
    "berlin52": (np.array([
        [565,575],[25,185],[345,750],[945,685],[845,655],[880,660],[25,230],
        [525,1000],[580,1175],[650,1130],[1605,620],[1220,580],[1465,200],
        [1530,5],[845,680],[725,370],[145,665],[415,635],[510,875],[560,365],
        [300,465],[520,585],[480,415],[835,625],[975,580],[1215,245],
        [1320,315],[1250,400],[660,180],[410,250],[420,555],[575,665],
        [1150,1160],[700,580],[685,595],[685,610],[770,610],[795,645],
        [720,635],[760,650],[475,960],[95,260],[875,920],[700,500],[555,815],
        [830,485],[1170,65],[830,610],[605,625],[595,360],[1340,725],
        [1740,245]
    ], dtype=float), 7542),

    "eil51": (np.array([
        [37,52],[49,49],[52,64],[20,26],[40,30],[21,47],[17,63],[31,62],
        [52,33],[51,21],[42,41],[31,32],[5,25],[12,42],[36,16],[52,41],
        [27,23],[17,33],[13,13],[57,58],[62,42],[42,57],[16,57],[8,52],
        [7,38],[27,68],[30,48],[43,67],[58,48],[58,27],[37,69],[38,46],
        [46,10],[61,33],[62,63],[63,69],[32,22],[45,35],[59,15],[5,6],
        [10,17],[21,10],[5,64],[30,15],[39,10],[32,39],[25,32],[25,55],
        [48,28],[56,37],[30,40]
    ], dtype=float), 426),
}


def load_embedded(name):
    """
    Load one of the embedded benchmark instances.

    Available: 'berlin52', 'eil51'.

    Parameters
    ----------
    name : str
        Instance name.

    Returns
    -------
    points : np.ndarray, shape (N, 2)
    optimal : int
        Known optimal tour length.
    """
    if name not in EMBEDDED_INSTANCES:
        raise ValueError(
            f"Unknown instance '{name}'. "
            f"Available: {list(EMBEDDED_INSTANCES)}"
        )
    points, optimal = EMBEDDED_INSTANCES[name]
    return points.copy(), optimal
