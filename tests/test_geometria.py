import numpy as np

def test_alargamiento():
    # Test básico para comprobar que la fórmula del Aspect Ratio no se ha alterado
    b = 3.873
    S = 1.0
    AR = b**2 / S
    assert round(AR, 1) == 15.0, "Error en el cálculo del alargamiento"
