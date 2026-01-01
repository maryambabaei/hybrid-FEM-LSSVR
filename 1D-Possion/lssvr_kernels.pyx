# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport pow, fabs

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] compute_legendre_A_matrix(
    cnp.ndarray[DTYPE_t, ndim=1] x_scaled,
    int M,
    double deriv_scale
):
    """
    Cython-optimized A matrix computation for Legendre polynomial second derivatives.
    Much faster than NumPy/Numba for repeated calls.
    """
    cdef int n_points = x_scaled.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=2] A = np.zeros((n_points, M), dtype=np.float64)
    cdef int i
    cdef double x, x2, x3, x4, x5, x6, x7, x8, x9, x10
    
    # Pre-compute powers for all points
    cdef cnp.ndarray[DTYPE_t, ndim=1] x2_arr = x_scaled * x_scaled
    cdef cnp.ndarray[DTYPE_t, ndim=1] x3_arr = x2_arr * x_scaled
    cdef cnp.ndarray[DTYPE_t, ndim=1] x4_arr = x2_arr * x2_arr
    cdef cnp.ndarray[DTYPE_t, ndim=1] x5_arr = x3_arr * x2_arr
    
    # P_0''(x) = 0, P_1''(x) = 0 (already zero)
    
    if M > 2:
        # P_2''(x) = 3
        for i in range(n_points):
            A[i, 2] = -3.0 * deriv_scale
    
    if M > 3:
        # P_3''(x) = 15x
        for i in range(n_points):
            A[i, 3] = -15.0 * x_scaled[i] * deriv_scale
    
    if M > 4:
        # P_4''(x) = 52.5x² - 7.5
        for i in range(n_points):
            A[i, 4] = (-52.5 * x2_arr[i] + 7.5) * deriv_scale
    
    if M > 5:
        # P_5''(x) = 157.5x³ - 52.5x
        for i in range(n_points):
            A[i, 5] = (-157.5 * x3_arr[i] + 52.5 * x_scaled[i]) * deriv_scale
    
    if M > 6:
        # P_6''(x) = 472.5x⁴ - 315x² + 13.5
        for i in range(n_points):
            A[i, 6] = (472.5 * x4_arr[i] - 315.0 * x2_arr[i] + 13.5) * deriv_scale
    
    if M > 7:
        # P_7''(x) = 1417.5x⁵ - 1260x³ + 189x
        for i in range(n_points):
            A[i, 7] = (1417.5 * x5_arr[i] - 1260.0 * x3_arr[i] + 189.0 * x_scaled[i]) * deriv_scale
    
    if M > 8:
        # P_8''(x) = 4252.5x⁶ - 4725x⁴ + 1260x² - 31.5
        x6_arr = x3_arr * x3_arr
        for i in range(n_points):
            A[i, 8] = (4252.5 * x6_arr[i] - 4725.0 * x4_arr[i] + 1260.0 * x2_arr[i] - 31.5) * deriv_scale
    
    if M > 9:
        # P_9''(x) = 12757.5x⁷ - 17325x⁵ + 6615x³ - 283.5x
        x7_arr = x4_arr * x3_arr
        for i in range(n_points):
            A[i, 9] = (12757.5 * x7_arr[i] - 17325.0 * x5_arr[i] + 6615.0 * x3_arr[i] - 283.5 * x_scaled[i]) * deriv_scale
    
    if M > 10:
        # P_10''(x) = 38272.5x⁸ - 62370x⁶ + 33075x⁴ - 3780x² + 63
        x6_arr = x3_arr * x3_arr
        x8_arr = x4_arr * x4_arr
        for i in range(n_points):
            A[i, 10] = (38272.5 * x8_arr[i] - 62370.0 * x6_arr[i] + 33075.0 * x4_arr[i] - 3780.0 * x2_arr[i] + 63.0) * deriv_scale
    
    if M > 11:
        # P_11''(x) = 114817.5x⁹ - 218295x⁷ + 155925x⁵ - 34650x³ + 1417.5x
        x7_arr = x4_arr * x3_arr
        x9_arr = x5_arr * x4_arr
        for i in range(n_points):
            A[i, 11] = (114817.5 * x9_arr[i] - 218295.0 * x7_arr[i] + 155925.0 * x5_arr[i] - 34650.0 * x3_arr[i] + 1417.5 * x_scaled[i]) * deriv_scale
    
    if M > 12:
        # P_12''(x) = 344452.5x¹⁰ - 759885x⁸ + 692835x⁶ - 269325x⁴ + 31185x² - 382.5
        x6_arr = x3_arr * x3_arr
        x8_arr = x4_arr * x4_arr
        x10_arr = x5_arr * x5_arr
        for i in range(n_points):
            A[i, 12] = (344452.5 * x10_arr[i] - 759885.0 * x8_arr[i] + 692835.0 * x6_arr[i] - 269325.0 * x4_arr[i] + 31185.0 * x2_arr[i] - 382.5) * deriv_scale
    
    return A


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[DTYPE_t, ndim=2] compute_legendre_C_matrix(
    double xmin_scaled,
    double xmax_scaled,
    int M
):
    """
    Cython-optimized C matrix computation for Legendre polynomial values at boundaries.
    """
    cdef cnp.ndarray[DTYPE_t, ndim=2] C = np.zeros((2, M), dtype=np.float64)
    cdef double xmin2, xmax2, xmin3, xmax3, xmin4, xmax4, xmin5, xmax5
    cdef double xmin6, xmax6, xmin7, xmax7, xmin8, xmax8, xmin9, xmax9
    cdef double xmin10, xmax10, xmin11, xmax11, xmin12, xmax12
    
    # Pre-compute powers for both boundaries
    xmin2 = xmin_scaled * xmin_scaled
    xmax2 = xmax_scaled * xmax_scaled
    xmin3 = xmin2 * xmin_scaled
    xmax3 = xmax2 * xmax_scaled
    xmin4 = xmin2 * xmin2
    xmax4 = xmax2 * xmax2
    xmin5 = xmin3 * xmin2
    xmax5 = xmax3 * xmax2
    
    # P_0(x) = 1
    C[0, 0] = 1.0
    C[1, 0] = 1.0
    
    if M > 1:
        # P_1(x) = x
        C[0, 1] = xmin_scaled
        C[1, 1] = xmax_scaled
    
    if M > 2:
        # P_2(x) = (3x²-1)/2
        C[0, 2] = 0.5 * (3.0 * xmin2 - 1.0)
        C[1, 2] = 0.5 * (3.0 * xmax2 - 1.0)
    
    if M > 3:
        # P_3(x) = (5x³-3x)/2
        C[0, 3] = 0.5 * (5.0 * xmin3 - 3.0 * xmin_scaled)
        C[1, 3] = 0.5 * (5.0 * xmax3 - 3.0 * xmax_scaled)
    
    if M > 4:
        # P_4(x) = (35x⁴ - 30x² + 3)/8
        C[0, 4] = 0.125 * (35.0 * xmin4 - 30.0 * xmin2 + 3.0)
        C[1, 4] = 0.125 * (35.0 * xmax4 - 30.0 * xmax2 + 3.0)
    
    if M > 5:
        # P_5(x) = (63x⁵ - 70x³ + 15x)/8
        C[0, 5] = 0.125 * (63.0 * xmin5 - 70.0 * xmin3 + 15.0 * xmin_scaled)
        C[1, 5] = 0.125 * (63.0 * xmax5 - 70.0 * xmax3 + 15.0 * xmax_scaled)
    
    if M > 6:
        # P_6(x) = (231x⁶ - 315x⁴ + 105x² - 5)/16
        xmin6 = xmin3 * xmin3
        xmax6 = xmax3 * xmax3
        C[0, 6] = 0.0625 * (231.0 * xmin6 - 315.0 * xmin4 + 105.0 * xmin2 - 5.0)
        C[1, 6] = 0.0625 * (231.0 * xmax6 - 315.0 * xmax4 + 105.0 * xmax2 - 5.0)
    
    if M > 7:
        # P_7(x) = (429x⁷ - 693x⁵ + 315x³ - 35x)/16
        xmin7 = xmin4 * xmin3
        xmax7 = xmax4 * xmax3
        C[0, 7] = 0.0625 * (429.0 * xmin7 - 693.0 * xmin5 + 315.0 * xmin3 - 35.0 * xmin_scaled)
        C[1, 7] = 0.0625 * (429.0 * xmax7 - 693.0 * xmax5 + 315.0 * xmax3 - 35.0 * xmax_scaled)
    
    if M > 8:
        # P_8(x) = (6435x⁸ - 12012x⁶ + 6930x⁴ - 1260x² + 35)/128
        xmin6 = xmin3 * xmin3
        xmax6 = xmax3 * xmax3
        xmin8 = xmin4 * xmin4
        xmax8 = xmax4 * xmax4
        C[0, 8] = 0.0078125 * (6435.0 * xmin8 - 12012.0 * xmin6 + 6930.0 * xmin4 - 1260.0 * xmin2 + 35.0)
        C[1, 8] = 0.0078125 * (6435.0 * xmax8 - 12012.0 * xmax6 + 6930.0 * xmax4 - 1260.0 * xmax2 + 35.0)
    
    if M > 9:
        # P_9(x) = (12155x⁹ - 25740x⁷ + 18018x⁵ - 4620x³ + 315x)/128
        xmin7 = xmin4 * xmin3
        xmax7 = xmax4 * xmax3
        xmin9 = xmin5 * xmin4
        xmax9 = xmax5 * xmax4
        C[0, 9] = 0.0078125 * (12155.0 * xmin9 - 25740.0 * xmin7 + 18018.0 * xmin5 - 4620.0 * xmin3 + 315.0 * xmin_scaled)
        C[1, 9] = 0.0078125 * (12155.0 * xmax9 - 25740.0 * xmax7 + 18018.0 * xmax5 - 4620.0 * xmax3 + 315.0 * xmax_scaled)
    
    if M > 10:
        # P_10(x) = (46189x¹⁰ - 109395x⁸ + 90090x⁶ - 30030x⁴ + 3465x² - 63)/256
        xmin6 = xmin3 * xmin3
        xmax6 = xmax3 * xmax3
        xmin8 = xmin4 * xmin4
        xmax8 = xmax4 * xmax4
        xmin10 = xmin5 * xmin5
        xmax10 = xmax5 * xmax5
        C[0, 10] = 0.00390625 * (46189.0 * xmin10 - 109395.0 * xmin8 + 90090.0 * xmin6 - 30030.0 * xmin4 + 3465.0 * xmin2 - 63.0)
        C[1, 10] = 0.00390625 * (46189.0 * xmax10 - 109395.0 * xmax8 + 90090.0 * xmax6 - 30030.0 * xmax4 + 3465.0 * xmax2 - 63.0)
    
    if M > 11:
        # P_11(x) = (88179x¹¹ - 230945x⁹ + 218790x⁷ - 90090x⁵ + 15015x³ - 693x)/256
        xmin7 = xmin4 * xmin3
        xmax7 = xmax4 * xmax3
        xmin9 = xmin5 * xmin4
        xmax9 = xmax5 * xmax4
        xmin11 = xmin6 * xmin5
        xmax11 = xmax6 * xmax5
        C[0, 11] = 0.00390625 * (88179.0 * xmin11 - 230945.0 * xmin9 + 218790.0 * xmin7 - 90090.0 * xmin5 + 15015.0 * xmin3 - 693.0 * xmin_scaled)
        C[1, 11] = 0.00390625 * (88179.0 * xmax11 - 230945.0 * xmax9 + 218790.0 * xmax7 - 90090.0 * xmax5 + 15015.0 * xmax3 - 693.0 * xmax_scaled)
    
    if M > 12:
        # P_12(x) = (676039x¹² - 1939938x¹⁰ + 2078505x⁸ - 1021020x⁶ + 225225x⁴ - 18018x² + 231)/1024
        xmin6 = xmin3 * xmin3
        xmax6 = xmax3 * xmax3
        xmin8 = xmin4 * xmin4
        xmax8 = xmax4 * xmax4
        xmin10 = xmin5 * xmin5
        xmax10 = xmax5 * xmax5
        xmin12 = xmin6 * xmin6
        xmax12 = xmax6 * xmax6
        C[0, 12] = 0.0009765625 * (676039.0 * xmin12 - 1939938.0 * xmin10 + 2078505.0 * xmin8 - 1021020.0 * xmin6 + 225225.0 * xmin4 - 18018.0 * xmin2 + 231.0)
        C[1, 12] = 0.0009765625 * (676039.0 * xmax12 - 1939938.0 * xmax10 + 2078505.0 * xmax8 - 1021020.0 * xmax6 + 225225.0 * xmax4 - 18018.0 * xmax2 + 231.0)
    
    return C
