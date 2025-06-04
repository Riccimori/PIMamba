#coding=utf8
import matplotlib.pyplot as plt
import numpy as np
import math
'''
Sand Production;
We use a total of 11 range parameters.

x_down_len: Theoretical baseline length of the dynamometer card      
LrLen: Theoretical left and right side lengths of the dynamometer card
Angle: Theoretical angle between the side and base of the dynamometer card      
comp_ud_len: Length difference between the top and base of the dynamometer card
amplitude, ax, pianyi: Amplitude, frequency, and initial phase of upper/lower load curve
P_control, k : Control point and curvature of the fitted piecewise quadratic Bezier curves.
               P_control is automatically calculated by the normal vector of the starting point.
'''


def plot_parallelogram(ud_lengths, lr_lengths, up_down_complen,lowamplitude,lowax,lowpianyi,upamplitude,upax,uppianyi,angle_Du,k):
    plt.figure(figsize=(2.24, 2.24),facecolor='black')


    # Lower Curve
    x0, y0 = 0, 0
    x1 = x0 + ud_lengths
    y1 = lowamplitude * np.cos(lowpianyi + lowax * x1)
    x1_point = np.linspace(x0, x1, 50)
    y1_point = lowamplitude * np.cos(lowpianyi + lowax * x1_point)
    plt.plot(x1_point, y1_point,'white')
    y1cos_qian = lowamplitude * np.cos(lowpianyi + lowax * x0)
    y1cos_hou = lowamplitude * np.cos(lowpianyi + lowax * x1)


    # Right Curve Point
    angle = np.radians(angle_Du)
    x2 = ud_lengths + lr_lengths * np.sin(angle)
    y2 = lr_lengths * np.cos(angle)


    # Upper curve
    x3 = x2 - ud_lengths + up_down_complen
    y3 = y2
    x3_point = np.linspace(x3, x2, 50)
    y3_point = y3 + upamplitude * np.sin(uppianyi + upax * x3_point)

    plt.plot(x3_point, y3_point, 'white')
    y3sin_hou = y3 + upamplitude * np.sin(uppianyi + upax * x2)
    y3sin_qian = y3 + upamplitude * np.sin(uppianyi + upax * x3)

    # Right quadratic Bezier
    x2_start = x1
    x2_end = x2
    y2_start = y1cos_hou
    y2_end = y3sin_hou
    P02 = np.array([x2_start, y2_start], dtype=float)
    P22 = np.array([x2_end, y2_end], dtype=float)

    # --- Automatically calculate control points -------------------------------------------------------
    d2 = P22 - P02  # Chord vector (end point minus start point)
    L2 = math.hypot(d2[0], d2[1])   # Chord Length

    # Normal vector: rotate the chord 90° counterclockwise; if you want the arc to bend to the other side, multiply n by -1
    n2 = np.array([-d2[1], d2[0]]) / L2

    k2 = k # Degree of curvature
    h2 = -k2 * L2   # True arch height (the vertical distance of the control point from the chord)

    M2 = (P02 + P22) / 2     # Midpoint of the chord
    M_arc2 = M2 + h2 * n2   # Push the midpoint along the unit normal vector n4 to h4, which is the arc top position
    P12 = 2 * M_arc2 - (P02 + P22) / 2  # P_control

    # --- Generate a quadratic Bézier curve -------------------------------------------------
    t2 = np.linspace(0, 1, 50)[:, None]
    B2 = (1 - t2) ** 2 * P02 + 2 * (1 - t2) * t2 * P12 + t2 ** 2 * P22
    xt2, yt2 = B2[:, 0], B2[:, 1]
    plt.plot(xt2, yt2, 'white')

    # Left quadratic Bezier
    x4_start = 0
    x4_end = x3
    y4_start = y1cos_qian
    y4_end = y3sin_qian
    P04 = np.array([x4_start, y4_start], dtype=float)
    P24 = np.array([x4_end, y4_end], dtype=float)
    # --- Automatically calculate control points -------------------------------------------------------
    d4 = P24 - P04               # Chord vector (end point minus start point)
    L4 = math.hypot(d4[0], d4[1])      # Chord Length

    # Normal vector: rotate the chord 90° counterclockwise; if you want the arc to bend to the other side, multiply n by -1
    n4 = np.array([-d4[1], d4[0]]) / L4

    k4 = k  # Degree of curvature
    h4 = k4 * L4    # True arch height (the vertical distance of the control point from the chord)

    M4 = (P04 + P24) / 2          # Midpoint of the chord
    M_arc4 = M4 + h4 * n4        # Push the midpoint along the unit normal vector n4 to h4, which is the arc top position
    P14 = 2 * M_arc4 - (P04 + P24) / 2  # P_control

    # --- Generate a quadratic Bézier curve -------------------------------------------------
    t4 = np.linspace(0, 1, 50)[:, None]
    B4 = (1 - t4) ** 2 * P04 + 2 * (1 - t4) * t4 * P14 + t4 ** 2 * P24
    xt4, yt4 = B4[:, 0], B4[:, 1]
    plt.plot(xt4, yt4, 'white')


    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('equal')
    # plt.savefig('./SP/zuoping-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}.png'
    #             .format(lr_lengths, angle_Du, up_down_complen, lowamplitude, upamplitude,lowax,upax, k))
    plt.show()
    plt.close()

if __name__ == '__main__':
    ud_len = 5
    for lr_len in np.arange(3 ,4 , 0.5, dtype='float64'):
        for angle in np.arange(10 ,60, 10, dtype='float64'):
            for comp_ud_len in np.arange(0, 1, 0.5, dtype='float64'):
                for low_amplitude in np.arange(0, 0.4, 0.1, dtype='float64'):
                    for up_amplitude in np.arange(0, 0.4 ,0.1, dtype='float64'):
                        if(low_amplitude==0 and up_amplitude==0): continue
                        for lowax in np.arange(16, 22, 2, dtype='float64'):
                            for upax in np.arange(16, 22, 2, dtype='float64'):
                                for k in np.arange(0, 0.2, 0.1, dtype='float64'):
                                    plot_parallelogram(ud_lengths=5, lr_lengths=lr_len, up_down_complen=comp_ud_len,
                                                                    lowamplitude=low_amplitude, lowax=lowax,lowpianyi=0,
                                                                    upamplitude=up_amplitude, upax=upax,uppianyi=0,
                                                                   angle_Du=angle, k=k)