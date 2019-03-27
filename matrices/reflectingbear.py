import matplotlib.pyplot as plt
import numpy as np

bear_white_fur  = np.array([[ 2.229253 ,  2.4680387,  2.7017972,  2.8299874,  2.8676905,
         2.7746897,  2.4052003,  2.2795237,  2.1639012,  1.847196 ,
         2.0030351,  2.229253 ,        np.nan,  1.8044659,  1.8974666,
         2.0491517,  2.1326865,  2.3175294,  2.5112591,  2.9028799,
         2.7520679,  2.5962289,  2.2367936,  1.9577914,  1.8044659],
       [-1.0902537, -1.0601388, -1.1881273, -1.4641809, -1.677495 ,
        -1.8431272, -2.028836 , -2.0363647, -1.9485295, -1.5043341,
        -1.3186252, -1.0902537,        np.nan, -2.3902152, -2.5527822,
        -2.5527822, -2.4463632, -2.4570104, -2.5527822, -2.5527822,
        -2.2446597, -2.1467861, -2.3575907, -2.4378972, -2.3902152]])

bear_black_fur = np.array([[ 2.0030351 ,  2.229253  ,  2.1639012 ,  2.0809546 ,  1.9728726 ,
         1.8974666 ,  1.8924396 ,  2.0030351 ,         np.nan,  2.7017972 ,
         2.8500957 ,  2.9707453 ,  3.0159889 ,  2.94561   ,  2.8299874 ,
         2.7017972 ,         np.nan,  2.1639012 ,  2.2317666 ,  2.3147132 ,
         2.299632  ,  2.2493613 ,  2.1890365 ,  2.1211711 ,  2.1337387 ,
         2.1639012 ,         np.nan,  2.4982011 ,  2.5610936 ,  2.6213642 ,
         2.633986  ,  2.5536071 ,  2.5057417 ,  2.4982011 ,         np.nan,
         2.2468478 ,  2.3247673 ,  2.4429034 ,  2.4303357 ,  2.3448755 ,
         2.2820372 ,  2.2468478 ,         np.nan,  2.1966706 ,  2.2722074 ,
         2.4055076 ,  2.481933  ,  2.449941  ,  2.4001756 ,  2.3237501 ,
         2.222442  ,  2.1984479 ,  2.1966706 ,         np.nan,  1.847196  ,
         1.7818441 ,  1.7290599 ,  1.6310321 ,  1.4575984 ,  1.3369488 ,
         1.2791375 ,  1.3671112 ,  1.8044659 ,  1.9577914 ,  2.2367936 ,
         2.5962289 ,  2.7520679 ,  2.9028799 ,  3.4005595 ,  3.3150993 ,
         3.0511783 ,  2.9531506 ,  2.8676905 ,  2.7746897 ,  2.4052003 ,
         2.2795237 ,  2.1639012 ,  1.847196  ,         np.nan,  2.0491517 ,
         2.5112591 ,  2.3175294 ,  2.1326865 ,  2.0491517 ],
       [-1.3186252 , -1.0902537 , -0.99238015, -0.96477475, -0.99488975,
        -1.1153494 , -1.2408283 , -1.3186252 ,         np.nan, -1.1881273 ,
        -1.0852346 , -1.1454645 , -1.3286636 , -1.4666904 , -1.4641808 ,
        -1.1881273 ,         np.nan, -1.5545256 , -1.5219011 , -1.4014413 ,
        -1.3512497 , -1.3412115 , -1.3989317 , -1.4917862 , -1.5419777 ,
        -1.5545256 ,         np.nan, -1.4265371 , -1.3964222 , -1.4968054 ,
        -1.6097363 , -1.64738   , -1.5545256 , -1.4265371 ,         np.nan,
        -1.6423608 , -1.6699662 , -1.677495  , -1.7176483 , -1.7477632 ,
        -1.7176483 , -1.6423608 ,         np.nan, -1.7223509 , -1.7622781 ,
        -1.7764744 , -1.7613908 , -1.8767359 , -1.9805465 , -1.9991791 ,
        -1.9672374 , -1.913114  , -1.7223509 ,         np.nan, -1.5043341 ,
        -1.5444873 , -1.486767  , -1.1504836 , -1.0626484 , -1.11284   ,
        -1.2558858 , -1.7452537 , -2.3902152 , -2.4378972 , -2.3575907 ,
        -2.1467861 , -2.2446597 , -2.5527822 , -2.5527822 , -2.1919586 ,
        -1.7828973 , -1.6850238 , -1.677495  , -1.8431272 , -2.028836  ,
        -2.0363647 , -1.9485295 , -1.5043341 ,         np.nan, -2.5527822 ,
        -2.5527822 , -2.4570104 , -2.4463632 , -2.5527822 ]])

bear_face = np.array([[ 2.2419927,  2.2526567,  2.3015334,  2.3477442,  2.441943 ,
               np.nan,  2.5258499,  2.5113971,  2.5327621,  2.5632387,
         2.5780058,  2.5726645,  2.5475292,  2.5258499,        np.nan,
         2.2858075,  2.2704121,  2.2402497,  2.2283105,  2.2484187,
         2.273554 ,  2.2858075],
       [-1.7605035, -1.9432811, -1.9707865, -1.9654629, -1.781798 ,
               np.nan, -1.4688862, -1.4942957, -1.5099806, -1.5112354,
        -1.4877081, -1.466063 , -1.4588479, -1.4688862,        np.nan,
        -1.4346933, -1.4506918, -1.4463002, -1.418381 , -1.4055194,
        -1.4083427, -1.4346933]])

blue1 = (0.27450980392156865, 0.396078431372549, 0.5372549019607843)
magenta = (0.9882352941176471, 0.4588235294117647, 0.8588235294117647)
green = (0.6862745098039216, 0.8588235294117647, 0.5215686274509804)
orange = (0.8549019607843137, 0.6705882352941176, 0.45098039215686275)
blue2 = (0.47843137254901963, 0.6823529411764706, 0.8431372549019608)
bear_white = (0.89, 0.856, 0.856)
bear_black = (0.141, 0.11, 0.11)

verySmallNumber = 1e-14


def gsBasis(A):
    B = np.array(A, dtype=np.float_)
    # for each columns of B
    for i in range(B.shape[1]):
        # substract the overlap from the previous columns
        for j in range(i):
            # substract the overlap
            B[:, i] = B[:, i] - B[:, i] @ B[:, j] * B[:, j]

        # Now normalise if possible
        if np.linalg.norm(B[:, i]) > verySmallNumber:
            B[:, i] = B[:, i] / np.linalg.norm(B[:, i])
        else:
            B[:, i] = np.zeros_like(B[:, i])

    return B


def draw_mirror(bearVectors):
    fig, ax = plt.subplots(figsize=(12, 12), dpi=80)
    ax.set_xlim([-3.50, 3.50])
    ax.set_ylim([-3.50, 3.50])
    ax.set_aspect(1)
    ax.set_facecolor(blue1)

    gs = gsBasis(bearVectors)
    ax.plot([gs[0, 0] * -5, gs[0, 0] * 5], [gs[1, 0] * -5, gs[1, 0] * 5], lw=2, color=green, zorder=4)
    ax.fill([
        -5 * gs[0, 0], -5 * gs[0, 0] - 5 * gs[0, 1], 5 * gs[0, 0] - 5 * gs[0, 1], 5 * gs[0, 0]
    ], [
        -5 * gs[1, 0], -5 * gs[1, 0] - 5 * gs[1, 1], 5 * gs[1, 0] - 5 * gs[1, 1], 5 * gs[1, 0]
    ], color=blue2, zorder=0)

    # draw bear basis vectors
    ax.arrow(0, 0, bearVectors[0, 0], bearVectors[1, 0], lw=3, color=orange, zorder=5, head_width=0.1)
    ax.arrow(0, 0, bearVectors[0, 1], bearVectors[1, 1], lw=3, color=orange, zorder=5, head_width=0.1)

    # draw orthonormal rebased of bear basis vectors
    ax.arrow(0, 0, gs[0, 0], gs[1, 0], lw=3, color=magenta, zorder=6, head_width=0.1)
    ax.arrow(0, 0, gs[0, 1], gs[1, 1], lw=3, color=magenta, zorder=6, head_width=0.1)
    return ax

# GRADED FUNCTION
# This is the cell you should edit and submit.

# In this function, you will return the transformation matrix T,
# having built it out of an orthonormal basis set E that you create from Bear's Basis
# and a transformation matrix in the mirror's coordinates TE.
def build_reflection_matrix(bearBasis) : # The parameter bearBasis is a 2×2 matrix that is passed to the function.
    # Use the gsBasis function on bearBasis to get the mirror's orthonormal basis.
    E = gsBasis(bearBasis)
    # Write a matrix in component form that perform's the mirror's reflection in the mirror's basis.
    # Recall, the mirror operates by negating the last component of a vector.
    # Replace a,b,c,d with appropriate values
    TE = np.array([[1, 0],
                   [0, -1]])
    # Combine the matrices E and TE to produce your transformation matrix.
    T = E @ TE @ inv(E)
    # Finally, we return the result. There is no need to change this line.
    return T

# This is the matrix of Bear's basis vectors.
bearBasis = np.array(
    [[1,   -1],
     [1.5, 2]])
# This line uses your code to build a transformation matrix for us to use.
T = build_reflection_matrix(bearBasis)

# Bear is drawn as a set of polygons, the vertices of which are placed as a matrix list of column vectors.
# We have three of these non-square matrix lists: bear_white_fur, bear_black_fur, and bear_face.
# We'll make new lists of vertices by applying the T matrix you've calculated.
reflected_bear_white_fur = T @ bear_white_fur
reflected_bear_black_fur = T @ bear_black_fur
reflected_bear_face = T @ bear_face

# This next line runs a code to set up the graphics environment.
ax = draw_mirror(bearBasis)

# We'll first plot Bear, his white fur, his black fur, and his face.
ax.fill(bear_white_fur[0], bear_white_fur[1], color=bear_white, zorder=1)
ax.fill(bear_black_fur[0], bear_black_fur[1], color=bear_black, zorder=2)
ax.plot(bear_face[0], bear_face[1], color=bear_white, zorder=3)

# Next we'll plot Bear's reflection.
ax.fill(reflected_bear_white_fur[0], reflected_bear_white_fur[1], color=bear_white, zorder=1)
ax.fill(reflected_bear_black_fur[0], reflected_bear_black_fur[1], color=bear_black, zorder=2)
ax.plot(reflected_bear_face[0], reflected_bear_face[1], color=bear_white, zorder=3)

plt.show()