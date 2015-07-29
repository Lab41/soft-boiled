import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap





def plot_gmm(gmm, true_ll=None, percent=None):
    """Plots the contour map of a GMM on top of a world map
    Partially from: http://matplotlib.org/basemap/users/examples.html
    Will also plot the best estimated location as well as a true location if passed in"""
    plt.figure()
    x = np.linspace(-180,180, num=3600)
    y = np.linspace(90,-90, num=180)
    #GMM uses lat, lon so must flip in to obtain the contours correctly
    X, Y = np.meshgrid(y,x)
    XX = np.array([X.ravel(), Y.ravel()]).T
    #Obtains the per-sample log probabilities of the data under the model
    Z = -gmm.score_samples(XX)[0]

    if percent:
        Z = np.exp2(gmm.score_samples(XX)[0])
        target = np.sum(Z)*percent
        z_sorted = sorted(Z, reverse=True)
        i = 0
        sum = 0
        while sum < target:
            sum += z_sorted[i]
            i += 1
        Z[Z < z_sorted[i]]  = 0
        Z = -np.log(Z)

    Z = Z.reshape(X.shape)

    #set up and draw the world map
    m = Basemap(projection='mill', lon_0=0)

    m.drawcountries()
    m.drawcoastlines()
    m.drawparallels(np.arange(-90.,120.,30.),labels=[1,0,0,0]) # draw parallels
    m.drawmeridians(np.arange(0.,420.,60.),labels=[0,0,0,1]) # draw meridians

    cmap = mpl.cm.pink

    #for plotting we want to be back in lon, lat space so flip back
    X, Y = m(Y,X)
    #plot the contour lines as well as a color bar
    CS = m.contourf(X, Y, Z, 25, cmap=cmap)
    CB = plt.colorbar(CS, shrink=0.8, extend='both')

    # Plot estimated location
    (best_lat, best_lon) = gmm.means_[np.argmax(gmm.weights_)]
    best_lat, best_lon = m(best_lon, best_lat)
    plt.plot(best_lat, best_lon, '*g')

    # If provided also plot the true lat/lon expected to come in as lat,lon
    if true_ll:
        lat, lon = m(true_ll[1], true_ll[0])
        plt.plot(lat, lon, '*b')

    #plots the center of each ellipse and weights the size relative to the weight of the ellipse on the model
    for i in range (0, gmm.n_components):
        lat, lon = gmm.means_[i]
        weight = gmm.weights_[i]
        x, y = m(lon, lat)
        plt.plot(x, y, 'ok', markersize=10*weight)

    plt.show()