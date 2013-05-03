# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:06:10 2013

@author: skywalker
"""

# ASSPLOT.PY
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Python script for importing data from a CSV file into NUMPY arrays 
# and automatic plotting of data in Assignment 1, ECON1102/7074.
#
# (c) 2013, T. Kam, Australian Notional University
# =============================================================================


# Import NUMPY and MATPLOTLIB libraries
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0. Create array of strings (of variable names)
# -----------------------------------------------------------------------------

labels = [ '(a) GDP per capita',
           '(b) Population, total',
           '(c) Income share (highest 10%)',
           '(d) Income share (lowest 10%)',
           '(e) GINI',
           '(f) Health expenditure (% of GDP)',
           '(g) Researchers in R&D (per mil people)',
           '(h) Secondary education',
           '(i) Sanitation (% pop. access)',
           '(j) US GDP per capita'
           ]
           
print "Make sure your variable names are ordered as per the data file!\n"

nvar = len(labels)

for i in range(0, nvar):
    print  i, ". ", labels[i], "\n" 

# -----------------------------------------------------------------------------
# 1. Import data from CSV file
# -----------------------------------------------------------------------------

# Use NUMPY's genfromtxt module: convert CSV data into NUMPY array:
data = np.genfromtxt('china_us.csv', delimiter=',')
nrow = data.shape[0]    # number of rows in DATA
ncol = data.shape[1]    # number of column in DATA

# Extract DATES series from DATA:
dates = data.astype(int)[0,:]

# Extract rest of time series data from DATA:
series = data[1:nrow,:]


# Check DATA is imported correctly
print "Data has (#variables, #observations) = ", str(series.shape)
print "Start date with Observation =", str(dates[0])
print "End date with non-empty observation(s) = ", str(dates[-1]),"\n"

# -----------------------------------------------------------------------------
# 2. Visual plot of raw data (deletin omitted observations)
# -----------------------------------------------------------------------------
# Now I want to get rid of missing obs and corresponding dates for each data 
# series. To do that I use NUMPY's ISNAN Boolean operator, for each series at 
# at a time (i.e. at each iteration of the FOR loop indexed by integer n):

# FIGURE: Loop over all data series
plt.figure()

for n in range(0, nvar):
    
    y = series[n,:]         # Current (n-th) y-axis variable
    picker = ~np.isnan(y)   # Logical vector to select non-NaN observations
    
    y = y[picker]           # Pick out non-NaN observations only
    x = dates[picker]       # % Pick out dates associate with non-NaN obs. only
    
    
    # Now we plot each n-th data series as we loop through "n":
    print "Generating figure #", str(n)
    
    h = plt.subplot(5,2,n+1)
    plt.plot(x,y, '-rs')  # Plot with Marker option r(ed) and o (circle)
    plt.title(labels[n], fontsize=8)
    #plt.xlabel('Year', fontsize=9)
    plt.setp(h.get_xticklabels(), rotation='horizontal', fontsize=8)
    plt.setp(h.get_yticklabels(), rotation='horizontal', fontsize=8)
    plt.autoscale(enable=True, axis='both', tight=True)
    h.margins(0.10)
    
    # Ensure enough spacing/padding between sub-plots
    plt.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.2)

# Save figure in EPS and PNG formats:
plt.savefig('data_chn_us.eps')
plt.savefig('data_chn_us.png')

plt.show()

# -----------------------------------------------------------------------------
# 3. Transform data for further analysis and plots
# -----------------------------------------------------------------------------

# Calculate new variables:
subdates = dates[0:-3]

ychina = series[0,:]
ychina = ychina[~np.isnan(ychina)]
growth_chi = np.divide( ychina[1:-1], ychina[0:-2] )
growth_chi = np.log( growth_chi )   # China's per period growth rate

yus = series[9,:]
yus = yus[0:ychina.size]
ygap = np.log(ychina[0:-2]) - np.log(yus[0:-2])

# Plot growth rates vs y-gap data:
start_date = 1960
start_ind = np.where(subdates == start_date)[0]
x1 = subdates[start_ind:-1]
y1 = ygap[start_ind:-1]
y2 = growth_chi[start_ind:-1]

# FIGURE:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.figure()
x = x1
y = y1
plt.subplot(2,1,1)
plt.plot(x, y, '-bo')
plt.autoscale(enable=True, axis='both', tight=True)
plt.margins(0.10)
plt.ylabel('$\ln (y_{china,t}/y_{US,t})$', fontsize=12)
#plt.xlabel('Growth Rate', fontsize=9)

plt.subplot(2,1,2)
x = x1
y = y2
plt.plot(x, y, '-bo')
plt.autoscale(enable=True, axis='both', tight=True)
plt.margins(0.10)
plt.ylabel('$\Delta \ln (y_{china,t+1})$', fontsize=12)
#plt.xlabel('Growth Rate', fontsize=9)

# Save figures in EPS and PNG formats:
plt.savefig('data_chn_us_convergence1.eps')
plt.savefig('data_chn_us_convergence1.png')




# FIGURE: Scatter plot ygap vs. growth (CHI)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.figure()
x = y1
y = y2
year = x1

# Linear best fit:
deg = 1 # OLS for linear best fit equation
bols = np.polyfit(x, y, deg, rcond=None, full=False)
yhat = bols[1] + bols[0]*x
print "\nNow we study the growth-vs.-ygap data\n"
print "\t\tLinear Best Fit (Slope, Intercept) = ", str(bols), "\n"

colorspec = np.random.random((x.size, 1))

t_labels = ['{0}'.format(t) for t in range(int(year[0]),int(year[-1]+1))]
plt.subplots_adjust(bottom = 0.1)
h = plt.scatter(x, y, marker = 'o', 
                c = colorspec[:, 0], s = 30,
                cmap = plt.get_cmap('Spectral'))

for label, xt, yt in zip(t_labels, x, y):
    plt.annotate(
        label, 
        xy = (xt, yt), 
        xytext = (-10, 10), 
        fontsize=8,
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'pink', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
        )

plt.plot(x, yhat, '-r') # Line of OLS best fit

plt.ylabel('$\Delta \ln (y_{china,t+1})$', fontsize=12)
plt.xlabel('$\ln (y_{china,t}/y_{US,t})$', fontsize=12)

# Save figures in EPS and PNG formats:
plt.savefig('data_chn_us_convergence2.eps')
plt.savefig('data_chn_us_convergence2.png')

plt.show()



