import numpy as np
from scipy.optimize import minimize
import pdb
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator, UnivariateSpline
import sys
import pickle
from ratio_x import ratio_x
import copy

def project_points(x, y, z, a, b, c):
    """
    Projects the points with coordinates x, y, z onto the plane
    defined by a*x + b*y + c*z = 1
    """
    vector_norm = a*a + b*b + c*c
    normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
    point_in_plane = np.array([a, b, c]) / vector_norm

    points = np.column_stack((x, y, z))
    points_from_point_in_plane = points - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None]*normal_vector)

    return point_in_plane + proj_onto_plane


def hill(K,s,b,x):
    """return a hill function value. 
    K- dissociation constant
    s- number of functional antibodies
    b- basal fluorescence
    x- antigen concentration"""
    return s*x/(K + x)+b


def make_x(K, amp, basal, fl):
    """Simulate flow cytometry and return sort fractions for built in test function
    returns sort fractions and mean for each bin
    K- dissociation rate
    amp- number of functional antibodies
    basal- mean fluorescence value at 0 antigen
    fl- antigen concentration """
    ab = np.random.randn(100000)
    ab = np.exp(ab)
    
    bound = ab*hill(K,amp,basal,fl)
    bound0 = ab*hill(K,amp,basal,0)
    bound1 = ab*(amp+basal)
    mu0 = np.nanmean(np.log(bound0))
    mu1 = np.nanmean(np.log(bound1))
    slope = (np.log(amp+basal) - np.log(basal))/(mu1-mu0)
    bound = slope*np.log(bound) + (np.log(basal) - slope*mu0)
    bound = np.exp(bound)

    boundaries = [0,10,100,1000, np.inf]
    usethis = lambda ii: (bound > boundaries[ii]) & (bound < boundaries[ii+1])
    c = [np.mean(bound[usethis(ii)]) for ii in range(4)]
    x = np.histogram(bound, bins=[0,10,100,1000, np.inf])[0]
    return x, c


def log_poiss(x, R, T, b):
    """returns the log poisson probability
    x- fraction of cells with sequence s sorted into a bin
    R- number of reads with sequence s in bin
    T- total number of reads in a bin
    b- coefficient """
    if np.sum(x<=0):
        return -np.inf
    if np.sum(R)==0:
        return -b*T*x
    return R*np.log(x*b*T)-b*T*x


def get_x3(x, k, mu, c):
    """internal function that calculates x3 based on constraints k and mu"""
    c=np.log10(c)
    d = c/k
    return (np.log10(mu)-k*d[3]-(d[0]-d[3])*x[0]-(d[1]-d[3])*x[1])/(d[2]-d[3])


def get_x(x, k, mu, c):
    """internal function
    given x1, x2, fraction of the population k, mean mu, bin values c, appends x3 and x4 """
    x3 = get_x3(x, k, mu, c)
    x4 = k - np.sum(x)-x3
    return np.array([x[0], x[1], x3, x4])


def LL(x, R, T, b, k, c, mu):
    """objective function, calculates x3 and x4, and then returns a negative log-likilihood of x1, x2, x3, x4 given data R, T, and scaling factor b"""
    x = get_x(x[:2],k,mu,c)
    out = np.sum(log_poiss(x,R,T,b))
    return out


def getML(x0, R,T,b,k,c, mu):
    """fitting functino, returns most likely x and negative log-likihood of x given the data (R,T,b, c), and model (k, mu)"""
    o = lambda x:-LL(x,R,T,b, k, c, mu)
    bnds = ((0,None),(0,None))
    cons = ({'type': 'ineq', 'fun': lambda x:  k-np.sum(x) - get_x3(x, k, mu, c)}, {'type': 'ineq', 'fun': lambda x:  get_x3(x, k, mu, c) })
    fit = minimize(o, x0, method='Nelder-mead', options = {'maxiter':1000})
    return get_x(fit['x'][:2], k, mu, c), fit['fun']


def closest_legit_x(x1, curr_mu, c, k):
    """internal function, finds the closest x value from x1 that satisfies constraints mean = mu, fraction = k"""
    try:
        
        if np.sum(x1<0):
            logc = np.log10(c)
            pfun = lambda a: np.exp(-a*logc)/np.sum(np.exp(-a*logc))
            o = lambda a: (np.log10(curr_mu)-pfun(a).dot(logc))**2
            fit = minimize(o, 1, method='nelder-mead',tol = 1e-20)
            a = fit['x']
            x0 = pfun(a)
            x0*=k
    
            dx = x0 - x1
            gammas = x0/dx
            gammas = gammas[gammas>=0]
            gamma = np.min(gammas)*(1-1e-6)
            x1 = x0 - gamma * dx
            x1*=k/x1.sum()

    except:
        print('invalid x1')
        print(x1)

    try:
        if np.sum(x1<0):
            logc = np.log10(c)
            pfun = lambda a: np.exp(-a*logc)/np.sum(np.exp(-a*logc))
            o = lambda a: (np.log10(curr_mu)-pfun(a).dot(logc))**2
            fit = minimize(o, 1, method='nelder-mead',tol = 1e-20)
            a = fit['x']
            x0 = pfun(a)
            x0*=k
    
            dx = x0 - x1
            gammas = x0/dx
            gammas = gammas[gammas>=0]
            gamma = np.min(gammas)*(1-1e-6)
            x1 = x0 - gamma * dx
            x1*=k/x1.sum()
    except:
        print('inalid x1')
    print(x1)
    return x1


def generate_counts(K, s, basal,k,fl):
    """internal test function that generates tite-seq data"""
    truex = []
    R = []
    T = []
    b = []
    c = []
    sorts = []
    for f in fl:
        x, temp_c = make_x(K, s, basal, f)
        x= np.array(x, dtype = float)/float(np.sum(x))
        x*=k/np.sum(x)
        truex.append(x)
        S = np.random.rand(4)*10000000
        Sx = x*S.sum()
        p = Sx / S
        p[p>1] = 1
        N = int(-np.log(np.random.rand())*100000)
        T_temp = np.random.rand(4)**(np.random.rand()*10)
        T_temp *= N/np.sum(T_temp)
        T_temp += 1
        scale = 10**(np.random.rand()*4-2)
        T.append(np.round(T_temp))
        R.append([float(np.random.binomial(int(T_temp[ii]/scale),p[ii])*scale) for ii in range(4)])
        c.append(temp_c)
        b.append(S.sum()/S)
        sorts.append(S.tolist())

    truex = np.array(truex)
    R = np.array(R)
    T = np.array(T)
    b= np.array(b)
    c = np.array(c)
    return R, T, b, truex, c, np.array(sorts)


def make_x_dict(R, T, b, k, c):
    """make_x_dict return 2 functions. 
    The first function returns the x1 and x2 pairs as function of mean fluorescence value. 
    The return x1 and x2 values maximize the log-likilood given the data R,T,b, c and model parameter fraction of the population k."""
    x_dict = dict()
    mu = np.logspace(np.log10(c[0]+1e-4),np.log10(c[-1]-1e-4), 50)
    curr_mu = mu[0]
    logc = np.log10(c)
    pfun = lambda a: np.exp(-a*logc)/np.sum(np.exp(-a*logc))
    o = lambda a: (np.log10(curr_mu)-pfun(a).dot(logc))**2
    fit = minimize(o, 1, method='nelder-mead',tol = 1e-20)
    a = fit['x']
    x0 = pfun(a)
    x0*=k
    temp_x, temp_obj = getML(x0[:2],R,T,b, k, c, curr_mu)
    last_mu = np.inf

    obj = []
    xout0 = []
    xout1 = []
    for curr_mu in mu:
        denom = k*(curr_mu - c[-1])
        x0 = project_points(temp_x[0], temp_x[1], temp_x[2], (c[0]-c[-1])/denom, (c[1]-c[-1])/denom, (c[2]-c[-1])/denom).flatten()
        x1 = get_x(x0[:2], k, curr_mu, c)
        logc = np.log10(c)
        pfun = lambda a: np.exp(-a*logc)/np.sum(np.exp(-a*logc))
        o = lambda a: (np.log10(curr_mu)-pfun(a).dot(logc))**2
        fit = minimize(o, a, method='nelder-mead',tol = 1e-20)
        a = fit['x']
        x0 = pfun(a)
        x0*=k
        
        if np.sum(R) == 0:
            temp_x1, temp_obj1 = getML(x0[:2],R,T,b, k, c, curr_mu)
            x_dict[(curr_mu)] = (temp_x, temp_obj)
            obj.append(temp_obj)
            xout0.append(temp_x[0])
            xout1.append(temp_x[1])
            last_mu = curr_mu
            continue
        
        if np.sum(x1<0):
            dx = x0 - x1
            gammas = x0/dx
            gammas = gammas[gammas>=0]
            gamma = np.min(gammas)*(0.5)
            x1 = x0 - gamma * dx
            
        temp_x, temp_obj = getML(x1[:2],R,T,b, k, c, curr_mu)

        temp_x1, temp_obj1 = getML(x0[:2],R,T,b, k, c, curr_mu)
        if temp_obj1<temp_obj:
            temp_x = temp_x1
            temp_obj = temp_obj1

        x_dict[(curr_mu)] = (temp_x, temp_obj)
        obj.append(temp_obj)
        xout0.append(temp_x[0])
        xout1.append(temp_x[1])
        last_mu = curr_mu
        
    mu = np.array(mu)
    obj = np.array(obj)
    usethis = np.where(np.isfinite(obj))[0]
    mu = mu[usethis]
    obj = obj[usethis]
    xout0 = np.array(xout0)[usethis]
    xout1 = np.array(xout1)[usethis]
    if obj.shape[0] < 3:
        myf = lambda x: 0*x
        x_fun = lambda x:[x*np.nan,x*np.nan]
    else:
        myf = Akima1DInterpolator(mu,obj)
        x_fun = lambda x:[float(Akima1DInterpolator(mu, xout0)(x)),float(Akima1DInterpolator(mu, xout1)(x))]
    
    return x_fun, myf


def x_star(R, T, b, k_scan, c, basal, KD_scan, s_scan, fl, unsorted = None):
    """if you fit the KD from another script, this is the function you should use
    R- MXN numpy array of read counts for sequence s. M is the number antigen concentration, and N is the number of bins
    T- MXN numpy array of total read counts.
    b- coefficent relating sequence sampling rate to cell sort rate. p(x)~(x*b*T)^R * e^(-x*b*T)/R!
    k_scan- numpy array of guess for the fraction of the population that has sequence s
    c- log fluorescence value of each bin
    basal- basal fluorescence of cells (without antigen)
    KD_scan- np.array of possible KD values
    s_scan- np.array of possible number of functional antibodies
    fl- fluorescein concentrations
    optional:
    unsorted- a list of 2-tuples containing unsorted float/int read counts. The first element in a tuple is the number of reads with a sequence. The second tuple element
    is the total number of reads from the unsorted population. Example: unsorted = [(10,100000), (1,200)] 
    the output is
    best_x, best_k, best_s, KD_sigma, get_obj, prob, log_p, best, best_frac
    best_x- the fit proportion of cells that gets sorted into bin
    best_k- the fit KD
    best_s- the fit number of functional antibodies
    KD_sigma- inferred standard deviation of uncertainty for log KD
    get_obj - list of functions corresponding to fl elements. Each function returns the objective given a mean fluorescence value.
    prob - np.array of the probability given KD, s in KD_scan and s_scan. 
    log_p - log10 array of the probability, should have more information than prob due to machine precision
    best - best objective function value
    best_frac - fit proportion of the population containing sequence s"""

    [kk, ss]= np.meshgrid(KD_scan, s_scan)
    
    best = np.inf
    bestx = R*1./T
    best_k = 1
    best_s = s_scan[-1]
    best_frac = np.nan
    all_LLs = []
    for k in k_scan:
        x_dict = []
        get_obj = []
        for ii in range(fl.shape[0]):
            xvals, xfun = make_x_dict(R[ii], T[ii], b[ii], k, c)
            get_obj.append(xfun)
            x_dict.append(xvals)

        prob = []
        o = lambda x:np.nansum([get_obj[ii](hill(x[0], x[1], basal, fl[ii])) for ii in range(fl.shape[0])])
        all_obj = kk.T*0
        for ii, KD in enumerate(KD_scan):
            for jj, s in enumerate(s_scan):
                all_obj[ii,jj] = o([KD,s])
                if all_obj[ii,jj] < best:
                    best = all_obj[ii,jj]
                    best_k = KD
                    best_s = s
                    best_frac = k
                    best_x_dict = copy.deepcopy(x_dict)
        
        if not(unsorted == None):
            for us in unsorted:
                all_obj += log_poiss(k, us[0], us[1], 1)

        log_likelihood = -all_obj
        log_likelihood[~np.isfinite(prob)] = -np.inf
        all_LLs.append(log_likelihood)
    
    max_LL = np.nanmax([np.nanmax(curr) for curr in all_LLs])
    all_LLs = [curr + 80. - max_LL for curr in all_LLs]
    all_probs = [np.exp(curr) for curr in all_LLs]
    denominator = np.nansum([np.nansum(curr) for curr in all_probs])
    all_probs = [curr/denominator for curr in all_probs]
    #################
    l2 = (np.log(kk)-np.log(best_k))**2
    KD_sigma = np.sqrt(np.nansum([l2*curr.T for curr in all_probs]))
    prob = 0
    for curr in all_probs:
        prob+=curr
    
    log_p = np.log(prob+1e-50)
    
    
    if isinstance(basal, float) or isinstance(basal, int):
        h = lambda ii:hill(best_k,best_s,basal, fl[ii])
    else:
        h = lambda ii:hill(best_k,best_s,basal[ii], fl[ii])
    
    full_x = [get_x(best_x_dict[ii](h(ii)), k, h(ii), c) for ii in range(fl.shape[0])]
    best_x = np.array([closest_legit_x(full_x[ii], h(ii), c, k) for ii in range(len(full_x))])
    return best_x, best_k, best_s, KD_sigma, get_obj, prob, log_p, best, best_frac


def test():
    """built in test for the algorithm"""
    basal = 10
    c = np.logspace(0.5,3.5,4)
    max_s = c[-1] - basal
    KD_scan = np.logspace(-10, -4, 71)
    s_scan = np.logspace(2, np.log10(max_s)-0.02, 70)

    fl = np.array([0,10**-9.5, 10**-9, 10**-8.5, 10**-8, 10**-7.5, 10**-7, 10**-6.5, 10**-6, 10**-5.5, 10**-5])

    bSSE = []
    rSSE = []
    true_K = []
    true_s = []
    fitk = []
    delta = []
    log_delta = []
    stds = []
    for kk in range(1000):
        print(str(kk))
        true_K.append(10**(-6*np.random.rand()-4))
        true_s.append(10**(2+1*np.random.rand()))
        k = np.random.rand()*0.001
        R, T, b, truex, bin_ave, S = generate_counts(true_K[-1], true_s[-1], basal, k, fl)
        x_ratio, k_guess, b, k_std = ratio_x(R,T,S)
        usethis = (T*10)<S
        usethis = usethis + ~usethis * (S/10.) / T
        T*=usethis
        R*=usethis
        
        xscan = np.linspace(basal,c[-1], 1000)
        fid = open('example.dat', 'wb')
        pickle.dump(R,fid)
        pickle.dump(T,fid)
        pickle.dump(b,fid)
        pickle.dump(k,fid)
        pickle.dump(c,fid)
        pickle.dump(basal,fid)
        pickle.dump(KD_scan,fid)
        pickle.dump(s_scan,fid)
        pickle.dump(fl,fid)
        pickle.dump(xscan, fid)
        fid.close()
        
        best_LL =np.inf
        k_scan = np.logspace(np.log10(k_guess) - 6*np.log10(k_std), np.log10(k_guess)+6*np.log10(k_std),25)
        print('k: '+str(k)+' guessed k: '+ str(k_guess) + 'k error: ' + str(np.abs(k-k_guess)/k))
        
        x, KD, s, KD_sigma, get_obj, prob, log_prob, try_LL, k_guess = x_star(R, T, b, k_scan, c, basal, KD_scan, s_scan, fl)

        
        print('k: '+str(k)+' guessed k: '+ str(k_guess) + 'k error: ' + str(np.abs(k-k_guess)/k))
        KD = np.max([KD,10**-10])
        KD = np.min([KD,10**-4])
        if np.sum(R)==0:
            fitk.append(1)
            bSSE.append(1)
            rSSE.append(1)
            delta.append(0)
            log_delta.append(0)
            stds.append(np.inf)
        
        else:    
            ###########################################################################    
            fitk.append(KD)
            stds.append(KD_sigma)
            bSSE.append(np.nanmean((x-truex)**2))
            rSSE.append(np.nanmean((x_ratio-truex)**2))
            delta.append(bSSE[-1]-rSSE[-1])
            log_delta.append(np.log(bSSE[-1]+1e-15)-np.log(rSSE[-1]+1e-15))
        
        print('Fit KD: ' + str(fitk[-1]) + ' true KD: ' + str(true_K[-1]))
        print('Fit x: ' + str(bSSE[-1]) +' Naive x: '+str(rSSE[-1]))
        
        error = [(np.log(tk)-np.log(fk))/kstd for tk,fk,kstd in zip(true_K, fitk, stds)]
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        ax.hist(error, np.linspace(-10,10,40), normed=1)
        plt.ylabel('frequency')
        plt.xlabel('(true_KD - fit_KD)/error')
        plt.savefig('./fit/KD_error_dist.pdf')
        plt.close()
        
        
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        ax.loglog(bSSE, rSSE,'.')
        ax.loglog([np.min(bSSE),np.max(bSSE)],[np.min(bSSE),np.max(bSSE)])
        plt.ylabel('ratio SSE')
        plt.xlabel('poisson SSE')
        plt.text(1e-8,1e-6,' log10 poisson SSE -  log10 ratio SSE:\n '+ str(np.round(np.mean(log_delta),3)))
        plt.savefig('./fit/smooth_poiss_x_fit.pdf')
        plt.close()

        sizes = 1./(np.array(stds)+.01)+0.1
        sizes*=100/np.max(sizes)
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        ax.scatter(true_K, fitk,s=sizes)
        ax.loglog([1e-10,10**-4],[1e-10,10**-4])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('true KD')
        plt.ylabel('fit KD')
        plt.savefig('./fit/smooth_poiss_KD_fit.pdf')
        plt.close()
    
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        ax.errorbar(np.log10(np.array(true_K)), np.log10(np.array(fitk)), yerr=stds,fmt='o')
        ax.plot([-10,-4],[-10,-4])
        plt.xlabel('log true KD')
        plt.ylabel('log fit KD')
        ax.set_xlim([-10,-4])
        ax.set_ylim([-10,-4])
        plt.savefig('./fit/smooth_poiss_KD_error.pdf')
        plt.close()
    
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        H, xedges, yedges = np.histogram2d(np.log10(np.array(true_K)),np.log10(fitk),bins = (np.linspace(-10,-4, 51),np.linspace(-10, -4, 50)))
        cax= ax.pcolor(yedges[1:], xedges[1:],H)
        ax.plot([-10,-4],[-10,-4],'k')
        plt.xlabel('fit log10 KD')
        plt.ylabel('true log10 KD')
        cbar = fig.colorbar(cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('frequency', fontsize=24)
        plt.savefig('./fit/smooth_poiss_KD_hist.pdf')
        plt.close()
    
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        x = np.linspace(basal,c[-1], 1000)
        lls = np.array([get_obj[ii](x) for ii in range(fl.shape[0])])
        lls[~np.isfinite(lls)] = np.nanmax(lls)
        lls = np.exp(-np.array([(ll-np.nanmin(ll)) for ll in lls]))

        this_x = np.array(fl)
        this_x[0] = 1e-10
        cax= ax.pcolor(lls.T)
        ax.plot(np.linspace(0.5,10.5, 11),hill(fitk[-1], s, 0, fl)*1000./c[-1],'r', lw=3)
        ax.plot(np.linspace(0.5,10.5, 11),hill(true_K[-1], true_s[-1], 1, fl)*1000./c[-1],'w', lw=5)
        cbar = fig.colorbar(cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('-log likelihood', fontsize=24)
        plt.savefig('./fit/smooth_error'+str(kk)+'.png')
        plt.close()
        
        [kdkd, ss]= np.meshgrid(KD_scan, s_scan)        
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        cax= ax.pcolor(np.log10(s_scan), np.log10(KD_scan), log_prob)
        ax.scatter([np.log10(s)], [np.log10(KD)],s=50, c=[1,0,0])
        ax.scatter([np.log10(true_s[-1])], [np.log10(true_K[-1])],s=50, c=[1,1,1])
        
        plt.xlabel('s')
        plt.ylabel(r'$K_D$')
        cbar = fig.colorbar(cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('objective', fontsize=24)
        plt.savefig('./fit/smooth_log_p_dist'+str(kk)+'.png')
        plt.close()
        
        fig = plt.figure()
        ax = fig.add_axes([.18, .18, .75, .75])
        cax= ax.pcolor(np.log10(s_scan), np.log10(KD_scan), prob)
        ax.scatter([np.log10(s)], [np.log10(KD)],s=50, c=[1,0,0])
        ax.scatter([np.log10(true_s[-1])], [np.log10(true_K[-1])],s=50, c=[1,1,1])
        
        plt.xlabel('s')
        plt.ylabel(r'$K_D$')
        cbar = fig.colorbar(cax, orientation='vertical')
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('objective', fontsize=24)
        plt.savefig('./fit/smooth_p_dist'+str(kk)+'.png')
        plt.close()
        
        fid = open('./fit/simulation.dat','wb')
        pickle.dump(true_K, fid)
        pickle.dump(fitk, fid)
        pickle.dump(stds, fid)
        pickle.dump(bSSE, fid)
        pickle.dump(rSSE, fid)
        fid.close()


def main():
    """if this is run as the main script, and no arguments are passed, this performs a test of the algorithm. Additionaly, the test will output input files for the script 
    that can be used to test the script.
    if this is run as the main script, and the name of input file and output file are known, the input file must be a binary pickled file as:
    R- MXN numpy array of read counts for sequence s. M is the number antigen concentration, and N is the number of bins
    T- MXN numpy array of total read counts.
    b- coefficent relating sequence sampling rate to cell sort rate. p(x)~(x*b*T)^R * e^(-x*b*T)/R!
    k_scan- numpy array of guess for the fraction of the population that has sequence s
    c- log fluorescence value of each bin
    basal- basal fluorescence of cells (without antigen)
    KD_scan- np.array of possible KD values
    s_scan- np.array of possible number of functional antibodies
    fl- fluorescein concentrations
    optional:
    unsorted- a list of 2-tuples containing unsorted float/int read counts. The first element in a tuple is the number of reads with a sequence. The second tuple element
    is the total number of reads from the unsorted population. Example: unsorted = [(10,100000), (1,200)] 
    the output file contains
    x- the fit proportion of cells that gets sorted into bin
    KD- the fit KD
    s- the fit number of functional antibodies
    KD_sigma-inferred standard deviation of uncertainty for log KD
    k_guess- inferred fraction of the population that has the sequence s """
    if len(sys.argv)==1:
        test()
    
    if len(sys.argv) == 3:
        fid = open(sys.argv[1], 'rb')
        R = pickle.load(fid)
        T = pickle.load(fid)
        b = pickle.load(fid)
        k_scan = pickle.load(fid)
        c = pickle.load(fid)
        basal = pickle.load(fid)
        KD_scan = pickle.load(fid)
        s_scan = pickle.load(fid)
        fl = pickle.load(fid)
        if np.sum(R)==0:
            x = np.zeros(R.shape)
            KD = np.nan
            s = np.nan
            KD_sigma = np.nan
        
        else:
            try:
                unsorted = pickle.load(fid)
                fid.close()
                x, KD, s, KD_sigma, get_obj, prob, log_prob, try_LL, k_guess = x_star(R, T, b, k_scan, c, basal, KD_scan, s_scan, fl, unsorted)
            except:
                x, KD, s, KD_sigma, get_obj, prob, log_prob, try_LL, k_guess = x_star(R, T, b, k_scan, c, basal, KD_scan, s_scan, fl)
        
        fid = open(sys.argv[2], 'wb')
        pickle.dump(x, fid)
        pickle.dump(KD, fid)
        pickle.dump(s, fid)
        pickle.dump(KD_sigma, fid)
        pickle.dump(k_guess, fid)
        fid.close()


if __name__ == "__main__":
    main()


