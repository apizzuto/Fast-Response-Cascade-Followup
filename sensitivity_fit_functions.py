import numpy as np
import scipy as sp
#from lmfit                import Model
from scipy.optimize       import curve_fit
from scipy.stats          import chi2
import pickle
from glob import glob
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import healpy as hp
import scipy.stats as st
 
palette = ['#7fc97f', '#beaed4', '#fdc086', '#ffff99', '#386cb0', '#f0027f']

def erfunc(x, a, b):
    return 0.5 + 0.5*sp.special.erf(a*x + b)

def chi2cdf(x,df1,loc,scale):
    func = chi2.cdf(x,df1,loc,scale)
    return func

def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

def incomplete_gamma(x, a, scale):
    return sp.special.gammaincc( scale*x, a)

def poissoncdf(x, mu, loc):
    func = sp.stats.poisson.cdf(x, mu, loc)
    return func

def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class CascadeCalculator():
    def __init__(self, delta_t):
        self.cascade_info = self.load_cascade_info()
        self.delta_t = delta_t
        self.base_trials = '/data/user/apizzuto/fast_response_skylab/fast-response/fast_response/cascades_results/'
        self.bg_path = self.base_trials + 'bg/'
        self.fit_path = self.base_trials + 'fits/'
        self.sens_path = self.base_trials + 'sensitivity/'

    def get_index_string(self, index):
        return f"index_{index}_run_{int(self.cascade_info['run'][index])}" \
            + f"_event_{int(self.cascade_info['event'][index])}"

    def get_trials_string(self, index):
        return self.get_index_string(index) + f"_time_{int(self.delta_t)}.pkl"

    def load_cascade_info(self):
        with open('/home/tgregoire/hese_cascades/archival_data.pkl', 'rb') as f: 
            cascade_info = pickle.load(f, encoding='latin1') 
        return cascade_info

    def background_distribution(self, index, key='ts_prior'):
        bg = np.load(self.bg_path + self.get_trials_string(index), 
            allow_pickle=True)
        return bg[key]

    def background(self, index):
        bg_trials = np.load(self.bg_path + self.get_trials_string(index), 
            allow_pickle=True)
        return bg_trials

    def n_to_flux(self, N, index):
        signal_trials = np.load(self.sens_path + self.get_trials_string(index),
            allow_pickle=True)
        fl_per_one = np.mean(np.array(signal_trials['flux']) / np.array(signal_trials['mean_ninj']))
        return fl_per_one * N

    def loc_of_map(self, index):
        return self.cascade_info['ra'][index], self.cascade_info['dec'][index]

    def signal_distribution(self, index, ns):
        signal_trials = np.load(self.sens_path + self.get_trials_string(index),
            allow_pickle=True)
        ret = {}
        msk = np.array(signal_trials['true_ns']) == ns
        for k, v in signal_trials.items():
            ret[k] = np.array(v)[msk]
        return ret

    def signal(self, index):
        signal_trials = np.load(self.sens_path + self.get_trials_string(index),
            allow_pickle=True)
        return signal_trials

    def fits(self, index):
        fit_trials = np.load(self.fit_path + self.get_trials_string(index),
            allow_pickle=True)
        return fit_trials
    
    def pass_vs_inj(self, index, threshold = 0.5, in_ns = True, with_err = True, trim=-1):
        bg_trials = np.load(self.bg_path + self.get_trials_string(index), 
            allow_pickle=True)
        bg_trials = bg_trials['ts_prior']

        signal_trials = np.load(self.sens_path + self.get_trials_string(index),
            allow_pickle=True)
        bg_thresh = np.percentile(bg_trials, threshold * 100.)
        
        signal_fluxes, signal_indices = np.unique(signal_trials['mean_ninj'], 
            return_index=True)
        signal_indices = np.append(signal_indices, len(signal_trials['ts_prior']))
        
        if trim != -1 and trim < 0:
            signal_indices = signal_indices[:trim]
            signal_fluxes = signal_fluxes[:trim]
        elif trim > 0:
            signal_indices = signal_indices[:trim + 1]
            signal_fluxes = signal_fluxes[:trim]
        
        passing = np.array([np.count_nonzero(
            signal_trials['ts_prior'][li:ri] > bg_thresh) / float(ri - li) \
            for li, ri in zip(signal_indices[:-1], signal_indices[1:])])
        if not with_err:
            return signal_fluxes, passing
        else:
            errs = np.array([np.sqrt(p*(1.-p) / float(ri - li)) \
                for p, li, ri in zip(passing, signal_indices[:-1], 
                signal_indices[1:])])
            ngen = np.array([float(ri - li) for li, ri in zip(
                signal_indices[:-1], signal_indices[1:])])
            ntrig = passing * ngen
            bound_case_pass = (ntrig + (1./3.)) / (ngen + (2./3.))
            bound_case_sigma = np.sqrt(bound_case_pass*(1. - bound_case_pass) / (ngen + 2))
            errs = np.maximum(errs, bound_case_sigma)
            return signal_fluxes, passing, errs

    def sensitivity_fit(self, signal_fluxes, passing, errs, fit_func, p0 = None, conf_lev = 0.9):
        try:
            name = fit_func.__name__
            name = name.replace("_", " ")
        except:
            name = 'fit'
        signal_scale_fac = np.max(signal_fluxes)
        signal_fls = signal_fluxes / signal_scale_fac
        popt, pcov = curve_fit(fit_func, signal_fls, 
            passing, sigma = errs, p0 = p0, maxfev=1000)
        fit_points = fit_func(signal_fls, *popt)
        chi2 = np.sum((fit_points - passing)**2. / errs**2.)
        dof = len(fit_points) - len(popt)
        xfit = np.linspace(np.min(signal_fls) - 0.5/signal_scale_fac, 
            np.max(signal_fls), 100)
        yfit = fit_func(xfit, *popt)
        pval = sp.stats.chi2.sf(chi2, dof)
        sens = xfit[find_nearest_idx(yfit, conf_lev)]*signal_scale_fac
        return {'popt': popt, 'pcov': pcov, 'chi2': chi2, 
                'dof': dof, 'xfit': xfit*signal_scale_fac, 'yfit': yfit, 
                'name': name, 'pval':pval, 'ls':'--', 'sens': sens}
    
    def sensitivity_curve(self, index, threshold = 0.5, in_ns = True, with_err = True, trim=-1, ax = None, 
                        p0 = None, fontsize = 13, conf_lev = 0.9, legend=True, text=True):
        signal_fluxes, passing, errs = self.pass_vs_inj(index, threshold=threshold, 
            in_ns=in_ns, with_err=with_err, trim=trim)
        fits, plist = [], []
        for ffunc in [erfunc, incomplete_gamma, fsigmoid]:
            try:
                fits.append(self.sensitivity_fit(signal_fluxes, 
                    passing, errs, ffunc, p0=p0, conf_lev=conf_lev))
                plist.append(fits[-1]['pval'])
            except:
                pass
        
        #Find best fit of the three, make it look different in plot
        plist = np.array(plist)
        if len(plist) > 0:
            best_fit_ind= np.argmax(plist)
            if fits[best_fit_ind]['chi2'] / fits[best_fit_ind]['dof'] < 5:
                fits[best_fit_ind]['ls'] = '-'
        
        if ax==None:
            fig, ax = plt.subplots()
        
        for fit_dict in fits:
            ax.plot(fit_dict['xfit'], fit_dict['yfit'], 
                    label = r'{}: $\chi^2$ = {:.2f}, d.o.f. = {}'.format(
                    fit_dict['name'], fit_dict['chi2'], fit_dict['dof']),
                    ls = fit_dict['ls'])
            if fit_dict['ls'] == '-':
                ax.axhline(conf_lev, color = palette[-1], linewidth = 0.3, linestyle = '-.')
                ax.axvline(fit_dict['sens'], color = palette[-1], linewidth = 0.3, 
                    linestyle = '-.')
                if text:
                    ax.text(np.median(signal_fluxes), 0.8, r'Sens. = {:.2f}'.format(fit_dict['sens']))
        if fits[best_fit_ind]['chi2'] / fits[best_fit_ind]['dof'] > 5:
            inter = np.interp(conf_lev, passing, signal_fluxes)
            ax.axhline(conf_lev, color = palette[-1], linewidth = 0.3, linestyle = '-.')
            ax.axvline(inter, color = palette[-1], linewidth = 0.3, linestyle = '-.')
        ax.errorbar(signal_fluxes, passing, yerr=errs, capsize = 3, linestyle='', 
            marker = 's', markersize = 2)
        if legend:
            ax.legend(loc=4, fontsize = fontsize)
    
    def calc_sensitivity(self, index, threshold = 0.5, in_ns = True, with_err = True, trim=-1, 
                        conf_lev = 0.9, p0=None):
        signal_fluxes, passing, errs = self.pass_vs_inj(index, threshold=threshold, 
            in_ns=in_ns, with_err=with_err, trim=trim)
        fits, plist = [], []
        for ffunc in [erfunc, incomplete_gamma, fsigmoid]:
            try:
                fits.append(self.sensitivity_fit(signal_fluxes, 
                    passing, errs, ffunc, p0=p0, conf_lev=conf_lev))
                plist.append(fits[-1]['pval'])
            except:
                pass
        #Find best fit of the three, make it look different in plot
        plist = np.array(plist)
        if len(plist) > 0:
            best_fit_ind= np.argmax(plist)
            if fits[best_fit_ind]['chi2'] / fits[best_fit_ind]['dof'] < 5:
                return fits[best_fit_ind]
        inter = np.interp(conf_lev, passing, signal_fluxes)
        return {'sens': inter, 'name': 'linear_interpolation'}

    def pvals_for_signal(self, index, ns = 1, sigma_units = False):
        bg_trials = self.background(index)['ts_prior']
        signal_trials = self.signal(index)
        pvals = [100. - sp.stats.percentileofscore(bg_trials, 
            ts, kind='strict') for ts in signal_trials['ts_prior']]
        pvals = np.array(pvals)*0.01
        pvals = np.where(pvals==0, 1e-6, pvals)
        if not sigma_units:
            return pvals
        else:
            return sp.stats.norm.ppf(1. - (pvals / 2.))

    def find_all_sens(self, with_disc=True, disc_conf=0.5, 
        disc_thresh=1.-0.0013, verbose=True):
        num_alerts = 16
        sensitivities = np.zeros(num_alerts)
        if with_disc:
            discoveries = np.zeros(num_alerts)
        for ind in range(0, 62, 4):
            if verbose:
                print(ind, end=' ')
            try:
                sens = self.n_to_flux(self.calc_sensitivity(ind)['sens'], ind)
                sensitivities[ind//4] = sens
                if with_disc:
                    disc = self.n_to_flux(self.calc_sensitivity(ind, threshold=disc_thresh, 
                            conf_lev=disc_conf)['sens'], ind)
                    discoveries[ind//4] = disc
                if sens*self.delta_t*1e6 < 1e-1:
                    if verbose:
                        print("Problem calculating sensitivity for alert index {}".format(ind))
            except (IOError, ValueError, IndexError) as err:
                print(err)
        if with_disc:
            return sensitivities, discoveries
        else:
            return sensitivities

    def ns_fits_contours(self, index, levs = [5., 25., 50., 75., 95.]):
        levs = np.array(levs)
        signal_trials = self.fits(index)
        true_inj = np.array(signal_trials['true_ns'])
        ns_fit = np.array(signal_trials['ns_prior'])
        ninjs = np.unique(true_inj)
        if max(ninjs) < 10:
            print('Index {} has max {}'.format(index, max(ninjs)))
        contours = np.stack([np.percentile(ns_fit[true_inj==ninj], levs) for ninj in ninjs])
        return ninjs, contours.T

    def ns_fits_contours_plot(self, index, levs=[5., 25., 50., 75., 95.],
                        show=False, col='navy green', custom_label = 'Median', ax=None,
                        xlabel=True, ylabel=True, legend=True):
        if ax is None:
            fig, ax = plt.subplots()
        ninj, fits = self.ns_fits_contours(index, levs=levs)
        ax.plot(ninj, fits[2], label = custom_label, color = sns.xkcd_rgb[col])
        ax.fill_between(ninj, fits[0], fits[-1], alpha=0.3,
                        label='Central 90\%', color = sns.xkcd_rgb[col], lw=0)
        ax.fill_between(ninj, fits[1], fits[-2], alpha=0.5,
                        label='Central 50\%', color = sns.xkcd_rgb[col], lw=0)
        expectation = ninj
        exp_col = 'dark grey'
        ax.plot(ninj, expectation, ls = '--', color = sns.xkcd_rgb[exp_col])
        if legend:
            ax.legend(loc=4)
        if xlabel:
            ax.set_xlabel(r'$n_{\mathrm{inj}}$')
        ax.set_xlim(0., max(ninj))
        ax.set_ylim(0., max(ninj))
        if ylabel:
            ax.set_ylabel(r'$\hat{n}_{s}$')
        if show:
            plt.show()

    def fitting_bias_summary(self, sigs=[2., 5., 10.], containment=50.):
        bias = {sig: [] for sig in sigs}; spread = {sig: [] for sig in sigs};
        levs = [50.-containment / 2., 50., 50.+containment / 2.]
        for ind in range(0, 62, 4):
            try:
                ninjs, contours = self.ns_fits_contours(ind, levs=levs)
            except:
                for sig in sigs:
                    bias[sig].append(0.0)
                    spread[sig].append(0.0)
                continue
            for sig in sigs:
                try:
                    n_ind = np.argwhere(ninjs == sig)[0][0]
                    bias[sig].append(contours[1][n_ind])
                    spread[sig].append(contours[-1][n_ind] - contours[0][n_ind])
                except:
                    bias[sig].append(0.0)
                    spread[sig].append(0.0)
        return bias, spread

    def plot_map_with_bg_scatter(self, ind):
        skymap = self.cascade_info['skymap'][ind]
        hp.mollview(skymap,
            title=f"Run {casc_calc.cascade_info['run'][ind]} " \
                    + f"Event {casc_calc.cascade_info['event'][ind]}",
            unit='Prob.')
        hp.graticule(dmer=30, dpar=30)
        bg = self.background(ind)
        zero_msk = np.asarray(bg['ts_prior']) == 0.
        non_zero_ra = np.asarray(bg['ra'])[~zero_msk]
        non_zero_dec = np.asarray(bg['dec'])[~zero_msk]
        hp.projscatter(np.pi/2. - non_zero_dec, non_zero_ra, s=5,
            marker='x', alpha=0.5, color = sns.xkcd_rgb['battleship grey'])

    def plot_zoom_from_map(self, ind, reso=1., cmap=None, draw_contour=True, ax=None, 
                        col_label= r'$\log_{10}$(prob.)'):
        s = self.cascade_info['skymap'][ind]
        nside = hp.get_nside(s)
        ra, dec = self.loc_of_map(ind)
        title = f"Run {self.cascade_info['run'][ind]}, " \
            + f"Event {self.cascade_info['event'][ind]}"
        if cmap is None:
            pdf_palette = sns.color_palette("Blues", 500)
            cmap = mpl.colors.ListedColormap(pdf_palette)
        skymap = np.log10(s)
        #min_color = np.min([0., 2.*max_color])
        max_color = max(skymap)
        min_color = max_color - 4.
        hp.gnomview(skymap, rot=(np.degrees(ra), np.degrees(dec), 0),
                        cmap=cmap,
                        max=max_color,
                        min=min_color,
                        reso=reso,
                        title=title,
                        notext=True,
                        cbar=False
                        #unit=r""
                        )

        plt.plot(4.95/3.*reso*np.radians([-1, 1, 1, -1, -1]), 4.95/3.*reso*np.radians([1, 1, -1, -1, 1]), color="k", ls="-", lw=3)
        hp.graticule(verbose=False)
        self.plot_labels(dec, ra, reso)
        self.plot_color_bar(cmap = cmap, labels = [min_color, max_color], 
            col_label = col_label)

    def plot_zoom_with_bg_scatter(self, ind, reso=20):
        self.plot_zoom_from_map(ind, reso=reso)
        bg = self.background(ind)
        zero_msk = np.asarray(bg['ts_prior']) == 0.
        non_zero_ra = np.asarray(bg['ra'])[~zero_msk]
        non_zero_dec = np.asarray(bg['dec'])[~zero_msk]
        hp.projscatter(np.pi/2. - non_zero_dec, non_zero_ra,
            marker='x', alpha=0.5, color = sns.xkcd_rgb['battleship grey'])

    def plot_labels(self, src_dec, src_ra, reso):
        """Add labels to healpy zoom"""
        fontsize = 20
        plt.text(-1*np.radians(1.75*reso),np.radians(0), r"%.2f$^{\circ}$"%(np.degrees(src_dec)),
                horizontalalignment='right',
                verticalalignment='center', fontsize=fontsize)
        plt.text(-1*np.radians(1.75*reso),np.radians(reso), r"%.2f$^{\circ}$"%(reso+np.degrees(src_dec)),
                horizontalalignment='right',
                verticalalignment='center', fontsize=fontsize)
        plt.text(-1*np.radians(1.75*reso),np.radians(-reso), r"%.2f$^{\circ}$"%(-reso+np.degrees(src_dec)),
                horizontalalignment='right',
                verticalalignment='center', fontsize=fontsize)
        plt.text(np.radians(0),np.radians(-1.75*reso), r"%.2f$^{\circ}$"%(np.degrees(src_ra)),
                horizontalalignment='center',
                verticalalignment='top', fontsize=fontsize)
        plt.text(np.radians(reso),np.radians(-1.75*reso), r"%.2f$^{\circ}$"%(-reso+np.degrees(src_ra)),
                horizontalalignment='center',
                verticalalignment='top', fontsize=fontsize)
        plt.text(np.radians(-reso),np.radians(-1.75*reso), r"%.2f$^{\circ}$"%(reso+np.degrees(src_ra)),
                horizontalalignment='center',
                verticalalignment='top', fontsize=fontsize)
        plt.text(-1*np.radians(2.4*reso), np.radians(0), r"declination",
                    ha='center', va='center', rotation=90, fontsize=fontsize)
        plt.text(np.radians(0), np.radians(-2*reso), r"right ascension",
                    ha='center', va='center', fontsize=fontsize)

    def plot_color_bar(self, labels=[0., 5e3], col_label=r'$-2 \Delta \mathrm{LLH}$',
                   range=[0,6], cmap=None, offset=-30):
        fig = plt.gcf()
        ax = fig.add_axes([0.95, 0.2, 0.03, 0.6])
        labels = ['0' if lab == 0. else '{:.1e}'.format(lab) for lab in labels]
        cb = mpl.colorbar.ColorbarBase(ax, orientation="vertical")
        cb.set_label(col_label, labelpad=offset)
        cb.set_ticks([0., 1.])
        cb.set_ticklabels(labels)
        cb.update_ticks()