#!/usr/bin/env python

import numpy as np
import healpy as hp
import argparse
from glob import glob
from astropy.time import Time
import pickle


from fast_response.FastResponseAnalysis import FastResponseAnalysis

parser = argparse.ArgumentParser(description='FRA Cascade sensitivity')
parser.add_argument('--index', type=int,default=None,
                    help='skymap index')
parser.add_argument('--deltaT', type=float, default=None,
                    help='Time Window in seconds')
parser.add_argument('--ntrials', type=int, default = 100,
                        help='Trials')
args = parser.parse_args()

output_paths = '/data/user/apizzuto/fast_response_skylab/fast-response/fast_response/cascades_results/sensitivity/'
with open('/home/tgregoire/hese_cascades/archival_data.pkl', 'rb') as f: 
        cascade_info = pickle.load(f, encoding='latin1') 

event_mjd = cascade_info['mjd'][args.index]
run_id = cascade_info['run'][args.index]
event_id = cascade_info['event'][args.index]

deltaT = args.deltaT / 86400.

start_mjd = event_mjd - (deltaT / 2.)
stop_mjd = event_mjd + (deltaT / 2.)
start = Time(start_mjd, format='mjd').iso
stop = Time(stop_mjd, format='mjd').iso

cascade_healpy_file = '/data/user/apizzuto/fast_response_skylab/' \
    + 'fast-response/fast_response/cascades_results/skymaps/IceCube-Cascade' \
    + '_{}_{}.hp'.format(int(run_id), int(event_id))

gammas = [2.5] #np.linspace(2., 3., 3)
nsigs = [1., 2., 3., 4., 6., 8., 10., 15., 20., 25., 30., 50.]
deltaT = args.deltaT / 86400.

trials_per_sig = args.ntrials

tsList_prior  = []
nsList_prior  = []
true_ns       = []
ra            = []
dec           = []
gammaList     = []
mean_ninj     = []
flux_list     = []
true_ras      = []
true_decs     = []

seed_counter = 0

for gamma in gammas:
    f = FastResponseAnalysis(cascade_healpy_file, start, stop, save=False, 
                        alert_event=True, smear=False, alert_type='cascade')
    inj = f.initialize_injector(gamma=gamma)
    scale_arr = []
    step_size = 10
    for i in range(1,20*step_size + 1, step_size):
        scale_arr.append([])
        for j in range(5):
            scale_arr[-1].append(inj.sample(i, poisson=False)[0][0])
    scale_arr = np.median(scale_arr, axis=1)
    try:
        scale_factor = np.min(np.argwhere(scale_arr > 0))*step_size + 1.
    except:
        print("Scale factor thing for prior injector didn't work")
        scale_factor = 1.
    for nsig in nsigs:
        for jj in range(trials_per_sig):
            seed_counter += 1
            ni, sample, true_ra, true_dec = inj.sample(nsig*scale_factor, poisson = True, return_position = True)
            if sample is not None:
                sample['time'] = event_mjd
            try:
                val = f.llh.scan(0.0, 0.0, scramble=True, seed = seed_counter,
                        spatial_prior = f.spatial_prior, time_mask = [deltaT / 2., event_mjd],
                        pixel_scan = [f.nside, 4.0], inject = sample)
                tsList_prior.append(val['TS_spatial_prior_0'].max())
                max_prior   = np.argmax(val['TS_spatial_prior_0'])
                nsList_prior.append(val['nsignal'][max_prior])
                true_ns.append(val['n_inj'][max_prior])
                ra.append(val['ra'][max_prior])
                dec.append(val['dec'][max_prior])
                gammaList.append(gamma)
                mean_ninj.append(nsig*scale_factor)
                flux_list.append(inj.mu2flux(nsig*scale_factor))
                true_ras.append(true_ra[0])
                true_decs.append(true_dec[0])
            except ValueError:
                tsList_prior.append(0.0)
                nsList_prior.append(0.0)
                true_ns.append(0.0)
                ra.append(0.0)
                dec.append(0.0)
                gammaList.append(gamma)
                mean_ninj.append(nsig*scale_factor)
                flux_list.append(inj.mu2flux(nsig*scale_factor))
                true_ras.append(true_ra[0])
                true_decs.append(true_dec[0])

results = {'ts_prior': tsList_prior, 'ns_prior': nsList_prior,
            'true_ns': true_ns, 'ra': ra, 'dec': dec,
            'gamma': gammaList, 'mean_ninj': mean_ninj, 'flux': flux_list,
            'true_ra': true_ras, 'true_dec': true_decs}

with open(output_paths + 'index_{}_run_{}_event_{}_time_{}.pkl'.format(args.index, run_id, event_id, args.deltaT), 'wb') as fi:
    pickle.dump(results, fi, protocol=pickle.HIGHEST_PROTOCOL)
