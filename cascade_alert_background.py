#!/usr/bin/env python

import numpy as np
import healpy as hp
import argparse
from glob import glob
from astropy.time import Time
import pickle

from fast_response.FastResponseAnalysis import FastResponseAnalysis

parser = argparse.ArgumentParser(description='FRA Cascade background')
parser.add_argument('--index', type=int,default=None,
                    help='skymap index')
parser.add_argument('--deltaT', type=float, default=None,
                    help='Time Window in seconds')
parser.add_argument('--ntrials', type=int, default = 10000,
                        help='Trials')
args = parser.parse_args()

output_paths = '/data/user/apizzuto/fast_response_skylab/fast-response/fast_response/cascades_results/bg/'
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

trials_per_sig = args.ntrials

tsList_prior  = []
nsList_prior  = []
ra            = []
dec           = []

seed_counter = 0

f = FastResponseAnalysis(cascade_healpy_file, 
    start, stop, save=False, alert_event=True, smear=False, alert_type='cascade')

inj = f.initialize_injector(gamma=2.5) #just put this here to initialize f.spatial_prior
for jj in range(trials_per_sig):
    seed_counter += 1
    try:
        val = f.llh.scan(0.0, 0.0, scramble=True, seed = seed_counter,
                spatial_prior = f.spatial_prior, time_mask = [deltaT / 2., event_mjd],
                pixel_scan = [f.nside, 4.0], inject = None)
        tsList_prior.append(val['TS_spatial_prior_0'].max())
        max_prior   = np.argmax(val['TS_spatial_prior_0'])
        nsList_prior.append(val['nsignal'][max_prior])
        ra.append(val['ra'][max_prior])
        dec.append(val['dec'][max_prior])
    except ValueError:
        tsList_prior.append(0.0)
        nsList_prior.append(0.0)
        ra.append(0.0)
        dec.append(0.0)

results = {'ts_prior': tsList_prior, 
    'ns_prior': nsList_prior,
    'ra': ra, 
    'dec': dec}

with open(output_paths + 'index_{}_run_{}_event_{}_time_{}.pkl'.format(args.index, int(run_id), int(event_id), int(args.deltaT)), 'wb') as fi:
    pickle.dump(results, fi, protocol=pickle.HIGHEST_PROTOCOL)
