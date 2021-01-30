import pycondor, argparse, sys, os.path
from glob import glob
import numpy as np
import pandas as pd

error = '/scratch/apizzuto/fast_response/condor/error'
output = '/scratch/apizzuto/fast_response/condor/output'
log = '/scratch/apizzuto/fast_response/condor/log'
submit = '/scratch/apizzuto/fast_response/condor/submit'

script_base = '/data/user/apizzuto/fast_response_skylab/alert_event_followup/cascade_scripts/'

dagman_short_timescale = pycondor.Dagman(
    'FRA_cascade_short_timescale_sens', submit=submit, verbose=2
    )
dagman_long_timescale = pycondor.Dagman(
    'FRA_cascade_long_timescale_sens', submit=submit, verbose=2
    )
dagman_fits = pycondor.Dagman(
    'FRA_cascade_fitting_bias', submit=submit, verbose=2
    )

background_short_time_jobs = pycondor.Job(
    'fra_background_short_time',
    script_base + 'cascade_alert_background.py',
    error=error, output=output, log=log, submit=submit,
    getenv=True, universe='vanilla',
    verbose=2, request_memory=6000,
    request_cpus=5,
    extra_lines= ['should_transfer_files = YES', 
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman_short_timescale
	)

background_long_time_jobs = pycondor.Job(
    'fra_long_short_time',
    script_base + 'cascade_alert_background.py',
    error=error, output=output, log=log, submit=submit,
    getenv=True, universe='vanilla',
    verbose=2, request_memory=8000,
    request_cpus=5,
    extra_lines= ['should_transfer_files = YES', 
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman_long_timescale
	)

sens_short_time_jobs = pycondor.Job(
    'fra_sens_short_time',
    script_base + 'cascade_alert_sensitivity.py',
    error=error, output=output, log=log, submit=submit,
    getenv=True, universe='vanilla',
    verbose=2, request_memory=6000,
    request_cpus=5,
    extra_lines= ['should_transfer_files = YES', 
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman_short_timescale
	)

sens_long_time_jobs = pycondor.Job(
    'fra_sens_long_time',
    script_base + 'cascade_alert_sensitivity.py',
    error=error, output=output, log=log, submit=submit,
    getenv=True, universe='vanilla',
    verbose=2, request_memory=8000,
    request_cpus=5,
    extra_lines= ['should_transfer_files = YES', 
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman_long_timescale
	)

fits_jobs = pycondor.Job(
    'fra_fits_jobs',
    script_base + 'cascade_alert_fits.py',
    error=error, output=output, log=log, submit=submit,
    getenv=True, universe='vanilla',
    verbose=2, request_memory=8000,
    request_cpus=5,
    extra_lines= ['should_transfer_files = YES', 
        'when_to_transfer_output = ON_EXIT'],
    dag=dagman_fits
	)

for deltaT, bg_job, sens_job in [(1000., background_short_time_jobs, sens_short_time_jobs),
    (2.*86400., background_long_time_jobs, sens_long_time_jobs)]:
    for index in range(0, 62, 4):
        if deltaT == 2.*86400. and index in [8, 16, 36, 44, 52]:
            bg_job.add_arg(f'--index={index} --deltaT={deltaT} --ntrials=2000')
            sens_job.add_arg(f'--index={index} --deltaT={deltaT} --ntrials=100')
        if deltaT == 2.*86400. and index in [44]:
            fits_jobs.add_arg(f'--index={index} --deltaT={deltaT} --ntrials=500')
  

background_short_time_jobs.add_child(sens_short_time_jobs)
background_long_time_jobs.add_child(sens_long_time_jobs)

dagman_short_timescale.build_submit()
dagman_long_timescale.build_submit()
dagman_fits.build_submit()
