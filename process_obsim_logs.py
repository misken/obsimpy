import re
from datetime import datetime
from pathlib import Path
import math

import numpy as np
import pandas as pd
from pandas import Timestamp
from statsmodels.stats.weightstats import DescrStatsW

import obnetwork2

def num_gt_0(column):
    return (column != 0).sum()


def get_stats(group, stub=''):
    if group.sum() == 0:
        return {stub+'count': group.count(), stub+'mean': 0.0,
                stub+'min': 0.0, stub+'num_gt_0': 0,
                stub+'max': 0.0, 'stdev': 0.0, 'sem': 0.0,
                stub+'var': 0.0, 'cv': 0.0,
                stub+'skew': 0.0, 'kurt': 0.0,
                stub+'p01': 0.0, stub+'p025': 0.0,
                stub+'p05': 0.0, stub+'p25': 0.0,
                stub+'p50': 0.0, stub+'p75': 0.0,
                stub+'p90': 0.0, stub+'p95': 0.0,
                stub+'p975': 0.0, stub+'p99': 0.0}
    else:
        return {stub+'count': group.count(), stub+'mean': group.mean(),
                stub+'min': group.min(), stub+'num_gt_0': num_gt_0(group),
                stub+'max': group.max(), 'stdev': group.std(), 'sem': group.sem(),
                stub+'var': group.var(), 'cv': group.std()/group.mean(),
                stub+'skew': group.skew(), 'kurt': group.kurt(),
                stub+'p01': group.quantile(0.01), stub+'p025': group.quantile(0.025),
                stub+'p05': group.quantile(0.05), stub+'p25': group.quantile(0.25),
                stub+'p50': group.quantile(0.5), stub+'p75': group.quantile(0.75),
                stub+'p90': group.quantile(0.9), stub+'p95': group.quantile(0.95),
                stub+'p975': group.quantile(0.975), stub+'p99': group.quantile(0.99)}


def process_obsim_logs(stop_log_path, occ_stats_path, output_path, run_time, warmup=0, output_file_stem='scenario_rep_stats_summary'):

    start_analysis = warmup
    end_analysis = run_time
    num_days = (run_time - warmup) / 24.0

    rx = re.compile(r'_scenario_([0-9]{1,4})_rep_([0-9]{1,4})')

    results = []

    for log_fn in stop_log_path.glob('unit_stop_log*.csv'):
        # Get scenario and rep numbers from filename
        m = re.search(rx, str(log_fn))
        scenario_name = m.group(0)
        scenario_num = int(m.group(1))
        rep_num = int(m.group(2))
        print(scenario_name, scenario_num, rep_num)

        # Read the log file and filter by included categories
        stops_df = pd.read_csv(log_fn)

        stops_df = stops_df[(stops_df['entry_ts'] <= end_analysis) & (stops_df['exit_ts'] >= start_analysis)]

        # LOS means and sds - planned and actual
        stops_df_grp_unit = stops_df.groupby(['unit'])
        plos_mean = stops_df_grp_unit['planned_los'].mean()
        plos_sd = stops_df_grp_unit['planned_los'].std()
        plos_skew = stops_df_grp_unit['planned_los'].skew()
        plos_kurt = stops_df_grp_unit['planned_los'].apply(pd.DataFrame.kurt)

        actlos_mean = stops_df_grp_unit['exit_enter'].mean()
        actlos_sd = stops_df_grp_unit['exit_enter'].std()
        actlos_skew = stops_df_grp_unit['exit_enter'].skew()
        actlos_kurt = stops_df_grp_unit['exit_enter'].apply(pd.DataFrame.kurt)

        grp_all = stops_df.groupby(['unit'])
        grp_blocked = stops_df[(stops_df['entry_tryentry'] > 0)].groupby(['unit'])

        blocked_uncond_stats = grp_all['entry_tryentry'].apply(get_stats, 'delay_')
        blocked_cond_stats = grp_blocked['entry_tryentry'].apply(get_stats, 'delay_')

        # Create new summary record as dict
        newrec = {'scenario': scenario_num}

        newrec['rep'] = rep_num
        newrec['num_days'] = num_days

        newrec['num_visits_obs'] = blocked_uncond_stats[('OBS', 'delay_count')]
        newrec['num_visits_ldr'] = blocked_uncond_stats[('LDR', 'delay_count')]
        newrec['num_visits_pp'] = blocked_uncond_stats[('PP','delay_count')]
        newrec['num_visits_csect'] = blocked_uncond_stats[('CSECT','delay_count')]

        # OBS LOS
        newrec['planned_los_mean_obs'] = plos_mean['OBS']
        newrec['actual_los_mean_obs'] = actlos_mean['OBS']
        newrec['planned_los_sd_obs'] = plos_sd['OBS']
        newrec['actual_los_sd_obs'] = actlos_sd['OBS']

        newrec['planned_los_cv2_obs'] = (plos_sd['OBS'] / plos_mean['OBS']) ** 2
        newrec['actual_los_cv2_obs'] = (actlos_sd['OBS'] / actlos_mean['OBS']) ** 2

        newrec['planned_los_skew_obs'] = plos_skew['OBS']
        newrec['actual_los_skew_obs'] = actlos_skew['OBS']
        newrec['planned_los_kurt_obs'] = plos_kurt['OBS']
        newrec['actual_los_kurt_obs'] = actlos_kurt['OBS']

        # LDR LOS
        newrec['planned_los_mean_ldr'] = plos_mean['LDR']
        newrec['actual_los_mean_ldr'] = actlos_mean['LDR']
        newrec['planned_los_sd_ldr'] = plos_sd['LDR']
        newrec['actual_los_sd_ldr'] = actlos_sd['LDR']

        newrec['planned_los_cv2_ldr'] = (plos_sd['LDR'] / plos_mean['LDR']) ** 2
        newrec['actual_los_cv2_ldr'] = (actlos_sd['LDR'] / actlos_mean['LDR']) ** 2

        newrec['planned_los_skew_ldr'] = plos_skew['LDR']
        newrec['actual_los_skew_ldr'] = actlos_skew['LDR']
        newrec['planned_los_kurt_ldr'] = plos_kurt['LDR']
        newrec['actual_los_kurt_ldr'] = actlos_kurt['LDR']

        # PP LOS
        newrec['planned_los_mean_pp'] = plos_mean['PP']
        newrec['actual_los_mean_pp'] = actlos_mean['PP']
        newrec['planned_los_sd_pp'] = plos_sd['PP']
        newrec['actual_los_sd_pp'] = actlos_sd['PP']

        newrec['planned_los_cv2_pp'] = (plos_sd['PP'] / plos_mean['PP']) ** 2
        newrec['actual_los_cv2_pp'] = (actlos_sd['PP'] / actlos_mean['PP']) ** 2

        newrec['planned_los_skew_pp'] = plos_skew['PP']
        newrec['actual_los_skew_pp'] = actlos_skew['PP']
        newrec['planned_los_kurt_pp'] = plos_kurt['PP']
        newrec['actual_los_kurt_pp'] = actlos_kurt['PP']

        # CSECT LOS
        newrec['planned_los_mean_csect'] = plos_mean['CSECT']
        newrec['actual_los_mean_csect'] = actlos_mean['CSECT']
        newrec['planned_los_sd_csect'] = plos_sd['CSECT']
        newrec['actual_los_sd_csect'] = actlos_sd['CSECT']

        newrec['planned_los_cv2_csect'] = (plos_sd['CSECT'] / plos_mean['CSECT']) ** 2
        newrec['actual_los_cv2_csect'] = (actlos_sd['CSECT'] / actlos_mean['CSECT']) ** 2

        newrec['planned_los_skew_csect'] = plos_skew['CSECT']
        newrec['actual_los_skew_csect'] = actlos_skew['CSECT']
        newrec['planned_los_kurt_csect'] = plos_kurt['CSECT']
        newrec['actual_los_kurt_csect'] = actlos_kurt['CSECT']

        # Interarrival time stats
        arrtimes_obs = stops_df.loc[stops_df.unit == 'OBS', 'request_entry_ts']
        arrtimes_ldr = stops_df.loc[stops_df.unit == 'LDR', 'request_entry_ts']
        arrtimes_csection = stops_df.loc[stops_df.unit == 'CSECT', 'request_entry_ts']
        arrtimes_pp = stops_df.loc[stops_df.unit == 'PP', 'request_entry_ts']

        # Make sure arrival times are sorted to compute interarrival times
        arrtimes_obs.sort_values(inplace=True)
        arrtimes_ldr.sort_values(inplace=True)
        arrtimes_csection.sort_values(inplace=True)
        arrtimes_pp.sort_values(inplace=True)

        iatimes_obs = arrtimes_obs.diff(1)[1:]
        iatimes_ldr = arrtimes_ldr.diff(1)[1:]
        iatimes_csection = arrtimes_csection.diff(1)[1:]
        iatimes_pp = arrtimes_pp.diff(1)[1:]

        # IA time stats
        newrec['iatime_mean_obs'] = iatimes_obs.mean()
        newrec['iatime_sd_obs'] = iatimes_obs.std()
        newrec['iatime_skew_obs'] = iatimes_obs.skew()
        newrec['iatime_kurt_obs'] = iatimes_obs.kurtosis()

        newrec['iatime_mean_ldr'] = iatimes_ldr.mean()
        newrec['iatime_sd_ldr'] = iatimes_ldr.std()
        newrec['iatime_skew_ldr'] = iatimes_ldr.skew()
        newrec['iatime_kurt_ldr'] = iatimes_ldr.kurtosis()

        newrec['iatime_mean_csection'] = iatimes_csection.mean()
        newrec['iatime_sd_csection'] = iatimes_csection.std()
        newrec['iatime_skew_csection'] = iatimes_csection.skew()
        newrec['iatime_kurt_csection'] = iatimes_csection.kurtosis()

        newrec['iatime_mean_pp'] = iatimes_pp.mean()
        newrec['iatime_sd_pp'] = iatimes_pp.std()
        newrec['iatime_skew_pp'] = iatimes_pp.skew()
        newrec['iatime_kurt_pp'] = iatimes_pp.kurtosis()

        # Get occ from occ stats summaries
        occ_stats_fn = Path(occ_stats_path) / f"unit_occ_stats_scenario_{scenario_num}_rep_{rep_num}.csv"
        occ_stats_df = pd.read_csv(occ_stats_fn, index_col=0)

        newrec['occ_mean_obs'] = occ_stats_df.loc['OBS']['mean_occ']
        newrec['occ_stdev_obs'] = occ_stats_df.loc['OBS']['sd_occ']
        newrec['occ_p05_obs'] = occ_stats_df.loc['OBS']['p05_occ']
        newrec['occ_p25_obs'] = occ_stats_df.loc['OBS']['p25_occ']
        newrec['occ_p50_obs'] = occ_stats_df.loc['OBS']['p50_occ']
        newrec['occ_p75_obs'] = occ_stats_df.loc['OBS']['p75_occ']
        newrec['occ_p95_obs'] = occ_stats_df.loc['OBS']['p95_occ']
        newrec['occ_p99_obs'] = occ_stats_df.loc['OBS']['p99_occ']
        newrec['occ_min_obs'] = occ_stats_df.loc['OBS']['min_occ']
        newrec['occ_max_obs'] = occ_stats_df.loc['OBS']['max_occ']

        newrec['occ_mean_ldr'] = occ_stats_df.loc['LDR']['mean_occ']
        newrec['occ_stdev_ldr'] = occ_stats_df.loc['LDR']['sd_occ']
        newrec['occ_p05_ldr'] = occ_stats_df.loc['LDR']['p05_occ']
        newrec['occ_p25_ldr'] = occ_stats_df.loc['LDR']['p25_occ']
        newrec['occ_p50_ldr'] = occ_stats_df.loc['LDR']['p50_occ']
        newrec['occ_p75_ldr'] = occ_stats_df.loc['LDR']['p75_occ']
        newrec['occ_p95_ldr'] = occ_stats_df.loc['LDR']['p95_occ']
        newrec['occ_p99_ldr'] = occ_stats_df.loc['LDR']['p99_occ']
        newrec['occ_min_ldr'] = occ_stats_df.loc['LDR']['min_occ']
        newrec['occ_max_ldr'] = occ_stats_df.loc['LDR']['max_occ']        #
        
        # # PP Occupancy
        newrec['occ_mean_pp'] = occ_stats_df.loc['PP']['mean_occ']
        newrec['occ_stdev_pp'] = occ_stats_df.loc['PP']['sd_occ']
        newrec['occ_p05_pp'] = occ_stats_df.loc['PP']['p05_occ']
        newrec['occ_p25_pp'] = occ_stats_df.loc['PP']['p25_occ']
        newrec['occ_p50_pp'] = occ_stats_df.loc['PP']['p50_occ']
        newrec['occ_p75_pp'] = occ_stats_df.loc['PP']['p75_occ']
        newrec['occ_p95_pp'] = occ_stats_df.loc['PP']['p95_occ']
        newrec['occ_p99_pp'] = occ_stats_df.loc['PP']['p99_occ']
        newrec['occ_min_pp'] = occ_stats_df.loc['PP']['min_occ']
        newrec['occ_max_pp'] = occ_stats_df.loc['PP']['max_occ']

        newrec['occ_mean_csect'] = occ_stats_df.loc['CSECT']['mean_occ']
        newrec['occ_stdev_csect'] = occ_stats_df.loc['CSECT']['sd_occ']
        newrec['occ_p05_csect'] = occ_stats_df.loc['CSECT']['p05_occ']
        newrec['occ_p25_csect'] = occ_stats_df.loc['CSECT']['p25_occ']
        newrec['occ_p50_csect'] = occ_stats_df.loc['CSECT']['p50_occ']
        newrec['occ_p75_csect'] = occ_stats_df.loc['CSECT']['p75_occ']
        newrec['occ_p95_csect'] = occ_stats_df.loc['CSECT']['p95_occ']
        newrec['occ_p99_csect'] = occ_stats_df.loc['CSECT']['p99_occ']
        newrec['occ_min_csect'] = occ_stats_df.loc['CSECT']['min_occ']
        newrec['occ_max_csect'] = occ_stats_df.loc['CSECT']['max_occ']


        newrec['pct_waitq_ldr'] = blocked_uncond_stats[('LDR', 'delay_num_gt_0')]/blocked_uncond_stats[('LDR', 'delay_count')]

        if ('LDR', 'delay_mean') in blocked_cond_stats.index:
            newrec['waitq_ldr_mean'] = blocked_cond_stats[('LDR', 'delay_mean')]
            newrec['waitq_ldr_p95'] = blocked_cond_stats[('LDR', 'delay_p95')]
        else:
            newrec['waitq_ldr_mean'] = 0.0
            newrec['waitq_ldr_p95'] = 0.0

        newrec['pct_blocked_by_pp'] = blocked_uncond_stats[('PP', 'delay_num_gt_0')]/blocked_uncond_stats[('PP', 'delay_count')]

        if ('PP', 'delay_mean') in blocked_cond_stats.index:
            newrec['blocked_by_pp_mean'] = blocked_cond_stats[('PP', 'delay_mean')]
            newrec['blocked_by_pp_p95'] = blocked_cond_stats[('PP', 'delay_p95')]
        else:
            newrec['blocked_by_pp_mean'] = 0.0
            newrec['blocked_by_pp_p95'] = 0.0

        newrec['timestamp'] = str(datetime.now())

        print(newrec)

        results.append(newrec)

        json_stats_path = output_path / 'json'
        json_stats_path.mkdir(exist_ok=True)

        output_json_file = json_stats_path / f'output_stats_scenario_{scenario_num}_rep_{rep_num}.json'
        with open(output_json_file, 'a') as json_output:
            json_output.write(str(newrec)+'\n')


    results_df = pd.DataFrame(results)
    cols = ["scenario","rep","timestamp", "num_days",
            "num_visits_obs", "num_visits_ldr", "num_visits_pp", "num_visits_csect",
            "planned_los_mean_obs", "actual_los_mean_obs", "planned_los_sd_obs", "actual_los_sd_obs",
            "planned_los_cv2_obs", "actual_los_cv2_obs",
            "planned_los_skew_obs", "actual_los_skew_obs", "planned_los_kurt_obs", "actual_los_kurt_obs",
            "planned_los_mean_ldr", "actual_los_mean_ldr", "planned_los_sd_ldr", "actual_los_sd_ldr",
            "planned_los_cv2_ldr", "actual_los_cv2_ldr",
            "planned_los_skew_ldr", "actual_los_skew_ldr", "planned_los_kurt_ldr", "actual_los_kurt_ldr",
            "planned_los_mean_pp", "actual_los_mean_pp", "planned_los_sd_pp", "actual_los_sd_pp",
            "planned_los_cv2_pp", "actual_los_cv2_pp",
            "planned_los_skew_pp", "actual_los_skew_pp", "planned_los_kurt_pp", "actual_los_kurt_pp",
            "planned_los_mean_csect", "actual_los_mean_csect", "planned_los_sd_csect", "actual_los_sd_csect",
            "planned_los_cv2_csect", "actual_los_cv2_csect",
            "planned_los_skew_csect", "actual_los_skew_csect", "planned_los_kurt_csect", "actual_los_kurt_csect",
            "iatime_mean_obs", "iatime_sd_obs", "iatime_skew_obs", "iatime_kurt_obs",
            "iatime_mean_ldr", "iatime_sd_ldr", "iatime_skew_ldr", "iatime_kurt_ldr",
            "iatime_mean_csection", "iatime_sd_csection", "iatime_skew_csection", "iatime_kurt_csection",
            "iatime_mean_pp", "iatime_sd_pp", "iatime_skew_pp", "iatime_kurt_pp",
            "occ_mean_obs", "occ_mean_ldr", "occ_mean_csect", "occ_mean_pp",
            "occ_p95_obs", "occ_p95_ldr", "occ_p95_csect", "occ_p95_pp",
            "pct_waitq_ldr", "waitq_ldr_mean", "waitq_ldr_p95",
            "pct_blocked_by_pp", "blocked_by_pp_mean", "blocked_by_pp_p95"]

    output_csv_file = output_path / f'{output_file_stem}.csv'
    results_df = results_df[cols]
    results_df = results_df.sort_values(by=['scenario', 'rep'])
    results_df.to_csv(output_csv_file, index=False)


def compute_occ_stats(units=['OBS', 'LDR', 'CSECT', 'PP'],
                      occ_log_path=None, occ_stats_path=None,
                      warmup=0,
                      quantiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99]):



    rx = re.compile(r'_scenario_([0-9]{1,4})_rep_([0-9]{1,4})')

    results = []

    for log_fn in occ_log_path.glob('unit_occ_log*.csv'):
        # Get scenario and rep numbers from filename
        m = re.search(rx, str(log_fn))
        scenario_name = m.group(0)
        scenario_num = m.group(1)
        rep_num = m.group(2)
        print(scenario_name, scenario_num, rep_num)

        occ_stats_file = Path(occ_stats_path) / f"unit_occ_stats_scenario_{scenario_num}_rep_{rep_num}.csv"

        # Read the log file and filter by included categories
        df = pd.read_csv(log_fn)

        occ_stats_dfs = []
        # Filter out recs before warmup
        for unit in units:
            unit_df = df[(df['timestamp'] > warmup) & (df['unit'] == unit)]

            weighted_stats = DescrStatsW(unit_df['occ'], weights=unit_df['occ_weight'], ddof=0)

            occ_quantiles = weighted_stats.quantile(quantiles)
            occ_stats_df = pd.DataFrame([{'unit': unit,
                                          'mean_occ': weighted_stats.mean, 'sd_occ': weighted_stats.std,
                                          'min_occ': unit_df['occ'].min(), 'max_occ': unit_df['occ'].max()}])

            quantiles_df = pd.DataFrame(occ_quantiles).transpose()
            quantiles_df.rename(columns = lambda x: f"p{100 * x:02.0f}_occ", inplace=True)

            occ_stats_df = pd.concat([occ_stats_df, quantiles_df], axis=1)
            occ_stats_dfs.append(occ_stats_df)


        occ_stats_df =  pd.concat(occ_stats_dfs)
        occ_stats_df.to_csv(occ_stats_file, index=False)
        header = output_header("Occupancy stats", 50, scenario_num, rep_num)
        print(header)
        print(occ_stats_df)


def output_header(msg, linelen, scenario, rep_num):
    header = f"\n{msg} (scenario={scenario} rep={rep_num})\n{'-' * linelen}\n"
    return header


def qng_approx(scenario_inputs_summary):
    results = []

    for row in scenario_inputs_summary.iterrows():
        scenario = row[1]['scenario']
        arr_rate = row[1]['arrival_rate']
        c_sect_prob = row[1]['c_sect_prob']
        ldr_mean_svctime = row[1]['mean_los_ldr']
        ldr_cv2_svctime = 1 / row[1]['num_erlang_stages_ldr']
        ldr_cap = row[1]['cap_ldr']
        pp_mean_svctime = c_sect_prob * row[1]['mean_los_pp_c'] + (1 - c_sect_prob) * row[1]['mean_los_pp_noc']

        rates = [1 / row[1]['mean_los_pp_c'], 1 / row[1]['mean_los_pp_noc']]
        probs = [c_sect_prob, 1 - c_sect_prob]
        stages = [int(row[1]['num_erlang_stages_pp']), int(row[1]['num_erlang_stages_pp'])]
        moments = [hyper_erlang_moment(rates, stages, probs, moment) for moment in [1, 2]]
        variance = moments[1] - moments[0] ** 2
        cv2 = variance / moments[0] ** 2

        pp_cv2_svctime = cv2

        pp_cap = row[1]['cap_pp']
        sim_mean_waitq_ldr_mean = row[1]['mean_waitq_ldr_mean']
        sim_mean_pct_waitq_ldr = row[1]['mean_pct_waitq_ldr']
        sim_actual_los_mean_mean_ldr = row[1]['actual_los_mean_mean_ldr']
        sim_mean_pct_blocked_by_pp = row[1]['mean_pct_blocked_by_pp']
        sim_mean_blocked_by_pp_mean = row[1]['mean_blocked_by_pp_mean']

        ldr_pct_blockedby_pp = obnetwork2.ldr_prob_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap, pp_cv2_svctime)
        ldr_meantime_blockedby_pp = obnetwork2.ldr_condmeantime_blockedby_pp_hat(arr_rate, pp_mean_svctime, pp_cap,
                                                                                 pp_cv2_svctime)
        (obs_meantime_blockedbyldr, ldr_effmean_svctime, obs_prob_blockedby_ldr, obs_condmeantime_blockedbyldr) = \
            obnetwork2.obs_blockedby_ldr_hats(arr_rate, c_sect_prob, ldr_mean_svctime, ldr_cv2_svctime, ldr_cap,
                                              pp_mean_svctime, pp_cv2_svctime, pp_cap)

        scen_results = {'scenario': scenario,
                        'arr_rate': arr_rate,
                        'prob_blockedby_ldr_approx': obs_prob_blockedby_ldr,
                        'prob_blockedby_ldr_sim': sim_mean_pct_waitq_ldr,

                        'condmeantime_blockedbyldr_approx': obs_condmeantime_blockedbyldr,
                        'condmeantime_blockedbyldr_sim': sim_mean_waitq_ldr_mean,
                        'ldr_effmean_svctime_approx': ldr_effmean_svctime,
                        'ldr_effmean_svctime_sim': sim_actual_los_mean_mean_ldr,
                        'prob_blockedby_pp_approx': ldr_pct_blockedby_pp,
                        'prob_blockedby_pp_sim': sim_mean_pct_blocked_by_pp,
                        'condmeantime_blockedbypp_approx': ldr_meantime_blockedby_pp,
                        'condmeantime_blockedbypp_sim': sim_mean_blocked_by_pp_mean}

        results.append(scen_results)

        # print("scenario {}\n".format(scenario))
        # print(results)

    results_df = pd.DataFrame(results)
    return results_df


def aggregate_over_reps(scenario_rep_summary_path, scenario_summary_path):
    """Compute summary stats by scenario (aggregating over replications)"""

    output_stats_summary_df = pd.read_csv(scenario_rep_summary_path).sort_values(by=['scenario', 'rep'])

    output_stats_summary_agg_df = output_stats_summary_df.groupby(['scenario']).agg(
        num_visits_obs_mean=pd.NamedAgg(column='num_visits_obs', aggfunc='mean'),
        num_visits_ldr_mean=pd.NamedAgg(column='num_visits_ldr', aggfunc='mean'),
        num_visits_csect_mean=pd.NamedAgg(column='num_visits_csect', aggfunc='mean'),
        num_visits_pp_mean=pd.NamedAgg(column='num_visits_pp', aggfunc='mean'),

        planned_los_mean_mean_obs=pd.NamedAgg(column='planned_los_mean_obs', aggfunc='mean'),
        planned_los_mean_mean_ldr=pd.NamedAgg(column='planned_los_mean_ldr', aggfunc='mean'),
        planned_los_mean_mean_csect=pd.NamedAgg(column='planned_los_mean_csect', aggfunc='mean'),
        planned_los_mean_mean_pp=pd.NamedAgg(column='planned_los_mean_pp', aggfunc='mean'),

        actual_los_mean_mean_obs=pd.NamedAgg(column='actual_los_mean_obs', aggfunc='mean'),
        actual_los_mean_mean_ldr=pd.NamedAgg(column='actual_los_mean_ldr', aggfunc='mean'),
        actual_los_mean_mean_csect=pd.NamedAgg(column='actual_los_mean_csect', aggfunc='mean'),
        actual_los_mean_mean_pp=pd.NamedAgg(column='actual_los_mean_pp', aggfunc='mean'),

        planned_los_mean_cv2_obs=pd.NamedAgg(column='planned_los_cv2_obs', aggfunc='mean'),
        planned_los_mean_cv2_ldr=pd.NamedAgg(column='planned_los_cv2_ldr', aggfunc='mean'),
        planned_los_mean_cv2_csect=pd.NamedAgg(column='planned_los_cv2_csect', aggfunc='mean'),
        planned_los_mean_cv2_pp=pd.NamedAgg(column='planned_los_cv2_pp', aggfunc='mean'),

        actual_los_mean_cv2_obs=pd.NamedAgg(column='actual_los_cv2_obs', aggfunc='mean'),
        actual_los_mean_cv2_ldr=pd.NamedAgg(column='actual_los_cv2_ldr', aggfunc='mean'),
        actual_los_mean_cv2_csect=pd.NamedAgg(column='actual_los_cv2_csect', aggfunc='mean'),
        actual_los_mean_cv2_pp=pd.NamedAgg(column='actual_los_cv2_pp', aggfunc='mean'),

        occ_mean_mean_obs=pd.NamedAgg(column='occ_mean_obs', aggfunc='mean'),
        occ_mean_mean_ldr=pd.NamedAgg(column='occ_mean_ldr', aggfunc='mean'),
        occ_mean_mean_csect=pd.NamedAgg(column='occ_mean_csect', aggfunc='mean'),
        occ_mean_mean_pp=pd.NamedAgg(column='occ_mean_pp', aggfunc='mean'),

        occ_mean_p95_obs=pd.NamedAgg(column='occ_p95_obs', aggfunc='mean'),
        occ_mean_p95_ldr=pd.NamedAgg(column='occ_p95_ldr', aggfunc='mean'),
        occ_mean_p95_csect=pd.NamedAgg(column='occ_p95_csect', aggfunc='mean'),
        occ_mean_p95_pp=pd.NamedAgg(column='occ_p95_pp', aggfunc='mean'),

        mean_pct_waitq_ldr=pd.NamedAgg(column='pct_waitq_ldr', aggfunc='mean'),
        mean_waitq_ldr_mean=pd.NamedAgg(column='waitq_ldr_mean', aggfunc='mean'),
        mean_waitq_ldr_p95=pd.NamedAgg(column='waitq_ldr_p95', aggfunc='mean'),

        mean_pct_blocked_by_pp=pd.NamedAgg(column='pct_waitq_ldr', aggfunc='mean'),
        mean_blocked_by_pp_mean=pd.NamedAgg(column='blocked_by_pp_mean', aggfunc='mean'),
        mean_blocked_by_pp_p95=pd.NamedAgg(column='blocked_by_pp_p95', aggfunc='mean'),
    )

    output_stats_summary_agg_df.to_csv(scenario_summary_path, index=True)


def hyper_erlang_moment(rates, stages, probs, moment):
    terms = [probs[i - 1] * math.factorial(stages[i - 1] + moment - 1) * (1 / math.factorial(stages[i - 1] - 1)) * (
                stages[i - 1] * rates[i - 1]) ** (-moment)
             for i in range(1, len(rates) + 1)]

    return sum(terms)


if __name__ == '__main__':

    scenario_inputs_path = Path('./input/exp10_tandem05_metainputs.csv')
    stop_log_path = Path('./logs')
    occ_log_path = Path('./logs')
    occ_stats_path = Path('./stats_warmup')
    output_path = Path('./output')
    scenario_rep_summary_stem = 'scenario_rep_stats_summary'
    scenario_summary_stem = 'scenario_stats_summary'
    scenario_inputs_summary_qng_stem = 'scenario_inputs_summary_qng'
    run_time = 60000
    warmup_time = 4800

    b_compute_occ_stats = False
    b_process_obsim_logs = False


    # From occupancy logs, compute occupancy summary stats
    if b_compute_occ_stats:
        compute_occ_stats(units=['OBS', 'LDR', 'CSECT', 'PP'],
                                occ_log_path=occ_log_path, occ_stats_path=occ_stats_path,
                                warmup=warmup_time, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

    if b_process_obsim_logs:
    # From the patient stop logs, compute summary stats by scenario by replication
        process_obsim_logs(stop_log_path, occ_stats_path, output_path, run_time, warmup=warmup_time,
                           output_file_stem=scenario_rep_summary_stem)

    # Compute summary stats by scenario (aggregating over the replications)
    scenario_rep_summary_path = output_path / f"{scenario_rep_summary_stem}.csv"
    scenario_summary_path = output_path / f"{scenario_summary_stem}.csv"
    aggregate_over_reps(scenario_rep_summary_path, scenario_summary_path)
    scenario_summary_df = pd.read_csv(scenario_summary_path)

    # Merge the scenario summary with the scenario inputs
    scenario_inputs_df = pd.read_csv(scenario_inputs_path)
    scenario_inputs_summary_df = scenario_inputs_df.merge(scenario_summary_df, on=['scenario'])

    # Using the scenario summary we just created
    qng_approx_df = qng_approx(scenario_inputs_summary_df)

    scenario_inputs_summary_qng_df = scenario_inputs_summary_df.merge(qng_approx_df, on=['scenario'])
    scenario_inputs_summary_qng_path = output_path / f"{scenario_inputs_summary_qng_stem}.csv"
    scenario_inputs_summary_qng_df.to_csv(scenario_inputs_summary_qng_path, index=False)