from pathlib import Path

import pandas as pd
import numpy as np
import yaml


def config_from_csv(scenarios_csv_path_, settings_path_, config_path_, bat_path_):

    # Read scenarios file in DataFrame
    scenarios_df = pd.read_csv(scenarios_csv_path_)
    # Read settings file
    with open(settings_path_, 'rt') as settings_file:
        settings = yaml.safe_load(settings_file)
        print(settings)

    global_vars = {}
    with open(bat_path_, 'w') as bat_file:
    # Iterate over rows in scenarios file
        for row in scenarios_df.iterrows():
            scenario = int(row[1]['scenario'].tolist())

            global_vars['arrival_rate'] = row[1]['arrival_rate'].tolist()

            global_vars['mean_los_obs'] = row[1]['mean_los_obs'].tolist()
            global_vars['num_erlang_stages_obs'] = int(row[1]['num_erlang_stages_obs'])

            global_vars['mean_los_ldr'] = float(row[1]['mean_los_ldr'])
            global_vars['num_erlang_stages_ldr'] = int(row[1]['num_erlang_stages_ldr'])

            global_vars['mean_los_pp_noc'] = float(row[1]['mean_los_pp_noc'])
            global_vars['mean_los_pp_c'] = float(row[1]['mean_los_pp_c'])
            global_vars['num_erlang_stages_pp'] = int(row[1]['num_erlang_stages_pp'])

            global_vars['mean_los_csect'] = float(row[1]['mean_los_csect'])
            global_vars['num_erlang_stages_csect'] = int(row[1]['num_erlang_stages_csect'])

            global_vars['c_sect_prob'] = float(row[1]['c_sect_prob'])

            config = {}
            config['locations'] = settings['locations']
            cap_obs = int(row[1]['cap_obs'].tolist())
            cap_ldr = int(row[1]['cap_ldr'].tolist())
            cap_pp = int(row[1]['cap_pp'].tolist())
            config['locations'][1]['capacity'] = cap_obs
            config['locations'][2]['capacity'] = cap_ldr
            config['locations'][4]['capacity'] = cap_pp


            # Write scenario config file

            config['scenario'] = scenario
            config['run_settings'] = settings['run_settings']
            config['paths'] = settings['paths']
            config['random_number_streams'] = settings['random_number_streams']

            config['routes'] = settings['routes']
            config['global_vars'] = global_vars

            config_file_path = Path(config_path_) / f'scenario_{scenario}.yaml'

            with open(config_file_path, 'w', encoding='utf-8') as config_file:
                yaml.dump(config, config_file)

            run_line = f"python obflow_6.py {config_file_path} --loglevel=WARNING\n"
            bat_file.write(run_line)


if __name__ == '__main__':
    scenarios_csv_path = Path('input/exp10_obflow06_metainputs.csv')
    settings_path = Path('input/exp11_obflow06_settings.yaml')
    config_path = Path('input/config/exp11')
    bat_path = Path('./run') / 'exp11_obflow06_run.sh'

    config_from_csv(scenarios_csv_path, settings_path, config_path, bat_path)