import os
import sys
import argparse
import random
import yaml
import numpy as np
import pandas as pd
from pyswmm import Simulation, Nodes, Links

# ── 1. Constants & Default Config ─────────────────────────────────────────────
THRESHOLD      = 5.0    # mg/L detection threshold
CARRIER_FLOW   = 0.005  # 5 L/s baseline inflow (m3/s)
CFS_TO_M3S     = 0.0283168

def load_config(config_path="config/default.yaml"):
    if not os.path.exists(config_path):
        return None
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# ── 2. Helper functions ───────────────────────────────────────────────────────
def format_time(hrs):
    return f"{hrs:.3f}"

# ── 3. Feature Engineering (Topology) ─────────────────────────────────────────
def build_topology_features(inp_path, high_risk_nodes=None):
    import networkx as nx
    G = nx.DiGraph()
    outfalls = []
    high_risk_nodes = set(high_risk_nodes or [])
    
    with open(inp_path) as f:
        section = None
        for line in f:
            s = line.strip().upper()
            if not s or s.startswith(';'): continue
            if s.startswith('['):
                section = s.strip('[]')
                continue
            parts = line.split()
            if section == 'CONDUITS' and len(parts) >= 3:
                G.add_edge(parts[1], parts[2])
            if section == 'OUTFALLS' and len(parts) >= 1:
                outfalls.append(parts[0])

    topo_depth = {}
    nodes = list(G.nodes())
    for node in nodes:
        min_d = 999
        for out in outfalls:
            try:
                d = nx.shortest_path_length(G, node, out)
                if d < min_d: min_d = d
            except: pass
        topo_depth[node] = min_d if min_d != 999 else 0

    rows = []
    for node in nodes:
        n_up = len(nx.ancestors(G, node))
        nt = 0
        if node.startswith('JI'): nt = 1
        elif 'AUX' in node.upper(): nt = 2
        elif node in outfalls: nt = 3

        rows.append({
            'node_id':         node,
            'topo_depth':      topo_depth[node],
            'n_upstream':      n_up,
            'node_type_code':  nt,
            'is_high_risk':    1 if node in high_risk_nodes else 0,
            'prior_contam_prob': 2.0 if node in high_risk_nodes else 1.0
        })
    
    df = pd.DataFrame(rows).set_index('node_id')
    df['prior_contam_prob'] /= df['prior_contam_prob'].sum()
    
    return df, topo_depth

# ── 4. Scenario Builder ───────────────────────────────────────────────────────
def build_scenario_inp(base_inp, tmp_inp, src, mass_kg, duration_hrs, start_offset_hrs=1.0, carrier_flow=0.005):
    vol_m3      = carrier_flow * (duration_hrs * 3600)
    conc_mg_l   = (mass_kg * 1e6) / vol_m3
    
    start_hrs = start_offset_hrs
    end_hrs   = start_offset_hrs + duration_hrs
    post_hrs  = end_hrs + 0.01

    def ts(val):
        return [
            (format_time(0.0),       0.0),
            (format_time(start_hrs), val),
            (format_time(end_hrs),   val),
            (format_time(post_hrs),  0.0),
        ]

    ts_flow = f'F_{src}'
    ts_conc = f'C_{src}'

    with open(base_inp) as f:
        content = f.read()

    # Handle external files (Record.dat, Site-Post.jpg)
    base_dir = os.path.dirname(os.path.abspath(base_inp))
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if line.strip() and not line.startswith('[') and not line.startswith(';'):
            parts = line.split()
            if len(parts) >= 2:
                p_upper = [p.upper() for p in parts]
                f_idx = -1
                if 'FILE' in p_upper: f_idx = p_upper.index('FILE')
                if f_idx != -1 and f_idx + 1 < len(parts):
                    fname_raw = parts[f_idx + 1]
                    fname = fname_raw.strip('"')
                    if not os.path.isabs(fname):
                        abs_fname = os.path.join(base_dir, fname)
                        line = line.replace(fname_raw, f'"{abs_fname}"')
        new_lines.append(line)
    content = '\n'.join(new_lines)

    # 1. Inject [POLLUTANTS]
    if '[POLLUTANTS]' not in content:
        pb = ('[POLLUTANTS]\n'
              ';;Name           Units  Crain      Cgw        Crdii      Kdecay     SnowOnly   Co-Pollut        Co-Fraction      DWF-Concen       Pipe-Exfil\n'
              'CONTAM           MG/L   0.0        0.0        0.0        0.0        NO         *                0.0              0.0              0.0\n\n')
        if '[JUNCTIONS]' in content: content = content.replace('[JUNCTIONS]', pb + '[JUNCTIONS]')
        else: content = pb + content

    # 2. Inject [TIMESERIES]
    ts_block = f';; Injection for {src}\n'
    for t, v in ts(carrier_flow): ts_block += f'{ts_flow:<16}{" ":<12}{t:<12}{v:<12}\n'
    for t, v in ts(conc_mg_l):   ts_block += f'{ts_conc:<16}{" ":<12}{t:<12}{v:<12}\n'

    if '[TIMESERIES]' in content:
        parts = content.split('[TIMESERIES]', 1)
        post_ts = parts[1].split('[', 1)
        if len(post_ts) > 1:
            content = parts[0] + '[TIMESERIES]\n' + ts_block + post_ts[0] + '[' + post_ts[1]
        else:
            content = parts[0] + '[TIMESERIES]\n' + ts_block + parts[1]
    else:
        content += f"\n[TIMESERIES]\n{ts_block}\n"

    # 3. Inject [INFLOWS]
    inflow_entry = (f'{src:<17}FLOW   {ts_flow:<20}DIRECT  1.0  1.0\n'
                    f'{src:<17}CONTAM {ts_conc:<20}CONCEN  1.0  1.0\n')
    if '[INFLOWS]' in content:
        parts = content.split('[INFLOWS]', 1)
        post_in = parts[1].split('[', 1)
        if len(post_in) > 1:
            content = parts[0] + '[INFLOWS]\n' + inflow_entry + post_in[0] + '[' + post_in[1]
        else:
            content = parts[0] + '[INFLOWS]\n' + inflow_entry + parts[1]
    else:
        content += f"\n[INFLOWS]\n{inflow_entry}\n"

    with open(tmp_inp, 'w') as f: f.write(content)

# ── 5. Run one scenario ────────────────────────────────────────────────────────
def run_scenario(tmp_inp, node_ids, threshold=5.0):
    results = {}
    try:
        with Simulation(tmp_inp) as sim:
            nodes_obj = Nodes(sim)
            links_obj = Links(sim)
            
            # Map nodes to connected links for velocity
            node_to_links = {nid: [] for nid in node_ids}
            for link in links_obj:
                lid = link.linkid
                if link.inlet_node in node_to_links: node_to_links[link.inlet_node].append(lid)
                if link.outlet_node in node_to_links: node_to_links[link.outlet_node].append(lid)

            peaks = {nid: 0.0 for nid in node_ids}
            peak_step = {nid: 0 for nid in node_ids}
            inflow_sum = {nid: 0.0 for nid in node_ids}
            vel_sum = {nid: 0.0 for nid in node_ids}
            step_n = 0
            
            for _ in sim:
                step_n += 1
                for nid in node_ids:
                    node = nodes_obj[nid]
                    c = node.pollut_quality['CONTAM']
                    if c > peaks[nid]:
                        peaks[nid] = c
                        peak_step[nid] = step_n
                    inflow_sum[nid] += abs(node.total_inflow)
                    
                    # Avg velocity of connected links (V = Q/A)
                    lids = node_to_links[nid]
                    if lids:
                        v_total = 0.0
                        for lid in lids:
                            link = links_obj[lid]
                            try:
                                q = abs(link.flow)
                                # Try to get area; if not available, use 1.0 (or 0)
                                a = getattr(link, 'ups_xsection_area', 0.0) 
                                if a <= 0: a = getattr(link, 'ds_xsection_area', 0.0)
                                if a > 0.0001:
                                    v_total += q / a
                            except: pass
                        vel_sum[nid] += (v_total / len(lids))
            
            for nid in node_ids:
                p = peaks[nid]
                # Convert ft/s to m/s (0.3048) if model is in US units
                # (Example 8/9 are US, CFS)
                v_ms = (vel_sum[nid] / max(step_n, 1)) * 0.3048
                results[nid] = {
                    'peak_conc': round(p, 4),
                    't_peak_min': round(peak_step[nid] * 1.0, 2) if p > 0 else None,
                    'mean_flow_m3s': round((inflow_sum[nid] / max(step_n, 1)) * CFS_TO_M3S, 6),
                    'mean_vel_ms': round(v_ms, 6),
                    'detected': 1 if p >= threshold else 0
                }
    except Exception as e:
        print(f"    WARNING: simulation failed -- {e}")
        for nid in node_ids:
            results[nid] = {'peak_conc': 0.0, 't_peak_min': None, 'mean_flow_m3s': 0.0, 'mean_vel_ms': 0.0, 'detected': 0}
    return results

# ── 6. Main ───────────────────────────────────────────────────────────────────
def main():
    config = load_config()
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',  default=config['dataset']['model_path'] if config else './dataset/Examples/Example8.inp')
    p.add_argument('--n_scenarios', type=int, default=config['dataset']['n_scenarios'] if config else 100)
    p.add_argument('--output_dir',  default=config['dataset']['output_dir'] if config else './output')
    p.add_argument('--seed',        type=int, default=config['dataset']['seed'] if config else 42)
    a = p.parse_args()

    # Load extended settings from config
    threshold = config['dataset'].get('threshold', 5.0) if config else 5.0
    carrier_flow = config['dataset'].get('carrier_flow', 0.005) if config else 0.005
    mass_range = config['dataset'].get('mass_range', [0.01, 0.5]) if config else [0.01, 0.5]
    dur_range = config['dataset'].get('duration_range', [0.25, 3.0]) if config else [0.25, 3.0]
    start_range = config['dataset'].get('start_range', [0.0, 6.0]) if config else [0.0, 6.0]
    high_risk = config['dataset'].get('high_risk_nodes', []) if config else []
    exclude = config['dataset'].get('exclude_sources', []) if config else []
    sampling_strategy = config['dataset'].get('sampling_strategy', 'weighted') if config else 'weighted'

    random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.output_dir, exist_ok=True)

    print("=" * 60)
    print("SWMM Dataset Generator  --  Mhango & Sambito (2026)")
    print("=" * 60)

    topo_df, topo_depth = build_topology_features(a.model_path, high_risk)
    node_ids = list(topo_df.index)
    candidate_sources = [n for n in node_ids if n not in exclude]
    
    print(f"\n[1/5] Network: {os.path.basename(a.model_path)}")
    print(f"      Nodes: {len(node_ids)}  |  Candidate sources: {len(candidate_sources)}")
    print(f"      Sampling: {sampling_strategy}")

    raw_rows = []
    tmp_inp  = "_scenario_tmp.inp"

    if sampling_strategy == 'uniform':
        source_probs = np.ones(len(candidate_sources)) / len(candidate_sources)
    else:
        source_probs = [topo_df.loc[n, 'prior_contam_prob'] for n in candidate_sources]
        source_probs = np.array(source_probs) / sum(source_probs)

    for i in range(a.n_scenarios):
        src = np.random.choice(candidate_sources, p=source_probs)
        mass = round(random.uniform(mass_range[0], mass_range[1]), 3)
        dur  = round(random.uniform(dur_range[0], dur_range[1]), 2)
        offset = round(random.uniform(start_range[0], start_range[1]), 2)
        
        if (i+1) % 10 == 0 or i == 0 or i == a.n_scenarios-1:
            print(f"  Scenario {i+1:>5}/{a.n_scenarios}  src={src:<8}  mass={mass}kg  dur={dur}h  start={offset}h", end='\r')

        build_scenario_inp(a.model_path, tmp_inp, src, mass, dur, offset, carrier_flow)
        res = run_scenario(tmp_inp, node_ids, threshold)

        for nid, data in res.items():
            raw_rows.append({
                'scen_id': f"{i+1:04d}", 'src_node': src, 'node_id': nid,
                'topo_depth': topo_depth.get(nid, 0),
                'peak_conc': data['peak_conc'], 't_peak_min': data['t_peak_min'],
                'mean_flow_m3s': data['mean_flow_m3s'], 'mean_vel_ms': data['mean_vel_ms'],
                'detected': data['detected']
            })

    # Final cleanup of temp file
    import time
    if os.path.exists(tmp_inp):
        for _ in range(5):
            try: os.remove(tmp_inp); break
            except: time.sleep(0.1)

    print(f"\n  Done. Total rows: {len(raw_rows):,}")

    raw_df = pd.DataFrame(raw_rows)
    raw_df['t_peak_min'] = pd.to_numeric(raw_df['t_peak_min'], errors='coerce')
    raw_path = os.path.join(a.output_dir, 'raw_scenarios.csv')
    raw_df.to_csv(raw_path, index=False)

    print("\n[5/5] Building node_features.csv...")
    grp = raw_df.groupby('node_id')
    node_agg = pd.DataFrame({
        'detection_freq':       grp['detected'].mean().round(4),
        'peak_conc_mean':       grp['peak_conc'].mean().round(4),
        'peak_conc_std':        grp['peak_conc'].std().round(4),
        'time_to_peak_mean':    grp['t_peak_min'].mean().round(2),
        'mean_flow_m3s':        grp['mean_flow_m3s'].mean().round(6),
        'mean_vel_ms':          grp['mean_vel_ms'].mean().round(6),
        'n_scenarios_detected': grp['detected'].sum().astype(int),
    })
    final_df = topo_df.join(node_agg, how='left').reset_index()
    feat_path = os.path.join(a.output_dir, 'node_features.csv')
    final_df.to_csv(feat_path, index=False)
    print(f"      {feat_path}  ({len(final_df.columns)} columns)")

if __name__ == '__main__':
    main()
