"""
The following code is used to analyse the traces in the folder ending with _frontend_traces.json.
The id of this batch experiment is "batch_11736941878".
File names are coded as "batch_<batch_id>_<sub_experiment_id>_<run_id>_frontend_traces.json".
Every sub experiment is a sub configuration of the batch experiment definition.
Every run is a single run of the sub experiment. We do multiple runs to get a better understanding of the performance.
Every file in the folder ending with _frontend_traces.json contains  a JSON array of traces.
This is an example of how a  single trace looks like:
{"index": 0, "trace_id": "89af8d8fbc1dfbfa14d37eedbe8f53c7", "span_id": "3b35790ffd7ce4ab", "operation": "router frontend egress", "start_time": 1736941985734492, "end_time": 1736941985795495, "duration": 61003, "service_name": "frontendproxy", "span_kind": "client", "req_status_code": "308", "ref_type": "CHILD_OF", "ref_type_span_ID": "e7418dcd2afce880", "ref_type_trace_ID": "89af8d8fbc1dfbfa14d37eedbe8f53c7", "add_security_context": "add_security_context", "delay_treatment": "delay_treatment"}, {"index": 1, "trace_id": "89af8d8fbc1dfbfa14d37eedbe8f53c7", "span_id": "e7418dcd2afce880", "operation": "ingress", "start_time": 1736941985734380, "end_time": 1736941985795546, "duration": 61166, "service_name": "frontendproxy", "span_kind": "server", "req_status_code": "308", "ref_type": "N/A", "ref_type_span_ID": "N/A", "ref_type_trace_ID": "N/A", "add_security_context": "add_security_context", "delay_treatment": "delay_treatment"}, {"index": 2, "trace_id": "89af8d8fbc1dfbfa14d37eedbe8f53c7", "span_id": "81e00097aeb04d2e", "operation": "GET", "start_time": 1736941985794000, "end_time": 1736941985794381, "duration": 381, "service_name": "frontend", "span_kind": "server", "req_status_code": 308, "ref_type": "CHILD_OF", "ref_type_span_ID": "3b35790ffd7ce4ab", "ref_type_trace_ID": "89af8d8fbc1dfbfa14d37eedbe8f53c7", "add_security_context": "add_security_context", "delay_treatment": "delay_treatment"}, {"index": 0, "trace_id": "29829605d938220d94cb62df8e970a94", "span_id": "ceb9eba7c0093a61", "operation": "ingress", "start_time": 1736942007015193, "end_time": 1736942007026395, "duration": 11202, "service_name": "frontendproxy", "span_kind": "server", "req_status_code": "200", "ref_type": "N/A", "ref_type_span_ID": "N/A", "ref_type_trace_ID": "N/A", "add_security_context": "add_security_context", "delay_treatment": "delay_treatment"}, {"index": 1, "trace_id": "29829605d938220d94cb62df8e970a94", "span_id": "3e6d0c7698349667", "operation": "router frontend egress", "start_time": 1736942007015296, "end_time": 1736942007026150, "duration": 10854, "service_name": "frontendproxy", "span_kind": "client", "req_status_code": "200", "ref_type": "CHILD_OF", "ref_type_span_ID": "ceb9eba7c0093a61", "ref_type_trace_ID": "29829605d938220d94cb62df8e970a94", "add_security_context": "add_security_context", "delay_treatment": "delay_treatment"}

The traces are annotated with the treatment that was applied to the trace.
In this case, the relevant treatment is "delay_treatment". The other treatment is "add_security_context", which is not relevant for this analysis.
If a trace was produced whilst there was no treatment applied, the treatment is "NoTreatment".

The files ending with _config.json contains the configuration of the sub experiment.

This is how it looks like:

{"id": "0", "name": "big_0", "status": "PENDING", "created_at": "2025-01-04T18:12:35.968327", "started_at": null, "completed_at": null, "error_message": null, "spec": {"experiment": {"name": "latest", "version": "0.0.1", "orchestrator": "kubernetes", "services": {"jaeger": {"name": "astronomy-shop-jaeger-query", "namespace": "system-under-evaluation"}, "prometheus": [{"name": "astronomy-shop-prometheus-server", "namespace": "system-under-evaluation", "target": "sue"}, {"name": "kube-prometheus-kube-prome-prometheus", "namespace": "oxn-external-monitoring", "target": "oxn"}]}, "responses": [{"name": "frontend_traces", "type": "trace", "service_name": "frontend", "left_window": "10s", "right_window": "10s", "limit": 1}, {"name": "system_CPU", "type": "metric", "metric_name": "sum(rate(container_cpu_usage_seconds_total{namespace=\"system-under-evaluation\"}[1m]))", "left_window": "10s", "right_window": "10s", "step": 1, "target": "oxn"}], "treatments": [{"add_security_context": {"action": "security_context_kubernetes", "params": {"namespace": "system-under-evaluation", "label_selector": "app.kubernetes.io/component", "label": "recommendationservice", "capabilities": {"add": ["NET_ADMIN"]}}}}, {"delay_treatment": {"action": "delay", "params": {"namespace": "system-under-evaluation", "label_selector": "app.kubernetes.io/name", "label": "astronomy-shop-recommendationservice", "delay_time": "90ms", "delay_jitter": "120ms", "duration": "1m", "interface": "eth0"}}}], "sue": {"compose": "opentelemetry-demo/docker-compose.yml", "exclude": ["loadgenerator"], "required": [{"namespace": "system-under-evaluation", "name": "astronomy-shop-prometheus-server"}]}, "loadgen": {"run_time": "4m", "max_users": 600, "spawn_rate": 50, "locust_files": ["/backend/locust/locust_basic_interaction.py", "/backend/locust/locust_otel_demo.py"], "target": {"name": "astronomy-shop-frontendproxy", "namespace": "system-under-evaluation", "port": 8080}}}}}

The different sub experiments in this batch differ in the delay_time and duration of the delay_treatment.
In this case we have 6 sub experiments, each with 3 runs.
Sub experiment ids range from 0 to 5.
Run ids range from 0 to 2.
"""

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def load_traces(batch_id):
    """Load all trace files for a given batch ID."""
    traces_data = []
    
    # Find all trace files for this batch
    trace_files = glob.glob(f"batch_{batch_id}_*_frontend_traces.json")
    
    for file_path in trace_files:
        # Extract sub_experiment_id and run_id from filename
        parts = file_path.split('_')
        sub_exp_id = int(parts[2]) # the third part
        run_id = int(parts[3]) # the fourth part
        report_file_path = f"batch_{batch_id}_{sub_exp_id}_report.yaml" # this file actually contains JSON...
        report_data = {}
        with open(report_file_path, 'r') as f:
            report_data = json.load(f)
            
        # Get list of run IDs in order they appear
        ordered_run_ids = list(report_data['report']['runs'].keys())

        with open(file_path, 'r') as f:
            traces = json.load(f)
            # Add metadata to each trace
            for trace in traces:
                trace['sub_experiment_id'] = sub_exp_id
                trace['run_id'] = run_id
                # Map run_id to the corresponding position in ordered_run_ids
                actual_run_id = ordered_run_ids[run_id]
                trace['loadgen_start_time'] = report_data['report']['runs'][actual_run_id]['loadgen']['loadgen_start_time']
                trace['loadgen_end_time'] = report_data['report']['runs'][actual_run_id]['loadgen']['loadgen_end_time']
            traces_data.extend(traces)
    
    return pd.DataFrame(traces_data)

def load_configs(batch_id):
    """Load all configuration files for a given batch ID."""
    configs = {}
    config_files = glob.glob(f"batch_{batch_id}_*_config.json")
    
    for file_path in config_files:
        sub_exp_id = int(file_path.split('_')[2])
        with open(file_path, 'r') as f:
            configs[sub_exp_id] = json.load(f)
    
    return configs

def analyze_batch(batch_id="11736941878"):
    # Load all data
    df = load_traces(batch_id)
    configs = load_configs(batch_id)
    
    # Convert duration from microseconds to milliseconds
    df['duration'] = df['duration'] / 1000
    
    # Create a mapping of sub_experiment_id to delay configuration
    delay_configs = {}
    counter = 0
    for sub_exp_id, config in configs.items():
        delay_treatment = next(t for t in config['spec']['experiment']['treatments'] 
                             if 'delay_treatment' in t)['delay_treatment']
        delay_configs[sub_exp_id] = {
            'delay_time': delay_treatment['params']['delay_time'],
            'delay_jitter': delay_treatment['params']['delay_jitter'],
            'duration': delay_treatment['params']['duration']
        }
        counter += 1
    print(delay_configs)
    print(counter)
    
    # Add configuration details to the DataFrame
    df['delay_time'] = df['sub_experiment_id'].map(
        lambda x: delay_configs[x]['delay_time'])
    
    df['delay_duration'] = df['sub_experiment_id'].map(
        lambda x: delay_configs[x]['duration'])
    
    # Analysis of duration by operation and delay configuration
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='delay_time', y='duration', hue='operation')
    plt.title('Operation Duration by Delay Configuration')
    plt.xlabel('Configured Delay Time')
    plt.ylabel('Duration (milliseconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('duration_analysis.png')
    
    # Calculate summary statistics
    summary_stats = df.groupby(['sub_experiment_id', 'delay_time', 'operation'])[['duration']].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).reset_index()
    
    # Print summary statistics
    #print("\nSummary Statistics by Sub-experiment and Operation (in milliseconds):")
    #print(summary_stats.to_string())

    # format of the timestamps here: 2025-01-15 12:05:04.558628
    print("Average duration of experiment runs per sub-experiment:")
    # First group by sub_experiment_id and run_id to get duration per run
    run_durations = df.groupby(['sub_experiment_id', 'run_id']).apply(
        lambda x: (pd.to_datetime(x['loadgen_end_time']).max() - 
                  pd.to_datetime(x['loadgen_start_time']).min()).total_seconds()
    )
    # Then calculate mean duration across runs for each sub-experiment
    avg_duration_per_subexp = run_durations.groupby('sub_experiment_id').mean()
    print("\nPer sub-experiment averages:")
    for sub_exp, duration in avg_duration_per_subexp.items():
        print(f"Sub-experiment {sub_exp}: {duration:.2f} seconds")
    
    print(f"\nOverall average: {avg_duration_per_subexp.mean():.2f} seconds")
    
    # Analyze impact on different services
    service_impact = df.groupby(['service_name', 'delay_time'])['duration'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    
    print("\nImpact on Different Services (in milliseconds):")
    print(service_impact.to_string())

    # Normalize time to be between 0 and 1 based on loadgen start/end times for each run
    df['normalized_time'] = df.groupby(['sub_experiment_id', 'run_id']).apply(
        lambda x: (x['start_time'] - x['start_time'].min()) / 
                 (x['start_time'].max() - x['start_time'].min())
    ).reset_index(level=[0,1], drop=True)
    # Filter for the recommendation service
    df_recommendation = df[df['service_name'] == 'recommendationservice']
    g = sns.FacetGrid(df_recommendation, col="delay_time", row="delay_duration", height=4, aspect=1)
    g.map(sns.scatterplot, data=df_recommendation, x="normalized_time", y="duration", hue="operation")
    g.set(ylim=(0, 2000))
    
    plt.savefig('duration_analysis_facet.png')


    # Lineplot of aggregated duration for recommendation service , considering treatment information.
    # One line for NoTreatment, one line for delay_treatment.

    g2 = sns.FacetGrid(df_recommendation, col="delay_time", row="delay_duration", height=4, aspect=1)
    g2.map(sns.lineplot, data=df_recommendation, x="normalized_time", y="duration", hue="delay_treatment")
    g2.set(ylim=(0, 2000))

    plt.savefig('duration_analysis_facet_recommendation.png')

    return df, configs

if __name__ == "__main__":
    df, configs = analyze_batch()