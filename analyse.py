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
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Any
import numpy as np

LOSS_TREATMENT_KEY = "loss_treatment"
DELAY_TREATMENT_KEY = "delay_treatment"
EXPERIMENT_METRICS_LOG = "delay-19/delay-metrics.log"

# Total cluster resources (replace with actual values)
total_cpu = 12000  # Milli CPU - 3 x e2-standard-4
total_memory = 48 * 1024 * 1024 * 1024  # Bytes - 48GB

class ExperimentAnalyzer:
    def __init__(self, directory: str, treatment_type: str, service_filter: List[str]):
        self.directory = Path(directory)
        self.batch_id = self._get_batch_id()
        self.params_mapping = self._load_params_mapping()
        self.traces_df = None
        self.configs = None
        self.treatment_type = treatment_type
        self.service_filter = service_filter
        self.cpu_usage = []
        self.cpu_usage_df = None
        self.memory_usage = []
        self.memory_usage_df = None
        self.metrics_df = None
        self.reports = None
    def _get_batch_id(self) -> str:
        """Extract batch ID from the first matching file in directory."""
        files = list(self.directory.glob("batch_*"))
        if not files:
            raise ValueError(f"No batch files found in {self.directory}")
        
        # Extract batch ID from first file
        batch_id = files[0].name.split('_')[1]
        return batch_id

    def _load_params_mapping(self) -> Dict[int, Dict[str, Any]]:
        """Load the parameters to sub-experiment ID mapping."""
        params_file = self.directory / f"batch_{self.batch_id}_params_to_id.json"
        if not params_file.exists():
            print(f"Warning: No params mapping file found at {params_file}")
            return {}
        
        with open(params_file) as f:
            mapping = json.load(f)
        
        # Convert to dictionary with sub_experiment_id as key
        return {item['sub_experiment_id']: item for item in mapping}

    def load_traces(self) -> pd.DataFrame:
        """Load all trace files for the batch."""
        traces_data = []
        trace_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_traces.json"))
        print(f"{len(trace_files)} trace files detected")
        for file_path in trace_files:
            # Extract sub_experiment_id and run_id from filename
            parts = file_path.stem.split('_')
            sub_exp_id = int(parts[2])
            run_id = int(parts[3])
            
            # Try to load corresponding report file
            report_file = self.directory / f"batch_{self.batch_id}_{sub_exp_id}_report.yaml"
            report_data = {}
            if report_file.exists():
                with open(report_file) as f:
                    report_data = json.load(f)
                ordered_run_ids = list(report_data['report']['runs'].keys())
            
            with open(file_path) as f:
                traces = json.load(f)
                for trace in traces:
                    trace['sub_experiment_id'] = sub_exp_id
                    trace['run_id'] = run_id
                    
                    # Add experiment parameters from mapping
                    if sub_exp_id in self.params_mapping:
                        for key, value in self.params_mapping[sub_exp_id].items():
                            if key != 'sub_experiment_id':
                                trace[key] = value
                    
                    # Add report data if available
                    if report_data and 'report' in report_data:
                        actual_run_id = ordered_run_ids[run_id]
                        run_data = report_data['report']['runs'][actual_run_id]['loadgen']
                        trace['loadgen_start_time'] = run_data.get('loadgen_start_time')
                        trace['loadgen_end_time'] = run_data.get('loadgen_end_time')
                
                traces_data.extend(traces)
        
        self.traces_df = pd.DataFrame(traces_data)
        return self.traces_df
    def load_reports(self):
        """Load all reports for the batch."""
        reports = {}
        report_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_report.yaml"))
        for file_path in report_files:
            sub_exp_id = int(file_path.stem.split('_')[2])
            with open(file_path) as f:
                reports[sub_exp_id] = json.load(f)
        self.reports = reports
        return self.reports
    def load_configs(self) -> Dict[int, Dict]:
        """Load all configuration files."""
        configs = {}
        config_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_config.json"))
        
        for file_path in config_files:
            sub_exp_id = int(file_path.stem.split('_')[2])
            with open(file_path) as f:
                configs[sub_exp_id] = json.load(f)
        
        self.configs = configs
        return configs
    
    def load_cpu_usage(self):
        """Load all cpu usage files for the batch."""
        # cpu usage files are named as "batch_<batch_id>_<sub_experiment_id>_<run_id>_cpu_usage.json"
        cpu_usage_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_cpu_usage.json"))
        print(f"{len(cpu_usage_files)} cpu usage files detected")
        for file_path in cpu_usage_files:
            # Extract sub_experiment_id and run_id from filename
            parts = file_path.stem.split('_')
            sub_exp_id = int(parts[2])
            run_id = int(parts[3])
            
            with open(file_path) as f:
                cpu_usage = json.load(f)
                for record in cpu_usage:
                    record['sub_experiment_id'] = sub_exp_id
                    record['run_id'] = run_id
                self.cpu_usage.extend(cpu_usage)
                
        self.cpu_usage_df = pd.DataFrame(self.cpu_usage)
        return self.cpu_usage_df
    
    def load_memory_usage(self):
        """Load all memory usage files for the batch."""
        # memory usage files are named as "batch_<batch_id>_<sub_experiment_id>_<run_id>_memory_available.json"
        memory_usage_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_memory_available.json"))
        print(f"{len(memory_usage_files)} memory usage files detected")
        for file_path in memory_usage_files:
            # Extract sub_experiment_id and run_id from filename
            parts = file_path.stem.split('_')
            sub_exp_id = int(parts[2])
            run_id = int(parts[3])
            
            with open(file_path) as f:
                memory_usage = json.load(f)
                for record in memory_usage:
                    record['sub_experiment_id'] = sub_exp_id
                    record['run_id'] = run_id
                self.memory_usage.extend(memory_usage)
        self.memory_usage_df = pd.DataFrame(self.memory_usage)
        return self.memory_usage_df
    def load_metrics_logs(self):
        """Load the metrics gathered using the kube metrics server"""

        # Read and parse log file
        log_file = EXPERIMENT_METRICS_LOG
        rows = []

        with open(log_file, "r") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "node" in data:
                        rows.append({
                            # 2025-01-19T14:51:47.895743172Z this is the format of the timestamp
                            "timestamp": pd.to_datetime(data["timestamp"]),
                            "node": data["node"],
                            "cpu": data["cpu"],  # Milli CPU
                            "memory": data["memory_bytes"],  # Memory in bytes
                            "type": "node"
                        })
                    elif "namespace" in data:
                        rows.append({
                            "timestamp": pd.to_datetime(data["timestamp"]),
                            "namespace": data["namespace"],
                            "pod": data["pod"],
                            "container": data["container"],
                            "cpu": data["cpu"],  # Milli CPU
                            "memory": data["memory_bytes"],  # Memory in bytes!!!!
                            "type": "container"
                        })
                except json.JSONDecodeError:
                    continue
        self.metrics_df = pd.DataFrame(rows)
        self.metrics_df["cpu_percentage"] = self.metrics_df["cpu"] / total_cpu * 100
        self.metrics_df["memory_percentage"] = self.metrics_df["memory"] / total_memory * 100
        return self.metrics_df

    def analyze(self):
        """Perform analysis on the loaded data."""
        if self.traces_df is None:
            self.load_traces()
        
        # Convert duration to milliseconds
        self.traces_df['duration'] = self.traces_df['duration'] / 1000

        # Convert loadgen_start_time and loadgen_end_time to datetime
        self.traces_df['loadgen_start_time'] = pd.to_datetime(self.traces_df['loadgen_start_time'])
        self.traces_df['loadgen_end_time'] = pd.to_datetime(self.traces_df['loadgen_end_time'])
        
        # Normalize time within each run
        self.traces_df['normalized_time'] = self.traces_df.groupby(
            ['sub_experiment_id', 'run_id'])['start_time'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        # Generate basic statistics
        self._generate_statistics()

    def _analyze_packet_loss_variations(self):
        """Analyze traces with packet loss treatment variations"""
        # Traces under loss treatment have the value "loss_treatment" in the "loss_treatment" column
        pass
    def analyze_loadgen_duration(self):
        """Analyze the duration of the loadgen and compare with experiment timing"""
        self.load_reports()
        
        for sub_exp_id, report in self.reports.items():
            # Get experiment-level timing
            exp_start = pd.to_datetime(report['report']['experiment_start'])
            exp_end = pd.to_datetime(report['report']['experiment_end']) 
            exp_duration = exp_end - exp_start
            
            print(f"\nSub Experiment {sub_exp_id}")
            print(f"Total experiment duration: {exp_duration}")
            print("Individual run durations:")
            
            for run_id, run_data in report['report']['runs'].items():
                start_time = pd.to_datetime(run_data['loadgen']['loadgen_start_time'])
                end_time = pd.to_datetime(run_data['loadgen']['loadgen_end_time'])
                loadgen_duration = end_time - start_time
                
                # Calculate offset from experiment start
                run_offset = start_time - exp_start
                
                print(f"  Run {run_id}:")
                print(f"    Duration: {loadgen_duration}")
                print(f"    Started at: {run_offset} from experiment start")

    def _generate_statistics(self):
        """Generate and print various statistics about the experiment."""
        # Basic duration statistics by service_name and treatment parameters
        if self.service_filter:
            stats = self.traces_df[self.traces_df['service_name'].isin(self.service_filter)].groupby(
                ['service_name'] + list(self.params_mapping[0].keys()) + [self.treatment_type]
            )['duration'].agg(['mean', 'std', 'count']).round(2)
        else:
            stats = self.traces_df.groupby(
                ['service_name'] + list(self.params_mapping[0].keys()) + [self.treatment_type]
            )['duration'].agg(['mean', 'std', 'count']).round(2)
        
        print("\nService Statistics:")
        print(stats.to_string())

    def _plot_latency_vs_time(self):
        """Plot latency vs time"""
        if self.service_filter:
            traces_df = self.traces_df[self.traces_df['service_name'].isin(self.service_filter)]
        else:
            traces_df = self.traces_df

        sns.lineplot(x='normalized_time', y='duration', hue=self.treatment_type, style="service_name", data=traces_df)
        plt.title('Latency vs Time')
        plt.xlabel('Normalized Time')
        plt.ylabel('Latency (ms)')
        plt.ylim(0, 1000)
        plt.savefig(f'latency_vs_time_{self.treatment_type}.png')
        plt.close()
    def _plot_cpu_usage_over_time(self):
        # Metric used is sum(rate(node_cpu_seconds_total{mode!=\"idle\"}[1m])) by (instance)
        self.load_cpu_usage()
        df = self.cpu_usage_df
        
        # Convert 'timestamp' to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Normalize time within each sub experiment
        df['normalized_time'] = df.groupby(
            ['sub_experiment_id', 'run_id'])['timestamp'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        
        # Sort by timestamp
        df = df.sort_values(by='timestamp')
        
        # Plotting
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='normalized_time', y='cpu_usage', hue="sub_experiment_id")
        plt.xlabel('Normalized Time')
        plt.ylabel('CPU Usage')
        plt.title('CPU Usage Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'cpu_usage_over_time_{self.treatment_type}.png')

    def _plot_node_cpu_mem_over_time(self):
        df = self.load_metrics_logs()
        # Filter node metrics
        node_metrics = df[df["type"] == "node"]

        # Clear any existing plots
        plt.clf()
        # Plot CPU usage
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=node_metrics, x="timestamp", y="cpu_percentage", hue="node")
        plt.title("CPU Usage per Node (%)")
        plt.ylabel("CPU Usage (%)")
        plt.xlabel("Time")
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'node_cpu_usage_{self.treatment_type}.png')
        plt.close()

        # Clear any existing plots
        plt.clf()
        # Plot Memory usage
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=node_metrics, x="timestamp", y="memory_percentage", hue="node")
        plt.title("Memory Usage per Node (%)")
        plt.ylabel("Memory Usage (%)")
        plt.xlabel("Time")
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'node_memory_usage_{self.treatment_type}.png')
        plt.close()

    def _plot_namespace_cpu_mem_over_time(self):
        df = self.load_metrics_logs()
        # Filter container metrics
        namespace_metrics = df[df["type"] == "container"]

        # Aggregate by namespace and timestamp
        namespace_agg = namespace_metrics.groupby(["timestamp", "namespace"]).sum().reset_index()

        # Clear any existing plots
        plt.clf()
        # Plot CPU usage
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=namespace_agg, x="timestamp", y="cpu_percentage", hue="namespace")
        plt.title("CPU Usage per Namespace (%)")
        plt.ylabel("CPU Usage (%)")
        plt.xlabel("Time")
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'namespace_cpu_usage_{self.treatment_type}.png')
        plt.close()

        # Clear any existing plots
        plt.clf()
        # Plot Memory usage
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=namespace_agg, x="timestamp", y="memory_percentage", hue="namespace")
        plt.title("Memory Usage per Namespace (%)")
        plt.ylabel("Memory Usage (%)")
        plt.xlabel("Time")
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'namespace_memory_usage_{self.treatment_type}.png')
        plt.close()

    def _plot_trace_duration_by_service(self):
        """Plot trace durations by service, processing one sub-experiment at a time."""
        plt.figure(figsize=(15, 8))
        
        # Get all unique sub-experiment IDs
        sub_exp_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_traces.json"))
        sub_exp_ids = sorted(set(int(f.stem.split('_')[2]) for f in sub_exp_files))
        
        # Process each sub-experiment separately
        for sub_exp_id in sub_exp_ids:
            print(f"Processing sub-experiment {sub_exp_id}")
            
            # Load traces for just this sub-experiment
            traces_data = []
            trace_files = list(self.directory.glob(f"batch_{self.batch_id}_{sub_exp_id}_*_traces.json"))
            
            for file_path in trace_files:
                run_id = int(file_path.stem.split('_')[3])
                
                with open(file_path) as f:
                    traces = json.load(f)
                    for trace in traces:
                        # Only include necessary fields
                        filtered_trace = {
                            'duration': float(trace['duration']) / 1000,  # Convert to ms
                            'start_time': float(trace['start_time']),
                            'service_name': trace['service_name'],
                            'sub_experiment_id': sub_exp_id,
                            'run_id': run_id
                        }
                        
                        # Add treatment parameters
                        if sub_exp_id in self.params_mapping:
                            for key, value in self.params_mapping[sub_exp_id].items():
                                if key != 'sub_experiment_id':
                                    filtered_trace[key] = value
                        
                        traces_data.append(filtered_trace)
            
            # Convert to DataFrame and process
            sub_exp_df = pd.DataFrame(traces_data)
            
            # Filter services if needed
            if self.service_filter:
                sub_exp_df = sub_exp_df[sub_exp_df['service_name'].isin(self.service_filter)]
            
            # Normalize time within each run
            sub_exp_df['normalized_time'] = sub_exp_df.groupby('run_id')['start_time'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            
            # Plot this sub-experiment's data
            for service in sub_exp_df['service_name'].unique():
                service_data = sub_exp_df[sub_exp_df['service_name'] == service]
                
                # Calculate rolling mean for smoother lines
                service_data = service_data.sort_values('normalized_time')
                rolling_mean = service_data.groupby('service_name')['duration'].rolling(
                    window=50, min_periods=1, center=True
                ).mean().reset_index(level=0, drop=True)
                
                plt.plot(
                    service_data['normalized_time'],
                    rolling_mean,
                    label=f"{service} (Sub-exp {sub_exp_id})",
                    alpha=0.7
                )
            
            # Clear memory
            del traces_data
            del sub_exp_df
        
        plt.title('Trace Duration by Service')
        plt.xlabel('Normalized Time')
        plt.ylabel('Duration (ms)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'trace_duration_by_service_{self.treatment_type}.png', bbox_inches='tight')
        plt.close()

    def _plot_treatment_comparison(self):
        """Plot comparison between traces with and without treatment as a scatter plot."""
        plt.figure(figsize=(15, 8))
        
        # Get all unique sub-experiment IDs
        sub_exp_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_traces.json"))
        sub_exp_ids = sorted(set(int(f.stem.split('_')[2]) for f in sub_exp_files))
        
        # Process each sub-experiment separately
        for sub_exp_id in sub_exp_ids:
            print(f"Processing sub-experiment {sub_exp_id}")
            
            # Load traces for just this sub-experiment
            traces_data = []
            trace_files = list(self.directory.glob(f"batch_{self.batch_id}_{sub_exp_id}_*_traces.json"))
            
            for file_path in trace_files:
                run_id = int(file_path.stem.split('_')[3])
                
                with open(file_path) as f:
                    traces = json.load(f)
                    for trace in traces:
                        # Only include necessary fields
                        filtered_trace = {
                            'duration': float(trace['duration']) / 1000,  # Convert to ms
                            'start_time': float(trace['start_time']),
                            'service_name': trace['service_name'],
                            'treatment_status': 'With Treatment' if trace.get(self.treatment_type) == self.treatment_type else 'No Treatment',
                            'sub_experiment_id': sub_exp_id,
                            'run_id': run_id
                        }
                        
                        # Add treatment parameters
                        if sub_exp_id in self.params_mapping:
                            for key, value in self.params_mapping[sub_exp_id].items():
                                if key != 'sub_experiment_id':
                                    filtered_trace[key] = value
                        
                        traces_data.append(filtered_trace)
            
            # Convert to DataFrame and process
            sub_exp_df = pd.DataFrame(traces_data)
            
            # Filter services if needed
            if self.service_filter:
                sub_exp_df = sub_exp_df[sub_exp_df['service_name'].isin(self.service_filter)]
            
            # Normalize time within each run
            sub_exp_df['normalized_time'] = sub_exp_df.groupby('run_id')['start_time'].transform(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
            
            # Plot this sub-experiment's data
            for service in sub_exp_df['service_name'].unique():
                for treatment in ['With Treatment', 'No Treatment']:
                    mask = (sub_exp_df['service_name'] == service) & (sub_exp_df['treatment_status'] == treatment)
                    service_data = sub_exp_df[mask]
                    
                    if not service_data.empty:
                        marker = 'o' if treatment == 'With Treatment' else 'x'
                        plt.scatter(
                            service_data['normalized_time'],
                            service_data['duration'],
                            label=f"{service} ({treatment}) - Sub-exp {sub_exp_id}",
                            marker=marker,
                            alpha=0.3,
                            s=20
                        )
            
            # Clear memory
            del traces_data
            del sub_exp_df
        
        plt.title('Recommendationservice: Treatment vs No Treatment aggregated over all sub experiments')
        plt.xlabel('Normalized Time')
        plt.ylabel('Duration (ms)')
        plt.legend(['Delay 120ms •', 'No Delay ×'], loc='upper right')
        plt.tight_layout()
        plt.savefig(f'treatment_comparison_{self.treatment_type}.png', bbox_inches='tight')
        plt.close()

    def _gather_fault_detection_data(self):
        """Gather and process fault detection data into a DataFrame."""
        detection_data = []
        
        for sub_exp_id in self.params_mapping.keys():
            fault_file = self.directory / f"batch_{self.batch_id}_{sub_exp_id}_fault_detection.json"
            if not fault_file.exists():
                continue
                
            with open(fault_file) as f:
                fault_data = json.load(f)
                
            # Find the relevant fault (matching treatment type)
            relevant_faults = [
                fault for fault in fault_data['results'] 
                if fault['fault_name'] == self.treatment_type
            ]
            
            if relevant_faults:
                fault = relevant_faults[0]  # Take the first matching fault
                params = self.params_mapping[sub_exp_id]
                
                detection_data.append({
                    'sub_experiment_id': sub_exp_id,
                    'detected': fault['detected'],
                    'detection_latency': fault.get('detection_latency', None),
                    'num_alerts': len(fault.get('alerts_triggered', [])),
                    'latency_threshold': float(params['experiment.treatments.0.kubernetes_prometheus_rules.params.latency_threshold']),
                    'evaluation_window': int(params['experiment.treatments.0.kubernetes_prometheus_rules.params.evaluation_window'].replace('s', '')),
                })
        
        return pd.DataFrame(detection_data)
    
    def _gather_raw_alerts_data(self):
        """Gather and process raw fault detection data into a DataFrame."""
        alerts_data = []
        
        # First load reports to get treatment timing information
        reports = {}
        for sub_exp_id in self.params_mapping.keys():
            print(f"Processing sub-experiment {sub_exp_id}")
            report_file = self.directory / f"batch_{self.batch_id}_{sub_exp_id}_report.yaml"
            if report_file.exists():
                print(f"Loading report file {report_file}")
                with open(report_file) as f:
                    report = json.load(f)
                    # Find the relevant treatment timing
                    for run_id, run_data in report['report']['runs'].items():
                        for interaction in run_data['interactions'].values():
                            # YOU NEED TO CHANGE THIS TO CHECK FOR THE CORRECT TREATMENT TYPE
                            if interaction['treatment_type'] == "KubernetesNetworkDelayTreatment":
                                print(f"Found treatment for sub-experiment {sub_exp_id}")
                                reports[sub_exp_id] = {
                                    'treatment_start': pd.to_datetime(interaction['treatment_start']),
                                    'treatment_end': pd.to_datetime(interaction['treatment_end'])
                                }
                                break
        # get and normalise the loadgen start and end times
        # TODO
        
        # Now process alerts data
        for sub_exp_id in self.params_mapping.keys():
            detections_file = self.directory / f"batch_{self.batch_id}_{sub_exp_id}_detections.json"
            if not detections_file.exists() or sub_exp_id not in reports:
                continue
                
            with open(detections_file) as f:
                detections = json.load(f)
                
            # Process each alert series
            for result in detections['detections']['data']['result']:
                metric = result['metric']
                values = result['values']
                
                # Convert timestamps and values to DataFrame
                alert_times = pd.DataFrame(values, columns=['timestamp', 'value'])
                alert_times['timestamp'] = pd.to_datetime(alert_times['timestamp'], unit='s')
                alert_times['value'] = alert_times['value'].astype(float)
                
                # Add metadata
                alert_times['sub_experiment_id'] = sub_exp_id
                alert_times['alertname'] = metric['alertname']
                alert_times['service'] = metric.get('job', '').split('/')[-1]
                alert_times['detection'] = metric.get('detection', '')
                
                # Classify alerts as true/false positives
                """ treatment_period = (
                    (alert_times['timestamp'] >= reports[sub_exp_id]['treatment_start']) &
                    (alert_times['timestamp'] <= reports[sub_exp_id]['treatment_end'])
                )
                alert_times['alert_type'] = 'false_positive'
                alert_times.loc[treatment_period, 'alert_type'] = 'true_positive'
                
                # Calculate detection time for first true positive
                first_true_positive = alert_times[
                    (alert_times['alert_type'] == 'true_positive') & 
                    (alert_times['value'] == 1)
                ]['timestamp'].min()
                
                if pd.notna(first_true_positive):
                    detection_time = (
                        first_true_positive - reports[sub_exp_id]['treatment_start']
                    ).total_seconds()
                else:
                    detection_time = None
                
                alert_times['detection_time'] = detection_time """
                
                alerts_data.append(alert_times)
        
        if alerts_data:
            return pd.concat(alerts_data, ignore_index=True)
        return pd.DataFrame()

    def _plot_fault_detection_latency_and_false_alerts(self):
        """Plot detection latency and false alerts analysis."""
        df = self._gather_raw_alerts_data()
        if df.empty:
            print("No alerts data available")
            return
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Detection Latency by Configuration
        detection_stats = df.groupby('sub_experiment_id').agg({
            'detection_time': 'first'  # Each sub-experiment has same detection time
        }).reset_index()
        
        # Add configuration parameters
        detection_stats['latency_threshold'] = detection_stats['sub_experiment_id'].map(
            lambda x: self.params_mapping[x]['experiment.treatments.0.kubernetes_prometheus_rules.params.latency_threshold']
        )
        detection_stats['evaluation_window'] = detection_stats['sub_experiment_id'].map(
            lambda x: int(self.params_mapping[x]['experiment.treatments.0.kubernetes_prometheus_rules.params.evaluation_window'].replace('s', ''))
        )
        
        sns.scatterplot(
            data=detection_stats,
            x='latency_threshold',
            y='detection_time',
            size='evaluation_window',
            ax=ax1
        )
        ax1.set_title('Detection Latency by Configuration')
        ax1.set_xlabel('Latency Threshold (ms)')
        ax1.set_ylabel('Detection Time (seconds)')
        
        # Plot 2: False Positives by Configuration
        false_positives = df[df['alert_type'] == 'false_positive'].groupby(
            'sub_experiment_id'
        ).size().reset_index(name='false_positive_count')
        
        # Add configuration parameters
        false_positives['latency_threshold'] = false_positives['sub_experiment_id'].map(
            lambda x: self.params_mapping[x]['experiment.treatments.0.kubernetes_prometheus_rules.params.latency_threshold']
        )
        false_positives['evaluation_window'] = false_positives['sub_experiment_id'].map(
            lambda x: int(self.params_mapping[x]['experiment.treatments.0.kubernetes_prometheus_rules.params.evaluation_window'].replace('s', ''))
        )
        
        sns.scatterplot(
            data=false_positives,
            x='latency_threshold',
            y='false_positive_count',
            size='evaluation_window',
            ax=ax2
        )
        ax2.set_title('False Positives by Configuration')
        ax2.set_xlabel('Latency Threshold (ms)')
        ax2.set_ylabel('Number of False Positives')
        
        plt.tight_layout()
        plt.savefig(f'fault_detection_analysis_{self.treatment_type}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_fault_detection_scatter(self):
        """Create a scatter plot showing fault detection performance across parameter variations."""
        plt.figure(figsize=(12, 8))
        
        # Get processed data
        df = self._gather_fault_detection_data()
        
        # Create scatter plot
        detected_mask = df['detected'] == True
        not_detected_mask = df['detected'] == False
        
        # Plot points where faults were detected
        plt.scatter(
            df[detected_mask]['latency_threshold'],
            df[detected_mask]['evaluation_window'],
            s=df[detected_mask]['detection_latency'] / 10,  # Size based on detection latency
            c='green',
            alpha=0.6,
            label='Detected',
            marker='o'
        )
        
        # Plot points where faults were not detected
        plt.scatter(
            df[not_detected_mask]['latency_threshold'],
            df[not_detected_mask]['evaluation_window'],
            c='red',
            alpha=0.6,
            label='Not Detected',
            marker='x',
            s=100  # Added size parameter to make points bigger
        )
        
        # Add labels for each point
        for _, row in df.iterrows():
            label = f"Latency: {row['detection_latency']:.1f}s\nAlerts: {row['num_alerts']}"
            plt.annotate(
                label,
                (row['latency_threshold'], row['evaluation_window']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=14
            )
        
        plt.title(f'Fault Detection Performance by Parameter Variations\nTreatment: {self.treatment_type}')
        plt.xlabel('Latency Threshold (ms)')
        plt.ylabel('Evaluation Window (seconds)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'fault_detection_scatter_{self.treatment_type}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_alerts_over_traces(self):
        """Plot trace durations as lines with alert firing times as points overlaid."""
        plt.figure(figsize=(15, 8))
        
        # First get the alerts data
        alerts_df = self._gather_raw_alerts_data()
        if alerts_df.empty:
            print("No alerts data available")
            return
            
        # Process each sub-experiment separately
        sub_exp_files = list(self.directory.glob(f"batch_{self.batch_id}_*_*_traces.json"))
        sub_exp_ids = sorted(set(int(f.stem.split('_')[2]) for f in sub_exp_files))
        
        # Create a color map for sub-experiments
        colors = plt.cm.rainbow(np.linspace(0, 1, len(sub_exp_ids)))
        color_map = dict(zip(sub_exp_ids, colors))
        
        for sub_exp_id in sub_exp_ids:
            print(f"Processing sub-experiment {sub_exp_id}")
            
            # Load traces for this sub-experiment
            traces_data = []
            trace_files = list(self.directory.glob(f"batch_{self.batch_id}_{sub_exp_id}_*_traces.json"))
            
            for file_path in trace_files:
                run_id = int(file_path.stem.split('_')[3])
                
                with open(file_path) as f:
                    traces = json.load(f)
                    for trace in traces:
                        filtered_trace = {
                            'duration': float(trace['duration']) / 1000,  # Convert to ms
                            'timestamp': pd.to_datetime(float(trace['start_time']), unit='us'),
                            'service_name': trace['service_name'],
                            'sub_experiment_id': sub_exp_id,
                            'run_id': run_id
                        }
                        traces_data.append(filtered_trace)
            
            # Convert to DataFrame and process
            sub_exp_df = pd.DataFrame(traces_data)
            
            # Filter services first
            if self.service_filter:
                sub_exp_df = sub_exp_df[sub_exp_df['service_name'].isin(self.service_filter)]
            
            if not sub_exp_df.empty:
                # Normalize timestamps within each run
                min_time = sub_exp_df['timestamp'].min()
                max_time = sub_exp_df['timestamp'].max()
                sub_exp_df['normalized_time'] = (sub_exp_df['timestamp'] - min_time) / (max_time - min_time)
                
                # Bin the data into 100 bins and calculate mean duration for each bin
                bins = np.linspace(0, 1, 100)
                binned_data = pd.cut(sub_exp_df['normalized_time'], bins, observed=True)
                mean_durations = sub_exp_df.groupby(binned_data)['duration'].mean()
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Plot smoothed trace durations
                plt.plot(bin_centers,
                        mean_durations,
                        label=f'Traces Sub-exp {sub_exp_id}',
                        color=color_map[sub_exp_id],
                        alpha=0.7,
                        linewidth=2)
                
                # Normalize and plot alert points for this sub-experiment
                sub_exp_alerts = alerts_df[
                    (alerts_df['sub_experiment_id'] == sub_exp_id) & 
                    (alerts_df['value'] == 1)  # Only plot when alert is firing
                ].copy()  # Create copy to avoid SettingWithCopyWarning
                
                if not sub_exp_alerts.empty:
                    # Normalize alert timestamps
                    sub_exp_alerts.loc[:, 'normalized_time'] = (
                        sub_exp_alerts['timestamp'] - min_time) / (max_time - min_time)
                    
                    plt.scatter(sub_exp_alerts['normalized_time'],
                              [mean_durations.max() * 1.1] * len(sub_exp_alerts),  # Position above the lines
                              marker='v',
                              color=color_map[sub_exp_id],
                              s=100,
                              label=f'Alerts Sub-exp {sub_exp_id}')
            
            # Clear memory
            del traces_data
            del sub_exp_df
        
        plt.title(f'Trace Durations with Alert Firing Times\nService: {", ".join(self.service_filter)}')
        plt.xlabel('Normalized Time')
        plt.ylabel('Duration (ms)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add configuration details to plot
        config_text = "Configurations:\n"
        for sub_exp_id in sub_exp_ids:
            params = self.params_mapping[sub_exp_id]
            config_text += (f"Sub-exp {sub_exp_id}: "
                          f"Threshold={params['experiment.treatments.0.kubernetes_prometheus_rules.params.latency_threshold']}ms, "
                          f"Window={params['experiment.treatments.0.kubernetes_prometheus_rules.params.evaluation_window']}\n")
        
        plt.figtext(0.02, 0.02, config_text, fontsize=8, va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'alerts_over_traces_{self.treatment_type}.png', bbox_inches='tight', dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment traces')
    parser.add_argument('directory', help='Directory containing experiment files')
    parser.add_argument('treatment_type', help='Type of treatment to analyze', choices=['loss_treatment', 'delay_treatment'])
    args = parser.parse_args()

    service_filter = ["recommendationservice"]

    analyzer = ExperimentAnalyzer(args.directory, args.treatment_type, service_filter)
    #analyzer._plot_fault_detection_latency_and_false_alerts()
    analyzer._plot_alerts_over_traces()
if __name__ == "__main__":
    main()