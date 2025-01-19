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
        trace_files = list(self.directory.glob(f"batch_{self.batch_id}_*_traces.json"))
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

    def load_configs(self) -> Dict[int, Dict]:
        """Load all configuration files."""
        configs = {}
        config_files = list(self.directory.glob(f"batch_{self.batch_id}_*_config.json"))
        
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
        # Plot trace duration by service
        sns.boxplot(x='service_name', y='duration', data=self.traces_df)
        plt.title('Trace Duration by Service')
        plt.xlabel('Service')
        plt.ylabel('Duration (ms)')
        plt.savefig(f'trace_duration_by_service_{self.treatment_type}.png')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment traces')
    parser.add_argument('directory', help='Directory containing experiment files')
    parser.add_argument('treatment_type', help='Type of treatment to analyze', choices=['loss_treatment', 'delay_treatment'])
    args = parser.parse_args()

    service_filter = ["recommendationservice", "frontend"]

    analyzer = ExperimentAnalyzer(args.directory, args.treatment_type, service_filter)
    analyzer.analyze()
    analyzer._plot_latency_vs_time()
    analyzer._plot_cpu_usage_over_time()
    analyzer._plot_node_cpu_mem_over_time()
    analyzer._plot_namespace_cpu_mem_over_time()
    analyzer._plot_trace_duration_by_service()
if __name__ == "__main__":
    main()