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
        self.params_mapping = self.load_params_mapping()
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
        self.alerts_df = None
        self.fault_detection_df = None
    def _get_batch_id(self) -> str:
        """Extract batch ID from the first matching file in directory."""
        files = list(self.directory.glob("batch_*"))
        if not files:
            raise ValueError(f"No batch files found in {self.directory}")
        
        # Extract batch ID from first file
        batch_id = files[0].name.split('_')[1]
        return batch_id

    def load_params_mapping(self) -> Dict[int, Dict[str, Any]]:
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

            actual_run_id = ordered_run_ids[run_id]
            run_data = report_data['report']['runs'][actual_run_id]['loadgen']
            st_timestamp = pd.to_datetime(run_data.get('loadgen_start_time'))
            et_timestamp = pd.to_datetime(run_data.get('loadgen_end_time'))
            # Start loading traces
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
                        trace['loadgen_start_time'] = st_timestamp
                        trace['loadgen_end_time'] = et_timestamp
                
                traces_data.extend(traces)
        
        self.traces_df = pd.DataFrame(traces_data)
        return self.traces_df
    def load_raw_alerts(self):
        """Load all alerts.
        Example structure:
        {
            "detections": {
                "status": "success",
                "data": {
                "result": [
                    {
                    "metric": {
                        "__name__": "LatencyHigh",
                        "alertname": "LatencyHigh",
                        "alertstate": "firing",
                        "container": "recommendationservice",
                        "detection": "rapid",
                        "instance": "10.244.0.245:9100",
                        "job": "kubernetes-service-endpoints/astronomy-shop-recommendationservice",
                        "namespace": "system-under-evaluation",
                        "pod": "astronomy-shop-recommendationservice-75b9c8b864-m57r6",
                        "service": "astronomy-shop-recommendationservice"
                    },
                    "values": [
                        [
                        1737519177,
                        "0"
                        ]
                    ]
                }
                ]
                }
                }
            }

        """
        alerts_files = list(self.directory.glob(f"batch_{self.batch_id}_*_detections.json"))
        alerts_data = []
        for file_path in alerts_files:
            sub_exp_id = int(file_path.stem.split('_')[2])
            with open(file_path) as f:
                d = json.load(f)
                for alert in d["detections"]["data"]["result"]:
                    #print("adding sub experiment id", sub_exp_id)
                    alert["sub_experiment_id"] = sub_exp_id
                alerts_data.extend(d["detections"]["data"]["result"])

        self.alerts_df = pd.DataFrame(alerts_data)
        # turn the timestamps into datetime
        # Explode the values array into separate rows
        self.alerts_df = self.alerts_df.explode('values')
        # Split timestamp and value into separate columns
        self.alerts_df[['timestamp', 'value']] = pd.DataFrame(self.alerts_df['values'].tolist(), index=self.alerts_df.index)
        # Convert Unix timestamp to datetime
        self.alerts_df['timestamp'] = pd.to_datetime(self.alerts_df['timestamp'], unit='s')
        # Drop the original values column as it's no longer needed
        self.alerts_df = self.alerts_df.drop('values', axis=1)
        return self.alerts_df
        
    def load_fault_detection(self, fault_filter_out: List[str] = []):
        """Load all fault detection files for the batch.
        {
            "results": [
                {
                    "fault_name": <fault_name>,
                    "detected": <true/false>,
                    "detection_time": "2025-01-29T20:17:54.655000",
                    "detection_latency": 120.73843,
                    "true_positives": [
                        {
                            "name": "CriticalServiceRPCLatencySpike",
                            "time": "2025-01-29T20:18:09.655000",
                            "severity": "critical",
                            "labels": {"__name__": "ALERTS", "alertname": "CriticalServiceRPCLatencySpike", "alertstate": "firing", "detection": "rapid", "job": "opentelemetry-demo/adservice", "rpc_method": "GetAds", "rpc_service": "oteldemo.AdService", "runb...
                        }
                    ],
                    "false_positives": [
                        {
                            "name": "CriticalServiceRPCLatencySpike",
                            "time": "2025-01-29T20:18:09.655000",
                            "severity": "critical",
                            "labels": {"__name__": "ALERTS", "alertname": "CriticalServiceRPCLatencySpike", "alertstate": "firing", "detection": "rapid", "job": "opentelemetry-demo/adservice", "rpc_method": "GetAds", "rpc_service": "oteldemo.AdService", "runb...
                        }
                    ]
                }
            ]
        }
        
        """
        fault_detection_files = list(self.directory.glob(f"batch_{self.batch_id}_*_fault_detection.json"))
        print(f"{len(fault_detection_files)} fault detection files detected")
        fault_detection_data = []
        for file_path in fault_detection_files:
            sub_exp_id = int(file_path.stem.split('_')[2])
            with open(file_path) as f:
                fault_detection = json.load(f)
                for result in fault_detection["results"]:
                    if result["fault_name"] not in fault_filter_out:
                        result["sub_experiment_id"] = sub_exp_id
                        if self.params_mapping is not None:
                            result["params"] = self.params_mapping[sub_exp_id]
                        fault_detection_data.append(result)
        self.fault_detection_df = pd.DataFrame(fault_detection_data)
        return self.fault_detection_df

    def load_reports(self):
        """Load all reports for the batch."""
        reports = {}
        report_files = list(self.directory.glob(f"batch_{self.batch_id}_*_report.yaml"))
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

    def analyse_fault_detection(self, fault_filter_out: List[str] = []):
        self.load_params_mapping()
        self.load_fault_detection(fault_filter_out)
        
        # Print how many true positives and false positives per subexperiment
        false_positives_total = 0
        for sub_exp_id, group in self.fault_detection_df.groupby("sub_experiment_id"):
            # Sum up all true positives and false positives across all fault detections
            true_positives_count = sum(len(row.true_positives) for row in group.itertuples())
            false_positives_count = sum(len(row.false_positives) for row in group.itertuples())
           # Calculate actual false positives for this experiment by subtracting previous total
            false_positives_real = false_positives_count - false_positives_total
            false_positives_total += false_positives_real
            print(f"Subexperiment {sub_exp_id} - Threshold: {self.params_mapping[sub_exp_id]['experiment.treatments.0.kubernetes_prometheus_rules.params.latency_threshold']} - Evaluation Window: {self.params_mapping[sub_exp_id]['experiment.treatments.0.kubernetes_prometheus_rules.params.evaluation_window']}")
            print(f"Detection latency: {group['detection_latency'].mean()}s")
            print(f"True Positives: {true_positives_count}")
            print(f"False Positives: {false_positives_real}")

    def plot_fault_detection_metrics(self):
        """Create visualizations to analyze fault detection performance metrics."""
        self.load_params_mapping()
        self.load_fault_detection(fault_filter_out=["kubernetes_prometheus_rules", "add_security_context"])
        
        # Prepare data for plotting
        plot_data = []
        false_positives_total = 0
        
        for sub_exp_id, group in self.fault_detection_df.groupby("sub_experiment_id"):
            true_positives_count = sum(len(row.true_positives) for row in group.itertuples())
            false_positives_count = sum(len(row.false_positives) for row in group.itertuples())
            false_positives_real = false_positives_count - false_positives_total
            false_positives_total += false_positives_real
            
            params = self.params_mapping[sub_exp_id]
            plot_data.append({
                'sub_experiment_id': sub_exp_id,
                'latency_threshold': float(params['experiment.treatments.0.kubernetes_prometheus_rules.params.latency_threshold']),
                'evaluation_window': float(params['experiment.treatments.0.kubernetes_prometheus_rules.params.evaluation_window'].rstrip('s')),
                'detection_latency': group['detection_latency'].mean(),
                'false_positives': false_positives_real,
                'true_positives': true_positives_count
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create a figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap showing detection latency vs threshold and window
        pivot_latency = df_plot.pivot(
            index='latency_threshold',
            columns='evaluation_window',
            values='detection_latency'
        )
        sns.heatmap(pivot_latency, annot=True, fmt='.1f', ax=ax1, cmap='YlOrRd')
        ax1.set_title('Detection Latency (seconds)')
        ax1.set_xlabel('Evaluation Window (seconds)')
        ax1.set_ylabel('Latency Threshold (seconds)')
        
        # 2. Heatmap showing false positives vs threshold and window
        pivot_fp = df_plot.pivot(
        index='latency_threshold',
        columns='evaluation_window',
        values='false_positives'
        )
        sns.heatmap(pivot_fp, annot=True, fmt='.0f', ax=ax2, cmap='YlOrRd')  # Changed fmt='d' to fmt='.0f'
        ax2.set_title('False Positives Count')
        ax2.set_xlabel('Evaluation Window (seconds)')
        ax2.set_ylabel('Latency Threshold (seconds)')
        
        # 3. Scatter plot with detection latency vs false positives
        sns.scatterplot(
            data=df_plot,
            x='detection_latency',
            y='false_positives',
            size='evaluation_window',
            hue='latency_threshold',
            ax=ax3
        )
        ax3.set_title('Detection Latency vs False Positives')
        ax3.set_xlabel('Detection Latency (seconds)')
        ax3.set_ylabel('False Positives Count')
        
       # 4. Bar plot showing the truepositive/falsepositive ratio for different thresholds
        thresholds = df_plot['latency_threshold'].unique()
        ratios = []
        for threshold in thresholds:
            threshold_data = df_plot[df_plot['latency_threshold'] == threshold]
            # Calculate average ratio for this threshold
            ratio = threshold_data['true_positives'].sum() / threshold_data['false_positives'].sum()
            ratios.append(ratio)
        
        # Create bar plot
        ax4.bar(
            thresholds,
            ratios,
            width=4, 
            color='orange',
            edgecolor='black'
        )
        
        # Add value labels on top of each bar
        for i, ratio in enumerate(ratios):
            ax4.text(thresholds[i], ratio, f'{ratio:.2f}', 
                    ha='center', va='bottom')
            
        ax4.set_title('True Positives/False Positives Ratio by Threshold')
        ax4.set_xlabel('Latency Threshold (seconds)')
        ax4.set_ylabel('True Positives/False Positives Ratio')
        
        # Add grid for better readability
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('plots/fault_detection_analysis.png')
        plt.close()



def main():
    parser = argparse.ArgumentParser(description='Analyze experiment traces')
    parser.add_argument('directory', help='Directory containing experiment files')
    parser.add_argument('treatment_type', help='Type of treatment to analyze', choices=['loss_treatment', 'delay_treatment'])
    args = parser.parse_args()

    service_filter = ["recommendationservice"]

    analyzer = ExperimentAnalyzer(args.directory, args.treatment_type, service_filter)
    #analyzer._plot_fault_detection_latency_and_false_alerts()
    #analyzer.analyse_fault_detection(fault_filter_out=["kubernetes_prometheus_rules", "add_security_context"])
    analyzer.plot_fault_detection_metrics()
if __name__ == "__main__":
    main()