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
        self.alerts_df = None
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

    def analyse_alerts(self):

        self.load_traces()
        print(f"Loaded {len(self.traces_df)} traces")
        self.load_raw_alerts()
        print(f"Loaded {len(self.alerts_df)} alerts")
        self.load_reports()
        print(f"Loaded {len(self.reports)} reports")

        # Annotate the alerts with the treatment that was applied
        self.alerts_df["treatment"] = "no_treatment"  # Default value
        self.alerts_df["detection_latency"] = 0
        for sub_exp_id, report in self.reports.items():
            mask = self.alerts_df["sub_experiment_id"] == sub_exp_id
            print(f"Found {mask.sum()} alerts for sub experiment {sub_exp_id}")
            for run_id, run_data in report["report"]["runs"].items():
                for interaction_id, interaction_data in run_data["interactions"].items():
                    if interaction_data["treatment_name"] == self.treatment_type:
                        # Convert treatment timestamps to datetime objects
                        treatment_start = pd.to_datetime(interaction_data["treatment_start"])
                        treatment_end = pd.to_datetime(interaction_data["treatment_end"])
                        
                        self.alerts_df.loc[mask, "treatment"] = self.alerts_df[mask]["timestamp"].apply(
                            lambda x: "treatment" if treatment_start <= x <= treatment_end else "no_treatment"
                        )
                        self.alerts_df.loc[mask, "detection_latency"] = self.alerts_df[mask]["timestamp"] - treatment_start
                        break
                break

        # Get the reference time (minimum from traces)
        reference_time = self.traces_df["loadgen_start_time"].min()

        # Normalise trace time
        self.traces_df["normalized_time"] = self.traces_df["loadgen_start_time"] - reference_time

        # Normalize alert times
        self.alerts_df["normalized_time"] = self.alerts_df["timestamp"] - reference_time

        print(f"Minimal detection latency: {self.alerts_df['detection_latency'].min()}")
        print(f"Maximal detection latency: {self.alerts_df['detection_latency'].max()}")

        # Plot the alerts over the traces
        plt.figure(figsize=(10, 6))
        sns.lineplot(x="normalized_time", y="detection_latency", data=self.alerts_df, hue="metric", style="metric", markers=True, dashes=False)
        plt.xlabel("Time (s)")
        plt.ylabel("Alert Value")
        plt.title("Alerts Over Traces")
        plt.legend(title="Metric")
        plt.savefig(f"{self.directory}/alerts_over_traces.png")

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment traces')
    parser.add_argument('directory', help='Directory containing experiment files')
    parser.add_argument('treatment_type', help='Type of treatment to analyze', choices=['loss_treatment', 'delay_treatment'])
    args = parser.parse_args()

    service_filter = ["recommendationservice"]

    analyzer = ExperimentAnalyzer(args.directory, args.treatment_type, service_filter)
    #analyzer._plot_fault_detection_latency_and_false_alerts()
    analyzer.analyse_alerts()
if __name__ == "__main__":
    main()