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
        self.plots_out_dir = None
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
        """Load all fault detection files for the batch."""
        fault_detection_files = list(self.directory.glob(f"batch_{self.batch_id}_*_fault_detection.json"))
        print(f"{len(fault_detection_files)} fault detection files detected")
        fault_detection_data = []
        for file_path in fault_detection_files:
            sub_exp_id = int(file_path.stem.split('_')[2])
            with open(file_path) as f:
                fault_detection = json.load(f)
                # Changed to handle fault_detection as a list directly
                for result in fault_detection:  # Removed ["results"] access
                    if result["fault_name"] not in fault_filter_out:
                        result["sub_experiment_id"] = sub_exp_id
                        if self.params_mapping is not None:
                            result["params"] = self.params_mapping[sub_exp_id]
                        fault_detection_data.append(result)
        self.fault_detection_df = pd.DataFrame(fault_detection_data)
        # turn the start_time and end_time into datetime
        self.fault_detection_df['start_time'] = pd.to_datetime(self.fault_detection_df['start_time'])
        self.fault_detection_df['end_time'] = pd.to_datetime(self.fault_detection_df['end_time'])
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
        validation_issues = []
        
        for sub_exp_id, group in self.fault_detection_df.groupby("sub_experiment_id"):
            true_positives_count = 0
            false_positives_count = 0
            
            for row in group.itertuples():
                fault_start = row.start_time
                fault_end = row.end_time
                
                # Validate true positives
                for tp in row.true_positives:
                    alert_time = pd.to_datetime(tp['time'])
                    if not (fault_start <= alert_time <= fault_end):
                        validation_issues.append(
                            f"Sub-exp {sub_exp_id}: True positive alert at {alert_time} "
                            f"outside fault interval [{fault_start}, {fault_end}]"
                        )
                true_positives_count += len(row.true_positives)
                
                # Validate false positives
                for fp in row.false_positives:
                    alert_time = pd.to_datetime(fp['time'])
                    if fault_start <= alert_time <= fault_end:
                        validation_issues.append(
                            f"Sub-exp {sub_exp_id}: False positive alert at {alert_time} "
                            f"inside fault interval [{fault_start}, {fault_end}]"
                        )
                false_positives_count += len(row.false_positives)
            
            print(f"\nSubexperiment {sub_exp_id} - Threshold: {self.params_mapping[sub_exp_id]['treatments.0.kubernetes_prometheus_rules.params.latency_threshold']} - Evaluation Window: {self.params_mapping[sub_exp_id]['treatments.0.kubernetes_prometheus_rules.params.evaluation_window']}")
            print(f"Detection latency: {group['detection_latency'].mean()}s")
            print(f"True Positives: {true_positives_count}")
            print(f"False Positives: {false_positives_count}")
    
        # Print validation issues if any were found
        if validation_issues:
            print("\nValidation Issues Found:")
            for issue in validation_issues:
                print(f"- {issue}")
        else:
            print("\nNo validation issues found - all alerts are correctly categorized!")

    def plot_fault_detection_metrics(self):
        """Create visualizations to analyze fault detection performance metrics."""
        self.load_params_mapping()
        self.load_fault_detection(fault_filter_out=["kubernetes_prometheus_rules", "add_security_context"])
        
        # Prepare data for plotting
        plot_data = []
        
        for sub_exp_id, group in self.fault_detection_df.groupby("sub_experiment_id"):
            true_positives_count = sum(len(row.true_positives) for row in group.itertuples())
            false_positives_count = sum(len(row.false_positives) for row in group.itertuples())
            
            params = self.params_mapping[sub_exp_id]
            plot_data.append({
                'sub_experiment_id': sub_exp_id,
                'latency_threshold': float(params['treatments.0.kubernetes_prometheus_rules.params.latency_threshold']),
                'evaluation_window': float(params['treatments.0.kubernetes_prometheus_rules.params.evaluation_window'].rstrip('s')),
                'detection_latency': group['detection_latency'].mean(),
                'false_positives': false_positives_count,

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
        plt.savefig(f'{self.plots_out_dir}/fault_detection_analysis.png')
        plt.close()

    def plot_trace_duration_by_service(self, service_filter=None):
        """Plot trace durations by service, aggregating across all sub-experiments.
        
        Args:
            service_filter (list): Optional list of service names to filter by
        """
        plt.figure(figsize=(15, 8))
        df = self.load_traces()
        reports_df = self.reports

        # Filter services if specified
        if service_filter:
            df = df[df['service_name'].isin(service_filter)]
        
        # Normalize time across all sub experiments
        df['normalized_time'] = df.groupby('sub_experiment_id')['start_time'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )
        
        # Convert duration from microseconds to milliseconds
        df['duration'] = df['duration'] / 1000

        # Plot data for each service
        for service in df['service_name'].unique():

            service_data = df[df['service_name'] == service]
            
            # Calculate rolling mean for smoother lines
            service_data = service_data.sort_values('normalized_time')
            rolling_mean = service_data.groupby('service_name')['duration'].rolling(
                window=100, min_periods=1, center=True
            ).mean().reset_index(level=0, drop=True)
            
            sns.lineplot(
                data=service_data,
                x='normalized_time',
                y='duration',
                hue='delay_treatment',
                alpha=0.7
            )
        
        plt.title('Trace Duration by Service')
        plt.xlabel('Normalized Time')
        plt.ylabel('Duration (ms)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{self.plots_out_dir}/trace_duration_by_service_{self.treatment_type}.png', bbox_inches='tight')
        plt.close()

    def analyse_raw_alerts(self):
        """Analyse the raw alerts and verify if they triggered during fault injection periods."""
        self.load_raw_alerts()
        self.load_fault_detection()
        
        # Group alerts by sub_experiment_id
        for sub_exp_id, alerts_group in self.alerts_df.groupby('sub_experiment_id'):
            print(f"\nAnalyzing alerts for sub-experiment {sub_exp_id}")
            
            # Get fault injection periods for this sub-experiment
            fault_periods = self.fault_detection_df[
                self.fault_detection_df['sub_experiment_id'] == sub_exp_id
            ]
            
            if fault_periods.empty:
                print(f"No fault periods found for sub-experiment {sub_exp_id}")
                continue
            
            total_alerts = len(alerts_group)
            alerts_during_faults = 0
            alerts_outside_faults = 0
            
            # Analyze each alert
            for _, alert in alerts_group.iterrows():
                alert_time = alert['timestamp']
                alert_in_fault_period = False
                
                # Check if alert falls within any fault period
                for _, fault in fault_periods.iterrows():
                    if fault['start_time'] <= alert_time <= fault['end_time']:
                        alert_in_fault_period = True
                        alerts_during_faults += 1
                        break
                
                if not alert_in_fault_period:
                    alerts_outside_faults += 1
            
            # Print statistics
            print(f"Total alerts: {total_alerts}")
            print(f"Alerts during fault injection: {alerts_during_faults}")
            print(f"Alerts outside fault injection: {alerts_outside_faults}")
            
            if total_alerts > 0:
                during_percentage = (alerts_during_faults / total_alerts) * 100
                outside_percentage = (alerts_outside_faults / total_alerts) * 100
                print(f"Percentage during faults: {during_percentage:.2f}%")
                print(f"Percentage outside faults: {outside_percentage:.2f}%")
                
                # Get experiment parameters if available
                if self.params_mapping and sub_exp_id in self.params_mapping:
                    params = self.params_mapping[sub_exp_id]
                    print("\nExperiment parameters:")
                    print(f"Latency threshold: {params.get('treatments.0.kubernetes_prometheus_rules.params.latency_threshold', 'N/A')}")
                    print(f"Evaluation window: {params.get('treatments.0.kubernetes_prometheus_rules.params.evaluation_window', 'N/A')}")
def main():
    parser = argparse.ArgumentParser(description='Analyze experiment traces')
    parser.add_argument('directory', help='Directory containing experiment files')
    parser.add_argument('treatment_type', help='Type of treatment to analyze', choices=['loss_treatment', 'delay_treatment'])
    args = parser.parse_args()


    service_filter = ["recommendationservice"]

    analyzer = ExperimentAnalyzer(args.directory, args.treatment_type, service_filter)
    analyzer.plots_out_dir = args.directory + "/plots"
    # Create plots directory if it doesn't exist
    Path(analyzer.plots_out_dir).mkdir(parents=True, exist_ok=True)
    #analyzer._plot_fault_detection_latency_and_false_alerts()
    #analyzer.analyse_fault_detection(fault_filter_out=["kubernetes_prometheus_rules", "add_security_context"])
    analyzer.analyse_fault_detection(fault_filter_out=["kubernetes_prometheus_rules", "add_security_context"])
    analyzer.plot_fault_detection_metrics()
    analyzer.plot_trace_duration_by_service(service_filter=service_filter)
    analyzer.analyse_raw_alerts()
if __name__ == "__main__":
    main()