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
ALERTS_FILTER_OUT = ["ImmediateHighHTTPLatency"]

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
                        # filter out the alerts that are in the ALERTS_FILTER_OUT list
                        result["true_positives"] = [tp for tp in result["true_positives"] if tp["name"] not in ALERTS_FILTER_OUT]
                        result["false_positives"] = [fp for fp in result["false_positives"] if fp["name"] not in ALERTS_FILTER_OUT]
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
                    if tp['name'] in ALERTS_FILTER_OUT:
                        continue
                    alert_time = pd.to_datetime(tp['time'])
                    if not (fault_start <= alert_time <= fault_end):
                        validation_issues.append(
                            f"Sub-exp {sub_exp_id}: True positive alert at {alert_time} "
                            f"outside fault interval [{fault_start}, {fault_end}]"
                        )
                true_positives_count += len(row.true_positives)
                
                # Validate false positives
                for fp in row.false_positives:
                    if fp['name'] in ALERTS_FILTER_OUT:
                        continue
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

        #self.plot_traces_with_fault_detection(self.load_traces(), 4)

    def plot_fault_detection_metrics(self):
        """Create visualizations to analyze fault detection performance metrics."""
        self.load_params_mapping()
        self.load_fault_detection(fault_filter_out=["kubernetes_prometheus_rules", "add_security_context"])
        
        # Prepare data for plotting
        plot_data = []
        
        # Group by sub_experiment_id. For each sub-experiment, sort the faults by start time,
        # then filter out false positives that fall within any earlier true-positive interval.
        for sub_exp_id, group in self.fault_detection_df.groupby("sub_experiment_id"):
            # Sort faults by start_time so we can properly exclude FP alerts already covered by previous faults.
            sorted_group = group.sort_values("start_time")
            true_positives_count = 0
            false_positives_count = 0
            previous_intervals = []  # List to store intervals (start_time, end_time) for faults with true positives.

            for row in sorted_group.itertuples():
                # Count true positives normally.
                tp_list = row.true_positives if row.true_positives is not None else []
                tp_count = len(tp_list)
                true_positives_count += tp_count

                # Process false positives: filter out alerts that fall in any previously recorded true positive fault interval.
                fp_list = row.false_positives if row.false_positives is not None else []
                fp_filtered = []
                for fp in fp_list:
                    alert_time = pd.to_datetime(fp['time'])
                    # If alert time is within any previously added fault interval, do not count it as a false positive.
                    if any(start <= alert_time <= end for (start, end) in previous_intervals):
                        continue
                    fp_filtered.append(fp)
                false_positives_count += len(fp_filtered)

                # If this fault record contained any true positives, record its interval so that later FPs can be filtered.
                if tp_count > 0:
                    fault_start = pd.to_datetime(row.start_time)
                    fault_end = pd.to_datetime(row.end_time)
                    previous_intervals.append((fault_start, fault_end))
            
            # Use experiment parameters from the params mapping for details.
            params = self.params_mapping[sub_exp_id]
            plot_data.append({
                'sub_experiment_id': sub_exp_id,
                'latency_threshold': float(params['treatments.0.kubernetes_prometheus_rules.params.latency_threshold']),
                'evaluation_window': float(params['treatments.0.kubernetes_prometheus_rules.params.evaluation_window'].rstrip('s')),
                'quantile': float(params['treatments.0.kubernetes_prometheus_rules.params.quantile']),
                'detection_latency': group['detection_latency'].mean(),
                'false_positives': false_positives_count,
                'true_positives': true_positives_count
            })
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create a figure with multiple subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Heatmap showing detection latency vs threshold and quantile.
        pivot_latency = df_plot.pivot(
            index='latency_threshold',
            columns='quantile',
            values='detection_latency'
        )
        sns.heatmap(pivot_latency, annot=True, fmt='.1f', ax=ax1, cmap='YlOrRd')
        ax1.set_title('Detection Latency (seconds)')
        ax1.set_xlabel('Quantile')
        ax1.set_ylabel('Latency Threshold (seconds)')
        
        # 2. Heatmap showing false positives vs threshold and quantile.
        pivot_fp = df_plot.pivot(
            index='latency_threshold',
            columns='quantile',
            values='false_positives'
        )
        sns.heatmap(pivot_fp, annot=True, fmt='.0f', ax=ax2, cmap='YlOrRd')
        ax2.set_title('False Positives Count')
        ax2.set_xlabel('Quantile')
        ax2.set_ylabel('Latency Threshold (seconds)')
        
        # 3. Scatter plot with detection latency vs false positives.
        sns.scatterplot(
            data=df_plot,
            x='detection_latency',
            y='false_positives',
            size='quantile',
            hue='latency_threshold',
            ax=ax3
        )
        ax3.set_title('Detection Latency vs False Positives')
        ax3.set_xlabel('Detection Latency (seconds)')
        ax3.set_ylabel('False Positives Count')
        
        # 4. Bar plot showing true positives and false positives.
        sns.set_theme(style="whitegrid")
        bar_plot_data = []
        thresholds = df_plot['latency_threshold'].unique()
        for threshold in thresholds:
            threshold_data = df_plot[df_plot['latency_threshold'] == threshold]
            bar_plot_data.append({
                'threshold': threshold,
                'true_positives': threshold_data['true_positives'].sum(),
                'false_positives': threshold_data['false_positives'].sum()
            })
        plot_df = pd.DataFrame(bar_plot_data)
        
        if plot_df['true_positives'].sum() < plot_df['false_positives'].sum():
            sns.set_color_codes("muted")
            sns.barplot(x="false_positives", y="threshold", data=plot_df,
                        label="False Positives", color="red", ax=ax4, orient="h")
        
            sns.set_color_codes("pastel")
            sns.barplot(x="true_positives", y="threshold", data=plot_df,
                        label="True Positives", color="green", ax=ax4, orient="h")
        else:
            sns.set_color_codes("pastel")
            sns.barplot(x="true_positives", y="threshold", data=plot_df,
                        label="True Positives", color="green", ax=ax4, orient="h")
            sns.set_color_codes("muted")
            sns.barplot(x="false_positives", y="threshold", data=plot_df,
                        label="False Positives", color="red", ax=ax4, orient="h")
        
        ax4.legend(ncol=2, loc="lower right", frameon=True)
        ax4.set(ylabel="Latency Threshold (seconds)", xlabel="Number of Alerts")
        ax4.set_title('True vs False Positives by Threshold')
        sns.despine(left=True, bottom=True, ax=ax4)
        
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
        #df['normalized_time'] = df.groupby('sub_experiment_id')['start_time'].transform(
        #    lambda x: (x - x.min()) / (x.max() - x.min())
        #)
        df['normalized_time'] = df['start_time']
        
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
                hue=self.treatment_type,
                alpha=0.7
            )
        
        plt.title('Trace Duration by Service')
        plt.xlabel('Timestamps')
        plt.ylabel('Duration (ms)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{self.plots_out_dir}/trace_duration_by_service_{self.treatment_type}.png', bbox_inches='tight')
        plt.close()

    def plot_traces_with_fault_detection(self, df, sub_exp_id):
        """Plot traces with fault detection for a single sub-experiment.

        This plot is similar to the one generated in plot_trace_duration_by_service but overlays vertical dashed lines
        at the fault start and end times indicated by the fault detection data (i.e. where alerts were triggered).
        
        Args:
            df (pd.DataFrame): The traces DataFrame.
            sub_exp_id (int): The sub-experiment ID for which to plot the traces.
        """
        # Filter traces for the provided sub_experiment_id
        df_sub = df[df['sub_experiment_id'] == sub_exp_id].copy()
        if df_sub.empty:
            print(f"No traces found for sub-experiment {sub_exp_id}.")
            return

        # Ensure start_time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sub['start_time']):
            # Convert microseconds since epoch to datetime
            df_sub['start_time'] = pd.to_datetime(df_sub['start_time'], unit='us')
        
        df_sub['normalized_time'] = df_sub['start_time']

        # Convert duration from microseconds to milliseconds
        df_sub['duration'] = df_sub['duration'] / 1000

        plt.figure(figsize=(15, 8))
        services = ["recommendationservice"]
        # Plot line for each service in the sub-experiment
        for service in df_sub['service_name'].unique():
            if service not in services:
                continue
            service_data = df_sub[df_sub['service_name'] == service].sort_values('normalized_time')
            sns.lineplot(
                data=service_data,
                x='normalized_time',
                y='duration',
                label=service,
                alpha=0.7
            )

        plt.title(f"Trace Duration with Fault Detection for Sub-experiment {sub_exp_id}")
        plt.xlabel("Timestamps")
        plt.ylabel("Duration (ms)")
        
        # Load fault detection data if not already loaded
        if self.fault_detection_df is None:
            self.load_fault_detection()
        
        # Filter fault detection data for the given sub_experiment_id
        fault_data = self.fault_detection_df[self.fault_detection_df['sub_experiment_id'] == sub_exp_id]
        if fault_data.empty:
            print(f"No fault detection data found for sub-experiment {sub_exp_id}.")
        else:
            fault_data = fault_data.copy()  # To avoid SettingWithCopyWarning
            fault_data['start_time'] = pd.to_datetime(fault_data['start_time'])
            fault_data['end_time'] = pd.to_datetime(fault_data['end_time'])
            # Plot vertical lines for fault detection start and end times.
            # We only label the first occurrence of the vertical line to avoid duplicate legend entries.
            fault_start_plotted = False
            fault_end_plotted = False
            fault_start = None
            seen_alert_times = set()
            for _, row in fault_data.iterrows():

                fault_start = row['start_time']
                fault_end = row['end_time']
                """ if not fault_start_plotted:
                    plt.axvline(
                        x=fault_start, color='red', linestyle='--', linewidth=1.5, label="Fault Start"
                    )
                    fault_start_plotted = True
                else:
                    plt.axvline(
                        x=fault_start, color='red', linestyle='--', linewidth=1.5
                    )
                if not fault_end_plotted:
                    plt.axvline(
                        x=fault_end, color='blue', linestyle='--', linewidth=1.5, label="Fault End"
                    )
                    fault_end_plotted = True
                else:
                    plt.axvline(
                        x=fault_end, color='blue', linestyle='--', linewidth=1.5
                    ) """
                # Plot the alerts (convert alert times to datetime)
                for true_positive in row['true_positives']:
                    alert_time = pd.to_datetime(true_positive['time'])
                    if alert_time not in seen_alert_times:
                        plt.axvline(
                            x=alert_time, color='green', linestyle='--', linewidth=1.5
                        )
                        seen_alert_times.add(alert_time)
                for false_positive in row['false_positives']:
                    alert_time = pd.to_datetime(false_positive['time'])
                    if alert_time not in seen_alert_times:
                        plt.axvline(
                            x=alert_time, color='red', linestyle='--', linewidth=1.5
                        )
                        seen_alert_times.add(alert_time)
                break
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        output_file = f"{self.plots_out_dir}/trace_with_fault_detection_subexp_{sub_exp_id}.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {output_file}")

    def analyse_raw_alerts(self):
        """Analyse the raw alerts and verify if they triggered during fault injection periods."""
        self.load_raw_alerts()
        self.load_fault_detection()

        all_alerts = []
        
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
                # Extract metric information
                metric = alert['metric']
                all_alerts.append({
                    'timestamp': alert['timestamp'],
                    'sub_experiment_id': alert['sub_experiment_id'],
                    'alert_name': metric['alertname'],
                    'alert_state': metric['alertstate'],
                    'value': alert['value']
                })
                
                alert_time = alert['timestamp'].tz_localize('UTC')
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

        # Convert all_alerts to DataFrame for plotting
        alerts_df = pd.DataFrame(all_alerts)
        
        # Only plot alerts that are firing (value == 1)
        firing_alerts = alerts_df[alerts_df['value'] == '1']

        # Analyse how many times per second each alert fires
        alert_rates = []
        for sub_exp_id, sub_exp_alerts in firing_alerts.groupby('sub_experiment_id'):
            # Get experiment parameters for context
            params = self.params_mapping[sub_exp_id]
            latency_threshold = params.get('treatments.0.kubernetes_prometheus_rules.params.latency_threshold', 'N/A')
            evaluation_window = params.get('treatments.0.kubernetes_prometheus_rules.params.evaluation_window', 'N/A')
            
            # Calculate time range for this sub-experiment
            time_range = (sub_exp_alerts['timestamp'].max() - sub_exp_alerts['timestamp'].min()).total_seconds()
            
            # Group by alert name and calculate rates
            for alert_name, alert_group in sub_exp_alerts.groupby('alert_name'):
                total_alerts = len(alert_group)
                rate = total_alerts / time_range if time_range > 0 else 0
                
                alert_rates.append({
                    'sub_experiment_id': sub_exp_id,
                    'alert_name': alert_name,
                    'total_alerts': total_alerts,
                    'time_range_seconds': time_range,
                    'alerts_per_second': rate,
                    'latency_threshold': latency_threshold,
                    'evaluation_window': evaluation_window
                })
        
        # Convert to DataFrame for analysis and plotting
        rates_df = pd.DataFrame(alert_rates)
        
        # Print summary statistics
        print("\nAlert Firing Rates Summary:")
        for sub_exp_id, group in rates_df.groupby('sub_experiment_id'):
            print(f"\nSub-experiment {sub_exp_id}:")
            print(f"Latency Threshold: {group['latency_threshold'].iloc[0]}")
            print(f"Evaluation Window: {group['evaluation_window'].iloc[0]}")
            print("\nAlert firing rates (alerts/second):")
            for _, row in group.iterrows():
                print(f"{row['alert_name']}: {row['alerts_per_second']:.3f} "
                      f"(Total: {row['total_alerts']} over {row['time_range_seconds']:.1f}s)")
        
        # plot the alerts during faults and outside faults
        plt.figure(figsize=(15, 8))
        sns.scatterplot(
            data=firing_alerts,
            x='timestamp',
            y='sub_experiment_id',
            hue='alert_state',
            style='alert_name',
            s=100
        )
        plt.title('Alert Firing Times by Sub-experiment')
        plt.xlabel('Time')
        plt.ylabel('Sub-experiment ID')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.plots_out_dir}/alerts_during_faults_and_outside_faults.png')
        plt.close()

    def analyse_traces(self):
        """Analyse the traces timing and counts per sub-experiment."""
        self.load_traces()
        
        print("\nTrace Analysis Summary:")
        print("=" * 50)
        
        # Group traces by sub-experiment
        for sub_exp_id, traces_group in self.traces_df.groupby('sub_experiment_id'):
            print(f"\nSub-experiment {sub_exp_id}:")
            
            # Get experiment parameters for context
            if self.params_mapping and sub_exp_id in self.params_mapping:
                params = self.params_mapping[sub_exp_id]
                print(f"Configuration:")
                print(f"- Latency threshold: {params.get('treatments.0.kubernetes_prometheus_rules.params.latency_threshold', 'N/A')}")
                print(f"- Evaluation window: {params.get('treatments.0.kubernetes_prometheus_rules.params.evaluation_window', 'N/A')}")
            
            # Time analysis
            start_time = pd.to_datetime(traces_group['start_time'].min(), unit='us')
            end_time = pd.to_datetime(traces_group['start_time'].max(), unit='us')
            duration = (end_time - start_time).total_seconds()
            
            print("\nTiming:")
            print(f"- Start time: {start_time}")
            print(f"- End time: {end_time}")
            print(f"- Duration: {duration:.2f} seconds")
            
            # Count analysis
            total_traces = len(traces_group)
            traces_per_second = total_traces / duration if duration > 0 else 0
            
            print("\nCounts:")
            print(f"- Total traces: {total_traces}")
            print(f"- Traces per second: {traces_per_second:.2f}")
            
            # Service breakdown
            print("\nService breakdown:")
            service_counts = traces_group['service_name'].value_counts()
            for service, count in service_counts.items():
                percentage = (count / total_traces) * 100
                print(f"- {service}: {count} traces ({percentage:.1f}%)")
            
            # Operation breakdown (top 5)
            print("\nTop 5 operations:")
            op_counts = traces_group['operation'].value_counts().head()
            for op, count in op_counts.items():
                percentage = (count / total_traces) * 100
                print(f"- {op}: {count} traces ({percentage:.1f}%)")
            
            # Duration statistics (in milliseconds)
            durations_ms = traces_group['duration'] / 1000
            print("\nDuration statistics (ms):")
            print(f"- Mean: {durations_ms.mean():.2f}")
            print(f"- Median: {durations_ms.median():.2f}")
            print(f"- 95th percentile: {durations_ms.quantile(0.95):.2f}")
            print(f"- Max: {durations_ms.max():.2f}")
            
            print("-" * 50)
        
        # Create summary visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Trace counts by sub-experiment and service
        plt.subplot(2, 1, 1)
        trace_counts = self.traces_df.groupby(['sub_experiment_id', 'service_name']).size().unstack()
        trace_counts.plot(kind='bar', stacked=True)
        plt.title('Trace Counts by Sub-experiment and Service')
        plt.xlabel('Sub-experiment ID')
        plt.ylabel('Number of Traces')
        plt.legend(title='Service', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Duration distribution by sub-experiment
        plt.subplot(2, 1, 2)
        sns.boxplot(
            data=self.traces_df,
            x='sub_experiment_id',
            y='duration',
            showfliers=False  # Exclude outliers for better visualization
        )
        plt.title('Trace Duration Distribution by Sub-experiment')
        plt.xlabel('Sub-experiment ID')
        plt.ylabel('Duration (microseconds)')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_out_dir}/trace_analysis_summary.png', bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment traces')
    parser.add_argument('directory', help='Directory containing experiment files')
    parser.add_argument('treatment_type', help='Type of treatment to analyze', choices=['packet_loss_treatment', 'delay_treatment'])
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
    #analyzer.analyse_raw_alerts()
    analyzer.analyse_traces()
if __name__ == "__main__":
    main()