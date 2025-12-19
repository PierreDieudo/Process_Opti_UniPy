import os
import pandas as pd
import numpy as np
from collections import deque
import openpyxl
from openpyxl import load_workbook

class OptimisationLogger:
    def __init__(self, log_path, log_interval=100, rolling_window=200):
        self.results_path = log_path
        os.makedirs(self.results_dir, exist_ok=True)

        # Paths
        self.success_log_path = self.results_path

        # Counters
        self.attempted_run = 0
        self.success = 0
        self.failed = 0

        # Buffers
        self.log_interval = log_interval
        self.rolling_window = rolling_window
        self.success_log_buffer = []
        self.rolling_log_buffer = []
        self.rolling_buffer = deque(maxlen=self.rolling_window)

        # Initialise summary dataframe
        self.opti_tracking = pd.DataFrame({
            "Total Evaluations": pd.Series(dtype=int),
            "Successful Evaluations": pd.Series(dtype=int),
            "Failed Evaluations": pd.Series(dtype=int),
            "Success Percentage": pd.Series(dtype=float)
        })

        # Track if sheets exist
        self.excel_initialised = {
            "Raw_Logs": False,
            f"Rolling_MA_{self.rolling_window}": False
        }

        # Initialise Excel file with empty sheets
        with pd.ExcelWriter(self.success_log_path, engine="openpyxl") as writer:
            pd.DataFrame(columns=[
                "Run", "Param_1","Param_2","Param_3","Param_4",
                "Param_5","Param_6","Param_7","Param_8",
                "Evaluation","Purity","Recovery","TAC_CC",
                "Power_Consumption","Cost_of_Capture","SPECCA"
            ]).to_excel(writer, sheet_name="Raw_Logs", index=False)
            pd.DataFrame(columns=[
                "Run","Param_1","Param_2","Param_3","Param_4",
                "Param_5","Param_6","Param_7","Param_8",
                "Evaluation","Purity","Recovery","TAC_CC",
                "Power_Consumption","Cost_of_Capture","SPECCA"
            ]).to_excel(writer, sheet_name=f"Rolling_MA_{self.rolling_window}", index=False)

    def log(self, params, Economics):
        self.attempted_run += 1

        # Check for success
        is_success = isinstance(Economics, dict) and Economics["Recovery"] <= 1 and Economics["Purity"] <= 1

        if is_success:
            self.success += 1
            evaluation_value = Economics["Evaluation"]

            # Raw entry
            raw_entry = {
                "Run": self.attempted_run,
                "Param_1": params[0],
                "Param_2": params[1],
                "Param_3": params[2],
                "Param_4": params[3],
                "Param_5": params[4],
                "Param_6": params[5],
                "Param_7": params[6],
                "Param_8": params[7],
                "Evaluation": Economics["Evaluation"],
                "Purity": Economics["Purity"],
                "Recovery": Economics["Recovery"],
                "TAC_CC": Economics["TAC_CC"],
                "Power_Consumption": Economics["Power_Consumption"],
                "Cost_of_Capture": Economics["Cost_of_Capture"],
                "SPECCA": Economics["SPECCA"],
            }

            self.rolling_buffer.append(raw_entry)

            # Compute rolling means
            rolling_means = {}
            for key in raw_entry:
                if key != "Run":
                    values = [x[key] for x in self.rolling_buffer]
                    rolling_means[f"{key}_MA{self.rolling_window}"] = float(np.mean(values))
            rolling_means["Run"] = self.attempted_run

            # Add to buffers
            self.success_log_buffer.append(raw_entry)
            self.rolling_log_buffer.append(rolling_means)

            # Flush to Excel automatically every log_interval
            if len(self.success_log_buffer) >= self.log_interval:
                self.flush_to_excel()

        else:
            self.failed += 1
            evaluation_value = 1e10

        # Update summary every 100 attempted runs
        if self.attempted_run % 100 == 0:
            success_percentage = self.success / self.attempted_run * 100
            new_row = pd.DataFrame([{
                "Total Evaluations": self.attempted_run,
                "Successful Evaluations": self.success,
                "Failed Evaluations": self.failed,
                "Success Percentage": success_percentage
            }])
            self.opti_tracking = pd.concat([self.opti_tracking, new_row], ignore_index=True)

        return evaluation_value


    def flush_to_excel(self):
        if not self.success_log_buffer and not self.rolling_log_buffer:
            return

        path = self.success_log_path

        # Ensure the file exists with proper structure
        if not os.path.exists(path):
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame(columns=[
                    "Run","Param_1","Param_2","Param_3","Param_4",
                    "Param_5","Param_6","Param_7","Param_8",
                    "Evaluation","Purity","Recovery","TAC_CC",
                    "Power_Consumption","Cost_of_Capture","SPECCA"
                ]).to_excel(writer, sheet_name="Raw_Logs", index=False)

                pd.DataFrame().to_excel(writer, sheet_name=f"Rolling_MA_{self.rolling_window}", index=False)

        # ---- Append RAW LOGS ----
        if self.success_log_buffer:
            existing = pd.read_excel(path, sheet_name="Raw_Logs")
            new_df = pd.DataFrame(self.success_log_buffer)
            combined = pd.concat([existing, new_df], ignore_index=True)

            with pd.ExcelWriter(path,
                                mode="a",
                                engine="openpyxl",
                                if_sheet_exists="replace") as writer:
                combined.to_excel(writer, sheet_name="Raw_Logs", index=False)

            self.success_log_buffer.clear()

        # ---- Append ROLLING MA ----
        if self.rolling_log_buffer:
            sheet_name = f"Rolling_MA_{self.rolling_window}"
            existing = pd.read_excel(path, sheet_name=sheet_name)
            new_df = pd.DataFrame(self.rolling_log_buffer)
            combined = pd.concat([existing, new_df], ignore_index=True)

            with pd.ExcelWriter(path,
                                mode="a",
                                engine="openpyxl",
                                if_sheet_exists="replace") as writer:
                combined.to_excel(writer, sheet_name=sheet_name, index=False)

            self.rolling_log_buffer.clear()
