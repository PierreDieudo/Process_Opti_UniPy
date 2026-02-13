import os
import inspect
from datetime import datetime
from collections import deque
import sys
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pyparsing import originalTextFor


class OptimisationLogger:
    """
    Logger for long optimisation runs.

    - Owns success/failure logic
    - Owns all counters
    - Creates a unique Excel file per run
    - Logs raw results and rolling averages
    """

    METRIC_KEYS = [
        "Evaluation",
        "Purity",
        "Recovery",
        "TAC_CC",
        "Power_Consumption",
        "Cost_of_Capture",
        "SPECCA",
    ]

    # =====================================================
    #                    INIT
    # =====================================================
    def __init__(
        self,
        log_dir,
        log_interval=100,
        rolling_window=100,
        failure_penalty=5e9,
        min_recovery=0.025,
    ):
        # Directory only — filename is auto-generated
        self.log_dir = log_dir or "."
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_interval = log_interval
        self.rolling_window = rolling_window
        self.failure_penalty = failure_penalty
        self.min_recovery = min_recovery

        # Determined on first call
        self.param_count = None
        self.path = None

        # Counters
        self.attempted_run = 0
        self.success = 0
        self.failed = 0

        # Buffers
        self.raw_buffer = []
        self.rolling_buffer = deque(maxlen=rolling_window)
        self.rolling_ma_buffer = []
        self.summary_buffer = []

    # =====================================================
    #               FILE INITIALISATION
    # =====================================================
    def _build_log_filename(self, param_count):
        script = os.path.basename(sys.argv[0]) or "interactive"
        date_str = datetime.now().strftime("%Y-%m-%d")

        base = f"Optimisation_{script}_P{param_count}_{date_str}"
        filename = base + ".xlsx"
        path = os.path.join(self.log_dir, filename)

        run_idx = 1
        while os.path.exists(path):
            run_idx += 1
            path = os.path.join(self.log_dir, f"{base}_run{run_idx}.xlsx")

        return path

    def _initialise_excel(self):
        with pd.ExcelWriter(self.path, engine="openpyxl") as writer:
            pd.DataFrame(columns=self._raw_columns()).to_excel(
                writer, sheet_name="Raw_Logs", index=False
            )
            pd.DataFrame(columns=self._rolling_columns()).to_excel(
                writer,
                sheet_name=f"Rolling_MA_{self.rolling_window}",
                index=False,
            )
            pd.DataFrame(
                columns=[
                    "Total Evaluations",
                    "Successful Evaluations",
                    "Failed Evaluations",
                    "Success Percentage",
                ]
            ).to_excel(writer, sheet_name="Summary", index=False)

    # =====================================================
    #                 COLUMN HELPERS
    # =====================================================
    def _param_columns(self):
        return [f"Param_{i+1}" for i in range(self.param_count)]

    def _raw_columns(self):
        return ["Run"] + self._param_columns() + self.METRIC_KEYS

    def _rolling_columns(self):
        return (
            ["Run"]
            + [f"{p}_MA{self.rolling_window}" for p in self._param_columns()]
            + [f"{m}_MA{self.rolling_window}" for m in self.METRIC_KEYS]
        )

    # =====================================================
    #              SUCCESS CRITERIA
    # =====================================================
    def _is_success(self, Economics):
        try:
            return (
                isinstance(Economics, dict)
                and isinstance(Economics.get("Evaluation"), (int, float))
                and np.isfinite(Economics["Evaluation"])
                and 0 <= Economics["Recovery"] <= 1
                and 0 <= Economics["Purity"] <= 1
                and Economics["SPECCA"] > 0
                and Economics["Recovery"] >= self.min_recovery
            )
        except Exception:
            return False

    # =====================================================
    #                   PUBLIC API
    # =====================================================
    def log(self, params, Economics):
        self.attempted_run += 1

        # First call: infer parameter count and create file
        if self.param_count is None:
            self.param_count = len(params)
            self.path = self._build_log_filename(self.param_count)
            self._initialise_excel()

        if not self._is_success(Economics):
            self.failed += 1
            self._update_summary()
            return self.failure_penalty

        # ---------- Successful run ----------
        self.success += 1
        raw_entry = {"Run": self.attempted_run}
        for i, v in enumerate(params):
            raw_entry[f"Param_{i+1}"] = v

        for k in self.METRIC_KEYS:
            raw_entry[k] = Economics.get(k, np.nan)

        self.raw_buffer.append(raw_entry)
        self.rolling_buffer.append(raw_entry)

        # ---------- Rolling averages ----------
        rolling_entry = {"Run": self.attempted_run}
        for key in raw_entry:
            if key == "Run":
                continue
            vals = [x.get(key, np.nan) for x in self.rolling_buffer]
            arr = np.array([v for v in vals if pd.notna(v)], dtype=float)
            rolling_entry[f"{key}_MA{self.rolling_window}"] = (
                np.mean(arr) if arr.size else np.nan
            )

        self.rolling_ma_buffer.append(rolling_entry)

        if len(self.raw_buffer) >= self.log_interval:
            self.flush()

        self._update_summary()
        return Economics["Evaluation"]

    def _update_summary(self):
        if self.attempted_run % 250 == 0:
            self.summary_buffer.append({
                "Total Evaluations": self.attempted_run,
                "Successful Evaluations": self.success,
                "Failed Evaluations": self.failed,
                "Success Percentage": 100 * self.success / self.attempted_run,
            })

    # =====================================================
    #                 EXCEL APPEND
    # =====================================================
    def _append_df(self, sheet_name, df):
        # Read existing sheet to find startrow
        try:
            existing = pd.read_excel(self.path, sheet_name=sheet_name)
            startrow = len(existing) + 1  # leave header
            header = False
        except FileNotFoundError:
            # File doesn’t exist yet
            startrow = 0
            header = True
        except ValueError:
            # Sheet doesn’t exist yet
            startrow = 0
            header = True

        # Use ExcelWriter in append mode
        with pd.ExcelWriter(
            self.path, engine="openpyxl", mode="a", if_sheet_exists="overlay"
        ) as writer:
            df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                header=header,
                startrow=startrow,
            )


    def flush(self):


        if self.raw_buffer:
            self._append_df("Raw_Logs", pd.DataFrame(self.raw_buffer))
            self.raw_buffer.clear()

        if self.rolling_ma_buffer:
            self._append_df(
                f"Rolling_MA_{self.rolling_window}",
                pd.DataFrame(self.rolling_ma_buffer),
            )
            self.rolling_ma_buffer.clear()

        if self.summary_buffer:
            self._append_df("Summary", pd.DataFrame(self.summary_buffer))
            self.summary_buffer.clear()

    def close(self):
        self.flush()
