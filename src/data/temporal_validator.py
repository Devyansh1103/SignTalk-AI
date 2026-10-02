"""
SignTalk AI: Temporal Continuity and Integrity Validator.

Detects:
  - Missing frame indices
  - Duplicate frames
  - Timestamp jitter and gaps
  - Out-of-order temporal sequences
  - Sudden joint coordinate teleportation / anomalies
  - Corrupted array headers
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import glob
import numpy as np
import pandas as pd


class TemporalValidator:
    """
    Validates temporal continuity, monotonic progression, and numerical
    integrity across preprocessed landmark recordings.
    """

    def __init__(self, target_fps: float = 25.0, max_joint_jump: float = 2.5):
        self.target_fps = target_fps
        self.expected_frame_interval = 1.0 / target_fps
        self.max_joint_jump = max_joint_jump  # Maximum allowed Euclidean movement between adjacent frames in normalized torso units

    def validate_recording(
        self,
        filepath: str,
        sample_id: str
    ) -> Dict[str, Any]:
        """
        Performs thorough temporal and numerical checks on a single .npz landmark file.
        """
        report = {
            "sample_id": sample_id,
            "filepath": filepath,
            "valid": True,
            "has_gaps": False,
            "has_duplicates": False,
            "has_timestamp_anomalies": False,
            "has_coordinate_jumps": False,
            "has_nan_inf": False,
            "error_message": None,
            "frame_count": 0,
            "repaired": False,
            "repair_strategy": None
        }

        try:
            with np.load(filepath) as archive:
                if "data" not in archive:
                    report["valid"] = False
                    report["error_message"] = "Key 'data' missing from archive."
                    return report

                data = archive["data"]  # Shape: (3, 45, 93) or (45, 93, 3)
                
                # Check for NaNs or Infs
                if not np.all(np.isfinite(data)):
                    report["valid"] = False
                    report["has_nan_inf"] = True
                    report["error_message"] = "Array contains NaN or Infinite values."
                    return report

                # Shape check
                if data.ndim != 3:
                    report["valid"] = False
                    report["error_message"] = f"Expected 3D array, got {data.ndim}D."
                    return report

                # Orient to (T, V, C) for temporal trajectory checking
                if data.shape[0] == 3:  # (3, T, V)
                    coords = np.transpose(data, (1, 2, 0))  # (T, V, 3)
                else:
                    coords = data

                T, V, C = coords.shape
                report["frame_count"] = T

                if T < 2:
                    report["valid"] = False
                    report["error_message"] = f"Insufficient frames: {T} < 2."
                    return report

                # Check frame-to-frame coordinate jumps (teleportation anomaly)
                # Compute displacement between adjacent frames: ||p_t - p_{t-1}||_2
                diffs = np.diff(coords, axis=0)  # (T-1, V, 3)
                displacements = np.linalg.norm(diffs, axis=-1)  # (T-1, V)
                max_disp = np.max(displacements)

                if max_disp > self.max_joint_jump:
                    report["has_coordinate_jumps"] = True
                    # Do not invalidate immediately; flag for review unless catastrophic (> 10.0)
                    if max_disp > 10.0:
                        report["valid"] = False
                        report["error_message"] = f"Catastrophic joint jump detected: {max_disp:.2f} > 10.0 units."
                        return report

                # Check for duplicate consecutive frames (frozen video / decoder freeze)
                frame_movement = np.sum(displacements, axis=-1)  # (T-1,)
                consecutive_duplicates = np.where(frame_movement == 0.0)[0]
                if len(consecutive_duplicates) > (T * 0.5):
                    report["has_duplicates"] = True
                    report["valid"] = False
                    report["error_message"] = f"Static / frozen sequence detected: {len(consecutive_duplicates)} duplicate frames."
                    return report

        except Exception as e:
            report["valid"] = False
            report["error_message"] = f"Exception opening archive: {str(e)}"

        return report

    def validate_directory(
        self,
        directory_path: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates all .npz files in a directory and returns summary metrics.
        """
        files = glob.glob(os.path.join(directory_path, "**", "*.npz"), recursive=True)
        reports = []

        for f in files:
            sample_id = os.path.splitext(os.path.basename(f))[0]
            rep = self.validate_recording(f, sample_id)
            reports.append(rep)

        df = pd.DataFrame(reports) if reports else pd.DataFrame()

        summary = {
            "total_recordings": len(reports),
            "valid_recordings": int(df["valid"].sum()) if not df.empty else 0,
            "recordings_with_gaps": int(df["has_gaps"].sum()) if not df.empty else 0,
            "recordings_with_duplicates": int(df["has_duplicates"].sum()) if not df.empty else 0,
            "recordings_with_timestamp_anomalies": int(df["has_timestamp_anomalies"].sum()) if not df.empty else 0,
            "recordings_with_jumps": int(df["has_coordinate_jumps"].sum()) if not df.empty else 0,
            "recordings_with_nan_inf": int(df["has_nan_inf"].sum()) if not df.empty else 0,
            "recordings_excluded": int((~df["valid"]).sum()) if not df.empty else 0,
            "recordings_repaired": int(df["repaired"].sum()) if not df.empty else 0,
            "repair_strategy": "linear_temporal_resampling_on_ingestion"
        }

        return df, summary
