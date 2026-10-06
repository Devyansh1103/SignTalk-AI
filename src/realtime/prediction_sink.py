"""
SignTalk AI - Prediction Sinks
Defines publishing destinations for real-time sign predictions, including
console logging, CSV diagnostics recording, and extensible callback handlers.
"""

from abc import ABC, abstractmethod
import os
import csv
import logging
from typing import List, Optional, Callable, Dict, Any

from src.realtime.types import PredictionResult

logger = logging.getLogger("SignTalk.RealTime.PredictionSink")


class BasePredictionSink(ABC):
    """Abstract base class for consuming real-time prediction events."""

    @abstractmethod
    def publish(self, prediction: PredictionResult) -> None:
        """Publishes a new prediction result."""
        pass

    def close(self) -> None:
        """Releases any allocated resources."""
        pass


class ConsolePredictionSink(BasePredictionSink):
    """
    Emits prediction results directly to stdout/console.
    """

    def __init__(self, show_top_k: bool = True, verbose: bool = False):
        self.show_top_k = show_top_k
        self.verbose = verbose

    def publish(self, prediction: PredictionResult) -> None:
        if self.verbose:
            print("\n" + "=" * 40)
            print(f"Prediction: {prediction.label.upper()}")
            print(f"Confidence: {prediction.confidence:.2f}")
            print(f"Gloss:      {prediction.gloss}")
            print(f"Latency:    {prediction.inference_latency_ms:.1f} ms")
            print(f"Quality:    {prediction.input_quality:.2f}")
            if self.show_top_k and prediction.top_k:
                print("Top predictions:")
                for rank, (cid, lbl, prob) in enumerate(prediction.top_k, 1):
                    print(f"  {rank}. {lbl.upper():<12} {prob:.2f}")
            print("=" * 40)
        else:
            top_str = " | ".join([f"{lbl.upper()}: {p:.2f}" for _, lbl, p in prediction.top_k])
            print(
                f"[SignTalk AI] Prediction: {prediction.label.upper()} "
                f"({prediction.confidence:.2f}) | {prediction.inference_latency_ms:.1f}ms | [{top_str}]"
            )


class PredictionLoggerSink(BasePredictionSink):
    """
    Appends temporal prediction records to a CSV file for benchmarking and error analysis.
    """

    CSV_HEADERS = [
        "timestamp",
        "window_start",
        "window_end",
        "prediction",
        "class_id",
        "confidence",
        "input_quality",
        "inference_latency_ms",
        "buffer_length",
        "device"
    ]

    def __init__(self, output_path: str = "results/realtime/predictions.csv"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        self._file = None
        self._writer = None
        self._init_csv()

    def _init_csv(self) -> None:
        file_exists = os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0
        self._file = open(self.output_path, mode="a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.CSV_HEADERS)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def publish(self, prediction: PredictionResult) -> None:
        if self._writer is None or self._file is None:
            return

        row = {
            "timestamp": f"{prediction.timestamp:.4f}",
            "window_start": f"{prediction.window_start_time:.4f}",
            "window_end": f"{prediction.window_end_time:.4f}",
            "prediction": prediction.label,
            "class_id": prediction.class_id,
            "confidence": f"{prediction.confidence:.4f}",
            "input_quality": f"{prediction.input_quality:.4f}",
            "inference_latency_ms": f"{prediction.inference_latency_ms:.2f}",
            "buffer_length": prediction.buffer_length,
            "device": prediction.device
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()
            self._file = None
            self._writer = None


class CallbackPredictionSink(BasePredictionSink):
    """
    Invokes a user-supplied callback function whenever a prediction is produced.
    """

    def __init__(self, callback: Callable[[PredictionResult], None]):
        self.callback = callback

    def publish(self, prediction: PredictionResult) -> None:
        self.callback(prediction)


class CompositePredictionSink(BasePredictionSink):
    """
    Dispatches predictions to multiple underlying sinks.
    """

    def __init__(self, sinks: Optional[List[BasePredictionSink]] = None):
        self.sinks = sinks or []

    def add_sink(self, sink: BasePredictionSink) -> None:
        self.sinks.append(sink)

    def publish(self, prediction: PredictionResult) -> None:
        for sink in self.sinks:
            try:
                sink.publish(prediction)
            except Exception as e:
                logger.error(f"Error publishing to {type(sink).__name__}: {e}")

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as e:
                logger.warning(f"Error closing sink {type(sink).__name__}: {e}")
