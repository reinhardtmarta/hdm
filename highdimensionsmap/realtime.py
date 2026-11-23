import time
from collections import deque
from typing import Callable, Optional, Dict, Any, List
import numpy as np

from .scanner import HDMScanner
from .tracker import RobustMotionTracker
from .tensor import TrajectoryMotionTensor


class RealtimeTrajectoryMonitor:
    """Monitor de trajetória em tempo real com janela limitada e travas contra travamento.

    Utiliza um buffer circular (window_size) para limitar o consumo de RAM e
    monitores de tempo (max_duration_sec e max_step_ms) para garantir execução previsível.
    """

    def __init__(
        self,
        scanner: HDMScanner,
        window_size: int = 128,
        max_duration_sec: float = 10.0,
        max_step_ms: float = 20.0,
    ):
        if int(window_size) < 3:
            raise ValueError("window_size deve ser >= 3 para calcular velocidade e curvatura.")
        if float(max_duration_sec) <= 0.0:
            raise ValueError("max_duration_sec deve ser > 0.")
        if float(max_step_ms) <= 0.0:
            raise ValueError("max_step_ms deve ser > 0.")

        self.scanner = scanner
        self.window_size = int(window_size)
        self.max_duration_sec = float(max_duration_sec)
        self.max_step_ms = float(max_step_ms)

        self.tracker = RobustMotionTracker(scanner)
        self.tensor = TrajectoryMotionTensor(scanner)
        self._buffer: deque = deque(maxlen=self.window_size)
        self.history_log: List[Dict[str, Any]] = []

    def push_step(self, vector: np.ndarray) -> Dict[str, Any]:
        """Insere um novo vetor no buffer circular e processa a janela atual com limite de tempo."""
        t0 = time.perf_counter()

        vec = np.asarray(vector, dtype=np.float64).ravel()
        if not np.isfinite(vec).all():
            raise ValueError("Vetor de entrada contém NaN ou Inf.")

        self._buffer.append(vec)
        current_len = len(self._buffer)

        # Aguarda pelo menos 3 passos no buffer para métricas cinemáticas completas
        if current_len < 3:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "status": "buffering",
                "buffered_steps": current_len,
                "step_time_ms": elapsed_ms,
                "overrun": elapsed_ms > self.max_step_ms,
            }

        window_matrix = np.vstack(self._buffer)
        kinematics = self.tracker.track(window_matrix)
        coherence = self.tensor.compute_coherence(kinematics["signatures"])

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        overrun = elapsed_ms > self.max_step_ms

        snapshot = {
            "status": "overrun_warning" if overrun else "ok",
            "buffered_steps": current_len,
            "latest_velocity": float(kinematics["velocity"][-1]),
            "latest_curvature": float(kinematics["curvature"][-1]),
            "latest_noise": float(kinematics["noise"][-1]),
            "global_coherence": coherence["global_coherence"],
            "is_structured": coherence["is_structured"],
            "step_time_ms": elapsed_ms,
            "overrun": overrun,
        }
        return snapshot

    def run_stream(
        self,
        data_source: Callable[[int], Optional[np.ndarray]],
        target_hz: float = 30.0,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Executa o monitoramento por tempo definido (max_duration_sec) sem travar a CPU.

        Parâmetros:
            data_source: Função que recebe o índice do passo (int) e retorna um vetor 1D (ou None para encerrar).
            target_hz: Frequência alvo de amostragem por segundo (controla o ritmo do loop).
            verbose: Se True, exibe alertas apenas quando ocorre anomalia ou atraso.
        """
        if target_hz <= 0.0:
            raise ValueError("target_hz deve ser > 0.")

        step_interval = 1.0 / float(target_hz)
        start_time = time.monotonic()
        step_idx = 0
        self.history_log.clear()

        while (time.monotonic() - start_time) < self.max_duration_sec:
            loop_start = time.monotonic()

            vec = data_source(step_idx)
            if vec is None:
                break

            snapshot = self.push_step(vec)
            snapshot["step_index"] = step_idx
            snapshot["elapsed_sec"] = round(time.monotonic() - start_time, 4)
            self.history_log.append(snapshot)

            if verbose and snapshot.get("overrun"):
                print(
                    f"[HDM-RT Alerta] Passo {step_idx}: tempo ({snapshot['step_time_ms']:.2f} ms) "
                    f"excedeu limite ({self.max_step_ms} ms)."
                )

            step_idx += 1

            # Cede tempo de CPU para não travar o sistema nem rodar em loop infinito vazio
            work_time = time.monotonic() - loop_start
            sleep_time = step_interval - work_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        if verbose:
            total_sec = time.monotonic() - start_time
            overruns = sum(1 for item in self.history_log if item.get("overrun"))
            print(
                f"\n[HDM-RT Concluído] Duração: {total_sec:.2f}s | Passos: {step_idx} | "
                f"Atrasos (overruns): {overruns}"
            )

        return self.history_log
