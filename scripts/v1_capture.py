#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V1 — Prueba de confianza del detector visual

Captura interactiva por posición:
  - El usuario coloca el objeto en una posición fija de la matriz de
    validación y pulsa ENTER.
  - El script abre una ventana de captura que se cierra cuando se han
    recogido N detecciones válidas del target (o cuando se agota un
    timeout de seguridad).
  - Cierra la ventana, calcula media y σ de confianza dentro de la
    ventana. Ese valor representa "la confianza en esa posición".
  - Se repite para las 20 posiciones de la matriz..

NO se evalúa aquí la precisión de localización 3D, eso es V2 y va en
otro script aparte.

Uso:
    python3 v1_capture.py --target cubo --positions 20 --samples 5 \\
        --out ~/ws_daniel/validation_data/v1/v1_cubo.csv

Precondiciones (en otras terminales, en este orden):
    ros2 launch depthai_ros_driver camera.launch.py \\
        namespace:=oak_cam pointcloud.enable:=true
    ros2 launch stereo_location ur5_perception.launch.py
"""
import argparse
import csv
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from stereo_location_interfaces.msg import ObjDetArray


# Clases válidas del modelo entrenado (ver memoria del proyecto).
VALID_CLASSES = {
    'bola', 'botella rosa', 'caballo', 'cubo', 'lechuga',
    'pina', 'prisma', 'refresco', 'tomate', 'vaca'
}


class V1Capture(Node):
    """Nodo suscriptor que acumula confianzas del target dentro de ventanas."""

    def __init__(self, target_class: str):
        super().__init__('v1_capture_node')
        self.target_class = target_class
        self._collecting = False
        self._samples = []
        self._lock = threading.Lock()

        # QoS verificado contra el publisher real: RELIABLE / VOLATILE.
        qos = QoSProfile(
            depth=20,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self.sub = self.create_subscription(
            ObjDetArray, '/object_tracker/detections', self._cb, qos
        )
        self.get_logger().info(
            f'Suscrito a /object_tracker/detections, target="{target_class}"'
        )

    def _cb(self, msg: ObjDetArray):
        if not self._collecting:
            return

        # Mejor detección del target dentro del array (por confianza).
        best = None
        for obj in msg.objects:
            if obj.class_name == self.target_class:
                if best is None or obj.confidence > best.confidence:
                    best = obj

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if best is not None:
            sample = {
                'stamp_sec': stamp,
                'detected': True,
                'confidence': float(best.confidence),
                'num_in_frame': len(msg.objects),
            }
        else:
            sample = {
                'stamp_sec': stamp,
                'detected': False,
                'confidence': math.nan,
                'num_in_frame': len(msg.objects),
            }

        with self._lock:
            self._samples.append(sample)

    def start_window(self):
        with self._lock:
            self._samples = []
        self._collecting = True

    def stop_window(self) -> list:
        self._collecting = False
        with self._lock:
            return list(self._samples)

    def num_detected_so_far(self) -> int:
        """Cuántas muestras con detected=True llevamos en la ventana actual."""
        with self._lock:
            return sum(1 for s in self._samples if s['detected'])

    def num_total_so_far(self) -> int:
        with self._lock:
            return len(self._samples)


def summarize_window(samples: list) -> dict:
    """Estadísticas de confianza dentro de la ventana de captura."""
    total = len(samples)
    detected = [s for s in samples if s['detected']]
    n_det = len(detected)

    if n_det == 0:
        return {
            'num_total': total, 'num_detected': 0,
            'window_detection_rate': 0.0, 'hit': 0,
            'conf_mean': math.nan, 'conf_std': math.nan,
        }

    confs = [s['confidence'] for s in detected]
    return {
        'num_total': total,
        'num_detected': n_det,
        'window_detection_rate': n_det / total,
        'hit': 1,
        'conf_mean': statistics.mean(confs),
        'conf_std': statistics.stdev(confs) if len(confs) > 1 else 0.0,
    }


def write_summary_csv(path: Path, summaries: list):
    fields = [
        'position_id', 'target_class', 'status',
        'num_total', 'num_detected', 'window_detection_rate', 'hit',
        'conf_mean', 'conf_std', 'reached_target', 'elapsed_s',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k, '') for k in fields})


def write_raw_csv(path: Path, raw_rows: list):
    fields = [
        'position_id', 'target_class', 'sample_idx', 'stamp_sec',
        'detected', 'confidence', 'num_in_frame',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in raw_rows:
            w.writerow({k: r.get(k, '') for k in fields})


def print_position_result(pos_id: int, summary: dict):
    if summary['hit']:
        tag = '✓ HIT' if summary['reached_target'] else '⚠ TIMEOUT (parcial)'
    else:
        tag = '✗ MISS (timeout sin detecciones)'
    conf_mean = summary['conf_mean']
    conf_std = summary['conf_std']
    mean_s = f'{conf_mean:.3f}' if not math.isnan(conf_mean) else 'nan'
    std_s = f'{conf_std:.3f}' if not math.isnan(conf_std) else 'nan'
    print(f'  → {tag}  '
          f'samples: {summary["num_detected"]}/{summary["num_total"]}  '
          f'conf: μ={mean_s}  σ={std_s}  '
          f't={summary["elapsed_s"]:.1f}s')
    print()


def print_global_summary(summaries: list, target: str):
    ok = [s for s in summaries if s.get('status') == 'ok']
    skipped = [s for s in summaries if s.get('status') == 'skipped']
    hits = [s for s in ok if s['hit'] == 1]

    print('\n' + '=' * 64)
    print(f'RESUMEN GLOBAL — target = "{target}"')
    print('=' * 64)
    print(f'Posiciones procesadas: {len(ok)}   skipped: {len(skipped)}')
    if not ok:
        print('Sin datos. Nada que reportar.')
        return

    detection_rate = len(hits) / len(ok)
    print(f'Tasa de detección (M4):  {len(hits)}/{len(ok)} = '
          f'{100 * detection_rate:.1f}%')

    if hits:
        position_means = [s['conf_mean'] for s in hits]
        gmean = statistics.mean(position_means)
        gstd = statistics.stdev(position_means) if len(position_means) > 1 else 0.0
        print(f'Confianza por posición → media:   {gmean:.4f}')
        print(f'Confianza por posición → desv.:   {gstd:.4f}')
        print(f'(n = {len(position_means)} posiciones detectadas)')
    print('=' * 64 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='V1: confianza del detector visual — captura interactiva.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--target', required=True,
                        help='Clase target (ej: cubo, vaca, pina, "botella rosa")')
    parser.add_argument('--positions', type=int, default=20,
                        help='Número de posiciones (default 20)')
    parser.add_argument('--samples', type=int, default=5,
                        help='Detecciones válidas a recoger por posición (default 5)')
    parser.add_argument('--timeout', type=float, default=15.0,
                        help='Timeout de seguridad por posición en segundos (default 15.0)')
    parser.add_argument('--out', required=True,
                        help='CSV de salida con resumen por posición')
    parser.add_argument('--out-raw', default=None,
                        help='CSV opcional con todas las samples (default: derivado de --out)')
    args = parser.parse_args()

    if args.target not in VALID_CLASSES:
        print(f'⚠  Clase "{args.target}" no está en VALID_CLASSES = {VALID_CLASSES}')
        print('   ¿Continúa de todos modos? [s/N]: ', end='', flush=True)
        if input().strip().lower() not in ('s', 'si', 'y', 'yes'):
            sys.exit(1)

    out_path = Path(args.out).expanduser().resolve()
    if args.out_raw:
        raw_path = Path(args.out_raw).expanduser().resolve()
    else:
        raw_path = out_path.with_name(f'{out_path.stem}_raw.csv')

    rclpy.init()
    node = V1Capture(args.target)

    # rclpy.spin en thread aparte para que input() del main no lo bloquee.
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('\n' + '=' * 64)
    print('V1 CAPTURE — confianza del detector visual')
    print('=' * 64)
    print(f'  Target class:    {args.target}')
    print(f'  Posiciones:      {args.positions}')
    print(f'  Samples/pos:     {args.samples} (timeout {args.timeout}s)')
    print(f'  CSV summary:     {out_path}')
    print(f'  CSV raw:         {raw_path}')
    print('=' * 64)
    print('Warm-up del suscriptor (2 s)...')
    time.sleep(2.0)
    print('Listo.\n')

    summaries = []
    raw_rows = []
    pos_id = 1

    try:
        while pos_id <= args.positions:
            print(f'─── Posición {pos_id}/{args.positions} ───')
            try:
                cmd = input(
                    f'Coloca el objeto en la posición {pos_id} y pulsa '
                    f'[ENTER] para capturar  ([s]kip, [r]epetir anterior, [q]uit): '
                ).strip().lower()
            except EOFError:
                cmd = 'q'

            if cmd == 'q':
                print('Guardando y saliendo...')
                break

            if cmd == 's':
                summaries.append({
                    'position_id': pos_id, 'target_class': args.target,
                    'status': 'skipped',
                    'num_total': 0, 'num_detected': 0,
                    'window_detection_rate': 0.0, 'hit': 0,
                    'conf_mean': math.nan, 'conf_std': math.nan,
                    'reached_target': 0, 'elapsed_s': 0.0,
                })
                print(f'  → posición {pos_id} marcada como SKIPPED.\n')
                pos_id += 1
                continue

            if cmd == 'r':
                if not summaries:
                    print('  → no hay posición previa. Continúa en la 1.\n')
                    continue
                last = summaries.pop()
                raw_rows = [r for r in raw_rows
                            if r['position_id'] != last['position_id']]
                pos_id = last['position_id']
                print(f'  → repitiendo posición {pos_id}.\n')
                continue

            # Captura: espera N muestras válidas o timeout.
            print(f'  Esperando {args.samples} detecciones válidas '
                  f'(timeout {args.timeout}s)...')
            node.start_window()
            t0 = time.monotonic()
            last_print = 0
            reached = False
            while True:
                n_ok = node.num_detected_so_far()
                elapsed = time.monotonic() - t0

                if n_ok >= args.samples:
                    reached = True
                    break
                if elapsed >= args.timeout:
                    reached = False
                    break

                # Feedback en consola sin spamear.
                if n_ok != last_print:
                    print(f'    {n_ok}/{args.samples} válidas '
                          f'({node.num_total_so_far()} msgs, '
                          f'{elapsed:.1f}s)', flush=True)
                    last_print = n_ok

                time.sleep(0.05)

            elapsed_final = time.monotonic() - t0
            window_samples = node.stop_window()

            summary = summarize_window(window_samples)
            summary['position_id'] = pos_id
            summary['target_class'] = args.target
            summary['status'] = 'ok'
            summary['reached_target'] = 1 if reached else 0
            summary['elapsed_s'] = elapsed_final
            summaries.append(summary)

            for i, s in enumerate(window_samples):
                raw_rows.append({
                    'position_id': pos_id,
                    'target_class': args.target,
                    'sample_idx': i,
                    'stamp_sec': s['stamp_sec'],
                    'detected': int(s['detected']),
                    'confidence': s['confidence'],
                    'num_in_frame': s['num_in_frame'],
                })

            print_position_result(pos_id, summary)
            pos_id += 1

    except KeyboardInterrupt:
        print('\nCtrl+C detectado. Guardando lo capturado...')
    finally:
        write_summary_csv(out_path, summaries)
        write_raw_csv(raw_path, raw_rows)
        print(f'\nSummary CSV: {out_path}')
        print(f'Raw CSV:     {raw_path}')
        print_global_summary(summaries, args.target)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
