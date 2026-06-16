#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 — Prueba de estabilidad del centroide 3D 

Captura interactiva por posición (objeto físicamente inmóvil):
  - El usuario coloca el objeto en una posición fija de un conjunto
    reducido de posiciones (típicamente 5 extremas + 1 central) y pulsa
    ENTER.
  - El script abre una ventana de captura que se cierra cuando se han
    recogido N detecciones válidas del target (con posición 3D no NaN).
  - Por cada posición se calculan, sobre las repeticiones:
        · centroide promedio (x̄, ȳ, z̄)
        · desviaciones estándar por eje σ_x, σ_y, σ_z
        · desviación estándar tridimensional σ_xyz = √(σ_x²+σ_y²+σ_z²)
  - Se repite para todas las posiciones.

NO evalúa confianza como métrica principal (eso es V1), aunque se
registra como diagnóstico complementario.

Las posiciones 3D se reportan en el frame en el que el nodo
`object_location` las publica (oak_rgb_camera_optical_frame). La
dispersión (σ) es invariante a transformaciones rígidas, así que no
hace falta transformar al frame del robot.

Uso:
    python3 v2_capture.py --target cubo --positions 6 --samples 25 \\
        --out ~/ws_daniel/validation_data/v2/v2_cubo.csv

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


def _is_valid_pos(p) -> bool:
    """Una posición es válida si sus componentes son finitas y z>0
    (el nodo trabaja en frame óptico de cámara, z<=0 es imposible)."""
    for v in (p.x, p.y, p.z):
        if math.isnan(v) or math.isinf(v):
            return False
    if p.z <= 0.0:
        return False
    return True


class V2Capture(Node):
    """Nodo suscriptor que acumula posiciones 3D del target dentro de ventanas."""

    def __init__(self, target_class: str):
        super().__init__('v2_capture_node')
        self.target_class = target_class
        self._collecting = False
        self._samples = []
        self._lock = threading.Lock()

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

        # Mejor detección del target por confianza.
        best = None
        for obj in msg.objects:
            if obj.class_name == self.target_class:
                if best is None or obj.confidence > best.confidence:
                    best = obj

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if best is None:
            sample = {
                'stamp_sec': stamp, 'detected': False, 'valid_pos': False,
                'x': math.nan, 'y': math.nan, 'z': math.nan,
                'confidence': math.nan, 'num_in_frame': len(msg.objects),
            }
        else:
            valid = _is_valid_pos(best.position)
            sample = {
                'stamp_sec': stamp,
                'detected': True,
                'valid_pos': valid,
                'x': float(best.position.x) if valid else math.nan,
                'y': float(best.position.y) if valid else math.nan,
                'z': float(best.position.z) if valid else math.nan,
                'confidence': float(best.confidence),
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

    def num_valid_so_far(self) -> int:
        """Cuántas muestras con detección Y posición válida llevamos."""
        with self._lock:
            return sum(1 for s in self._samples if s['valid_pos'])

    def num_total_so_far(self) -> int:
        with self._lock:
            return len(self._samples)


def summarize_position(samples: list) -> dict:
    """Estadísticas 3D dentro de la ventana de una posición."""
    total = len(samples)
    valid = [s for s in samples if s['valid_pos']]
    n = len(valid)

    if n == 0:
        return {
            'num_total': total, 'num_valid': 0, 'hit': 0,
            'mean_x': math.nan, 'mean_y': math.nan, 'mean_z': math.nan,
            'std_x': math.nan, 'std_y': math.nan, 'std_z': math.nan,
            'std_3d': math.nan,
            'conf_mean': math.nan, 'conf_std': math.nan,
        }

    xs = [s['x'] for s in valid]
    ys = [s['y'] for s in valid]
    zs = [s['z'] for s in valid]
    confs = [s['confidence'] for s in valid]

    mean_x, mean_y, mean_z = statistics.mean(xs), statistics.mean(ys), statistics.mean(zs)

    if n > 1:
        std_x = statistics.stdev(xs)
        std_y = statistics.stdev(ys)
        std_z = statistics.stdev(zs)
        conf_std = statistics.stdev(confs)
    else:
        std_x = std_y = std_z = 0.0
        conf_std = 0.0

    std_3d = math.sqrt(std_x ** 2 + std_y ** 2 + std_z ** 2)

    return {
        'num_total': total,
        'num_valid': n,
        'hit': 1,
        'mean_x': mean_x, 'mean_y': mean_y, 'mean_z': mean_z,
        'std_x': std_x, 'std_y': std_y, 'std_z': std_z,
        'std_3d': std_3d,
        'conf_mean': statistics.mean(confs), 'conf_std': conf_std,
    }


def write_summary_csv(path: Path, summaries: list):
    fields = [
        'position_id', 'target_class', 'status',
        'num_total', 'num_valid', 'hit',
        'mean_x', 'mean_y', 'mean_z',
        'std_x', 'std_y', 'std_z', 'std_3d',
        'conf_mean', 'conf_std',
        'reached_target', 'elapsed_s',
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
        'detected', 'valid_pos', 'x', 'y', 'z',
        'confidence', 'num_in_frame',
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
        tag = '✗ MISS (timeout sin posiciones válidas)'

    def fmt(v, prec=4):
        return f'{v:.{prec}f}' if not math.isnan(v) else 'nan'

    print(f'  → {tag}  '
          f'valid: {summary["num_valid"]}/{summary["num_total"]}  '
          f't={summary["elapsed_s"]:.1f}s')
    if summary['hit']:
        print(f'    centroide:  x={fmt(summary["mean_x"])}  '
              f'y={fmt(summary["mean_y"])}  z={fmt(summary["mean_z"])}')
        print(f'    σ por eje:  σx={fmt(summary["std_x"])}  '
              f'σy={fmt(summary["std_y"])}  σz={fmt(summary["std_z"])}')
        print(f'    σ_xyz:      {fmt(summary["std_3d"])}  '
              f'(conf μ={fmt(summary["conf_mean"], 3)})')
    print()


def print_global_summary(summaries: list, target: str):
    ok = [s for s in summaries if s.get('status') == 'ok']
    skipped = [s for s in summaries if s.get('status') == 'skipped']
    hits = [s for s in ok if s['hit'] == 1]

    print('\n' + '=' * 64)
    print(f'RESUMEN GLOBAL — target = "{target}"')
    print('=' * 64)
    print(f'Posiciones procesadas: {len(ok)}   skipped: {len(skipped)}')
    if not hits:
        print('Sin posiciones con datos válidos. Nada que reportar.')
        return

    detection_rate = len(hits) / len(ok)
    print(f'Posiciones con datos:   {len(hits)}/{len(ok)} = '
          f'{100 * detection_rate:.1f}%')

    # Estadísticas entre posiciones.
    stds_x = [s['std_x'] for s in hits]
    stds_y = [s['std_y'] for s in hits]
    stds_z = [s['std_z'] for s in hits]
    stds_3d = [s['std_3d'] for s in hits]

    def stats(vs):
        if len(vs) == 0:
            return math.nan, math.nan
        if len(vs) == 1:
            return vs[0], 0.0
        return statistics.mean(vs), statistics.stdev(vs)

    mx, sx = stats(stds_x)
    my, sy = stats(stds_y)
    mz, sz = stats(stds_z)
    m3, s3 = stats(stds_3d)

    print(f'σ_x  entre posiciones (n={len(stds_x)}):  '
          f'media={mx:.5f}  desv.={sx:.5f}')
    print(f'σ_y  entre posiciones (n={len(stds_y)}):  '
          f'media={my:.5f}  desv.={sy:.5f}')
    print(f'σ_z  entre posiciones (n={len(stds_z)}):  '
          f'media={mz:.5f}  desv.={sz:.5f}')
    print(f'σ_xyz entre posiciones (n={len(stds_3d)}): '
          f'media={m3:.5f}  desv.={s3:.5f}')
    print('=' * 64 + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='V2: estabilidad del centroide 3D — captura interactiva.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--target', required=True,
                        help='Clase target (ej: cubo, vaca, pina, "botella rosa")')
    parser.add_argument('--positions', type=int, default=6,
                        help='Número de posiciones (default 6 = 5 extremas + 1 central)')
    parser.add_argument('--samples', type=int, default=25,
                        help='Detecciones válidas por posición (default 25)')
    parser.add_argument('--timeout', type=float, default=90.0,
                        help='Timeout de seguridad por posición en segundos (default 90.0)')
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
    node = V2Capture(args.target)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print('\n' + '=' * 64)
    print('V2 CAPTURE — estabilidad del centroide 3D')
    print('=' * 64)
    print(f'  Target class:    {args.target}')
    print(f'  Posiciones:      {args.positions}')
    print(f'  Samples/pos:     {args.samples} (timeout {args.timeout}s)')
    print(f'  CSV summary:     {out_path}')
    print(f'  CSV raw:         {raw_path}')
    print('=' * 64)
    print('IMPORTANTE: el objeto debe estar FÍSICAMENTE INMÓVIL durante la ventana.')
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
                    f'Coloca el objeto en la posición {pos_id} '
                    f'(estática, sin mover) y pulsa '
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
                    'num_total': 0, 'num_valid': 0, 'hit': 0,
                    'mean_x': math.nan, 'mean_y': math.nan, 'mean_z': math.nan,
                    'std_x': math.nan, 'std_y': math.nan, 'std_z': math.nan,
                    'std_3d': math.nan,
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

            # Captura.
            print(f'  Esperando {args.samples} detecciones válidas '
                  f'(timeout {args.timeout}s)...')
            node.start_window()
            t0 = time.monotonic()
            last_print = 0
            reached = False
            while True:
                n_ok = node.num_valid_so_far()
                elapsed = time.monotonic() - t0

                if n_ok >= args.samples:
                    reached = True
                    break
                if elapsed >= args.timeout:
                    reached = False
                    break

                if n_ok != last_print:
                    print(f'    {n_ok}/{args.samples} válidas '
                          f'({node.num_total_so_far()} msgs, '
                          f'{elapsed:.1f}s)', flush=True)
                    last_print = n_ok

                time.sleep(0.05)

            elapsed_final = time.monotonic() - t0
            window_samples = node.stop_window()

            summary = summarize_position(window_samples)
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
                    'valid_pos': int(s['valid_pos']),
                    'x': s['x'], 'y': s['y'], 'z': s['z'],
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
