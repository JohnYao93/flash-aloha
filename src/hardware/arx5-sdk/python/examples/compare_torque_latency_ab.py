"""Compare Kd=0 and Kd>0 torque-burst recordings without accessing hardware."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import click

from record_torque_latency import compare_motor_kd_frequency_response


def write_comparison_csv(path: Path, comparison: dict) -> None:
    rows = comparison["frequency_response"]
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_html(path: Path, comparison: dict) -> None:
    summary = comparison["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{item['frequency_hz']:.1f}</td>"
        f"<td>{item['baseline_velocity_gain_rad_s_per_nm']:.4g}</td>"
        f"<td>{item['baseline_velocity_phase_deg']:.1f}</td>"
        f"<td>{item['damped_velocity_gain_rad_s_per_nm']:.4g}</td>"
        f"<td>{item['damped_velocity_phase_deg']:.1f}</td>"
        f"<td>{100.0 * item['damping_effect_fraction']:.1f}%</td>"
        f"<td>{item['inferred_internal_vs_reported_gain']:.3f}</td>"
        f"<td>{item['inferred_internal_vs_reported_phase_deg']:.1f}</td>"
        f"<td>{item['fresher_velocity_expected_phase_deg']:.1f}</td>"
        f"<td>{item['same_reported_velocity_phase_error_deg']:.1f}</td>"
        f"<td>{item['fresher_velocity_phase_error_deg']:.1f}</td>"
        "</tr>"
        for item in comparison["frequency_response"]
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Local motor Kd velocity-path comparison</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccd3dd;padding:5px 9px;text-align:right}}
pre{{background:#f6f8fa;padding:12px;overflow:auto}}
</style></head><body><h1>Local motor Kd velocity-path comparison</h1>
<p>Joint {comparison['joint']}; motor Kd {comparison['baseline_motor_kd']:.4g}
versus {comparison['damped_motor_kd']:.4g}.</p>
<p><strong>Closer phase hypothesis: {html.escape(summary['closer_phase_hypothesis'])}</strong>.
Same-reported-velocity phase RMSE: {summary['same_reported_velocity_phase_rmse_deg']:.1f}°;
fresher-velocity phase RMSE: {summary['fresher_velocity_phase_rmse_deg']:.1f}°.
Median Kd-induced transfer change: {100.0 * summary['median_damping_effect_fraction']:.1f}%.
{html.escape(summary['interpretation_note'])}</p>
<p><code>{html.escape(comparison['equation'])}</code></p>
<table><thead><tr><th>Hz</th><th>G0 gain</th><th>G0 phase °</th>
<th>Gd gain</th><th>Gd phase °</th><th>Kd effect</th><th>inferred Hi/Hr gain</th>
<th>inferred Hi/Hr phase °</th><th>fresh expected phase °</th>
<th>same phase error °</th><th>fresh phase error °</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Complete result</h2><pre>{html.escape(json.dumps(comparison, indent=2))}</pre>
</body></html>"""
    path.write_text(document)


def write_local_host_comparison_html(path: Path, comparison: dict) -> None:
    summary = comparison["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{item['frequency_hz']:.1f}</td>"
        f"<td>{item['local_velocity_gain_rad_s_per_nm']:.4g}</td>"
        f"<td>{item['local_velocity_phase_deg']:.1f}</td>"
        f"<td>{item['host_velocity_gain_rad_s_per_nm']:.4g}</td>"
        f"<td>{item['host_velocity_phase_deg']:.1f}</td>"
        f"<td>{item['host_over_local_gain']:.3f}</td>"
        f"<td>{item['host_minus_local_phase_deg']:.1f}</td>"
        f"<td>{item['host_minus_local_equivalent_delay_ms']:.2f}</td>"
        "</tr>"
        for item in comparison["frequency_response"]
    )
    document = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Local versus host joint damping</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#172033}}
table{{border-collapse:collapse}}th,td{{border:1px solid #ccd3dd;padding:5px 9px;text-align:right}}
pre{{background:#f6f8fa;padding:12px;overflow:auto}}
</style></head><body><h1>Local versus host joint damping</h1>
<p>Joint {comparison['joint']}; local motor Kd {comparison['local_motor_kd']:.4g};
host B {comparison['host_joint_damping_nm_s_per_rad']:.4g} Nm·s/rad.</p>
<p>Median host/local velocity gain: {summary['median_host_over_local_gain']:.3f};
median host−local phase: {summary['median_host_minus_local_phase_deg']:.1f}°;
phase RMS: {summary['host_minus_local_phase_rms_deg']:.1f}°.
{html.escape(summary['interpretation_note'])}</p>
<table><thead><tr><th>Hz</th><th>local gain</th><th>local phase °</th>
<th>host gain</th><th>host phase °</th><th>host/local gain</th>
<th>host−local phase °</th><th>equiv. delay ms</th></tr></thead>
<tbody>{rows}</tbody></table>
<h2>Complete result</h2><pre>{html.escape(json.dumps(comparison, indent=2))}</pre>
</body></html>"""
    path.write_text(document)


@click.command()
@click.argument(
    "baseline_json", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "damped_json", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--output-prefix", type=click.Path(dir_okay=False, path_type=Path), default=None
)
def main(baseline_json: Path, damped_json: Path, output_prefix: Path | None):
    """Compare a local-Kd=0 BASELINE_JSON with a local-Kd>0 DAMPED_JSON."""
    try:
        baseline = json.loads(baseline_json.read_text())
        damped = json.loads(damped_json.read_text())
        comparison = compare_motor_kd_frequency_response(baseline, damped)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if output_prefix is None:
        output_prefix = baseline_json.with_name(
            f"{baseline_json.stem}_vs_{damped_json.stem}_motor_kd"
        )
    output_prefix = output_prefix.resolve()
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    html_path = output_prefix.with_suffix(".html")
    json_path.write_text(json.dumps(comparison, indent=2) + "\n")
    write_comparison_csv(csv_path, comparison)
    write_comparison_html(html_path, comparison)

    summary = comparison["summary"]
    click.echo(f"Closer phase hypothesis: {summary['closer_phase_hypothesis']}")
    click.echo(
        "Phase RMSE: "
        f"same reported velocity {summary['same_reported_velocity_phase_rmse_deg']:.1f} deg; "
        f"fresher velocity {summary['fresher_velocity_phase_rmse_deg']:.1f} deg."
    )
    click.echo(
        "Median Kd-induced transfer change: "
        f"{100.0 * summary['median_damping_effect_fraction']:.1f}%."
    )
    click.echo(f"CSV:    {csv_path}")
    click.echo(f"Report: {html_path}")
    click.echo(f"Meta:   {json_path}")


if __name__ == "__main__":
    main()
