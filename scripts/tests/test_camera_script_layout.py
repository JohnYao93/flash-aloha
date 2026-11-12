"""Camera command relocation checks; never open devices or start services."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
TOOLS = sorted(
    path.name for path in (ROOT / 'scripts/cameras').glob('*.py')
)


@pytest.mark.parametrize('name', TOOLS)
def test_camera_help_outside_repository(name, tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/cameras' / name), '--help'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()


@pytest.mark.parametrize('relative,args', [
    ('src/camera_utils/camera_server.py', ['--help']),
])
def test_direct_server_and_benchmark_help(relative, args, tmp_path):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, str(ROOT / relative), *args], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout.lower()


def test_manager_resolves_repository_and_existing_worker_entrypoint(tmp_path):
    relative = Path('scripts/cameras/camera_servers_tmux.sh')
    fixture = tmp_path / relative
    fixture.parent.mkdir(parents=True)
    (tmp_path / 'scripts/cameras').mkdir(parents=True, exist_ok=True)
    # Execute only assignments before the first function; no tmux or camera code.
    prefix = (ROOT / relative).read_text().split('\nusage() {', 1)[0]
    fixture.write_text(prefix + '\nprintf "%s\\n" "$PROJECT_ROOT" "$SCRIPT_PATH"\n')
    result = subprocess.run(['bash', str(fixture)], cwd='/tmp',
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        str(tmp_path), str(tmp_path / 'scripts/cameras/camera_servers_tmux.sh'),
    ]


def test_control_launcher_keeps_commands_and_working_directory(tmp_path):
    # All device commands and Python launches are replaced with logging stubs.
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    for name in ('v4l2-ctl', 'uv'):
        stub = bin_dir / name
        stub.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys\n'
            'print(json.dumps([os.path.basename(sys.argv[0]), os.getcwd(), sys.argv[1:]]))\n'
        )
        stub.chmod(0o755)
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/cameras/configure_camera_controls.sh')],
        cwd=tmp_path, env=dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}"),
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in result.stdout.splitlines()] == [
        *[['v4l2-ctl', str(tmp_path), ['-d', f'/dev/video_{camera}',
           '--set-ctrl=exposure_dynamic_framerate=0']] for camera in ('left', 'right', 'overhead')],
    ]


def test_camera_helpers_share_implementation_and_resource_paths(monkeypatch):
    from cameras import tune_cameras

    monkeypatch.delenv('FLASH_ALOHA_CAMERA_PARAMS', raising=False)
    assert Path(tune_cameras.REPO_ROOT) == ROOT
