import sys
import subprocess
from typing import Union

def run_command(
    cmd:                str,
    error_msg:          str,
    subprocess_args:    dict = {},
    cwd:                Union[str, None] = None
):
    """Run command-line command via subprocess.run with error handling

    Parameters
    ----------
    cmd : str
        The command to run
    error_msg : str
        Error message to print.
    subprocess_args : dict, optional
        Additional arguments for subprocess.run, by default empty dict.
    cwd : str, optional
        Working directory if command should be run in a different directory, by default None, i.e. the current directory is used.
    """
    try:
        # Set default arguments for subprocess.run
        if 'shell' not in subprocess_args:
            subprocess_args['shell'] = True
        if 'check' not in subprocess_args:
            subprocess_args['check'] = True
        if 'cwd' not in subprocess_args:
            subprocess_args['cwd'] = cwd
        if 'executable' not in subprocess_args:
            subprocess_args['executable'] = '/bin/bash'
        if 'capture_output' in subprocess_args and subprocess_args['capture_output']:
            return subprocess.run(cmd, **subprocess_args)
        else:
            subprocess.run(cmd, **subprocess_args)

    except subprocess.CalledProcessError as error:
        if error.returncode == 1 and cmd.startswith("diff"):
            # Files differ, not a real error for diff
            return error
        print(f"ERROR ({error.returncode}): {error_msg}")
        sys.exit(error.returncode)
