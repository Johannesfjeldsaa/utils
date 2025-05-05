#!/usr/bin/env python3
import os
import time
import random
import shutil
import getpass
import argparse
import subprocess
from pathlib import Path
from utils import run_command, timed_input


argparser = argparse.ArgumentParser(
    description="Script to create a case directory, set up and build the case, and submit it to the queue."
)
argparser.add_argument(
    '--USER',
    type=str,
    default=getpass.getuser(),
    help="User name for the project (default: current user)"
)
argparser.add_argument(
    '--PROJECT',
    type=str,
    default="nn9560k",
    help="Project name (default: nn9560k)"
)
argparser.add_argument(
    '--SHORTCASEDESCRIPT',
    type=str,
    default= f"{random.randint(0, 999999):06d}",
    help="Short case description (default: random number between 000000 and 999999)"
)
argparser.add_argument(
    '--COMPSET',
    type=str,
    default="NF1850norbc",
    help="Component set (default: NF1850norbc)"
)
argparser.add_argument(
    '--RES',
    type=str,
    default="f19_f19",
    help="Resolution (default: f19_f19)"
)
argparser.add_argument(
    '--DATE',
    type=str,
    default=time.strftime("%Y%m%d", time.localtime()),
    help="Date (default: today's date in YYYYMMDD format)"
)
argparser.add_argument(
    '--SRCROOT',
    type=str,
    default=None,
    help="Source root directory (default: /cluster/projects/<project>/<user>/NorESM_workdir/src/NorESM)"
)
argparser.add_argument(
    '--CASENAME',
    type=str,
    default=None,
    help="Case name (default: <COMPSET>_<RES>_<DATE>_<SHORTCASEDESCRIPT>)"
)
argparser.add_argument(
    '--CASEDIR',
    type=str,
    default=None,
    help="Case directory (default: /cluster/projects/<PROJECT>/<USER>/NorESM_workdir/cases-cam_output_reduction/<CASENAME>)"
)
argparser.add_argument(
    '--RUNDIR',
    type=str,
    default=None,
    help="Run directory (default: /cluster/projects/<PROJECT>/<USER>/NorESM_workdir/cases-cam_output_reduction/run_scripts)"
)
argparser.add_argument(
    '--MACHINE',
    type=str,
    default="betzy",
    help="Machine name (default: betzy)"
)
argparser.add_argument(
    '--user_nl_cam_path',
    type=str,
    default=None,
    help="Path to user_nl_cam file (default: <RUNDIR>/user_nl_cam)"
)
args = argparser.parse_args()

USER = args.USER
PROJECT = args.PROJECT
SHORTCASEDESCRIPT = args.SHORTCASEDESCRIPT
COMPSET = args.COMPSET
RES = args.RES
DATE = args.DATE
SRCROOT = Path(args.SRCROOT).resolve() if args.SRCROOT is not None else Path(
    f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/src/NorESM"
).resolve()
CASENAME = args.CASENAME if args.CASENAME is not None else f"{COMPSET}_{RES}_{DATE}_{SHORTCASEDESCRIPT}"
CASEDIR = Path(args.CASEDIR).resolve() if args.CASEDIR is not None else Path(
    f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/cases-cam_output_reduction/{CASENAME}"
).resolve()
RUNDIR = Path(args.RUNDIR).resolve() if args.RUNDIR is not None else Path(
    f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/cases-cam_output_reduction/run_scripts"
).resolve()
MACHINE = args.MACHINE
user_nl_cam_path = Path(args.user_nl_cam_path).resolve() if args.user_nl_cam_path is not None else Path(
    os.path.join(RUNDIR, 'user_nl_cam_allfalse')
).resolve()

def main():
    # Get current script path
    script_path = Path(__file__).resolve()

    # Create case directory if needed
    try:
        os.chdir(SRCROOT)
        if not Path(CASEDIR).exists():
            cn_args = f"--case {CASEDIR} --compset {COMPSET} --res {RES} --project {PROJECT} --mach {MACHINE} --run-unsupported"
            run_command(
                f"./cime/scripts/create_newcase {cn_args}",
                f"Running create_newcase with args {cn_args}"
            )
            print(f"\n*** Case directory created: {CASEDIR}\n")

        # Setup case
        os.chdir(CASEDIR)
        if not Path(os.path.join(CASEDIR, "CaseStatus")).exists():
            run_command(
                "./case.setup",
                "Running case.setup"
            )
            print("\n*** Case setup finished\n")

        # XML changes
        run_command(
            "./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=00:45:00",
            "Setting wallclock time"
        )
        run_command(
            "./xmlchange STOP_OPTION=nmonths,STOP_N=1",
            "Setting stop options"
        )
        run_command(
            './xmlchange CAM_CONFIG_OPTS="-phys cam6 -chem trop_mam_oslo -camnor  -chem trop_mam_oslo -camnor"',
            "removing -cosp from CAM_CONFIG_OPTS"
        )

        print("\n*** Case XML edits finished\n")

        # Copy user_nl_cam
        if CASEDIR.joinpath('user_nl_cam').exists():

            diff = run_command(
                f"diff {user_nl_cam_path} {os.path.join(CASEDIR, 'user_nl_cam')}",
                "Checking for differences between user_nl_cam files",
                subprocess_args={
                    'capture_output': True,
                    'text': True,
                    'cwd': CASEDIR
                }
            )
            if not diff.stdout:
                copy_user_nl_cam = 'n'
            else:
                print(f"\n*** user_nl_cam already exists in \n {CASEDIR}"
                      f"\n and differs from \n {user_nl_cam_path} \n in: {diff.stdout}")
                copy_user_nl_cam = timed_input(
                    prompt="Do you want to overwrite it? (Yy/[Nn])",
                    timeout=60,
                    default_output='n'
                )
            if copy_user_nl_cam:
                shutil.copy2(
                    user_nl_cam_path,
                    os.path.join(CASEDIR, 'user_nl_cam'),
                    follow_symlinks=True
                )
                print(f"\n*** user_nl_cam copied to {CASEDIR}\n")
            else:
                print(f"\n*** user_nl_cam not copied to {CASEDIR}\n")

            print("\n*** user_nl_cam edits finished\n")

        # Build and submit
        run_command(
            "./case.build",
            "Building case"
        )
        print("\n*** Case build finished\n")
        run_command(
            "./case.submit",
            "Submitting case"
        )
        print("\n*** Case submitted\n")

        # Copy runscript symlink
        case_script = Path(os.path.join(RUNDIR, f"{CASENAME}_runscript.py"))
        dest_symlink = Path(os.path.join(CASEDIR, f"{CASENAME}_runscript.py"))

        if not case_script.exists():
            shutil.copy2(script_path, case_script)
            print(f'\n*** Copied runscript to {RUNDIR} with the name {CASENAME}_runscript.py')

        if not dest_symlink.exists() or ( dest_symlink.exists() and not dest_symlink.is_symlink()):
            dest_symlink.symlink_to(case_script)
            print(f"\n*** Runscript symlink created: {dest_symlink} -> {case_script}\n")
        else:
            answer = timed_input(
                prompt="Runscript symlink already exists, do you want to update the symlink? (Yy/[Nn]) ",
                timeout=15
            )
            answer = answer.strip().lower() if answer is not None else answer

            if answer == "y":
                try:
                    dest_symlink.unlink()  # Remove old file/symlink
                    dest_symlink.symlink_to(case_script)
                    print(f"\n*** Runscript symlink created: {dest_symlink} -> {case_script}\n")
                except Exception as error:
                    print(f"Error updating runscript symlink: {str(error)}")
            else:
                print("\n*** Runscript symlink not modified\n")

        # Show queue status
        time.sleep(3)
        subprocess.run(["squeue", "-u", USER])

    except Exception as error:
        print(f"Fatal error: {str(error)}")
        exit(1)

main()
