#!/usr/bin/env python3
import os
import time
import shutil
import getpass
import subprocess
from pathlib import Path
from utils import run_command, timed_input

# Experiment configuration
USER = os.environ.get("USER", getpass.getuser())
PROJECT = os.environ.get("PROJECT", "nn9560k")
SHORTCASEDESCRIPT = os.environ.get("SHORTCASEDESCRIPT", "history_amwg")
COMPSET = os.environ.get("COMPSET", "NF1850norbc")
RES = os.environ.get("RES", "f19_f19")
DATE = os.environ.get("DATE", "20250429")
SRCROOT = os.environ.get("SRCROOT", f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/src/NorESM")
CASENAME = f"{COMPSET}_{RES}_{DATE}_{SHORTCASEDESCRIPT}"
CASEDIR = Path(os.environ.get("CASEDIR",
    f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/cases-cam_output_reduction/{CASENAME}"))
RUNDIR = Path(os.environ.get("RUNDIR",
    f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/cases-cam_output_reduction/run_scripts"))
MACHINE = os.environ.get("MACHINE", "betzy")

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
            "./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=06:00:00",
            "Setting wallclock time"
        )
        run_command(
            "./xmlchange STOP_OPTION=nyear,STOP_N=2",
            "Setting stop options"
        )
        run_command(
            './xmlchange CAM_AEROCOM=TRUE',
            "Setting CAM_AEROCOM=TRUE"
        )

        print("\n*** Case XML edits finished\n")

        # Copy user_nl_cam
        shutil.copy2(
            os.path.join(RUNDIR, 'user_nl_cam'),
            os.path.join(CASEDIR, 'user_nl_cam'),
            follow_symlinks=True
        )

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

if __name__ == "__main__":
    main()
