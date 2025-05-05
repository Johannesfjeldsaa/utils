#!/usr/bin/env python3
import os
import time
import shutil
import getpass
import subprocess
from pathlib import Path
from utils import run_command, timed_input

# Experiment configuration
USER = getpass.getuser()
PROJECT = "nn9560k"
SHORTCASEDESCRIPT = "aerosol_budgets"
COMPSET = "NF1850norbc"
RES = "f19_f19"
DATE = "20250430"
SRCROOT = Path(f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/src/NorESM").resolve()
CASENAME = f"{COMPSET}_{RES}_{DATE}_{SHORTCASEDESCRIPT}"
CASEDIR = Path(f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/cases-aerosol_budgets/{CASENAME}").resolve()
RUNDIR = Path(f"/cluster/work/users/{USER}/noresm/{CASENAME}/run").resolve()
RUNSCRIPTDIR = Path(f"/cluster/projects/{PROJECT}/{USER}/NorESM_workdir/cases-aerosol_budgets/run_scripts")

RUN_REFDIR = Path(RUNDIR).resolve()
RUN_REFCASE = "NF1850norbc_f19_f19_20241127_test16"
RUN_REFDATE = "0031-01-01"
RUN_REFTOD = "00000"
REFFILES_DIR = Path(f'/nird/datalake/NS9560K/noresm2.3/cases/{RUN_REFCASE}/rest/{RUN_REFDATE}-{RUN_REFTOD}').resolve()

MACHINE = "betzy"


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
            "./xmlchange RUN_TYPE=hybrid",
            "Setting RUN_TYPE=hybrid"
        )
        run_command(
            f"./xmlchange RUN_STARTDATE={RUN_REFDATE},START_TOD={RUN_REFTOD}",
            f"Setting RUN_STARTDATE={RUN_REFDATE},START_TOD={RUN_REFTOD}"
        )
        run_command(
            f"./xmlchange RUN_REFDIR={RUN_REFDIR},RUN_REFCASE={RUN_REFCASE},RUN_REFDATE={RUN_REFDATE},RUN_REFTOD={RUN_REFTOD}",
            f"Setting RUN_REFDIR={RUN_REFDIR},RUN_REFCASE={RUN_REFCASE},RUN_REFDATE={RUN_REFDATE},RUN_REFTOD={RUN_REFTOD}"
        )
        run_command(
            "./xmlchange STOP_OPTION=nyear,STOP_N=1",
            "Setting stop options"
        )
        run_command(
            "./xmlchange REST_OPTION=nmonth,REST_N=1",
            "Setting model restart file write options"
        )
        run_command(
            "./xmlchange DOUT_S_SAVE_INTERIM_RESTART_FILES=TRUE",
            "Setting interim restart file writing logical"
        )
        run_command(
            "./xmlchange --subgroup case.run JOB_WALLCLOCK_TIME=03:00:00",
            "Setting wallclock time"
        )

        print("\n*** Case XML edits finished\n")

        # Copy user_nl_cam
        shutil.copy2(
            os.path.join(RUNSCRIPTDIR, f'user_nl_cam_{CASENAME}'),
            os.path.join(CASEDIR, 'user_nl_cam'),
            follow_symlinks=True
        )

        print("\n*** user_nl_cam copied from {RUNSCRIPTDIR} to {CASEDIR}\n")

        # Build and submit
        run_command(
            "./case.build",
            "Building case"
        )
        print("\n*** Case build finished\n")

        # copy restart files from REFFILES_DIR to RUN_REFDIR
        if not REFFILES_DIR.exists():
            print(f"Error: REFFILES_DIR does not exist: {REFFILES_DIR}")
            exit(1)
        else:
            if not RUN_REFDIR.exists():
                os.mkdir(RUN_REFDIR)
            for file in REFFILES_DIR.iterdir():
                shutil.copy2(file, RUN_REFDIR, follow_symlinks=True)

            for file in RUN_REFDIR.iterdir():
                if file.name.endswith('.gz'):
                    run_command(
                        f'gzip -d {file}',
                        f'Unzipping {file}'
                    )

            copied = [file.name for file in RUN_REFDIR.iterdir()]
        print(
            "\n*** Restart files copied from {REFFILES_DIR} to {RUN_REFDIR}. Copied files:\n ",
            ',\n'.join(copied)
        )

        run_command(
            "./case.submit",
            "Submitting case"
        )
        print("\n*** Case submitted\n")

        # Copy runscript symlink
        case_script = Path(os.path.join(RUNSCRIPTDIR, f"{CASENAME}_runscript.py"))
        dest_symlink = Path(os.path.join(CASEDIR, f"{CASENAME}_runscript.py"))

        if not case_script.exists():
            shutil.copy2(script_path, case_script)
            print(f'\n*** Copied runscript to {RUNSCRIPTDIR} with the name {CASENAME}_runscript.py')

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
