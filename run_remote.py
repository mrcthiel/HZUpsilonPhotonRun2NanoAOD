#!/usr/bin/env python

import os
import typer

def execute_command(command):
    print(f"--> Will execute:\n {command}")
    os.system(command)

def run_analysis(hostname: str, username: str, working_dir: str, uerj_usr: bool = False):
    inner_commands = f"cd {working_dir} ; conda activate ../HZUpsilonPhotonRun2NanoAOD_env ; ./run_analysis.py all"
    # inner_commands = f"cd {working_dir} ; conda activate ../HZUpsilonPhotonRun2NanoAOD_env ; pwd"
    full_command = f"ssh {username}@{hostname} '{inner_commands}'"
    if uerj_usr:
        full_command = f"ssh {username}@{hostname} 'ssh uerj-usr \"{inner_commands}\"'"

    execute_command(full_command)



def sync_working_directories(hostname: str, username: str, working_dir: str, uerj_usr: bool = False):
    if uerj_usr:
        create_temp_dir = f"ssh {username}@{hostname} 'mkdir -p /tmp/{username}/analysis_temp_dir'"
        execute_command(create_temp_dir)
        sync_command = f"rsync -azP --delete --exclude={{'plots','outputs','.git'}} ./ {username}@lxplus.cern.ch:/tmp/{username}/analysis_temp_dir"
        execute_command(sync_command)
        re_sync_command = f"ssh {username}@{hostname} 'rsync -azP --exclude={{'plots','outputs','.git'}} --delete /tmp/{username}/analysis_temp_dir/ uerj-usr:{working_dir}'"
        execute_command(re_sync_command)
    
    else:
        sync_command = f"rsync -azP --delete --exclude={'outputs','.git'} ./ {username}@{hostname}:{working_dir}"
        execute_command(sync_command)

def sync_outputs(hostname: str, username: str, working_dir: str, uerj_usr: bool = False):
    if uerj_usr:
        create_temp_dir = f"ssh {username}@{hostname} 'mkdir -p /tmp/{username}/analysis_temp_dir/outputs'"
        execute_command(create_temp_dir)
        sync_command = f"ssh {username}@{hostname} 'rsync -azP uerj-usr:{working_dir}/outputs/ /tmp/{username}/analysis_temp_dir/outputs'"
        execute_command(sync_command)
        re_sync_command = f"rsync -azP {username}@lxplus.cern.ch:/tmp/{username}/analysis_temp_dir/outputs/ ./outputs "
        execute_command(re_sync_command)
        
    
    else:
        create_outputs_dir = f"mkdir -p outputs"
        execute_command(create_outputs_dir)
        sync_command = f"rsync -azP {username}@{hostname}:{working_dir}/outputs/ ./outputs "
        execute_command(sync_command)

def main(hostname: str, username: str, working_dir: str, uerj_usr: bool = False, outputs: bool = False):
    sync_working_directories(hostname, username, working_dir, uerj_usr)
    run_analysis(hostname, username, working_dir, uerj_usr)
    if outputs:
        sync_outputs(hostname, username, working_dir, uerj_usr)

if __name__ == "__main__":
    typer.run(main)

    