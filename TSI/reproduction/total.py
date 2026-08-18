import subprocess
import shutil
import os

scripts = [
    '1-1que_2que.py',
    '2-extract_1ans_2ans.py',
    '3-1ans_2ans_sim_score_for_CMT_and_MIP.py',
    '4-1que_2que_kdg.py',
    '5-3que.py',
    '6-calcu.py'
]

parameters = [
   {"1": True, "4": True, "5": True},
   {"1": False, "4": False, "5": False},
   {"1": True, "4": True, "5": False},
    {"1": False, "4": False, "5": True},
]

for params in parameters:
    for rep in range(1,3):
        print(f"\nRunning {params}, repetition {rep}")

        for script in scripts:
            print(f"Running {script}")
            if script == '1-1que_2que.py' and params["1"]:
                command = ["python", script, "--think"]
            elif script == '4-1que_2que_kdg.py' and params["4"]:
                command = ["python", script, "--think"]
            elif script == '5-3que.py' and params["5"]:
                command = ["python", script, "--think"]
            else:
                command = ["python", script]
            subprocess.run(command, check=True)

        destination = f"{params['1']}{params['4']}{params['5']}_rep{rep}"
        shutil.move('results', f"{destination}/results")
        shutil.move('results-similar-score', f"{destination}/results-similar-score")
        

