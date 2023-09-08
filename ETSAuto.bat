@echo off
call conda activate
start python Selfdrive/User/user_launcher.py
start python Selfdrive/Perception/perception_launcher.py
start python Selfdrive/Condition/condition_launcher.py
start python Selfdrive/Planning/Planning_launcher.py
start python Selfdrive/Control/control_launcher.py
