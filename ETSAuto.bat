@echo off
call conda activate
start python User/user_launcher.py
start python main.py
start python Condition/condition_launcher.py
start python Control/control_launcher.py
