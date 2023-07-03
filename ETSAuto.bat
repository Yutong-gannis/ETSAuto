@echo off
call conda activate
start python main.py
start python User/user_launcher.py
start python Condition/speedocr_launcher.py
start python Control/control_launcher.py