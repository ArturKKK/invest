#!/bin/bash
cd /Users/a.s.tabakov/Developer/invest
nohup ./venv/bin/python3.10 _research_r48_cont.py >> results_r48_cont.log 2>&1 &
echo "PID=$!" > results_r48_cont_pid.txt
echo "Started PID=$(cat results_r48_cont_pid.txt)"
