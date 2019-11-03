# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
sys.path.insert(0, os.getcwd())

from code.common import logging, run_command
import platform
import subprocess

# Check if MPS is currently turned on.
def check_mps_status():
    # Xavier does not have MPS
    if platform.processor() == "aarch64":
        return False
    # Check by printing out currently running processes and grepping nvidia-cuda-mps-control.
    cmd = "ps -ef | grep nvidia-cuda-mps-control | grep -c -v grep"
    logging.debug("Checking if MPS is running with command: {:}".format(cmd))
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p.wait()
    output = p.stdout.readlines()
    return int(output[0]) >= 1

# Turn off MPS.
def turn_off_mps():
    if platform.processor() != "aarch64" and check_mps_status():
        cmd = "echo quit | nvidia-cuda-mps-control"
        logging.info("Turn off MPS.")
        run_command(cmd)

# Turn on MPS.
def turn_on_mps(active_sms):
    if platform.processor() != "aarch64":
        turn_off_mps()
        cmd = "export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={:d} && nvidia-cuda-mps-control -d".format(active_sms)
        logging.info("Turn on MPS with active_sms = {:d}.".format(active_sms))
        run_command(cmd)

class ScopedMPS:
    def __init__(self, active_sms):
        self.active_sms = active_sms

    def __enter__(self):
        turn_on_mps(self.active_sms)

    def __exit__(self, type, value, traceback):
        turn_off_mps()
