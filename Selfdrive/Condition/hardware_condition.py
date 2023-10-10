import platform
import psutil
import GPUtil
import numpy as np
import os
import sys
from loguru import logger
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
from Common.log import condition_data_level, condition_info_level, condition_warning_level

        
class Hardware:
    """This is class record the condition of hardware"""
    def __init__(self):  # L:wheel base
        uname = platform.uname()
        self.system = uname.system
        if self.system != "Windows":
            logger.log("ConditionError", "Sorry, This program can only work on windows system!!!")
            sys.exit(1)
        self.system_version = uname.version
        
        self.cpu_num = psutil.cpu_count(logical=True)
        self.cpu_freq_max = psutil.cpu_freq().max
        self.cpu_freq_current = psutil.cpu_freq().current
        self.cpu_usage = psutil.cpu_percent() * 0.01  # decimal
        
        svmem = psutil.virtual_memory()
        self.mem_total = svmem.total / 1024 ** 3  # GB
        self.mem_free = svmem.available / 1024 ** 3  # GB
        self.mem_used = svmem.used / 1024 ** 3  # GB
        self.mem_percentage = svmem.percent * 0.01  # decimal
        
        gpu = GPUtil.getGPUs()[0]
        self.gpu_name = gpu.name
        self.gpu_load = gpu.load  # decimal
        self.gpu_mem_free = gpu.memoryFree  # GB
        self.gpu_mem_used = gpu.memoryUsed  # GB
        self.gpu_mem_total = gpu.memoryTotal  # GB
        self.gpu_temp = gpu.temperature  # °C
        
        logger.log("ConditionData", 
                   "Hardware Information: \n" +
                   "system: {}\t system_version: {}\n" + 
                   "cpu_num: {}\t cpu_freq_max: {}\t mem_total: {}\n" +
                   "gpu_name: {}\t gpu_mem_total: {}",
                   self.system, self.system_version, self.cpu_num, self.cpu_freq_max, self.mem_total, self.gpu_name, self.gpu_mem_total)
        

    def update(self):
        """This is function update the hardware's condition
        """
        self.cpu_freq_current = psutil.cpu_freq().current
        self.cpu_usage = psutil.cpu_percent() * 0.01  # decimal
        
        svmem = psutil.virtual_memory()
        self.mem_free = svmem.available / 1024 ** 3  # GB
        self.mem_used = svmem.used / 1024 ** 3  # GB
        self.mem_percentage = svmem.percent * 0.01  # decimal
        
        gpu = GPUtil.getGPUs()[0]
        self.gpu_load = gpu.load  # decimal
        self.gpu_mem_free = gpu.memoryFree  # GB
        self.gpu_mem_used = gpu.memoryUsed  # GB
        self.gpu_temp = gpu.temperature  # °C
        logger.log("ConditionData", 
                   "Hardware Condition: \n" +
                   "cpu_freq_current: {}\t cpu_usage: {}\n" + 
                   "mem_free: {}\t mem_used: {}\t mem_percentage: {}\n" +
                   "gpu_load: {}\t gpu_mem_free: {}\t gpu_mem_used: {}\t gpu_temp: {}",
                   self.cpu_freq_current, self.cpu_usage, self.mem_free, self.mem_used, self.mem_percentage, 
                   self.gpu_load, self.gpu_mem_free, self.gpu_mem_used, self.gpu_temp)
        logger.log("ConditionInfo", "Condition update finish.")

    def publish(self):
        """This is function to publish hardware condition data
        """
        condition_dict_pub = SharedMemoryDict(name='condition', size=1024)
        condition_dict_pub['cpu_freq_current'] =  self.cpu_freq_current
        condition_dict_pub['cpu_usage'] =  self.cpu_usage
        condition_dict_pub['mem_free'] =  self.mem_free
        condition_dict_pub['mem_percentage'] =  self.mem_percentage
        condition_dict_pub['gpu_load'] =  self.gpu_load
        condition_dict_pub['gpu_mem_free'] =  self.gpu_mem_free
        logger.log("ConditionInfo", "Hardware condition publish finish.")
        