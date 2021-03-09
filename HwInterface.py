#!/usr/bin/python3
# Hardware interface code for Gaitsync project


# System libraries
import time
import sys

# 3rd party libraries
# ADC converter (link: https://github.com/adafruit/Adafruit_Python_MCP3008/blob/master/Adafruit_MCP3008/MCP3008.py)
from Adafruit_MCP3008 import MCP3008
# IMU (link: https://github.com/adafruit/Adafruit_Python_BNO055/blob/master/Adafruit_BNO055/BNO055.py)
from Adafruit_BNO055 import BNO055


# Globals
# Software SPI configuration for ADC (MCP3008)
CLK  = 18
MISO = 23
MOSI = 24
CS   = 25

# Serial UART and RST configuration for IMU (BNO055)
SERIAL_PORT = '/dev/serial0'
RST = 11

NUM_FORCE_SENSORS = 6
NUM_DECIMALS = 3


class HwInterface:
    '''
    Hardware Interface API class for GaitSync Project
    '''
    def __init__(self):
        '''
        Initializes ADC (MCP3008) and IMU (BNO055) instances with given parameters
        '''
        try:
            self.adc = MCP3008(clk = CLK, cs = CS, miso = MISO, mosi = MOSI)
            self.imu = BNO055.BNO055(serial_port = SERIAL_PORT, rst = RST)
            if not self.imu.begin():
                raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')
        except Exception as e:
            print("Error occurred while initializing the sensors: {}".format(e))
            sys.exit(1)
    
    def readSensors(self):
        sensor_values = [0] * NUM_FORCE_SENSORS

        for i in range(NUM_FORCE_SENSORS):
            # The read_adc function will get the value of the specified channel (0-7).
            sensor_values[i] = self.adc.read_adc(i)
        
        sensor_values.extend(round(num, NUM_DECIMALS) for num in list(self.imu.read_gyroscope()))
        sensor_values.extend(list(self.imu.read_euler()))
        sensor_values.extend(list(self.imu.read_linear_acceleration()))
	#time.sleep(1)

        return sensor_values

def main():
    hwApi = HwInterface()
    
    while True:
        try:
            output = hwApi.readSensors()
            print(output)
        except Exception as e:
            print("Error occurred in the main loop: {}".format(e))
            sys.exit(1)

if __name__ == "__main__":
    main()
