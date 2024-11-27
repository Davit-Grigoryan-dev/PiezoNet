import xmlrpc.client as xml
import os
import time
import pyvisa
import GW_RIGOL
import serial

plecs = xml.Server('http://localhost:1080/RPC2')

rm = pyvisa.ResourceManager()


psu = None
load = None

com_port_psu = '3'
com_port_load = '4'
psu = GW_RIGOL.initialize_power_supply(rm, com_port_psu)
load = GW_RIGOL.initialize_load(rm, com_port_load)
psu.timeout = 10000
load.timeout = 10000


GW_RIGOL.setup_power_supply(psu, 60, 0.5)  # Set the power supply to 60 V and 0.5 A
GW_RIGOL.setup_load(load, 20, "VOLTAGE")  # Set the load to 30 V and CV mode 

GW_RIGOL.control_output(psu, "ON")
GW_RIGOL.control_load_input(load, "ON")
time.sleep(2)




def calculate_efficiency(psu_voltage, psu_current, load_voltage, load_current):
    try:
        input_power = psu_voltage * psu_current
        output_power = load_voltage * load_current
        efficiency = (output_power / input_power) * 100 if input_power > 0 else 0
        return efficiency
    except ZeroDivisionError:
        return 0

def sweep_and_find_best_efficiency(start, end, step, psu, load):
    time.sleep(5)
    values = []
    highest_efficiency = 0
    best_value = None

    current_value = start
    while current_value <= end:
        # Set the current value to the system being tested
        plecs.plecs.set('control_system_6_submodules_60hz_Strobe/PHASE', 'Value', str(current_value))
        time.sleep(2)

        psu_voltage_readings = []
        psu_current_readings = []
        load_voltage_readings = []
        load_current_readings = []

              # Measure 10 times and take average
        for i in range(10):  # Average over 10 measurements
            psu_current_readings.append(float(psu.query('MEAS:CURR?')))
            load_current_readings.append(float(load.query('MEAS:CURR?')))
            time.sleep(0.1)

        # Calculate averages
        avg_psu_voltage = 60
        avg_psu_current = sum(psu_current_readings) / len(psu_current_readings)
        avg_load_voltage = 20
        avg_load_current = sum(load_current_readings) / len(load_current_readings)



        # Calculate efficiency
        efficiency = calculate_efficiency(avg_psu_voltage, avg_psu_current, avg_load_voltage, avg_load_current)
        print(f"Current value: {current_value}, Efficiency: {efficiency}%")

        # Store the value and check if it's the highest efficiency found
        values.append((current_value, efficiency))
        if efficiency > highest_efficiency:
            highest_efficiency = efficiency
            best_value = current_value

        # Move to the next step
        current_value += step
        time.sleep(2)

    print(f"Highest Efficiency: {highest_efficiency}% at value: {best_value}")
    return best_value



BEST_PHASE = sweep_and_find_best_efficiency(0.35, 0.65, 0.01, psu, load)