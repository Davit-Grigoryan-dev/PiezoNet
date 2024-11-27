import pyvisa
import time

def initialize_resource_manager():
    try:
        print("Initializing VISA resource manager with PyVISA-py backend...")
        rm = pyvisa.ResourceManager('@py')
        resources = rm.list_resources()
        print("Available VISA resources:", resources)
        return rm
    except pyvisa.errors.VisaIOError as e:
        print(f"Error initializing VISA resource manager: {e}")
        return None

def initialize_power_supply(rm, com_port):
    try:
        if not rm:
            print("Resource manager is not initialized.")
            return None

        resource_name = f'ASRL{com_port}::INSTR'
        if resource_name not in rm.list_resources():
            print(f"Error: {resource_name} is not available.")
            return None

        print(f"Opening resource {resource_name}...")
        psu = rm.open_resource(resource_name)
        print("Resource opened successfully.")
        return psu
    except pyvisa.errors.VisaIOError as e:
        print(f"Error initializing power supply on {com_port}: {e}")
        return None

def setup_power_supply(psu, voltage, current):
    try:
        psu.write(f'VOLT {voltage}')
        psu.write(f'CURR {current}')
        print(f"Voltage set to: {voltage} V")
        print(f"Current set to: {current} A")
    except pyvisa.errors.VisaIOError as e:
        print(f"Error setting up power supply: {e}")

def query_power_supply(psu):
    try:
        identity = psu.query('*IDN?')
        print("Power Supply Identity:", identity)

        voltage = psu.query('VOLT?')
        print("Set Voltage:", voltage)

        current = psu.query('CURR?')
        print("Set Current:", current)

        output_state = psu.query('OUTP?')
        print("Output State:", output_state)
    except pyvisa.errors.VisaIOError as e:
        print(f"Error querying power supply: {e}")

def control_output(psu, state):
    try:
        if state == "ON":
            psu.write('OUTP ON')
            print("Output turned ON")
        elif state == "OFF":
            psu.write('OUTP OFF')
            print("Output turned OFF")
        else:
            print("Invalid state. Use ON or OFF.")
    except pyvisa.errors.VisaIOError as e:
        print(f"Error controlling output: {e}")

def initialize_load(rm, usb_resource_name):
    try:
        if not rm:
            print("Resource manager is not initialized.")
            return None

        if usb_resource_name not in rm.list_resources():
            print(f"Error: {usb_resource_name} is not available.")
            return None

        print(f"Opening resource {usb_resource_name}...")
        load = rm.open_resource(usb_resource_name)
        print("Resource opened successfully.")
        return load
    except pyvisa.errors.VisaIOError as e:
        print(f"Error initializing electronic load on {usb_resource_name}: {e}")
        return None

def setup_load(load, value, mode):
    try:
        if mode == "VOLTAGE":
            load.write('FUNC VOLT')
            load.write(f'VOLT {value}')
        elif mode == "CURRENT":
            load.write('FUNC CURR')
            load.write(f'CURR {value}')
        elif mode == "POWER":
            load.write('FUNC POW')
            load.write(f'POW {value}')
        print(f"Load {mode.lower()} set to: {value}")
    except pyvisa.errors.VisaIOError as e:
        print(f"Error setting up electronic load: {e}")

def query_load(load):
    try:
        voltage = load.query('MEAS:VOLT?')
        current = load.query('MEAS:CURR?')
        return float(voltage), float(current)
    except pyvisa.errors.VisaIOError as e:
        print(f"Error querying electronic load: {e}")
        return voltage, current

def control_load_input(load, state):
    try:
        if state == "ON":
            load.write('INP ON')
            print("Load input turned ON")
        elif state == "OFF":
            load.write('INP OFF')
            print("Load input turned OFF")
        else:
            print("Invalid state. Use ON or OFF.")
    except pyvisa.errors.VisaIOError as e:
        print(f"Error controlling load input: {e}")

def close_resource(rm, resource):
    try:
        if resource:
            resource.write('OUTP OFF')
            resource.close()
            print("Resource session closed.")
        if rm:
            rm.close()
            print("Resource manager closed.")
    except pyvisa.errors.VisaIOError as e:
        print(f"Error closing resource: {e}")



def query_power_supply_current(psu):
    try:
        current = psu.query('CURR?')
        print("Power Supply Current:", current)
        return float(current)
    except pyvisa.errors.VisaIOError as e:
        print(f"Error querying power supply current: {e}")
        return None

def query_load_current(load):
    try:
        current = load.query('MEAS:CURR?')
        print("Electronic Load Current:", current)
        return float(current)
    except pyvisa.errors.VisaIOError as e:
        print(f"Error querying electronic load current: {e}")
        return None
