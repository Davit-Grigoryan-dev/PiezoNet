import pyvisa
import numpy as np
import time
import math

def init_oscilloscope(resource_name=None, timeout=20000, ip_address=None):
    # """
    # Initializes the Rigol DHO800 oscilloscope interface over LAN (LXI) or USB.

    # Parameters:
    # - resource_name: The VISA resource name of the oscilloscope. If None and `ip_address` is provided, 
    #                  it uses LXI LAN with the provided IP address.
    # - ip_address: The IP address of the oscilloscope if using LXI LAN. Ignored if `resource_name` is provided.
    # - timeout: Communication timeout in milliseconds.

    # Returns:
    # - instrument: The VISA instrument object.
    # """
    rm = pyvisa.ResourceManager()

    if resource_name is None:
        if ip_address is None:
            # List resources if no resource name or IP is provided
            resources = rm.list_resources()
            if not resources:
                raise Exception("No VISA resources found.")
            resource_name = resources[0]  # Modify index if multiple devices are connected
        else:
            # Use LXI LAN with the provided IP address
            resource_name = f"TCPIP::{ip_address}::INSTR"

    # Open the connection to the oscilloscope
    instrument = rm.open_resource(resource_name)
    instrument.timeout = timeout

    # Identify the instrument
    idn = instrument.query("*IDN?")
    print(f"Connected to: {idn.strip()}")

    return instrument

import numpy as np
import time
import pyvisa

def get_waveform(instrument, channel):
    try:
        instrument.write("*CLS")  # Clear any previous errors
        time.sleep(0.1)  # Allow time for the command to complete
        
        # Ensure the oscilloscope is stopped if using RAW mode
        instrument.write(":STOP")
        time.sleep(0.1)
        
        # Set the data source to the specified channel, mode, and format
        instrument.write(f":WAV:SOUR {channel}")
        instrument.write(":WAV:MODE RAW")  # Use RAW for full memory depth
        instrument.write(":WAV:FORM BYTE")  # Set data format to BYTE (1 byte per point)
        
        # Set start and stop points to control the amount of data retrieved
        instrument.write(":WAV:STAR 1")
        instrument.write(":WAV:STOP 100000")  # Adjust based on needed resolution or buffer size
        
        # Retrieve and parse waveform preamble for scaling factors
        preamble_str = instrument.query(":WAV:PRE?")
        preamble = preamble_str.strip().split(',')
        if len(preamble) < 10:
            raise Exception("Failed to get valid waveform preamble.")

        # Parse preamble values for time and voltage scaling
        x_increment = float(preamble[4])
        x_origin = float(preamble[5])
        x_reference = float(preamble[6])
        y_increment = float(preamble[7])
        y_origin = float(preamble[8])
        y_reference = float(preamble[9])

        # Set an extended timeout for large data transfers
        instrument.timeout = 30000  # Extend to 30 seconds if necessary for data transfer

        # Attempt to read waveform data, with retry logic
        for attempt in range(3):
            try:
                instrument.write(":WAV:DATA?")
                raw_data = instrument.read_raw()
                break
            except pyvisa.errors.VisaIOError as e:
                print(f"Data transfer timeout, attempt {attempt + 1}")
                if attempt == 2:  # If last attempt fails, raise error
                    raise e
                time.sleep(0.5)  # Short delay before retrying

        # Parse binary data block
        if raw_data[0:1] != b'#':
            raise Exception("Invalid binary block header.")
        header_len = int(raw_data[1:2])
        data_len = int(raw_data[2:2 + header_len])
        waveform_data = raw_data[2 + header_len:2 + header_len + data_len]

        # Convert to numpy array and apply scaling factors
        waveform = np.frombuffer(waveform_data, dtype=np.uint8)
        voltage = (waveform - y_reference) * y_increment + y_origin
        time_array = (np.arange(len(voltage)) - x_reference) * x_increment + x_origin

        # Clear the oscilloscope’s status to remove any remaining flags
        instrument.write("*CLS")


        return time_array, voltage

    except Exception as e:
        print(f"Error in get_waveform function: {e}")
        return None, None


def calculate_vertical_div(max_voltage, vertical_divisions=4, scale_factor=0.8, channel=1, instrument=None):
    # """
    # Sets the vertical scale of the specified channel based on the max voltage.

    # Parameters:
    # - max_voltage: The maximum voltage of the input signal.
    # - vertical_divisions: The number of vertical divisions from 0 on the oscilloscope (default is 4).
    # - scale_factor: Factor to ensure the signal fits well on the screen (default is 0.8).
    # - channel: The channel number (e.g., 1 for CHAN1).
    # - instrument: The VISA instrument object representing the oscilloscope.
    # """

    # Calculate the required division value
    div = math.ceil(((max_voltage / vertical_divisions) / scale_factor))
    
    # Set the vertical scale on the oscilloscope
    if instrument:
        instrument.write(f":CHAN{channel}:SCAL {div}")




def close_oscilloscope(instrument):
    if instrument is not None:
        instrument.close()
        print("Oscilloscope connection closed.")
