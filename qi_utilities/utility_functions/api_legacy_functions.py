"""
NOTE: this file contains legacy code which was used originally with Quantum Inspire (QI) 1.0,
and has now been translated in order to be used with QI 2.0.
Apart from the translation to QI 2.0, this code and format are not being currently supported.
"""

import os
import numpy as np
from datetime import datetime
import pandas as pd
from qiskit_quantuminspire.qi_provider import QIProvider

def prepare_file(basename: str="",
                 suffix: str="",
                 doraw: int=0):
    """
    Creates the file name according to the basename and suffix that you provide.
    """

    histname="hist_"+basename
    circuitname="circuit_"+basename
    rawname="Raw_"+basename
    if (len(suffix)>0):
        histname+='_'+suffix
        circuitname+='_'+suffix
        rawname+='_'+suffix
    histname+='_API.txt'
    circuitname+='_API.pdf'
    rawname+="_API"

    file=open(histname,'w')
    file=open(circuitname,'w')
    
    if (doraw==0):
        return histname, circuitname
    else:
        return histname, circuitname, rawname

def GetTimeStamp():
    """
    Returns the timestamp of the current date and time
    """

    current_date = datetime.datetime.now()
    thisyear=str(current_date.year)
    thismonth="0"+str(current_date.month)
    thisday="0"+str(current_date.day)
    thishour="0"+str(current_date.hour)
    thisminute="0"+str(current_date.minute)
    thissecond="0"+str(current_date.second)
    timestamp=thisyear[-2:]+thismonth[-2:]+thisday[-2:]+"_"+thishour[-2:]+thisminute[-2:]+thissecond[-2:]
    return timestamp

def create_new_data_folder(datadir: str):
    """
    Crates the folder with the current date in the specified path.
    """

    data_folder_path = datadir + "/Data"
    try:
        os.makedirs(data_folder_path, exist_ok=False)
    except:
        print("Data folder already exists")

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d")
    today_path = data_folder_path + f"/{timestamp}"
    try:
        os.makedirs(today_path, exist_ok=False)
    except:
        print(f"Folder with timestamp {timestamp} already exists")
    
    os.chdir(today_path)  # change the current working directory to the specified path
    return today_path

def api_run_and_save(param,
                     Qcircuit,
                     histname: str="hist.txt",
                     circuit_name: str="cqasm.txt",
                     shots: int=16384,
                     backend_name: str='Tuna-9',
                     get_results: bool=True,
                     get_hist_data: bool=False,
                     measurement_list: list=[],
                     get_raw_data: bool=False,
                     rawdata_filename: str="rawdata",
                     timeout: int = 1200):
    """
    Runs QI with qiskit program and returns histogram and the raw data
    A copy of the cqasm program is saved to file circuit_name.

    param:              a reference number that you are free to choose.
    Qcircuit:           Qiskit quantum circuit.
    histname:           file name where you want to save the histogram data.                        
    circuit_name:       name of the file in which you want to save the quantum circuit.
    shots:              desired number of shots. For Tuna-9, the max is 16384.
    backend_name:       specify the name of the backend that you want to use.
    get_results:        False: do not return the measurement result
                        True: return the measurement result
    get_hist_data:      False: do not return the histogram data
                        True: return the histogram data (if this is True make sure to specify the measurement_list)
    measurement_list:   each entry of the list is equal to the number of measurements done simultaneously in the algorithm.
                        e.g. measurement_list = [4, 2, 1], it means that the rightmost entry of the classical bit string is the result of a single measurement,
                        the second and the third entries of the classical bit string are measured together and the last four are measured together.                        
    get_raw_data:       False: do not return the raw data
                        True: return the raw data
    rawdata_filename:   name of the raw data file you want to save
    """
    
    # Set the backend
    provider = QIProvider()
    backend = provider.get_backend(name = backend_name)
    
    Qcircuit.draw('mpl', filename = circuit_name)

    # Run the job
    job = backend.run(Qcircuit, shots = shots, memory = get_raw_data)
    results = job.result(timeout = timeout) # get the results

    # Get and save the histogram data
    if get_hist_data:
        histogram_data = results.get_counts()
        histogram_keys = dict()

        for entry_index, entry in enumerate(histogram_data):
            additional_len = Qcircuit.num_clbits - len(entry)
            for i in range(additional_len):
                entry = ('0' + (entry))
            histogram_keys[entry_index] = 'd' + entry

        data = np.column_stack([list(histogram_keys.values()), list(histogram_data.values())])

        process_data_and_save(data, measurement_list, histname)

    # Get and save the raw data (if asked)
    if get_raw_data:
        raw_data = results.get_memory()

        for entry in range(len(raw_data)):
            additional_len = Qcircuit.num_clbits - len(raw_data[entry])
            for i in range(additional_len):
                raw_data[entry] = '0' + raw_data[entry]

        for nr_shots in range(len(raw_data)):
            raw_data[nr_shots] = 'd'+str(raw_data[nr_shots])

        df = pd.DataFrame({ 
                                "Raw data values": raw_data
                            })
        # Save to a csv file
        output_file_rawdata = rawdata_filename+"_"+str(param)+".csv"
        df.to_csv(output_file_rawdata, index = False)

    if get_results:
        return results

def process_data_and_save(data, q, filename):
    q_cumsum = np.cumsum(q)  # Cumulative sum of q to determine slicing indices
    column_dicts = [{} for _ in q]  # Create an empty dictionary for each entry in q

    for row in data:
        bitstring = row[0][1:]  # Remove the 'd' at the start
        count = int(row[1])    # Count of the bitstring
        
        for i, length in enumerate(q):
            if i == 0:
                sliced = bitstring[:length]  # First slice
            else:
                sliced = bitstring[q_cumsum[i-1]:q_cumsum[i]]  # Subsequent slices
            
            # Populate the dictionary for this column
            if sliced in column_dicts[i]:
                column_dicts[i][sliced] += count  # Increment count if key exists
            else:
                column_dicts[i][sliced] = count  # Initialize key with count
    
    column_dicts_reversed = column_dicts[::-1]

    # Write results to a file
    with open(filename, 'w') as file:
        for i, col_dict in enumerate(column_dicts_reversed):
            file.write(f"{i}:")
            file.write(r"{")
            for key, value in col_dict.items():
                file.write(f"'{key}': {value}")
            file.write(r"}")
            file.write("\n")  # Add a blank line between dictionaries