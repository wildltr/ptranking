import os
import pickle

## due to the restriction of 4GB ##
max_bytes = 2**31 - 1

def pickle_save(target, file):
    bytes_out = pickle.dumps(target, protocol=4)
    with open(file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def pickle_load(file):
    file_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        bytes_in = bytearray(0)
        for _ in range(0, file_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data