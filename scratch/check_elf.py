import struct

filename = '/home/jv24b/miniconda3/envs/lorenzo/lib/python3.11/site-packages/numbalsoda/libsolve_ivp.so'

with open(filename, 'rb') as f:
    f.seek(404)
    flags = struct.unpack('<I', f.read(4))[0]
    print(f"Flags at 404: {flags}")

# Also check p_type just to be sure
with open(filename, 'rb') as f:
    f.seek(400)
    p_type = struct.unpack('<I', f.read(4))[0]
    print(f"Type at 400: {hex(p_type)}")
