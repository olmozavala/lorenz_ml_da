import struct
import os

filename = '/home/jv24b/miniconda3/envs/lorenzo/lib/python3.11/site-packages/numbalsoda/libsolve_ivp.so'
backup = filename + '.bak'

if not os.path.exists(backup):
    import shutil
    shutil.copy2(filename, backup)
    print(f"Created backup at {backup}")

with open(filename, 'r+b') as f:
    # Double check we are at the right place
    f.seek(400)
    p_type = struct.unpack('<I', f.read(4))[0]
    if p_type != 0x6474e551:
        print(f"Error: PT_GNU_STACK not found at 400. Found {hex(p_type)}")
        exit(1)
    
    f.seek(404)
    flags = struct.unpack('<I', f.read(4))[0]
    print(f"Current flags: {flags}")
    
    if flags == 7:
        f.seek(404)
        f.write(struct.pack('<I', 6))
        print("Successfully patched flags from 7 to 6")
    else:
        print("Flags are not 7, skipping patch.")
