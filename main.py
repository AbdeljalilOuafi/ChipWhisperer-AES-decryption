import numpy as np
import matplotlib.pyplot as plt
import chipwhisperer as cw

scope = cw.scope()
target = cw.target(scope, cw.targets.SimpleSerial)
scope.default_setup()

# AES ISBOX
ISBOX = np.array([
0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
0xfc,0x56,0x3e,0x4e,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
0xa0,0xe1,0x3d,0x3f,0xd0,0xce,0x4d,0xaa,0xfa,0x13,0xfb,0x0b,0x05,0xb9,0xb7,0x6d,
0x58,0xc5,0x8c,0x73,0xdc,0x39,0x5e,0xd8,0xfe,0x78,0xcd,0x5a,0xf4,0x1f,0xdd,0xa8], dtype=np.uint8)

HW = np.unpackbits(np.arange(256, dtype=np.uint8)[:,None], axis=1).sum(axis=1)

# Hamming Distance table
def create_hd_table():
    hd = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            hd[i, j] = HW[i ^ j]
    return hd

HD_TABLE = create_hd_table()

print("Capturing traces...")
N_traces = 2000
traces_all = []
ciphertexts_all = []

for i in range(N_traces):
    pt = bytearray(np.random.randint(0, 256, size=(16,), dtype=np.uint8).tolist())
    scope.arm()
    target.simpleserial_write('p', pt)
    ok = scope.capture()
    
    if not ok:
        trace = scope.get_last_trace()
        ct = target.simpleserial_read('r', 16, timeout=500)
        traces_all.append(trace.copy())
        if ct:
            ciphertexts_all.append(bytes(ct))
    
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{N_traces}")

traces_all = np.array(traces_all, dtype=np.float32)
ciphertexts_all = np.array(ciphertexts_all)

print(f"Traces captured: {traces_all.shape}")

# Normalize
traces_mean = traces_all.mean(axis=0, keepdims=True)
traces_std = traces_all.std(axis=0, keepdims=True)
traces_std[traces_std < 1e-10] = 1.0
traces_norm = (traces_all - traces_mean) / traces_std

n_traces = traces_all.shape[0]

print("\nAttacking last round (Hamming Distance)...")

best_key = np.zeros(16, dtype=np.uint8)

for byte_idx in range(16):
    print(f"Byte {byte_idx}...", end=" ")
    
    ct_column = np.array([ct[byte_idx] if byte_idx < len(ct) else 0 
                         for ct in ciphertexts_all], dtype=np.uint8)
    
    max_corr = np.zeros(256, dtype=np.float32)
    
    for kguess in range(256):
        inter = ISBOX[np.bitwise_xor(ct_column, kguess)]
        hyp = HD_TABLE[inter, 0].astype(np.float32)
        
        hyp_c = hyp - hyp.mean()
        hyp_std = hyp_c.std()
        
        if hyp_std < 1e-10:
            continue
        
        hyp_n = hyp_c / hyp_std
        corr = np.abs(np.dot(hyp_n, traces_norm)) / n_traces
        max_corr[kguess] = np.max(corr)
    
    best = np.argmax(max_corr)
    best_key[byte_idx] = best
    
    top3 = np.argsort(max_corr)[-3:][::-1]
    print(f"0x{top3[0]:02x}({max_corr[top3[0]]:.4f})")

key_hex = "".join(f"{b:02x}" for b in best_key)
print(f"\nRecovered key: {key_hex}")
print(f"True key:      2b7e151628aed2a6abf7158809cf4f3c")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for byte_idx in range(2):
    ct_column = np.array([ct[byte_idx] if byte_idx < len(ct) else 0 
                         for ct in ciphertexts_all], dtype=np.uint8)
    
    inter = ISBOX[np.bitwise_xor(ct_column, best_key[byte_idx])]
    hyp = HD_TABLE[inter, 0].astype(np.float32)
    hyp_c = hyp - hyp.mean()
    hyp_n = hyp_c / (hyp_c.std() + 1e-10)
    
    corr_curve = np.abs(np.dot(hyp_n, traces_norm)) / n_traces
    peak_idx = np.argmax(corr_curve)
    peak_val = corr_curve[peak_idx]
    
    # Key distribution
    ax = axes[0, byte_idx]
    max_corr = np.zeros(256)
    for k in range(256):
        inter = ISBOX[np.bitwise_xor(ct_column, k)]
        hyp = HD_TABLE[inter, 0].astype(np.float32)
        hyp_c = hyp - hyp.mean()
        hyp_std = hyp_c.std()
        if hyp_std > 1e-10:
            hyp_n = hyp_c / hyp_std
            corr = np.abs(np.dot(hyp_n, traces_norm)) / n_traces
            max_corr[k] = np.max(corr)
    
    ax.bar(range(256), max_corr, alpha=0.7)
    ax.axvline(best_key[byte_idx], color='r', linestyle='--', linewidth=2)
    ax.set_title(f"Byte {byte_idx}: Key=0x{best_key[byte_idx]:02x}")
    ax.set_ylabel("Correlation")
    
    # Correlation curve
    ax = axes[1, byte_idx]
    ax.plot(corr_curve, linewidth=1.5)
    ax.axvline(peak_idx, color='r', linestyle='--', alpha=0.7)
    ax.plot(peak_idx, peak_val, 'r*', markersize=15)
    ax.set_title(f"Peak={peak_val:.4f} @ sample {peak_idx}")
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Sample")

plt.tight_layout()
plt.savefig("cpa_last_round.png", dpi=150)
plt.show()