import numpy as np
import matplotlib.pyplot as plt
import chipwhisperer as cw
import time

# =============================================================================
# Setup ChipWhisperer Lite with XMEGA CW308 Target
# =============================================================================
scope = cw.scope()
target = cw.target(scope, cw.targets.SimpleSerial)
scope.default_setup()

# Optimized scope settings for XMEGA target
scope.clock.clkgen_freq = 7370000  # 7.37 MHz for XMEGA
scope.adc.samples = 24000          # Capture enough samples for full AES
scope.adc.offset = 0
scope.adc.basic_mode = "rising_edge"
scope.trigger.triggers = "tio4"
scope.io.tio1 = "serial_rx"
scope.io.tio2 = "serial_tx"
scope.io.hs2 = "clkgen"

# Reset target
if hasattr(scope, 'io'):
    scope.io.nrst = 'low'
    time.sleep(0.05)
    scope.io.nrst = 'high_z'
    time.sleep(0.05)

# =============================================================================
# AES S-Box and Inverse S-Box
# =============================================================================
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
], dtype=np.uint8)

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
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
], dtype=np.uint8)

# Hamming Weight lookup table
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# =============================================================================
# Trace Capture
# =============================================================================
print("=" * 60)
print("CPA Attack on AES Last Round")
print("Target: XMEGA CW308 with TinyAES128")
print("Power Model: Hamming Distance")
print("=" * 60)

print("\nCapturing traces...")
N_traces = 5000
traces_all = []
ciphertexts_all = []
plaintexts_all = []

for i in range(N_traces):
    pt = bytearray(np.random.randint(0, 256, size=(16,), dtype=np.uint8).tolist())
    scope.arm()
    target.simpleserial_write('p', pt)
    ok = scope.capture()
    
    if not ok:
        trace = scope.get_last_trace()
        ct = target.simpleserial_read('r', 16, timeout=500)
        if ct and len(ct) == 16:
            traces_all.append(trace.copy())
            ciphertexts_all.append(list(ct))
            plaintexts_all.append(list(pt))
    
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{N_traces} traces captured")

traces_all = np.array(traces_all, dtype=np.float64)
ciphertexts_all = np.array(ciphertexts_all, dtype=np.uint8)
plaintexts_all = np.array(plaintexts_all, dtype=np.uint8)

n_traces, n_samples = traces_all.shape
print(f"\nTotal valid traces: {n_traces}")
print(f"Samples per trace: {n_samples}")

# =============================================================================
# Preprocessing: Normalize traces
# =============================================================================
print("\nPreprocessing traces...")
traces_mean = traces_all.mean(axis=0)
traces_std = traces_all.std(axis=0)
traces_std[traces_std < 1e-10] = 1.0
traces_norm = (traces_all - traces_mean) / traces_std

# =============================================================================
# CPA Attack - Last Round with Hamming Distance
# =============================================================================
# 
# HAMMING DISTANCE MODEL FOR LAST ROUND:
# 
# In the last round of AES, the S-box output transitions to the final ciphertext.
# The power consumption is proportional to the number of bit flips during this
# transition.
#
# For each byte:
#   - S-box input: CT[i] ^ k[i] (before InvSubBytes in reverse)
#   - S-box output: InvSBox(CT[i] ^ k[i])
#   
# Hamming Distance = HW(S-box_input XOR S-box_output)
#                  = HW((CT[i] ^ k[i]) XOR InvSBox(CT[i] ^ k[i]))
#
# This models the bit transitions during the SubBytes operation.
# =============================================================================

print("\n" + "=" * 60)
print("Attacking Last Round (Hamming Distance Model)")
print("HD = HW( (CT^k) XOR InvSBox(CT^k) )")
print("=" * 60)

expected_key = "2b7e151628aed2a6abf7158809cf4f3c"
expected_bytes = bytes.fromhex(expected_key)

best_key = np.zeros(16, dtype=np.uint8)
best_correlations = np.zeros(16, dtype=np.float64)
all_correlations = np.zeros((16, 256), dtype=np.float64)
correlation_traces = {}

attack_start_time = time.time()

for byte_idx in range(16):
    byte_start = time.time()
    print(f"\nByte {byte_idx:2d}/15: ", end="", flush=True)
    
    ct_column = ciphertexts_all[:, byte_idx]
    max_corr = np.zeros(256, dtype=np.float64)
    best_corr_trace = None
    best_k = 0
    
    # Progress indicator
    for kguess in range(256):
        if kguess % 64 == 0:
            print(".", end="", flush=True)
        
        # S-box input (before InvSubBytes)
        sbox_input = ct_column ^ kguess
        
        # S-box output (after InvSubBytes)  
        sbox_output = ISBOX[sbox_input]
        
        # HAMMING DISTANCE: bit transitions during S-box operation
        # This is the key difference - we model HD not HW
        hyp = HW[sbox_input ^ sbox_output].astype(np.float64)
        
        # Normalize hypothesis
        hyp_mean = hyp.mean()
        hyp_centered = hyp - hyp_mean
        hyp_std = hyp_centered.std()
        
        if hyp_std < 1e-10:
            continue
        
        hyp_norm = hyp_centered / hyp_std
        
        # Pearson correlation (vectorized for speed)
        # corr = sum(hyp_norm * trace_norm) / n_traces for each sample
        corr = np.dot(hyp_norm, traces_norm) / n_traces
        corr_abs = np.abs(corr)
        max_corr[kguess] = np.max(corr_abs)
        
        if max_corr[kguess] > max_corr[best_k]:
            best_k = kguess
            best_corr_trace = corr_abs.copy()
    
    best = np.argmax(max_corr)
    best_key[byte_idx] = best
    best_correlations[byte_idx] = max_corr[best]
    all_correlations[byte_idx] = max_corr
    correlation_traces[byte_idx] = best_corr_trace
    
    # Results for this byte
    top3 = np.argsort(max_corr)[-3:][::-1]
    match = "[OK]" if best == expected_bytes[byte_idx] else "[X]"
    byte_time = time.time() - byte_start
    
    print(f" 0x{best:02x} {match} (corr={max_corr[best]:.4f}) "
          f"[2nd: 0x{top3[1]:02x}, 3rd: 0x{top3[2]:02x}] ({byte_time:.1f}s)")

attack_time = time.time() - attack_start_time

# =============================================================================
# Results Summary
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

key_hex = "".join(f"{b:02x}" for b in best_key)
matches = sum(1 for i in range(16) if best_key[i] == expected_bytes[i])

print(f"\nRecovered key: {key_hex}")
print(f"Expected key:  {expected_key}")
print(f"\nMatching bytes: {matches}/16")
print(f"Average correlation: {best_correlations.mean():.4f}")
print(f"Min correlation: {best_correlations.min():.4f} (byte {np.argmin(best_correlations)})")
print(f"Max correlation: {best_correlations.max():.4f} (byte {np.argmax(best_correlations)})")
print(f"Attack time: {attack_time:.1f} seconds")

if matches == 16:
    print("\n*** KEY SUCCESSFULLY RECOVERED! ***")

# =============================================================================
# FIGURE 1: Power Consumption Traces
# =============================================================================
print("\n" + "=" * 60)
print("Generating Graphs...")
print("=" * 60)

print("\n1. Power consumption traces...")
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))

# Overlaid traces
ax = axes1[0]
for i in range(min(50, n_traces)):
    ax.plot(traces_all[i], alpha=0.3, linewidth=0.5, color='blue')
ax.set_xlabel('Sample Number', fontsize=12)
ax.set_ylabel('Power (ADC counts)', fontsize=12)
ax.set_title(f'Power Consumption Traces (50 of {n_traces} traces)', fontsize=14)
ax.grid(True, alpha=0.3)

# Mean trace
ax = axes1[1]
ax.plot(traces_mean, 'b-', linewidth=1, label='Mean trace')
ax.fill_between(range(n_samples), 
                traces_mean - traces_all.std(axis=0), 
                traces_mean + traces_all.std(axis=0),
                alpha=0.3, color='blue', label='+/- 1 std dev')
ax.set_xlabel('Sample Number', fontsize=12)
ax.set_ylabel('Power (ADC counts)', fontsize=12)
ax.set_title('Mean Power Trace with Standard Deviation', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("power_traces.png", dpi=150, bbox_inches='tight')
print("   Saved: power_traces.png")

# =============================================================================
# FIGURE 2: Correlation Traces for Last Round
# =============================================================================
print("2. Correlation traces...")

fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12))

for byte_idx in range(16):
    row, col = byte_idx // 4, byte_idx % 4
    ax = axes2[row, col]
    
    corr_trace = correlation_traces[byte_idx]
    if corr_trace is not None:
        ax.plot(corr_trace, 'b-', linewidth=0.5)
        peak_idx = np.argmax(corr_trace)
        peak_val = corr_trace[peak_idx]
        ax.plot(peak_idx, peak_val, 'r*', markersize=8)
        ax.axhline(peak_val, color='red', linestyle='--', alpha=0.3, linewidth=0.5)
    
    match = "[OK]" if best_key[byte_idx] == expected_bytes[byte_idx] else "[X]"
    ax.set_title(f'Byte {byte_idx}: 0x{best_key[byte_idx]:02X} {match} (r={best_correlations[byte_idx]:.3f})', fontsize=9)
    ax.set_xlabel('Sample', fontsize=8)
    ax.set_ylabel('|Correlation|', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Correlation Traces - Last Round CPA (Hamming Distance Model)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_traces.png", dpi=150, bbox_inches='tight')
print("   Saved: correlation_traces.png")

# =============================================================================
# FIGURE 3: Key Candidate Correlations
# =============================================================================
print("3. Key candidate correlations...")

fig3, axes3 = plt.subplots(4, 4, figsize=(16, 12))

for byte_idx in range(16):
    row, col = byte_idx // 4, byte_idx % 4
    ax = axes3[row, col]
    
    correlations = all_correlations[byte_idx]
    colors = ['steelblue'] * 256
    colors[best_key[byte_idx]] = 'red'
    if expected_bytes[byte_idx] != best_key[byte_idx]:
        colors[expected_bytes[byte_idx]] = 'green'
    
    ax.bar(range(256), correlations, color=colors, alpha=0.7, width=1.0)
    ax.axvline(best_key[byte_idx], color='red', linestyle='--', linewidth=1)
    
    ax.set_title(f'Byte {byte_idx}: 0x{best_key[byte_idx]:02X} (exp: 0x{expected_bytes[byte_idx]:02X})', fontsize=9)
    ax.set_xlabel('Key Guess', fontsize=8)
    ax.set_ylabel('Max |Corr|', fontsize=8)
    ax.set_xlim([0, 255])

plt.suptitle('Key Candidate Correlations (Red=Recovered, Green=Expected)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("key_correlations.png", dpi=150, bbox_inches='tight')
print("   Saved: key_correlations.png")

# =============================================================================
# FIGURE 4: Summary for Professor
# =============================================================================
print("4. Summary figure...")

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

# Power traces
ax = axes4[0, 0]
for i in range(min(20, n_traces)):
    ax.plot(traces_all[i], alpha=0.4, linewidth=0.5)
ax.set_xlabel('Sample Number')
ax.set_ylabel('Power (ADC counts)')
ax.set_title('Power Consumption Traces')
ax.grid(True, alpha=0.3)

# Correlation trace for byte 0
ax = axes4[0, 1]
if correlation_traces[0] is not None:
    ax.plot(correlation_traces[0], 'b-', linewidth=1)
    peak_idx = np.argmax(correlation_traces[0])
    ax.plot(peak_idx, correlation_traces[0][peak_idx], 'r*', markersize=12)
ax.set_xlabel('Sample Number')
ax.set_ylabel('|Correlation|')
ax.set_title(f'Correlation Trace - Byte 0 (key=0x{best_key[0]:02X})')
ax.grid(True, alpha=0.3)

# Key candidates for byte 0
ax = axes4[1, 0]
ax.bar(range(256), all_correlations[0], color='steelblue', alpha=0.7, width=1.0)
ax.axvline(best_key[0], color='red', linestyle='--', linewidth=2, label=f'Best: 0x{best_key[0]:02X}')
ax.set_xlabel('Key Guess (0-255)')
ax.set_ylabel('Max |Correlation|')
ax.set_title('Key Candidates - Byte 0')
ax.legend()
ax.grid(True, alpha=0.3)

# Correlation per byte
ax = axes4[1, 1]
colors = ['green' if best_key[i] == expected_bytes[i] else 'red' for i in range(16)]
ax.bar(range(16), best_correlations, color=colors, alpha=0.8)
ax.axhline(best_correlations.mean(), color='blue', linestyle='--', label=f'Mean: {best_correlations.mean():.4f}')
ax.set_xlabel('Key Byte Index')
ax.set_ylabel('Best Correlation')
ax.set_title('Correlation per Byte (Green=Correct, Red=Wrong)')
ax.set_xticks(range(16))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

fig4.text(0.5, 0.02, 
          f'Recovered: {key_hex} | Expected: {expected_key} | Matches: {matches}/16',
          ha='center', fontsize=11, fontweight='bold', family='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'CPA Attack Summary - Last Round (Hamming Distance)\n{n_traces} traces, {n_samples} samples/trace', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("cpa_summary.png", dpi=150, bbox_inches='tight')
print("   Saved: cpa_summary.png")

plt.show()

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
print("\nOutput files:")
print("  - power_traces.png")
print("  - correlation_traces.png")
print("  - key_correlations.png")
print("  - cpa_summary.png")
