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

# Scope settings for XMEGA target
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
# AES Tables and Key Schedule
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

# Hamming Weight lookup
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# AES Round Constants
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

# InvShiftRows mapping: tells which ciphertext byte corresponds to which state byte
# In the last round: CT = ShiftRows(SubBytes(state9)) XOR key10
# To undo ShiftRows when attacking, we need this mapping
INVSHIFT = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]

def key_schedule_rounds(key, start_round, end_round):
    """
    Apply AES key schedule from start_round to end_round.
    If start_round > end_round, applies inverse key schedule.
    """
    key = np.array(list(key), dtype=np.uint8)
    
    if start_round < end_round:
        # Forward key schedule
        for r in range(start_round, end_round):
            # RotWord + SubWord + Rcon for first column
            temp = np.array([
                SBOX[key[13]] ^ RCON[r],
                SBOX[key[14]],
                SBOX[key[15]],
                SBOX[key[12]]
            ], dtype=np.uint8)
            
            # XOR with previous round key
            new_key = np.zeros(16, dtype=np.uint8)
            new_key[0:4] = key[0:4] ^ temp
            new_key[4:8] = key[4:8] ^ new_key[0:4]
            new_key[8:12] = key[8:12] ^ new_key[4:8]
            new_key[12:16] = key[12:16] ^ new_key[8:12]
            key = new_key
    else:
        # Inverse key schedule (going backwards)
        for r in range(start_round - 1, end_round - 1, -1):
            new_key = np.zeros(16, dtype=np.uint8)
            new_key[12:16] = key[12:16] ^ key[8:12]
            new_key[8:12] = key[8:12] ^ key[4:8]
            new_key[4:8] = key[4:8] ^ key[0:4]
            
            # Inverse of RotWord + SubWord + Rcon
            temp = np.array([
                SBOX[new_key[13]] ^ RCON[r],
                SBOX[new_key[14]],
                SBOX[new_key[15]],
                SBOX[new_key[12]]
            ], dtype=np.uint8)
            new_key[0:4] = key[0:4] ^ temp
            key = new_key
    
    return bytes(key)

# =============================================================================
# Trace Capture
# =============================================================================
print("=" * 60)
print("CPA Attack on AES")
print("Target: XMEGA CW308 with TinyAES128")
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

# Normalize traces
print("Preprocessing traces...")
traces_mean = traces_all.mean(axis=0)
traces_std = traces_all.std(axis=0)
traces_std[traces_std < 1e-10] = 1.0
traces_norm = (traces_all - traces_mean) / traces_std

# =============================================================================
# First Round Attack: HW(SBox(PT ^ k))
# =============================================================================
print("\n" + "=" * 60)
print("FIRST ROUND ATTACK")
print("Power Model: HW(SBox(PT ^ k))")
print("=" * 60)

expected_key = "2b7e151628aed2a6abf7158809cf4f3c"
expected_bytes = bytes.fromhex(expected_key)

first_round_key = np.zeros(16, dtype=np.uint8)
first_round_corr = np.zeros(16, dtype=np.float64)
first_round_all_corr = np.zeros((16, 256), dtype=np.float64)
first_round_traces = {}

for byte_idx in range(16):
    print(f"Byte {byte_idx:2d}: ", end="", flush=True)
    
    pt_column = plaintexts_all[:, byte_idx]
    max_corr = np.zeros(256, dtype=np.float64)
    best_trace = None
    
    for kguess in range(256):
        if kguess % 64 == 0:
            print(".", end="", flush=True)
        
        # First round: HW(SBox(PT ^ k))
        inter = SBOX[pt_column ^ kguess]
        hyp = HW[inter].astype(np.float64)
        
        hyp_c = hyp - hyp.mean()
        hyp_std = hyp_c.std()
        if hyp_std < 1e-10:
            continue
        hyp_n = hyp_c / hyp_std
        
        corr = np.abs(np.dot(hyp_n, traces_norm) / n_traces)
        max_corr[kguess] = np.max(corr)
        
        if max_corr[kguess] >= max_corr.max():
            best_trace = corr.copy()
    
    best = np.argmax(max_corr)
    first_round_key[byte_idx] = best
    first_round_corr[byte_idx] = max_corr[best]
    first_round_all_corr[byte_idx] = max_corr
    first_round_traces[byte_idx] = best_trace
    
    match = "[OK]" if best == expected_bytes[byte_idx] else "[X]"
    print(f" 0x{best:02x} {match} (corr={max_corr[best]:.4f})")

first_key_hex = "".join(f"{b:02x}" for b in first_round_key)
first_matches = sum(1 for i in range(16) if first_round_key[i] == expected_bytes[i])
print(f"\nFirst Round Key: {first_key_hex}")
print(f"Expected:        {expected_key}")
print(f"Matches: {first_matches}/16, Avg correlation: {first_round_corr.mean():.4f}")

# =============================================================================
# Last Round Attack: HW(InvSBox(CT ^ k))
# Recovers Round 10 key, then applies inverse key schedule
# =============================================================================
print("\n" + "=" * 60)
print("LAST ROUND ATTACK")
print("Power Model: HW(InvSBox(CT ^ k))")
print("Note: Recovers round 10 key, then applies inverse key schedule")
print("=" * 60)

# Focus on last portion of traces where round 10 occurs
# The last round is at the END of the AES operation
poi_start = int(n_samples * 0.8)  # Last 20% of trace
poi_end = n_samples
print(f"Using samples {poi_start} to {poi_end} (last round region)")

traces_last = traces_all[:, poi_start:poi_end]
traces_last_mean = traces_last.mean(axis=0)
traces_last_std = traces_last.std(axis=0)
traces_last_std[traces_last_std < 1e-10] = 1.0
traces_last_norm = (traces_last - traces_last_mean) / traces_last_std

last_round_key = np.zeros(16, dtype=np.uint8)
last_round_corr = np.zeros(16, dtype=np.float64)
last_round_all_corr = np.zeros((16, 256), dtype=np.float64)
last_round_traces = {}

for byte_idx in range(16):
    print(f"Byte {byte_idx:2d}: ", end="", flush=True)
    
    # Get the correct ciphertext byte considering InvShiftRows
    # When attacking byte i of the key, we use ct[INVSHIFT[i]]
    ct_byte_idx = INVSHIFT[byte_idx]
    ct_column = ciphertexts_all[:, ct_byte_idx]
    
    max_corr = np.zeros(256, dtype=np.float64)
    best_trace = None
    
    for kguess in range(256):
        if kguess % 64 == 0:
            print(".", end="", flush=True)
        
        # Last round: HW(InvSBox(CT ^ k))
        # This is the state BEFORE the last SubBytes (state9)
        inter = ISBOX[ct_column ^ kguess]
        hyp = HW[inter].astype(np.float64)
        
        hyp_c = hyp - hyp.mean()
        hyp_std = hyp_c.std()
        if hyp_std < 1e-10:
            continue
        hyp_n = hyp_c / hyp_std
        
        corr = np.abs(np.dot(hyp_n, traces_last_norm) / n_traces)
        max_corr[kguess] = np.max(corr)
        
        if max_corr[kguess] >= max_corr.max():
            best_trace = corr.copy()
    
    best = np.argmax(max_corr)
    last_round_key[byte_idx] = best
    last_round_corr[byte_idx] = max_corr[best]
    last_round_all_corr[byte_idx] = max_corr
    last_round_traces[byte_idx] = best_trace
    
    print(f" 0x{best:02x} (corr={max_corr[best]:.4f})")

last_key_hex = "".join(f"{b:02x}" for b in last_round_key)
print(f"\nRecovered Round 10 Key: {last_key_hex}")
print(f"Avg correlation: {last_round_corr.mean():.4f}")

# Apply inverse key schedule to get round 0 key
print("\nApplying inverse key schedule (round 10 -> round 0)...")
try:
    original_key = key_schedule_rounds(last_round_key, 10, 0)
    original_key_hex = original_key.hex()
    last_matches = sum(1 for i in range(16) if original_key[i] == expected_bytes[i])
    print(f"Original Key (round 0): {original_key_hex}")
    print(f"Expected:               {expected_key}")
    print(f"Matches: {last_matches}/16")
except Exception as e:
    print(f"Key schedule error: {e}")
    original_key_hex = "error"
    last_matches = 0

# =============================================================================
# Results Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nFirst Round Attack:")
print(f"  Key: {first_key_hex}")
print(f"  Matches: {first_matches}/16")
print(f"  Avg Correlation: {first_round_corr.mean():.4f}")

print(f"\nLast Round Attack:")
print(f"  Round 10 Key: {last_key_hex}")
print(f"  Original Key: {original_key_hex}")
print(f"  Matches: {last_matches}/16")
print(f"  Avg Correlation: {last_round_corr.mean():.4f}")

if first_matches == 16:
    print("\n*** FIRST ROUND ATTACK SUCCESSFUL! ***")
if last_matches == 16:
    print("\n*** LAST ROUND ATTACK SUCCESSFUL! ***")

# =============================================================================
# Generate Graphs
# =============================================================================
print("\n" + "=" * 60)
print("Generating Graphs...")
print("=" * 60)

# Figure 1: Power Traces
print("1. Power consumption traces...")
fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))

ax = axes1[0]
for i in range(min(50, n_traces)):
    ax.plot(traces_all[i], alpha=0.3, linewidth=0.5, color='blue')
ax.axvline(poi_start, color='red', linestyle='--', linewidth=2, label=f'Last Round Start ({poi_start})')
ax.set_xlabel('Sample Number')
ax.set_ylabel('Power (ADC counts)')
ax.set_title(f'Power Consumption Traces ({min(50, n_traces)} traces)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes1[1]
ax.plot(traces_mean, 'b-', linewidth=1)
ax.axvline(poi_start, color='red', linestyle='--', linewidth=2, label='Last Round Region')
ax.axvspan(poi_start, poi_end, alpha=0.2, color='red')
ax.set_xlabel('Sample Number')
ax.set_ylabel('Power (ADC counts)')
ax.set_title('Mean Power Trace')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("power_traces.png", dpi=150, bbox_inches='tight')
print("   Saved: power_traces.png")

# Figure 2: First Round Correlation Traces
print("2. First round correlation traces...")
fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12))
for byte_idx in range(16):
    row, col = byte_idx // 4, byte_idx % 4
    ax = axes2[row, col]
    if first_round_traces[byte_idx] is not None:
        ax.plot(first_round_traces[byte_idx], 'b-', linewidth=0.5)
        peak_idx = np.argmax(first_round_traces[byte_idx])
        ax.plot(peak_idx, first_round_traces[byte_idx][peak_idx], 'r*', markersize=8)
    match = "[OK]" if first_round_key[byte_idx] == expected_bytes[byte_idx] else "[X]"
    ax.set_title(f'Byte {byte_idx}: 0x{first_round_key[byte_idx]:02X} {match}', fontsize=9)
    ax.set_xlabel('Sample', fontsize=8)
    ax.set_ylabel('|Corr|', fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle(f'First Round CPA - Correlation Traces\nKey: {first_key_hex}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_traces_first_round.png", dpi=150, bbox_inches='tight')
print("   Saved: correlation_traces_first_round.png")

# Figure 3: Last Round Correlation Traces
print("3. Last round correlation traces...")
fig3, axes3 = plt.subplots(4, 4, figsize=(16, 12))
for byte_idx in range(16):
    row, col = byte_idx // 4, byte_idx % 4
    ax = axes3[row, col]
    if last_round_traces[byte_idx] is not None:
        ax.plot(last_round_traces[byte_idx], 'b-', linewidth=0.5)
        peak_idx = np.argmax(last_round_traces[byte_idx])
        ax.plot(peak_idx, last_round_traces[byte_idx][peak_idx], 'r*', markersize=8)
    ax.set_title(f'Byte {byte_idx}: 0x{last_round_key[byte_idx]:02X} (r={last_round_corr[byte_idx]:.3f})', fontsize=9)
    ax.set_xlabel('Sample', fontsize=8)
    ax.set_ylabel('|Corr|', fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle(f'Last Round CPA - Correlation Traces\nRound 10 Key: {last_key_hex}', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_traces_last_round.png", dpi=150, bbox_inches='tight')
print("   Saved: correlation_traces_last_round.png")

# Figure 4: Summary
print("4. Summary figure...")
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

# Power traces
ax = axes4[0, 0]
for i in range(min(20, n_traces)):
    ax.plot(traces_all[i], alpha=0.4, linewidth=0.5)
ax.axvspan(poi_start, poi_end, alpha=0.2, color='red', label='Last Round')
ax.set_xlabel('Sample')
ax.set_ylabel('Power')
ax.set_title('Power Traces')
ax.legend()
ax.grid(True, alpha=0.3)

# First round correlation for byte 0
ax = axes4[0, 1]
if first_round_traces[0] is not None:
    ax.plot(first_round_traces[0], 'b-', linewidth=1)
ax.set_xlabel('Sample')
ax.set_ylabel('|Correlation|')
ax.set_title(f'First Round - Byte 0 (0x{first_round_key[0]:02X})')
ax.grid(True, alpha=0.3)

# First round correlations per byte
ax = axes4[1, 0]
colors = ['green' if first_round_key[i] == expected_bytes[i] else 'red' for i in range(16)]
ax.bar(range(16), first_round_corr, color=colors, alpha=0.8)
ax.set_xlabel('Key Byte')
ax.set_ylabel('Max Correlation')
ax.set_title(f'First Round: {first_matches}/16 correct')
ax.set_xticks(range(16))
ax.grid(True, alpha=0.3, axis='y')

# Last round correlations per byte
ax = axes4[1, 1]
ax.bar(range(16), last_round_corr, color='steelblue', alpha=0.8)
ax.set_xlabel('Key Byte')
ax.set_ylabel('Max Correlation')
ax.set_title(f'Last Round: Avg corr = {last_round_corr.mean():.4f}')
ax.set_xticks(range(16))
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('CPA Attack Summary', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("cpa_summary.png", dpi=150, bbox_inches='tight')
print("   Saved: cpa_summary.png")

plt.show()

print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
