import numpy as np
import matplotlib.pyplot as plt
import chipwhisperer as cw

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
scope.trigger.triggers = "tio4"    # Trigger on TIO4
scope.io.tio1 = "serial_rx"
scope.io.tio2 = "serial_tx"
scope.io.hs2 = "clkgen"

# Reset target
if hasattr(scope, 'io'):
    scope.io.nrst = 'low'
    import time
    time.sleep(0.05)
    scope.io.nrst = 'high_z'
    time.sleep(0.05)

# =============================================================================
# AES Inverse S-Box (for last round attack)
# =============================================================================
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
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d
], dtype=np.uint8)

# =============================================================================
# Hamming Weight and Hamming Distance Tables
# =============================================================================
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

def create_hd_table():
    """Create 256x256 Hamming Distance lookup table."""
    hd = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            hd[i, j] = HW[i ^ j]
    return hd

HD_TABLE = create_hd_table()

# =============================================================================
# Trace Capture
# =============================================================================
print("=" * 60)
print("CPA Attack on AES Last Round (Hamming Distance Model)")
print("Target: XMEGA CW308 with TinyAES128")
print("=" * 60)

print("\nCapturing traces...")
N_traces = 5000  # Increased for better statistical significance
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
            ciphertexts_all.append(bytes(ct))
            plaintexts_all.append(bytes(pt))
    
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{N_traces} traces captured")

traces_all = np.array(traces_all, dtype=np.float64)
ciphertexts_all = np.array(ciphertexts_all)
plaintexts_all = np.array(plaintexts_all)

n_traces, n_samples = traces_all.shape
print(f"\nTotal traces captured: {n_traces}")
print(f"Samples per trace: {n_samples}")

# =============================================================================
# Point of Interest (POI) Selection for Last Round
# =============================================================================
# The last round of AES occurs at the end of the trace
# We focus on the last ~25% of the trace where round 10 operations happen
poi_start = int(n_samples * 0.75)  # Start at 75% of trace
poi_end = n_samples                 # End at 100%
print(f"\nPOI Selection: samples {poi_start} to {poi_end} (last round region)")

# Extract POI region for attack
traces_poi = traces_all[:, poi_start:poi_end]

# Normalize traces (z-score normalization)
traces_mean = traces_poi.mean(axis=0, keepdims=True)
traces_std = traces_poi.std(axis=0, keepdims=True)
traces_std[traces_std < 1e-10] = 1.0
traces_norm = (traces_poi - traces_mean) / traces_std

# Also normalize full traces for visualization
traces_full_mean = traces_all.mean(axis=0, keepdims=True)
traces_full_std = traces_all.std(axis=0, keepdims=True)
traces_full_std[traces_full_std < 1e-10] = 1.0
traces_full_norm = (traces_all - traces_full_mean) / traces_full_std

# =============================================================================
# CPA Attack on Last Round using Correct Hamming Distance Model
# =============================================================================
print("\n" + "=" * 60)
print("Attacking last round (Hamming Distance Model)")
print("Power Model: HD(CT, InvSBox(CT XOR k))")
print("=" * 60 + "\n")

best_key = np.zeros(16, dtype=np.uint8)
best_correlations = np.zeros(16, dtype=np.float64)
all_correlations = np.zeros((16, 256), dtype=np.float64)
correlation_traces = {}  # Store correlation traces for plotting

for byte_idx in range(16):
    print(f"Attacking byte {byte_idx:2d}...", end=" ")
    
    # Extract ciphertext column for this byte
    ct_column = np.array([ct[byte_idx] for ct in ciphertexts_all], dtype=np.uint8)
    
    max_corr = np.zeros(256, dtype=np.float64)
    best_corr_trace = None
    best_k = 0
    
    for kguess in range(256):
        # Compute intermediate value: InvSBox(CT XOR k)
        inter = ISBOX[ct_column ^ kguess]
        
        # CORRECT Hamming Distance Model for last round:
        # HD(CT, InvSBox(CT XOR k)) - models the register transition during AddRoundKey
        hyp = HD_TABLE[ct_column, inter].astype(np.float64)
        
        # Normalize hypothesis
        hyp_c = hyp - hyp.mean()
        hyp_std = hyp_c.std()
        
        if hyp_std < 1e-10:
            continue
        
        hyp_n = hyp_c / hyp_std
        
        # Compute Pearson correlation with all samples in POI region
        corr = np.abs(np.dot(hyp_n, traces_norm)) / n_traces
        max_corr[kguess] = np.max(corr)
        
        if max_corr[kguess] > max_corr[best_k]:
            best_k = kguess
            best_corr_trace = corr.copy()
    
    best = np.argmax(max_corr)
    best_key[byte_idx] = best
    best_correlations[byte_idx] = max_corr[best]
    all_correlations[byte_idx] = max_corr
    correlation_traces[byte_idx] = best_corr_trace
    
    # Show top 3 candidates
    top3 = np.argsort(max_corr)[-3:][::-1]
    print(f"Best: 0x{best:02x} (corr={max_corr[best]:.4f}) | "
          f"2nd: 0x{top3[1]:02x} ({max_corr[top3[1]]:.4f}) | "
          f"3rd: 0x{top3[2]:02x} ({max_corr[top3[2]]:.4f})")

# =============================================================================
# Results Summary
# =============================================================================
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

key_hex = "".join(f"{b:02x}" for b in best_key)
expected_key = "2b7e151628aed2a6abf7158809cf4f3c"

print(f"\nRecovered key: {key_hex}")
print(f"Expected key:  {expected_key}")
print(f"\nAverage correlation: {best_correlations.mean():.4f}")
print(f"Min correlation: {best_correlations.min():.4f}")
print(f"Max correlation: {best_correlations.max():.4f}")

# Check if key matches
if key_hex == expected_key:
    print("\n*** KEY SUCCESSFULLY RECOVERED! ***")
else:
    # Count matching bytes
    expected_bytes = bytes.fromhex(expected_key)
    matches = sum(1 for i in range(16) if best_key[i] == expected_bytes[i])
    print(f"\nMatching bytes: {matches}/16")

# =============================================================================
# FIGURE 1: Power Consumption Traces
# =============================================================================
print("\nGenerating power consumption trace graph...")

fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1a: Multiple overlaid traces
ax = axes1[0]
num_traces_to_plot = min(50, n_traces)
for i in range(num_traces_to_plot):
    ax.plot(traces_all[i], alpha=0.3, linewidth=0.5, color='blue')
ax.axvline(poi_start, color='red', linestyle='--', linewidth=2, label=f'POI Start (sample {poi_start})')
ax.set_xlabel('Sample Number', fontsize=12)
ax.set_ylabel('Power (ADC counts)', fontsize=12)
ax.set_title(f'Power Consumption Traces ({num_traces_to_plot} traces overlaid)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 1b: Mean trace with standard deviation
ax = axes1[1]
mean_trace = traces_all.mean(axis=0)
std_trace = traces_all.std(axis=0)
samples = np.arange(n_samples)
ax.plot(samples, mean_trace, 'b-', linewidth=1, label='Mean trace')
ax.fill_between(samples, mean_trace - std_trace, mean_trace + std_trace, 
                alpha=0.3, color='blue', label='±1 std dev')
ax.axvline(poi_start, color='red', linestyle='--', linewidth=2, label=f'Last Round Region Start')
ax.axvspan(poi_start, poi_end, alpha=0.1, color='red', label='Last Round POI')
ax.set_xlabel('Sample Number', fontsize=12)
ax.set_ylabel('Power (ADC counts)', fontsize=12)
ax.set_title('Mean Power Trace with Standard Deviation', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("power_traces.png", dpi=150, bbox_inches='tight')
print("  Saved: power_traces.png")

# =============================================================================
# FIGURE 2: Correlation Traces for Last Round Attack
# =============================================================================
print("Generating correlation trace graphs...")

fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12))

for byte_idx in range(16):
    row = byte_idx // 4
    col = byte_idx % 4
    ax = axes2[row, col]
    
    corr_trace = correlation_traces[byte_idx]
    if corr_trace is not None:
        samples_poi = np.arange(poi_start, poi_end)
        ax.plot(samples_poi, corr_trace, 'b-', linewidth=0.8)
        
        # Mark peak
        peak_idx = np.argmax(corr_trace)
        peak_val = corr_trace[peak_idx]
        ax.plot(samples_poi[peak_idx], peak_val, 'r*', markersize=10)
        ax.axhline(peak_val, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    
    ax.set_title(f'Byte {byte_idx}: 0x{best_key[byte_idx]:02X} (r={best_correlations[byte_idx]:.3f})', 
                 fontsize=10)
    ax.set_xlabel('Sample', fontsize=8)
    ax.set_ylabel('|Correlation|', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(0.1, best_correlations.max() * 1.2)])

plt.suptitle('Correlation Traces for Last Round CPA Attack\nHD(CT, InvSBox(CT ⊕ k)) Model', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_traces.png", dpi=150, bbox_inches='tight')
print("  Saved: correlation_traces.png")

# =============================================================================
# FIGURE 3: Key Candidate Correlations (Bar Charts)
# =============================================================================
print("Generating key candidate correlation graphs...")

fig3, axes3 = plt.subplots(4, 4, figsize=(16, 12))

expected_bytes = bytes.fromhex(expected_key)

for byte_idx in range(16):
    row = byte_idx // 4
    col = byte_idx % 4
    ax = axes3[row, col]
    
    correlations = all_correlations[byte_idx]
    colors = ['blue'] * 256
    colors[best_key[byte_idx]] = 'red'  # Highlight recovered key
    if byte_idx < len(expected_bytes) and expected_bytes[byte_idx] != best_key[byte_idx]:
        colors[expected_bytes[byte_idx]] = 'green'  # Highlight expected key if different
    
    ax.bar(range(256), correlations, color=colors, alpha=0.7, width=1.0)
    ax.axvline(best_key[byte_idx], color='red', linestyle='--', linewidth=1.5)
    
    title = f'Byte {byte_idx}: Recovered=0x{best_key[byte_idx]:02X}'
    if byte_idx < len(expected_bytes):
        title += f' (Expected=0x{expected_bytes[byte_idx]:02X})'
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Key Guess', fontsize=8)
    ax.set_ylabel('Max |Correlation|', fontsize=8)
    ax.set_xlim([0, 255])

plt.suptitle('Key Candidate Correlations - Last Round CPA Attack\nRed=Recovered, Green=Expected (if different)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("key_correlations.png", dpi=150, bbox_inches='tight')
print("  Saved: key_correlations.png")

# =============================================================================
# FIGURE 4: Summary Figure for Professor
# =============================================================================
print("Generating summary figure...")

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 4a: Sample power traces
ax = axes4[0, 0]
for i in range(min(20, n_traces)):
    ax.plot(traces_all[i], alpha=0.4, linewidth=0.5)
ax.axvspan(poi_start, poi_end, alpha=0.2, color='red', label='Last Round Region')
ax.set_xlabel('Sample Number', fontsize=11)
ax.set_ylabel('Power (ADC counts)', fontsize=11)
ax.set_title('Power Consumption Traces (20 traces)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4b: Correlation trace for byte 0
ax = axes4[0, 1]
byte_idx = 0
corr_trace = correlation_traces[byte_idx]
if corr_trace is not None:
    samples_poi = np.arange(poi_start, poi_end)
    ax.plot(samples_poi, corr_trace, 'b-', linewidth=1)
    peak_idx = np.argmax(corr_trace)
    peak_val = corr_trace[peak_idx]
    ax.plot(samples_poi[peak_idx], peak_val, 'r*', markersize=15, label=f'Peak: {peak_val:.4f}')
ax.set_xlabel('Sample Number', fontsize=11)
ax.set_ylabel('|Correlation|', fontsize=11)
ax.set_title(f'Correlation Trace - Key Byte 0 (0x{best_key[0]:02X})', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4c: Key candidates for byte 0
ax = axes4[1, 0]
correlations = all_correlations[0]
ax.bar(range(256), correlations, alpha=0.7, width=1.0, color='steelblue')
ax.axvline(best_key[0], color='red', linestyle='--', linewidth=2, label=f'Best: 0x{best_key[0]:02X}')
ax.set_xlabel('Key Guess (0-255)', fontsize=11)
ax.set_ylabel('Max |Correlation|', fontsize=11)
ax.set_title('Key Candidate Correlations - Byte 0', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4d: All bytes correlation summary
ax = axes4[1, 1]
byte_indices = np.arange(16)
bars = ax.bar(byte_indices, best_correlations, color='steelblue', alpha=0.8)
ax.set_xlabel('Key Byte Index', fontsize=11)
ax.set_ylabel('Best Correlation', fontsize=11)
ax.set_title('Correlation per Key Byte', fontsize=12)
ax.set_xticks(byte_indices)
ax.set_xticklabels([f'{i}' for i in range(16)])
ax.axhline(best_correlations.mean(), color='red', linestyle='--', 
           label=f'Mean: {best_correlations.mean():.4f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add recovered key as text
fig4.text(0.5, 0.02, f'Recovered Key: {key_hex}', ha='center', fontsize=12, 
          fontweight='bold', family='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('CPA Attack on AES Last Round - Summary\n'
             f'Target: XMEGA CW308 | Traces: {n_traces} | Model: HD(CT, InvSBox(CT⊕k))', 
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("cpa_last_round_summary.png", dpi=150, bbox_inches='tight')
print("  Saved: cpa_last_round_summary.png")

plt.show()

print("\n" + "=" * 60)
print("ATTACK COMPLETE")
print("=" * 60)
print("\nGenerated figures:")
print("  1. power_traces.png - Power consumption traces")
print("  2. correlation_traces.png - Correlation traces for all 16 bytes")
print("  3. key_correlations.png - Key candidate correlations")
print("  4. cpa_last_round_summary.png - Summary figure for professor")
