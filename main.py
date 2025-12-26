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

# =============================================================================
# Hamming Weight Table
# =============================================================================
HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# =============================================================================
# Trace Capture
# =============================================================================
print("=" * 60)
print("CPA Attack on AES - Multiple Power Models")
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
        # Only accept complete 16-byte ciphertexts
        if ct and len(ct) == 16:
            traces_all.append(trace.copy())
            ciphertexts_all.append(list(ct))  # Store as list for consistent indexing
            plaintexts_all.append(list(pt))
    
    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{N_traces} traces captured")

# Convert to numpy arrays
traces_all = np.array(traces_all, dtype=np.float64)
ciphertexts_all = np.array(ciphertexts_all, dtype=np.uint8)
plaintexts_all = np.array(plaintexts_all, dtype=np.uint8)

n_traces, n_samples = traces_all.shape
print(f"\nTotal valid traces: {n_traces}")
print(f"Samples per trace: {n_samples}")

# Normalize traces (z-score normalization)
traces_mean = traces_all.mean(axis=0, keepdims=True)
traces_std = traces_all.std(axis=0, keepdims=True)
traces_std[traces_std < 1e-10] = 1.0
traces_norm = (traces_all - traces_mean) / traces_std

# =============================================================================
# CPA Attack Function - Supports Multiple Power Models
# =============================================================================
def cpa_attack(traces_norm, data, attack_type='last_round', model='hw'):
    """
    Perform CPA attack with different power models.
    
    attack_type: 'last_round' (use ciphertexts) or 'first_round' (use plaintexts)
    model: 'hw' (Hamming Weight) or 'hd' (Hamming Distance)
    """
    n_traces = traces_norm.shape[0]
    best_key = np.zeros(16, dtype=np.uint8)
    best_correlations = np.zeros(16, dtype=np.float64)
    all_correlations = np.zeros((16, 256), dtype=np.float64)
    correlation_traces = {}
    
    for byte_idx in range(16):
        data_column = data[:, byte_idx]
        max_corr = np.zeros(256, dtype=np.float64)
        best_corr_trace = None
        best_k = 0
        
        for kguess in range(256):
            if attack_type == 'last_round':
                # Last round: intermediate = InvSBox(CT XOR k)
                inter = ISBOX[data_column ^ kguess]
            else:
                # First round: intermediate = SBox(PT XOR k)
                inter = SBOX[data_column ^ kguess]
            
            # Power model selection
            if model == 'hw':
                # Hamming Weight of intermediate value
                hyp = HW[inter].astype(np.float64)
            elif model == 'hd':
                # Hamming Distance: HD(data, intermediate)
                hyp = HW[data_column ^ inter].astype(np.float64)
            elif model == 'hd_sbox':
                # Hamming Distance through S-box: HD(input, output)
                sbox_input = data_column ^ kguess
                hyp = HW[sbox_input ^ inter].astype(np.float64)
            else:
                hyp = HW[inter].astype(np.float64)
            
            # Normalize hypothesis
            hyp_c = hyp - hyp.mean()
            hyp_std = hyp_c.std()
            
            if hyp_std < 1e-10:
                continue
            
            hyp_n = hyp_c / hyp_std
            
            # Compute Pearson correlation
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
    
    return best_key, best_correlations, all_correlations, correlation_traces

# =============================================================================
# Try Multiple Attack Configurations
# =============================================================================
print("\n" + "=" * 60)
print("Testing Multiple Power Models")
print("=" * 60)

expected_key = "2b7e151628aed2a6abf7158809cf4f3c"
expected_bytes = bytes.fromhex(expected_key)

attack_configs = [
    ('last_round', 'hw', ciphertexts_all, 'Last Round - Hamming Weight: HW(InvSBox(CT^k))'),
    ('last_round', 'hd', ciphertexts_all, 'Last Round - Hamming Distance: HD(CT, InvSBox(CT^k))'),
    ('last_round', 'hd_sbox', ciphertexts_all, 'Last Round - S-box HD: HD(CT^k, InvSBox(CT^k))'),
    ('first_round', 'hw', plaintexts_all, 'First Round - Hamming Weight: HW(SBox(PT^k))'),
    ('first_round', 'hd', plaintexts_all, 'First Round - Hamming Distance: HD(PT, SBox(PT^k))'),
]

results = {}

for attack_type, model, data, description in attack_configs:
    print(f"\n{'-' * 60}")
    print(f"Attack: {description}")
    print(f"{'-' * 60}")
    
    best_key, best_corrs, all_corrs, corr_traces = cpa_attack(
        traces_norm, data, attack_type, model
    )
    
    key_hex = "".join(f"{b:02x}" for b in best_key)
    avg_corr = best_corrs.mean()
    
    # Count matching bytes
    matches = sum(1 for i in range(16) if best_key[i] == expected_bytes[i])
    
    print(f"Recovered: {key_hex}")
    print(f"Expected:  {expected_key}")
    print(f"Matching bytes: {matches}/16")
    print(f"Avg correlation: {avg_corr:.4f} (min: {best_corrs.min():.4f}, max: {best_corrs.max():.4f})")
    
    results[f"{attack_type}_{model}"] = {
        'key': best_key,
        'correlations': best_corrs,
        'all_correlations': all_corrs,
        'correlation_traces': corr_traces,
        'matches': matches,
        'avg_corr': avg_corr,
        'description': description
    }

# Find best attack configuration
best_config = max(results.keys(), key=lambda k: (results[k]['matches'], results[k]['avg_corr']))
best_result = results[best_config]

print("\n" + "=" * 60)
print("BEST RESULT")
print("=" * 60)
print(f"Best attack: {best_result['description']}")
print(f"Recovered key: {''.join(f'{b:02x}' for b in best_result['key'])}")
print(f"Matching bytes: {best_result['matches']}/16")
print(f"Average correlation: {best_result['avg_corr']:.4f}")

# Use best result for plotting
best_key = best_result['key']
best_correlations = best_result['correlations']
all_correlations = best_result['all_correlations']
correlation_traces = best_result['correlation_traces']

# Print per-byte results
print("\nPer-byte results:")
for byte_idx in range(16):
    corrs = all_correlations[byte_idx]
    top3 = np.argsort(corrs)[-3:][::-1]
    expected_byte = expected_bytes[byte_idx]
    recovered = best_key[byte_idx]
    match = "[OK]" if recovered == expected_byte else "[X]"
    print(f"Byte {byte_idx:2d}: 0x{recovered:02x} (exp: 0x{expected_byte:02x}) {match} | "
          f"corr={corrs[recovered]:.4f} | 2nd: 0x{top3[1]:02x}({corrs[top3[1]]:.4f})")

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
ax.set_xlabel('Sample Number', fontsize=12)
ax.set_ylabel('Power (ADC counts)', fontsize=12)
ax.set_title(f'Power Consumption Traces ({num_traces_to_plot} traces overlaid)', fontsize=14)
ax.grid(True, alpha=0.3)

# Plot 1b: Mean trace with standard deviation
ax = axes1[1]
mean_trace = traces_all.mean(axis=0)
std_trace = traces_all.std(axis=0)
samples = np.arange(n_samples)
ax.plot(samples, mean_trace, 'b-', linewidth=1, label='Mean trace')
ax.fill_between(samples, mean_trace - std_trace, mean_trace + std_trace, 
                alpha=0.3, color='blue', label='±1 std dev')
ax.set_xlabel('Sample Number', fontsize=12)
ax.set_ylabel('Power (ADC counts)', fontsize=12)
ax.set_title('Mean Power Trace with Standard Deviation', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("power_traces.png", dpi=150, bbox_inches='tight')
print("  Saved: power_traces.png")

# =============================================================================
# FIGURE 2: Correlation Traces for All 16 Bytes
# =============================================================================
print("Generating correlation trace graphs...")

fig2, axes2 = plt.subplots(4, 4, figsize=(16, 12))

for byte_idx in range(16):
    row = byte_idx // 4
    col = byte_idx % 4
    ax = axes2[row, col]
    
    corr_trace = correlation_traces[byte_idx]
    if corr_trace is not None:
        ax.plot(corr_trace, 'b-', linewidth=0.8)
        
        # Mark peak
        peak_idx = np.argmax(corr_trace)
        peak_val = corr_trace[peak_idx]
        ax.plot(peak_idx, peak_val, 'r*', markersize=10)
        ax.axhline(peak_val, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
    
    match = "[OK]" if best_key[byte_idx] == expected_bytes[byte_idx] else "[X]"
    ax.set_title(f'Byte {byte_idx}: 0x{best_key[byte_idx]:02X} {match} (r={best_correlations[byte_idx]:.3f})', 
                 fontsize=10)
    ax.set_xlabel('Sample', fontsize=8)
    ax.set_ylabel('|Correlation|', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Correlation Traces - {best_result["description"]}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_traces.png", dpi=150, bbox_inches='tight')
print("  Saved: correlation_traces.png")

# =============================================================================
# FIGURE 3: Key Candidate Correlations (Bar Charts)
# =============================================================================
print("Generating key candidate correlation graphs...")

fig3, axes3 = plt.subplots(4, 4, figsize=(16, 12))

for byte_idx in range(16):
    row = byte_idx // 4
    col = byte_idx % 4
    ax = axes3[row, col]
    
    correlations = all_correlations[byte_idx]
    colors = ['steelblue'] * 256
    colors[best_key[byte_idx]] = 'red'  # Highlight recovered key
    if expected_bytes[byte_idx] != best_key[byte_idx]:
        colors[expected_bytes[byte_idx]] = 'green'  # Highlight expected key if different
    
    ax.bar(range(256), correlations, color=colors, alpha=0.7, width=1.0)
    ax.axvline(best_key[byte_idx], color='red', linestyle='--', linewidth=1.5, 
               label=f'Recovered: 0x{best_key[byte_idx]:02X}')
    if expected_bytes[byte_idx] != best_key[byte_idx]:
        ax.axvline(expected_bytes[byte_idx], color='green', linestyle=':', linewidth=1.5,
                   label=f'Expected: 0x{expected_bytes[byte_idx]:02X}')
    
    ax.set_title(f'Byte {byte_idx}', fontsize=10)
    ax.set_xlabel('Key Guess', fontsize=8)
    ax.set_ylabel('Max |Corr|', fontsize=8)
    ax.set_xlim([0, 255])
    ax.legend(fontsize=6, loc='upper right')

plt.suptitle('Key Candidate Correlations\nRed=Recovered, Green=Expected (if different)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("key_correlations.png", dpi=150, bbox_inches='tight')
print("  Saved: key_correlations.png")

# =============================================================================
# FIGURE 4: Summary Comparison of All Models
# =============================================================================
print("Generating model comparison summary...")

fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))

# Plot comparison of all models
ax = axes4[0, 0]
model_names = []
model_avg_corrs = []
model_matches = []
for key, result in results.items():
    short_name = key.replace('_', '\n')
    model_names.append(short_name)
    model_avg_corrs.append(result['avg_corr'])
    model_matches.append(result['matches'])

x = np.arange(len(model_names))
bars = ax.bar(x, model_avg_corrs, color='steelblue', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=8)
ax.set_ylabel('Average Correlation', fontsize=11)
ax.set_title('Comparison of Power Models - Avg Correlation', fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

# Add match counts on bars
for i, (bar, matches) in enumerate(zip(bars, model_matches)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
            f'{matches}/16', ha='center', va='bottom', fontsize=9)

# Plot: Power traces sample
ax = axes4[0, 1]
for i in range(min(20, n_traces)):
    ax.plot(traces_all[i], alpha=0.4, linewidth=0.5)
ax.set_xlabel('Sample Number', fontsize=11)
ax.set_ylabel('Power (ADC counts)', fontsize=11)
ax.set_title('Power Consumption Traces (20 samples)', fontsize=12)
ax.grid(True, alpha=0.3)

# Plot: Best model correlation trace for byte 0
ax = axes4[1, 0]
corr_trace = correlation_traces[0]
if corr_trace is not None:
    ax.plot(corr_trace, 'b-', linewidth=1)
    peak_idx = np.argmax(corr_trace)
    peak_val = corr_trace[peak_idx]
    ax.plot(peak_idx, peak_val, 'r*', markersize=15, label=f'Peak: {peak_val:.4f} @ {peak_idx}')
ax.set_xlabel('Sample Number', fontsize=11)
ax.set_ylabel('|Correlation|', fontsize=11)
ax.set_title(f'Correlation Trace - Key Byte 0 (Best Model)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot: Correlation per byte for best model
ax = axes4[1, 1]
byte_indices = np.arange(16)
colors = ['green' if best_key[i] == expected_bytes[i] else 'red' for i in range(16)]
bars = ax.bar(byte_indices, best_correlations, color=colors, alpha=0.8)
ax.set_xlabel('Key Byte Index', fontsize=11)
ax.set_ylabel('Best Correlation', fontsize=11)
ax.set_title('Correlation per Key Byte (Green=Correct, Red=Wrong)', fontsize=12)
ax.set_xticks(byte_indices)
ax.axhline(best_correlations.mean(), color='blue', linestyle='--', 
           label=f'Mean: {best_correlations.mean():.4f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

key_hex = "".join(f"{b:02x}" for b in best_key)
fig4.text(0.5, 0.02, f'Best Model: {best_result["description"]}\nRecovered Key: {key_hex} ({best_result["matches"]}/16 correct)', 
          ha='center', fontsize=11, fontweight='bold', family='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle(f'CPA Attack Summary - {n_traces} Traces', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig("cpa_summary.png", dpi=150, bbox_inches='tight')
print("  Saved: cpa_summary.png")

plt.show()

print("\n" + "=" * 60)
print("ATTACK COMPLETE")
print("=" * 60)
print("\nGenerated figures:")
print("  1. power_traces.png - Power consumption traces")
print("  2. correlation_traces.png - Correlation traces for all 16 bytes")
print("  3. key_correlations.png - Key candidate correlations")
print("  4. cpa_summary.png - Model comparison summary")
