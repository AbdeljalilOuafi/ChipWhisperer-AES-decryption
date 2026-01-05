"""
CPA ATTACK - ROUND 10 (Last Round)
Target: S-Box output in the final round using ciphertext
Recovers K10 (last round key), then derives K0 via inverse key schedule
"""

import numpy as np
import chipwhisperer as cw
from tqdm import trange

print("=" * 70)
print("CPA ATTACK - ROUND 10 (Last Round, S-Box Output HW)")
print("=" * 70)

# Load project
project = cw.open_project("aes_fresh_traces.cwp")
print(f"Loaded: {len(project.traces)} traces\n")

traces = np.array([t.wave for t in project.traces], dtype=np.float32)
plaintexts = np.array([list(t.textin) for t in project.traces], dtype=np.uint8)
ciphertexts = np.array([list(t.textout) for t in project.traces], dtype=np.uint8)
n_traces, n_samples = traces.shape

print(f"Shape: {n_traces} x {n_samples}")

# SBOX, Inverse SBOX, and Hamming Weight
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

# Inverse SBOX for verification
INV_SBOX = np.array([SBOX.tolist().index(i) for i in range(256)], dtype=np.uint8)

HW = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)

# AES-128 Key Schedule functions to derive K0 from K10
RCON = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36]

def inv_key_schedule(k10):
    """Reverse the AES-128 key schedule: given K10, recover K0"""
    # Work backwards through 10 rounds
    key = list(k10)
    
    for round_num in range(9, -1, -1):
        # Reverse the key schedule for this round
        # w[i] = w[i-4] XOR w[i-1] for i not divisible by 4
        # w[i] = w[i-4] XOR SubWord(RotWord(w[i-1])) XOR Rcon for i divisible by 4
        
        # Undo columns 3, 2, 1 (simple XOR)
        for col in range(3, 0, -1):
            for row in range(4):
                key[col*4 + row] ^= key[(col-1)*4 + row]
        
        # Undo column 0 (involves SubWord, RotWord, Rcon)
        # Original: w[0] = w_prev[0] XOR SubWord(RotWord(w_prev[3])) XOR Rcon
        # So: w_prev[0] = w[0] XOR SubWord(RotWord(w[3])) XOR Rcon
        # But we need the PREVIOUS w[3], which after undoing cols 1-3 is now in key[12:16]
        
        # After undoing cols 1-3, key[12:16] contains the previous round's column 3
        rot_word = [key[13], key[14], key[15], key[12]]  # RotWord
        sub_word = [SBOX[b] for b in rot_word]  # SubWord
        
        key[0] ^= sub_word[0] ^ RCON[round_num]
        key[1] ^= sub_word[1]
        key[2] ^= sub_word[2]
        key[3] ^= sub_word[3]
    
    return bytes(key)

# Expected keys for verification
expected_k0 = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")

# Compute expected K10 from K0 for verification
def key_schedule(k0):
    """Forward AES-128 key schedule: given K0, compute K10"""
    key = list(k0)
    for round_num in range(10):
        # Compute next round key
        rot_word = [key[13], key[14], key[15], key[12]]
        sub_word = [SBOX[b] for b in rot_word]
        
        new_key = [0] * 16
        new_key[0] = key[0] ^ sub_word[0] ^ RCON[round_num]
        new_key[1] = key[1] ^ sub_word[1]
        new_key[2] = key[2] ^ sub_word[2]
        new_key[3] = key[3] ^ sub_word[3]
        
        for col in range(1, 4):
            for row in range(4):
                new_key[col*4 + row] = key[col*4 + row] ^ new_key[(col-1)*4 + row]
        
        key = new_key
    return bytes(key)

expected_k10 = key_schedule(expected_k0)
print(f"Expected K0:  {expected_k0.hex()}")
print(f"Expected K10: {expected_k10.hex()}\n")

# Compute means and stds for traces
t_mean = np.mean(traces, axis=0)
t_std = np.std(traces, axis=0)

print("Running CPA attack on LAST ROUND (K10)...\n")

recovered_k10 = []
correlations_list = []

for byte_idx in trange(16, desc="K10 bytes"):
    best_corr = 0
    best_guess = 0
    
    for key_guess in range(256):
        # LAST ROUND MODEL:
        # Ciphertext[i] = SBOX[State9[...]] XOR K10[i]  (after ShiftRows)
        # So: SBOX_output = Ciphertext[i] XOR K10[i]
        # We target the Hamming Weight of this S-Box output
        
        hws = np.array([
            HW[ct[byte_idx] ^ key_guess]
            for ct in ciphertexts
        ], dtype=np.float32)
        
        hw_mean = np.mean(hws)
        hw_std = np.std(hws)
        
        if hw_std > 0:
            correlation = np.sum((traces - t_mean) * (hws.reshape(-1, 1) - hw_mean), axis=0) / (t_std * hw_std * n_traces)
            max_corr = np.max(np.abs(correlation))
        else:
            max_corr = 0
        
        if max_corr > best_corr:
            best_corr = max_corr
            best_guess = key_guess
    
    recovered_k10.append(best_guess)
    correlations_list.append(best_corr)

# Results
print("\n" + "=" * 70)
print("LAST ROUND KEY (K10) RECOVERY")
print("=" * 70)

recovered_k10_bytes = bytes(recovered_k10)
print(f"Expected K10:  {expected_k10.hex()}")
print(f"Recovered K10: {recovered_k10_bytes.hex()}")
print(f"Average correlation: {np.mean(correlations_list):.4f}")

matches_k10 = sum(1 for i in range(16) if recovered_k10[i] == expected_k10[i])
print(f"\n✓ K10 Matches: {matches_k10}/16")

for i in range(16):
    match_char = "[OK]" if recovered_k10[i] == expected_k10[i] else "[X]"
    print(f"  {i:2d}: {match_char} 0x{expected_k10[i]:02x} -> 0x{recovered_k10[i]:02x} (corr={correlations_list[i]:.4f})")

# Derive K0 from recovered K10
print("\n" + "=" * 70)
print("DERIVING ORIGINAL KEY (K0) FROM K10")
print("=" * 70)

recovered_k0 = inv_key_schedule(recovered_k10_bytes)
print(f"Expected K0:  {expected_k0.hex()}")
print(f"Derived K0:   {recovered_k0.hex()}")

matches_k0 = sum(1 for i in range(16) if recovered_k0[i] == expected_k0[i])
print(f"\n✓ K0 Matches: {matches_k0}/16")

if recovered_k0 == expected_k0:
    print("\n*** SUCCESS: Original AES key fully recovered! ***")
else:
    print("\n*** PARTIAL: Some key bytes incorrect ***")

print("=" * 70)