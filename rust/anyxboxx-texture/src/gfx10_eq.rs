//! Modes d'agencement GFX10 / RDNA 2, configuration mémoire (pipes / PKRs) et équations binaires.
//!
//! Porté des sources publiques MIT d'AMD addrlib dans Mesa (`src/amd/addrlib/src/gfx10` :
//! `gfx10SwizzlePattern.h` et `gfx10addrlib.cpp`).
//!
//! Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
//! SPDX-License-Identifier: MIT

const PIPE_INTERLEAVE_LOG2: u32 = 8;
const COLUMN_BITS: u32 = 2;
const MAX_BANK_XOR_BITS: u32 = 4;

/// Modes d'agencement GFX10 (numéros matériels `ADDR_SW_*` d'addrlib).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwizzleMode {
    Linear,
    Sw256BS,
    Sw256BD,
    Sw4KS,
    Sw4KD,
    Sw64KS,
    Sw64KD,
    Sw64KST,
    Sw64KDT,
    Sw4KSX,
    Sw4KDX,
    Sw64KZX,
    Sw64KSX,
    Sw64KDX,
    Sw64KRX,
}

impl SwizzleMode {
    pub fn from_hw(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::Linear,
            1 => Self::Sw256BS,
            2 => Self::Sw256BD,
            5 => Self::Sw4KS,
            6 => Self::Sw4KD,
            9 => Self::Sw64KS,
            10 => Self::Sw64KD,
            17 => Self::Sw64KST,
            18 => Self::Sw64KDT,
            21 => Self::Sw4KSX,
            22 => Self::Sw4KDX,
            24 => Self::Sw64KZX,
            25 => Self::Sw64KSX,
            26 => Self::Sw64KDX,
            27 => Self::Sw64KRX,
            _ => return None,
        })
    }

    pub const fn hw_code(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Sw256BS => 1,
            Self::Sw256BD => 2,
            Self::Sw4KS => 5,
            Self::Sw4KD => 6,
            Self::Sw64KS => 9,
            Self::Sw64KD => 10,
            Self::Sw64KST => 17,
            Self::Sw64KDT => 18,
            Self::Sw4KSX => 21,
            Self::Sw4KDX => 22,
            Self::Sw64KZX => 24,
            Self::Sw64KSX => 25,
            Self::Sw64KDX => 26,
            Self::Sw64KRX => 27,
        }
    }

    pub fn parse_name(name: &str) -> Result<Self, String> {
        let key = name.trim().to_ascii_lowercase().replace('_', "-");
        let key = key.strip_prefix("sw-").unwrap_or(&key);
        match key {
            "linear" => Ok(Self::Linear),
            "256b-s" | "256-s" => Ok(Self::Sw256BS),
            "256b-d" | "256-d" => Ok(Self::Sw256BD),
            "4k-s" | "4kb-s" => Ok(Self::Sw4KS),
            "4k-d" | "4kb-d" => Ok(Self::Sw4KD),
            "64k-s" | "64kb-s" => Ok(Self::Sw64KS),
            "64k-d" | "64kb-d" => Ok(Self::Sw64KD),
            "64k-s-t" | "64kb-s-t" => Ok(Self::Sw64KST),
            "64k-d-t" | "64kb-d-t" => Ok(Self::Sw64KDT),
            "4k-s-x" | "4kb-s-x" => Ok(Self::Sw4KSX),
            "4k-d-x" | "4kb-d-x" => Ok(Self::Sw4KDX),
            "64k-z-x" | "64kb-z-x" => Ok(Self::Sw64KZX),
            "64k-s-x" | "64kb-s-x" => Ok(Self::Sw64KSX),
            "64k-d-x" | "64kb-d-x" => Ok(Self::Sw64KDX),
            "64k-r-x" | "64kb-r-x" => Ok(Self::Sw64KRX),
            _ => Err(format!("mode GFX10 inconnu : {name}")),
        }
    }

    pub const fn block_log2(self) -> u32 {
        match self {
            Self::Linear => 0,
            Self::Sw256BS | Self::Sw256BD => 8,
            Self::Sw4KS | Self::Sw4KD | Self::Sw4KSX | Self::Sw4KDX => 12,
            Self::Sw64KS
            | Self::Sw64KD
            | Self::Sw64KST
            | Self::Sw64KDT
            | Self::Sw64KZX
            | Self::Sw64KSX
            | Self::Sw64KDX
            | Self::Sw64KRX => 16,
        }
    }

    pub const fn block_bytes(self) -> usize {
        if matches!(self, Self::Linear) { 0 } else { 1usize << self.block_log2() }
    }

    pub const fn is_xor(self) -> bool {
        self.hw_code() >= 16
    }

    pub(crate) const fn is_z_order(self) -> bool {
        matches!(self, Self::Sw64KZX)
    }

    const fn is_micro_xor(self) -> bool {
        matches!(self, Self::Sw64KST | Self::Sw64KDT)
    }

    const fn is_display(self) -> bool {
        matches!(
            self,
            Self::Sw256BD | Self::Sw4KD | Self::Sw64KD | Self::Sw64KDT | Self::Sw4KDX | Self::Sw64KDX
        )
    }
}

/// Configuration mémoire RDNA 2 (`GB_ADDR_CONFIG` : nombre de pipes et de packers).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipeConfig {
    pub pipes_log2: u32,
    pub pkrs_log2: u32,
}

impl PipeConfig {
    /// RDNA 2 RB+ 16 pipes / 8 PKRs (configuration par défaut pour les modes `_X`).
    pub const SCARLETT: Self = Self { pipes_log2: 4, pkrs_log2: 3 };

    /// Configuration sans entrelacement multi-pipe (1 pipe, 1 PKR).
    pub const SINGLE_PIPE: Self = Self { pipes_log2: 0, pkrs_log2: 0 };

    pub fn new(pipes: u32, pkrs: u32) -> Result<Self, String> {
        if !pipes.is_power_of_two() || pipes > 64 {
            return Err(format!("nombre de pipes invalide (1..=64, puissance de 2) : {pipes}"));
        }
        if !pkrs.is_power_of_two() || pkrs > 32 || pkrs > pipes {
            return Err(format!("nombre de PKRs invalide (1..=32, <= pipes) : {pkrs}"));
        }
        Ok(Self { pipes_log2: pipes.trailing_zeros(), pkrs_log2: pkrs.trailing_zeros() })
    }

    fn bank_xor_bits(self, block_log2: u32) -> u32 {
        let base = PIPE_INTERLEAVE_LOG2 + self.pipes_log2 + COLUMN_BITS;
        if block_log2 > base { (block_log2 - base).min(MAX_BANK_XOR_BITS) } else { 0 }
    }

    /// Masque `pipeBankXor` effectif (décalé de 8 bits et borné au bloc), selon `ComputeSurfaceAddrFromCoordMacroTiled`.
    pub fn effective_pipe_bank_xor(self, mode: SwizzleMode, pipe_bank_xor: u32) -> u32 {
        if !mode.is_xor() {
            return 0;
        }
        let block_log2 = mode.block_log2();
        let block_mask = (1u32 << block_log2) - 1;
        let pipe_mask = (1u32 << self.pipes_log2) - 1;
        let bank_bits = self.bank_xor_bits(block_log2);
        let bank_mask = ((1u32 << bank_bits) - 1) << (self.pipes_log2 + COLUMN_BITS);
        ((pipe_bank_xor & (pipe_mask | bank_mask)) << PIPE_INTERLEAVE_LOG2) & block_mask
    }
}

/// Terme binaire d'un bit d'adresse : parité XOR des bits sélectionnés sur X, Y et Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BitTerm {
    pub x: u16,
    pub y: u16,
    pub z: u8,
}

impl BitTerm {
    const fn new(x: u16, y: u16, z: u8) -> Self {
        Self { x, y, z }
    }

    const fn xor(self, other: Self) -> Self {
        Self { x: self.x ^ other.x, y: self.y ^ other.y, z: self.z ^ other.z }
    }
}

/// Équation binaire complète (jusqu'à 16 bits d'adresse dans un bloc de 64 Kio).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Equation {
    bits: Vec<BitTerm>,
}

impl Equation {
    /// Construit l'équation GFX10 RB+ pour un mode 2D (1xAA), une taille d'élément et une configuration pipe/PKR.
    pub fn for_mode(mode: SwizzleMode, element_bytes: u32, pipes: PipeConfig) -> Result<Self, String> {
        if !element_bytes.is_power_of_two() || element_bytes > 16 {
            return Err(format!("taille d'élément invalide (1, 2, 4, 8 ou 16) : {element_bytes}"));
        }
        if matches!(mode, SwizzleMode::Linear) {
            return Ok(Self { bits: Vec::new() });
        }
        let elem_log2 = element_bytes.trailing_zeros() as usize;
        let block_bits = mode.block_log2() as usize;
        match mode {
            SwizzleMode::Sw64KRX => build_rx_zx(elem_log2, pipes, false),
            SwizzleMode::Sw64KZX => build_rx_zx(elem_log2, pipes, true),
            _ if !mode.is_xor() => Ok(Self { bits: build_sd_base(mode.is_display(), elem_log2, block_bits) }),
            _ => Ok(Self {
                bits: build_sd_xor(mode.is_display(), elem_log2, block_bits, pipes, mode.is_micro_xor()),
            }),
        }
    }

    /// Décalage en octets dans la tuile pour les coordonnées d'élément `(x, y, slice)`.
    pub fn eval_byte_offset(&self, x: u32, y: u32, slice: u32) -> usize {
        let mut offset = 0usize;
        for (bit, term) in self.bits.iter().enumerate() {
            let parity = (x & u32::from(term.x)).count_ones()
                + (y & u32::from(term.y)).count_ones()
                + (slice & u32::from(term.z)).count_ones();
            offset |= ((parity & 1) as usize) << bit;
        }
        offset
    }

    pub fn terms(&self) -> &[BitTerm] {
        &self.bits
    }
}

/// Motifs 256 octets (`GFX10_SW_PATTERN_NIBBLE01`) pour Standard (S), Display (D) et Z-order (Z).
const MICRO_S: [&[u8; 16]; 5] = [
    b"x0x1x2x3y0y1y2y3",
    b"--x0x1x2y0y1y2x3",
    b"----x0x1y0y1y2x2",
    b"------x0y0y1x1x2",
    b"--------y0y1x0x1",
];

const MICRO_D: [&[u8; 16]; 5] = [
    b"x0x1x2y1y0y2x3y3",
    b"--x0x1x2y0y1y2x3",
    b"----x0x1y0y1x2y2",
    b"------x0y0x1x2y1",
    b"--------x0y0x1y1",
];

const MICRO_Z: [&[u8; 16]; 5] = [
    b"x0y0x1y1x2y2x3y3",
    b"--x0y0x1y1x2y2x3",
    b"----x0y0x1y1x2y2",
    b"------x0y0x1y1x2",
    b"--------x0y0x1y1",
];

fn parse_micro(pattern: &[u8; 16]) -> [BitTerm; 8] {
    let mut terms = [BitTerm::default(); 8];
    for bit in 0..8 {
        terms[bit] = match pattern[bit * 2] {
            b'x' => BitTerm::new(1u16 << (pattern[bit * 2 + 1] - b'0'), 0, 0),
            b'y' => BitTerm::new(0, 1u16 << (pattern[bit * 2 + 1] - b'0'), 0),
            _ => BitTerm::default(),
        };
    }
    terms
}

fn build_sd_base(is_d: bool, elem_log2: usize, num_bits: usize) -> Vec<BitTerm> {
    let micro = parse_micro(if is_d { MICRO_D[elem_log2] } else { MICRO_S[elem_log2] });
    let mut terms = Vec::with_capacity(num_bits);
    terms.extend_from_slice(&micro[..num_bits.min(8)]);
    let mut nx = micro.iter().filter(|t| t.x != 0).count();
    let mut ny = micro.iter().filter(|t| t.y != 0).count();
    for b in 8..num_bits {
        if (b - 8) % 2 == 0 {
            terms.push(BitTerm::new(0, 1u16 << ny, 0));
            ny += 1;
        } else {
            terms.push(BitTerm::new(1u16 << nx, 0, 0));
            nx += 1;
        }
    }
    terms
}

fn build_sd_xor(is_d: bool, elem_log2: usize, block_bits: usize, pipes: PipeConfig, is_micro_xor: bool) -> Vec<BitTerm> {
    let (p, k) = (pipes.pipes_log2 as usize, pipes.pkrs_log2 as usize);
    let base = build_sd_base(is_d, elem_log2, block_bits.max(8 + 2 * p));
    let mut out = base[..block_bits].to_vec();
    if is_micro_xor {
        for j in 0..p {
            let (dst, src) = (8 + j, 8 + 2 * p - 1 - j);
            if dst < block_bits && src < block_bits {
                out[dst] = out[dst].xor(base[src]);
            }
        }
    } else {
        let max_xor = block_bits - 8;
        let xy_bits = p.min(max_xor);
        let z_shift = k.saturating_sub(max_xor);
        let (eff_p, eff_k) = (p - z_shift, k - z_shift);
        for j in 0..xy_bits {
            let mut term = out[8 + j].xor(base[8 + 2 * xy_bits - 1 - j]);
            if eff_p > j && (eff_p - 1 - j) < eff_k {
                term.z ^= 1u8 << (eff_p - 1 - j);
            }
            out[8 + j] = term;
        }
    }
    out
}

/// Bits 12..15 de `SW_64K_R_X` et `SW_64K_Z_X` (1xAA, RB+ 16 pipes / 8 PKRs) selon `log2(bpe)`.
const RX_ZX_NIBBLE3_16P_8PKR: [[BitTerm; 4]; 5] = [
    [BitTerm::new(64, 0, 0), BitTerm::new(0, 64, 0), BitTerm::new(128, 256, 0), BitTerm::new(256, 128, 0)],
    [BitTerm::new(0, 8, 0), BitTerm::new(64, 0, 0), BitTerm::new(128, 128, 0), BitTerm::new(256, 64, 0)],
    [BitTerm::new(8, 0, 0), BitTerm::new(0, 8, 0), BitTerm::new(64, 128, 0), BitTerm::new(128, 64, 0)],
    [BitTerm::new(0, 4, 0), BitTerm::new(8, 0, 0), BitTerm::new(128, 8, 0), BitTerm::new(64, 64, 0)],
    [BitTerm::new(4, 0, 0), BitTerm::new(0, 4, 0), BitTerm::new(64, 8, 0), BitTerm::new(8, 64, 0)],
];

fn build_rx_zx(elem_log2: usize, pipes: PipeConfig, is_z: bool) -> Result<Equation, String> {
    if pipes != PipeConfig::SCARLETT {
        return Err(format!(
            "SW_64K_{}_X ne prend en charge que la configuration RB+ 16 pipes / 8 PKRs",
            if is_z { "Z" } else { "R" }
        ));
    }
    let micro = if is_z {
        parse_micro(MICRO_Z[elem_log2])
    } else if elem_log2 == 0 {
        parse_micro(MICRO_S[0])
    } else {
        parse_micro(MICRO_D[elem_log2])
    };
    let z_bits: [u8; 4] = if is_z && elem_log2 <= 2 { [0, 2, 1, 0] } else { [0, 4, 2, 1] };
    let nibble2 = [
        BitTerm::new(128, 144, z_bits[0]),
        BitTerm::new(16, 16, z_bits[1]),
        BitTerm::new(64, 32, z_bits[2]),
        BitTerm::new(32, 64, z_bits[3]),
    ];
    let mut bits = Vec::with_capacity(16);
    bits.extend_from_slice(&micro);
    bits.extend_from_slice(&nibble2);
    bits.extend_from_slice(&RX_ZX_NIBBLE3_16P_8PKR[elem_log2]);
    Ok(Equation { bits })
}
