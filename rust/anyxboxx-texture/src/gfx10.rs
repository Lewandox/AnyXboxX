//! Agencement de surfaces 2D et tableaux de textures 2D GFX10 / RDNA 2 :
//! dimensions de blocs, queues de mips (4 Kio et 64 Kio), tranches de tableau,
//! adressage par élément et copies bidirectionnelles tuilé ↔ linéaire.
//!
//! Porté des sources publiques MIT d'AMD addrlib dans Mesa (`src/amd/addrlib` :
//! `gfx10addrlib.cpp` et `addrlib2.cpp`).
//!
//! Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
//! SPDX-License-Identifier: MIT

pub use crate::gfx10_eq::{BitTerm, Equation, PipeConfig, SwizzleMode};

const MAX_EXTENT: u32 = 1 << 14;
const MAX_MIPS: u32 = 16;

/// Dimensions `(largeur, hauteur)` d'une micro-tuile de 256 octets selon `log2(element_bytes)`.
const BLOCK_256_2D: [(u32, u32); 5] = [(16, 16), (16, 8), (8, 8), (8, 4), (4, 4)];

/// Description d'une surface 2D (ou d'un tableau de textures 2D) GFX10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceDesc {
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub array_size: u32,
    /// Octets par élément : 1, 2, 4, 8 ou 16 (16 pour BC7 / BC3 / BC5 / BC6H).
    pub element_bytes: u32,
    /// Texels par côté d'élément : 1 (non compressé) ou 4 (blocs BCn).
    pub block_side: u32,
    pub swizzle: SwizzleMode,
    pub pipes: PipeConfig,
    pub pipe_bank_xor: u32,
}

impl SurfaceDesc {
    /// Raccourci pour une surface BC7 (blocs 4×4 de 16 octets) en configuration Scarlett.
    pub fn bc7(width: u32, height: u32, swizzle: SwizzleMode) -> Self {
        Self {
            width,
            height,
            mip_levels: 1,
            array_size: 1,
            element_bytes: 16,
            block_side: 4,
            swizzle,
            pipes: PipeConfig::SCARLETT,
            pipe_bank_xor: 0,
        }
    }
}

/// Position d'un niveau dans la tuile partagée de la queue de mips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipTail {
    /// Niveau dans la queue (`0` = premier niveau entré dans la queue).
    pub level: u32,
    /// Décalage en octets de ce niveau dans la tuile de queue (`mipTailOffset`).
    pub byte_offset: u32,
    /// Origine X en éléments dans la tuile de queue (`mipTailCoordX`).
    pub x: u32,
    /// Origine Y en éléments dans la tuile de queue (`mipTailCoordY`).
    pub y: u32,
}

/// Placement calculé d'un niveau de mip dans une tranche du tableau.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MipLayout {
    pub width_elements: u32,
    pub height_elements: u32,
    pub pitch_elements: u32,
    pub rows_elements: u32,
    /// Décalage en octets du bloc de ce mip dans la tranche (`0` si dans la queue).
    pub offset: u64,
    pub tail: Option<MipTail>,
}

/// Agencement complet d'une surface 2D ou d'un tableau de textures 2D GFX10.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurfaceLayout {
    pub desc: SurfaceDesc,
    pub block_width: u32,
    pub block_height: u32,
    pub alignment: u64,
    pub slice_size: u64,
    pub total_size: u64,
    pub first_tail_mip: u32,
    pub mips: Vec<MipLayout>,
    equation: Equation,
    effective_xor: usize,
}

fn data_dim(texels: u32, mip: u32, block_side: u32) -> u32 {
    (texels >> mip).max(1).div_ceil(block_side)
}

fn padded_dim(texels: u32, mip: u32, block_side: u32) -> u32 {
    texels.max(1).div_ceil(block_side).div_ceil(1 << mip)
}

fn max_mips_in_tail(block_log2: u32) -> u32 {
    if block_log2 <= 11 { 1 + (1u32 << (block_log2 - 9)) } else { block_log2 - 4 }
}

fn mip_tail_coords(mip_offset: u32, elem_log2: usize) -> (u32, u32) {
    let mip_x = ((mip_offset >> 9) & 1)
        | ((mip_offset >> 10) & 2)
        | ((mip_offset >> 11) & 4)
        | ((mip_offset >> 12) & 8)
        | ((mip_offset >> 13) & 16)
        | ((mip_offset >> 14) & 32);
    let mip_y = ((mip_offset >> 8) & 1)
        | ((mip_offset >> 9) & 2)
        | ((mip_offset >> 10) & 4)
        | ((mip_offset >> 11) & 8)
        | ((mip_offset >> 12) & 16)
        | ((mip_offset >> 13) & 32);
    let (mw, mh) = BLOCK_256_2D[elem_log2];
    (mip_x * mw, mip_y * mh)
}

impl SurfaceLayout {
    pub fn compute(desc: SurfaceDesc) -> Result<Self, String> {
        if desc.width == 0 || desc.height == 0 || desc.width > MAX_EXTENT || desc.height > MAX_EXTENT {
            return Err("dimensions hors limites (1..=16384)".into());
        }
        if desc.mip_levels == 0 || desc.mip_levels > MAX_MIPS {
            return Err("nombre de mips invalide (1..=16)".into());
        }
        if desc.array_size == 0 || desc.array_size > MAX_EXTENT {
            return Err("taille de tableau invalide (1..=16384)".into());
        }
        if !matches!(desc.block_side, 1 | 4) {
            return Err("taille de bloc texel invalide (1 ou 4)".into());
        }
        let equation = Equation::for_mode(desc.swizzle, desc.element_bytes, desc.pipes)?;
        if matches!(desc.swizzle, SwizzleMode::Linear) {
            return Ok(Self::compute_linear(desc, equation));
        }
        Ok(Self::compute_tiled(desc, equation))
    }

    fn compute_linear(desc: SurfaceDesc, equation: Equation) -> Self {
        let align = 256 / desc.element_bytes;
        let bpe = u64::from(desc.element_bytes);
        let mut mips = Vec::with_capacity(desc.mip_levels as usize);
        let mut offset = 0u64;
        for i in (0..desc.mip_levels).rev() {
            let pitch = padded_dim(desc.width, i, desc.block_side).next_multiple_of(align);
            let rows = padded_dim(desc.height, i, desc.block_side);
            mips.push(MipLayout {
                width_elements: data_dim(desc.width, i, desc.block_side),
                height_elements: data_dim(desc.height, i, desc.block_side),
                pitch_elements: pitch,
                rows_elements: rows,
                offset,
                tail: None,
            });
            offset += u64::from(pitch) * u64::from(rows) * bpe;
        }
        mips.reverse();
        Self {
            desc,
            block_width: align,
            block_height: 1,
            alignment: 256,
            slice_size: offset,
            total_size: offset * u64::from(desc.array_size),
            first_tail_mip: desc.mip_levels,
            mips,
            equation,
            effective_xor: 0,
        }
    }

    fn compute_tiled(desc: SurfaceDesc, equation: Equation) -> Self {
        let block_log2 = desc.swizzle.block_log2();
        let elem_log2 = desc.element_bytes.trailing_zeros();
        let bpe = u64::from(desc.element_bytes);
        let block_bytes = 1u64 << block_log2;
        let elements_log2 = block_log2 - elem_log2;
        let log2_w = elements_log2.div_ceil(2);
        let (block_width, block_height) = (1u32 << log2_w, 1u32 << (elements_log2 - log2_w));

        let mut mips = vec![
            MipLayout {
                width_elements: 0,
                height_elements: 0,
                pitch_elements: 0,
                rows_elements: 0,
                offset: 0,
                tail: None,
            };
            desc.mip_levels as usize
        ];

        let mut first_tail_mip = desc.mip_levels;
        let mut slice_size = 0u64;
        let mut mip_sizes = vec![0u64; desc.mip_levels as usize];
        let (tail_max_w, tail_max_h) = (block_width >> 1, block_height);
        let (fixed_tail_w, fixed_tail_h) = if desc.swizzle.is_z_order() && elem_log2 <= 1 {
            let (ew, eh) = BLOCK_256_2D[elem_log2 as usize];
            let (w4, h4) = BLOCK_256_2D[2];
            (tail_max_w / (ew / w4), tail_max_h / (eh / h4))
        } else {
            (tail_max_w, tail_max_h)
        };
        let has_tail = desc.mip_levels > 1 && block_log2 >= 12;
        let max_tail = if has_tail { max_mips_in_tail(block_log2) } else { 0 };

        for i in 0..desc.mip_levels {
            let w_pad = padded_dim(desc.width, i, desc.block_side);
            let h_pad = padded_dim(desc.height, i, desc.block_side);
            if has_tail && w_pad <= fixed_tail_w && h_pad <= fixed_tail_h && (desc.mip_levels - i) <= max_tail {
                first_tail_mip = i;
                slice_size += block_bytes;
                break;
            }
            let pitch = w_pad.next_multiple_of(block_width);
            let rows = h_pad.next_multiple_of(block_height);
            let size = u64::from(pitch) * u64::from(rows) * bpe;
            mip_sizes[i as usize] = size;
            slice_size += size;
            mips[i as usize] = MipLayout {
                width_elements: data_dim(desc.width, i, desc.block_side),
                height_elements: data_dim(desc.height, i, desc.block_side),
                pitch_elements: pitch,
                rows_elements: rows,
                offset: 0,
                tail: None,
            };
        }

        let mut offset = if first_tail_mip < desc.mip_levels { block_bytes } else { 0 };
        for i in (0..first_tail_mip).rev() {
            mips[i as usize].offset = offset;
            offset += mip_sizes[i as usize];
        }

        if first_tail_mip < desc.mip_levels {
            let (mw, mh) = BLOCK_256_2D[elem_log2 as usize];
            let (mut pitch, mut rows) = (tail_max_w, tail_max_h);
            for i in first_tail_mip..desc.mip_levels {
                let level = i - first_tail_mip;
                let m = max_tail - 1 - level;
                let byte_offset = if m > 6 { 16u32 << m } else { m << 8 };
                let (x, y) = mip_tail_coords(byte_offset, elem_log2 as usize);
                mips[i as usize] = MipLayout {
                    width_elements: data_dim(desc.width, i, desc.block_side),
                    height_elements: data_dim(desc.height, i, desc.block_side),
                    pitch_elements: pitch,
                    rows_elements: rows,
                    offset: 0,
                    tail: Some(MipTail { level, byte_offset, x, y }),
                };
                pitch = (pitch >> 1).max(mw);
                rows = (rows >> 1).max(mh);
            }
        }

        let effective_xor = desc.pipes.effective_pipe_bank_xor(desc.swizzle, desc.pipe_bank_xor) as usize;
        Self {
            desc,
            block_width,
            block_height,
            alignment: block_bytes,
            slice_size,
            total_size: slice_size * u64::from(desc.array_size),
            first_tail_mip,
            mips,
            equation,
            effective_xor,
        }
    }

    /// Décalage en octets de l'élément `(x, y)` pour `(mip, slice)` dans le tampon tuilé.
    pub fn element_offset(&self, mip: u32, slice: u32, x: u32, y: u32) -> Result<usize, String> {
        let m = self
            .mips
            .get(mip as usize)
            .ok_or_else(|| format!("mip {mip} hors limites (0..{})", self.mips.len()))?;
        if slice >= self.desc.array_size {
            return Err(format!("tranche {slice} hors limites (0..{})", self.desc.array_size));
        }
        if x >= m.width_elements || y >= m.height_elements {
            return Err(format!(
                "coordonnée ({x}, {y}) hors limites pour le mip {mip} ({}×{})",
                m.width_elements, m.height_elements
            ));
        }
        let slice_base = self.slice_size * u64::from(slice);
        if matches!(self.desc.swizzle, SwizzleMode::Linear) {
            let off = slice_base
                + m.offset
                + (u64::from(y) * u64::from(m.pitch_elements) + u64::from(x))
                    * u64::from(self.desc.element_bytes);
            return Ok(off as usize);
        }
        let (ex, ey, block_index) = match m.tail {
            Some(tail) => (x + tail.x, y + tail.y, 0u64),
            None => {
                let pitch_blocks = u64::from(m.pitch_elements / self.block_width);
                (x, y, u64::from(y / self.block_height) * pitch_blocks + u64::from(x / self.block_width))
            }
        };
        let blk_offset = self.equation.eval_byte_offset(ex, ey, slice) ^ self.effective_xor;
        let addr = slice_base + m.offset + (block_index << self.desc.swizzle.block_log2()) + blk_offset as u64;
        Ok(addr as usize)
    }

    /// Désentrelace une sous-ressource `(mip, slice)` vers un tampon linéaire contigu.
    pub fn detile_subresource(&self, tiled: &[u8], mip: u32, slice: u32) -> Result<Vec<u8>, String> {
        let (m, w, h, bpe) = self.validate_subresource(tiled.len(), mip, slice)?;
        let mut linear = vec![0u8; w * h * bpe];
        if matches!(self.desc.swizzle, SwizzleMode::Linear) {
            let base = (self.slice_size * u64::from(slice) + m.offset) as usize;
            let row_stride = m.pitch_elements as usize * bpe;
            let row_bytes = w * bpe;
            for y in 0..h {
                let src = base + y * row_stride;
                let dst = y * row_bytes;
                linear[dst..dst + row_bytes].copy_from_slice(&tiled[src..src + row_bytes]);
            }
            return Ok(linear);
        }
        self.copy_tiled_subresource(m, slice, w, h, bpe, |tiled_off, linear_off| {
            linear[linear_off..linear_off + bpe].copy_from_slice(&tiled[tiled_off..tiled_off + bpe]);
        });
        Ok(linear)
    }

    /// Range une sous-ressource linéaire `(mip, slice)` dans le tampon tuilé.
    pub fn tile_subresource(&self, linear: &[u8], tiled: &mut [u8], mip: u32, slice: u32) -> Result<(), String> {
        let (m, w, h, bpe) = self.validate_subresource(tiled.len(), mip, slice)?;
        let expected_linear = w * h * bpe;
        if linear.len() != expected_linear {
            return Err(format!(
                "tampon linéaire invalide : {} octets reçus, {expected_linear} attendus",
                linear.len()
            ));
        }
        if matches!(self.desc.swizzle, SwizzleMode::Linear) {
            let base = (self.slice_size * u64::from(slice) + m.offset) as usize;
            let row_stride = m.pitch_elements as usize * bpe;
            let row_bytes = w * bpe;
            for y in 0..h {
                let dst = base + y * row_stride;
                let src = y * row_bytes;
                tiled[dst..dst + row_bytes].copy_from_slice(&linear[src..src + row_bytes]);
            }
            return Ok(());
        }
        self.copy_tiled_subresource(m, slice, w, h, bpe, |tiled_off, linear_off| {
            tiled[tiled_off..tiled_off + bpe].copy_from_slice(&linear[linear_off..linear_off + bpe]);
        });
        Ok(())
    }

    fn validate_subresource(&self, tiled_len: usize, mip: u32, slice: u32) -> Result<(MipLayout, usize, usize, usize), String> {
        if (tiled_len as u64) < self.total_size {
            return Err(format!(
                "tampon tuilé trop court : {tiled_len} octets reçus, {} attendus",
                self.total_size
            ));
        }
        let m = *self.mips.get(mip as usize).ok_or_else(|| format!("mip {mip} hors limites"))?;
        if slice >= self.desc.array_size {
            return Err(format!("tranche {slice} hors limites"));
        }
        Ok((m, m.width_elements as usize, m.height_elements as usize, self.desc.element_bytes as usize))
    }

    /// Parcourt tous les éléments `(x, y)` d'une sous-ressource tuilée en factorisant les termes X
    /// (précalculés une fois par colonne) et les termes Y/Z (une fois par ligne).
    fn copy_tiled_subresource(
        &self,
        m: MipLayout,
        slice: u32,
        w: usize,
        h: usize,
        bpe: usize,
        mut copy_elem: impl FnMut(usize, usize),
    ) {
        let base = (self.slice_size * u64::from(slice) + m.offset) as usize;
        let block_log2 = self.desc.swizzle.block_log2();
        let z_xor = self.equation.eval_z(slice) ^ self.effective_xor;
        let (tail_x, tail_y, in_tail) = match m.tail {
            Some(t) => (t.x, t.y, true),
            None => (0, 0, false),
        };
        let pitch_blocks = if in_tail { 0 } else { (m.pitch_elements / self.block_width) as usize };

        // Pour les très petits mips (ex. queue de mips), évite l'allocation de la table X.
        if w * h <= 64 {
            for y in 0..m.height_elements {
                let y_blk = if in_tail { 0 } else { ((y / self.block_height) as usize * pitch_blocks) << block_log2 };
                let y_swiz = self.equation.eval_y(y + tail_y) ^ z_xor;
                let row_off = y as usize * w * bpe;
                for x in 0..m.width_elements {
                    let x_blk = if in_tail { 0 } else { ((x / self.block_width) as usize) << block_log2 };
                    let tiled_off = base + y_blk + x_blk + (self.equation.eval_x(x + tail_x) ^ y_swiz);
                    copy_elem(tiled_off, row_off + x as usize * bpe);
                }
            }
            return;
        }

        let x_table: Vec<(usize, usize)> = (0..m.width_elements)
            .map(|x| {
                let x_blk = if in_tail { 0 } else { ((x / self.block_width) as usize) << block_log2 };
                (x_blk, self.equation.eval_x(x + tail_x))
            })
            .collect();

        for y in 0..m.height_elements {
            let y_blk = if in_tail { 0 } else { ((y / self.block_height) as usize * pitch_blocks) << block_log2 };
            let y_swiz = self.equation.eval_y(y + tail_y) ^ z_xor;
            let row_base = base + y_blk;
            let mut linear_off = y as usize * w * bpe;
            for &(x_blk, x_swiz) in &x_table {
                copy_elem(row_base + x_blk + (x_swiz ^ y_swiz), linear_off);
                linear_off += bpe;
            }
        }
    }
}
