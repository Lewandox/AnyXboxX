//! Agencements GFX10 (4 Kio, 64 Kio, modes XOR) des blocs BC7 de 16 octets.
//!
//! `SW_4K_S` est porté de `tools/bc7_tiles.py` de Xbox2Windows (R0Y00, licence MIT), d'après les
//! tables publiques d'AMD (addrlib, `gfx10SwizzlePattern.h`). Les variantes 64 Kio, XOR (`_T`, `_X`)
//! et tableaux de textures s'appuient sur `crate::gfx10`.
use crate::bc7::BLOCK_BYTES;
use crate::gfx10::{PipeConfig, SurfaceDesc, SurfaceLayout, SwizzleMode};

pub const TILE_BYTES: usize = 4096;
pub const TILE_64K_BYTES: usize = 65_536;
const TILE_BLOCKS: usize = 16;
const TILE_64K_BLOCKS: usize = 64;
const MIP_TAIL_LEVELS: usize = 8;
const MIP_TAIL_64K_LEVELS: usize = 12;

/// Position (colonne, ligne) de chaque niveau de la queue de mips dans la tuile de 16 × 16 blocs.
const TAIL_ORIGINS: [(usize, usize); MIP_TAIL_LEVELS] =
    [(8, 0), (4, 8), (0, 12), (0, 8), (4, 4), (4, 0), (0, 4), (0, 0)];

/// Position (colonne, ligne) de chaque niveau de la queue de mips dans une tuile 64 Kio (64 × 64 blocs).
const TAIL_64K_ORIGINS: [(usize, usize); MIP_TAIL_64K_LEVELS] = [
    (32, 0),
    (0, 32),
    (16, 0),
    (0, 16),
    (8, 0),
    (4, 8),
    (0, 12),
    (0, 8),
    (4, 4),
    (4, 0),
    (0, 4),
    (0, 0),
];

pub fn x_bits(x: usize) -> usize {
    ((x & 3) << 2) | ((x & 4) << 3) | ((x & 8) << 4)
}

pub fn y_bits(y: usize) -> usize {
    (y & 3) | ((y & 4) << 2) | ((y & 8) << 3)
}

pub fn x_bits_64k(x: usize) -> usize {
    x_bits(x) | ((x & 16) << 5) | ((x & 32) << 6)
}

pub fn y_bits_64k(y: usize) -> usize {
    y_bits(y) | ((y & 16) << 4) | ((y & 32) << 5)
}

/// Blocs d'une surface rangée en tuiles SW_4K_S complètes, remis ligne par ligne.
pub fn linearize_4k_s(payload: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("dimensions nulles".into());
    }
    let (columns, rows) = (width.div_ceil(4), height.div_ceil(4));
    let tile_pitch = columns.div_ceil(TILE_BLOCKS);
    if payload.len() != tile_pitch * rows.div_ceil(TILE_BLOCKS) * TILE_BYTES {
        return Err("tuiles de 4 Kio incomplètes pour cette surface".into());
    }
    let mut output = vec![0u8; columns * rows * BLOCK_BYTES];
    for y in 0..rows {
        let tile_row = (y / TILE_BLOCKS) * tile_pitch;
        for x in 0..columns {
            let inside = y_bits(y % TILE_BLOCKS) | x_bits(x % TILE_BLOCKS);
            let source = (tile_row + x / TILE_BLOCKS) * TILE_BYTES + inside * BLOCK_BYTES;
            let target = (y * columns + x) * BLOCK_BYTES;
            output[target..target + BLOCK_BYTES].copy_from_slice(&payload[source..source + BLOCK_BYTES]);
        }
    }
    Ok(output)
}

/// Un niveau de la queue de mips partagée dans une tuile de 4 Kio (niveau 0 = le plus grand).
pub fn linearize_4k_s_tail(tile: &[u8], width: usize, height: usize, level: usize) -> Result<Vec<u8>, String> {
    if level >= MIP_TAIL_LEVELS || width == 0 || height == 0 {
        return Err("niveau ou dimensions invalides pour la queue de mips".into());
    }
    let (columns, rows) = (width.div_ceil(4), height.div_ceil(4));
    if columns > (8 >> level).max(1) || rows > (16 >> level).max(1) {
        return Err("ce niveau dépasse sa place dans la tuile".into());
    }
    if tile.len() != TILE_BYTES {
        return Err("une tuile de queue de mips fait 4 Kio".into());
    }
    let (x0, y0) = TAIL_ORIGINS[level];
    let mut output = Vec::with_capacity(columns * rows * BLOCK_BYTES);
    for row in y0..y0 + rows {
        for column in x0..x0 + columns {
            let start = (y_bits(row) | x_bits(column)) * BLOCK_BYTES;
            output.extend_from_slice(&tile[start..start + BLOCK_BYTES]);
        }
    }
    Ok(output)
}

/// Blocs d'une surface rangée en tuiles SW_64K_S complètes (64 × 64 blocs par tuile), remis ligne par ligne.
pub fn linearize_64k_s(payload: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err("dimensions nulles".into());
    }
    let (columns, rows) = (width.div_ceil(4), height.div_ceil(4));
    let tile_pitch = columns.div_ceil(TILE_64K_BLOCKS);
    if payload.len() != tile_pitch * rows.div_ceil(TILE_64K_BLOCKS) * TILE_64K_BYTES {
        return Err("tuiles de 64 Kio incomplètes pour cette surface".into());
    }
    let mut output = vec![0u8; columns * rows * BLOCK_BYTES];
    for y in 0..rows {
        let tile_row = (y / TILE_64K_BLOCKS) * tile_pitch;
        for x in 0..columns {
            let inside = y_bits_64k(y % TILE_64K_BLOCKS) | x_bits_64k(x % TILE_64K_BLOCKS);
            let source = (tile_row + x / TILE_64K_BLOCKS) * TILE_64K_BYTES + inside * BLOCK_BYTES;
            let target = (y * columns + x) * BLOCK_BYTES;
            output[target..target + BLOCK_BYTES].copy_from_slice(&payload[source..source + BLOCK_BYTES]);
        }
    }
    Ok(output)
}

/// Un niveau de la queue de mips partagée dans une tuile de 64 Kio (niveau 0 = le plus grand, jusqu'à 12 niveaux).
pub fn linearize_64k_s_tail(tile: &[u8], width: usize, height: usize, level: usize) -> Result<Vec<u8>, String> {
    if level >= MIP_TAIL_64K_LEVELS || width == 0 || height == 0 {
        return Err("niveau ou dimensions invalides pour la queue de mips 64 Kio".into());
    }
    let (columns, rows) = (width.div_ceil(4), height.div_ceil(4));
    if columns > (32 >> level).max(1) || rows > (64 >> level).max(1) {
        return Err("ce niveau dépasse sa place dans la tuile de 64 Kio".into());
    }
    if tile.len() != TILE_64K_BYTES {
        return Err("une tuile de queue de mips 64 Kio fait 64 Kio".into());
    }
    let (x0, y0) = TAIL_64K_ORIGINS[level];
    let mut output = Vec::with_capacity(columns * rows * BLOCK_BYTES);
    for row in y0..y0 + rows {
        for column in x0..x0 + columns {
            let start = (y_bits_64k(row) | x_bits_64k(column)) * BLOCK_BYTES;
            output.extend_from_slice(&tile[start..start + BLOCK_BYTES]);
        }
    }
    Ok(output)
}

/// Désentrelace une surface BC7 mono-mip pour n'importe quel mode GFX10 (4 Kio, 64 Kio, `_T`, `_X`).
pub fn linearize_bc7(
    payload: &[u8],
    width: u32,
    height: u32,
    mode: SwizzleMode,
    pipes: PipeConfig,
    pipe_bank_xor: u32,
) -> Result<Vec<u8>, String> {
    let mut desc = SurfaceDesc::bc7(width, height, mode);
    desc.pipes = pipes;
    desc.pipe_bank_xor = pipe_bank_xor;
    let layout = SurfaceLayout::compute(desc)?;
    if payload.len() as u64 != layout.total_size {
        return Err(format!(
            "taille de surface tuilée invalide : {} octets reçus, {} attendus",
            payload.len(),
            layout.total_size
        ));
    }
    layout.detile_subresource(payload, 0, 0)
}
