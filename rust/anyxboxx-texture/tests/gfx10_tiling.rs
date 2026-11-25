//! Tests d'agencement GFX10 / RDNA 2 : 256 o, 4 Kio, 64 Kio, modes XOR (`_T`, `_X`, `R_X`, `Z_X`),
//! queues de mips 4 Kio et 64 Kio, tableaux de textures 2D et réciprocité tuilage / désentrelacement.
use std::collections::HashSet;

use anyxboxx_texture::bc7;
use anyxboxx_texture::gfx10::{Equation, PipeConfig, SurfaceDesc, SurfaceLayout, SwizzleMode};
use anyxboxx_texture::{decode_surface, tiles};

const ALL_TILED_MODES: [SwizzleMode; 14] = [
    SwizzleMode::Sw256BS,
    SwizzleMode::Sw256BD,
    SwizzleMode::Sw4KS,
    SwizzleMode::Sw4KD,
    SwizzleMode::Sw64KS,
    SwizzleMode::Sw64KD,
    SwizzleMode::Sw64KST,
    SwizzleMode::Sw64KDT,
    SwizzleMode::Sw4KSX,
    SwizzleMode::Sw4KDX,
    SwizzleMode::Sw64KZX,
    SwizzleMode::Sw64KSX,
    SwizzleMode::Sw64KDX,
    SwizzleMode::Sw64KRX,
];

#[test]
fn sw_4k_s_matches_legacy_tiles_module_byte_for_byte() {
    let tile: Vec<u8> = (0..256u32)
        .flat_map(|n| [n.to_le_bytes(), [0; 4], [0; 4], [0; 4]].concat())
        .collect();
    let legacy = tiles::linearize_4k_s(&tile, 64, 64).unwrap();
    let generic = tiles::linearize_bc7(&tile, 64, 64, SwizzleMode::Sw4KS, PipeConfig::SCARLETT, 0).unwrap();
    assert_eq!(generic, legacy);

    // Queue de mips 4 Kio : un mip 32×64 texels (8×16 blocs) jusqu'à 1×1 (8 niveaux).
    let mut desc = SurfaceDesc::bc7(32, 64, SwizzleMode::Sw4KS);
    desc.mip_levels = 8;
    let layout = SurfaceLayout::compute(desc).unwrap();
    assert_eq!(layout.first_tail_mip, 0);
    assert_eq!(layout.total_size, 4096);
    for level in 0..8usize {
        let mw = (32usize >> level).max(1);
        let mh = (64usize >> level).max(1);
        let legacy_tail = tiles::linearize_4k_s_tail(&tile, mw, mh, level).unwrap();
        let generic_tail = layout.detile_subresource(&tile, level as u32, 0).unwrap();
        assert_eq!(generic_tail, legacy_tail, "niveau de queue 4K {level}");
    }
}

#[test]
fn sw_64k_s_bit_order_and_legacy_helper() {
    // 1 tuile de 64 Kio = 4 096 blocs BC7 (64 × 64). Bits : y0 y1 x0 x1 y2 x2 y3 x3 y4 x4 y5 x5.
    let tile: Vec<u8> = (0..4096u32)
        .flat_map(|n| [n.to_le_bytes(), [0; 4], [0; 4], [0; 4]].concat())
        .collect();
    let linear = tiles::linearize_64k_s(&tile, 256, 256).unwrap();
    let generic = tiles::linearize_bc7(&tile, 256, 256, SwizzleMode::Sw64KS, PipeConfig::SCARLETT, 0).unwrap();
    assert_eq!(linear, generic);

    let at = |x: usize, y: usize| u32::from_le_bytes(linear[(y * 64 + x) * 16..][..4].try_into().unwrap());
    assert_eq!(at(0, 0), 0);
    assert_eq!(at(0, 1), 1); // y0 -> bit 0
    assert_eq!(at(1, 0), 4); // x0 -> bit 2
    assert_eq!(at(0, 4), 16); // y2 -> bit 4
    assert_eq!(at(4, 0), 32); // x2 -> bit 5
    assert_eq!(at(0, 16), 256); // y4 -> bit 8
    assert_eq!(at(16, 0), 512); // x4 -> bit 9
    assert_eq!(at(0, 32), 1024); // y5 -> bit 10
    assert_eq!(at(32, 0), 2048); // x5 -> bit 11
    assert_eq!(at(63, 63), 4095);

    // Queue de mips 64 Kio : niveau 0 en (32, 0) -> bloc 2048 (0x8000 / 16), niveau 11 en (0, 0) -> bloc 0.
    let tail0 = tiles::linearize_64k_s_tail(&tile, 4, 4, 0).unwrap();
    assert_eq!(u32::from_le_bytes(tail0[..4].try_into().unwrap()), 2048);
    let tail1 = tiles::linearize_64k_s_tail(&tile, 4, 4, 1).unwrap();
    assert_eq!(u32::from_le_bytes(tail1[..4].try_into().unwrap()), 1024);
    let tail11 = tiles::linearize_64k_s_tail(&tile, 4, 4, 11).unwrap();
    assert_eq!(u32::from_le_bytes(tail11[..4].try_into().unwrap()), 0);
    assert!(tiles::linearize_64k_s_tail(&tile, 256, 256, 0).is_err());
    assert!(tiles::linearize_64k_s(&tile[..65_000], 256, 256).is_err());
}

#[test]
fn mesa_rbplus_golden_equations_match_exact_bit_positions() {
    // Vérifie les vecteurs exacts tirés de `gfx10SwizzlePattern.h` (16 pipes / 8 PKRs, 16 bpe).
    // SW_64K_S_X : bits 4..15 = Y0 Y1 X0 X1 (Y2^X5) (X2^Z2^Y5) (Z1^Y3^X4) (Z0^X3^Y4) Y4 X4 Y5 X5
    let eq_sx = Equation::for_mode(SwizzleMode::Sw64KSX, 16, PipeConfig::SCARLETT).unwrap();
    // x = 32 (X5=1) active bit 8 (Y2^X5) et bit 15 (X5) -> (1<<8) | (1<<15) = 0x8100
    assert_eq!(eq_sx.eval_byte_offset(32, 0, 0), 0x8100);
    // y = 32 (Y5=1) active bit 9 (X2^Z2^Y5) et bit 14 (Y5) -> (1<<9) | (1<<14) = 0x4200
    assert_eq!(eq_sx.eval_byte_offset(0, 32, 0), 0x4200);
    // slice = 1 (Z0=1) -> bit 11 (0x0800), slice = 2 (Z1=1) -> bit 10 (0x0400), slice = 4 (Z2=1) -> bit 9 (0x0200)
    assert_eq!(eq_sx.eval_byte_offset(0, 0, 1), 0x0800);
    assert_eq!(eq_sx.eval_byte_offset(0, 0, 2), 0x0400);
    assert_eq!(eq_sx.eval_byte_offset(0, 0, 4), 0x0200);
    assert_eq!(eq_sx.eval_byte_offset(0, 0, 7), 0x0E00);

    // SW_64K_S_T : identique à S_X sur X/Y, mais sans XOR de tranche Z.
    let eq_st = Equation::for_mode(SwizzleMode::Sw64KST, 16, PipeConfig::SCARLETT).unwrap();
    assert_eq!(eq_st.eval_byte_offset(32, 0, 0), 0x8100);
    assert_eq!(eq_st.eval_byte_offset(0, 0, 7), 0x0000);

    // SW_64K_D_X : micro-tuile Display (X0 Y0 X1 Y1) puis mêmes bits XOR 8..15.
    let eq_dx = Equation::for_mode(SwizzleMode::Sw64KDX, 16, PipeConfig::SCARLETT).unwrap();
    assert_eq!(eq_dx.eval_byte_offset(1, 0, 0), 0x0010); // X0 -> bit 4
    assert_eq!(eq_dx.eval_byte_offset(0, 1, 0), 0x0020); // Y0 -> bit 5
    assert_eq!(eq_dx.eval_byte_offset(32, 0, 1), 0x8900);

    // SW_64K_R_X (16 bpe, 16 pipes / 8 PKRs) : bit 12 = X2, bit 13 = Y2, bit 9 = Z2^X4^Y4, bit 11 = Z0^X5^Y6
    let eq_rx = Equation::for_mode(SwizzleMode::Sw64KRX, 16, PipeConfig::SCARLETT).unwrap();
    assert_eq!(eq_rx.eval_byte_offset(4, 0, 0), 0x1000); // X2 -> bit 12
    assert_eq!(eq_rx.eval_byte_offset(0, 4, 0), 0x2000); // Y2 -> bit 13
    assert_eq!(eq_rx.eval_byte_offset(16, 0, 0), 0x0200); // X4 -> bit 9
    assert_eq!(eq_rx.eval_byte_offset(0, 0, 1), 0x0800); // Z0 -> bit 11

    // SW_64K_Z_X (4 bpe, 16 pipes / 8 PKRs) : Z0 -> bit 10, Z1 -> bit 9 (décalé d'un bit pour bpe <= 4)
    let eq_zx4 = Equation::for_mode(SwizzleMode::Sw64KZX, 4, PipeConfig::SCARLETT).unwrap();
    assert_eq!(eq_zx4.eval_byte_offset(0, 0, 1), 0x0400);
    assert_eq!(eq_zx4.eval_byte_offset(0, 0, 2), 0x0200);
}

#[test]
fn every_tiled_mode_and_bpe_is_bijective_within_a_block() {
    for mode in ALL_TILED_MODES {
        for element_bytes in [1u32, 2, 4, 8, 16] {
            let desc = SurfaceDesc {
                width: 256,
                height: 256,
                mip_levels: 1,
                array_size: 4,
                element_bytes,
                block_side: 1,
                swizzle: mode,
                pipes: PipeConfig::SCARLETT,
                pipe_bank_xor: 0xC5,
            };
            let layout = SurfaceLayout::compute(desc).unwrap();
            let block_count = (layout.block_width * layout.block_height) as usize;
            for slice in [0u32, 1, 3] {
                let mut seen = HashSet::with_capacity(block_count);
                for y in 0..layout.block_height {
                    for x in 0..layout.block_width {
                        let off = layout.element_offset(0, slice, x, y).unwrap();
                        let rel = off - (layout.slice_size * u64::from(slice)) as usize;
                        assert_eq!(rel % element_bytes as usize, 0);
                        assert!(rel < mode.block_bytes(), "{mode:?} bpe={element_bytes} rel={rel}");
                        assert!(seen.insert(rel), "collision pour {mode:?} bpe={element_bytes} slice={slice}");
                    }
                }
                assert_eq!(seen.len(), block_count);
            }
        }
    }
}

#[test]
fn mip_tail_64k_all_twelve_levels_are_disjoint_and_round_trip() {
    // 1024×1024 BC7 (256×256 blocs), 11 niveaux : mips 0..=1 standards (256×256, 128×128),
    // mips 2..=10 (64×64 -> 1×1, soit 9 niveaux mais 64×64 > tail_max_w=32 -> mip 2 standard,
    // mips 3..=10 dans la queue 64 Kio !). Testons aussi 128×256 (32×64 blocs, 12 niveaux tous en queue).
    let mut desc = SurfaceDesc::bc7(1024, 1024, SwizzleMode::Sw64KSX);
    desc.mip_levels = 11;
    desc.pipe_bank_xor = 0x4B;
    let layout = SurfaceLayout::compute(desc).unwrap();
    assert_eq!(layout.block_width, 64);
    assert_eq!(layout.block_height, 64);
    assert_eq!(layout.first_tail_mip, 3);
    // Offsets croissants du plus petit mip standard au plus grand :
    // queue = 64 Kio (offset 0), mip 2 (64×64 = 1 bloc) en 64 Kio, mip 1 (128×128 = 4 blocs) en 128 Kio,
    // mip 0 (256×256 = 16 blocs) en 384 Kio.
    assert_eq!(layout.mips[2].offset, 65_536);
    assert_eq!(layout.mips[1].offset, 2 * 65_536);
    assert_eq!(layout.mips[0].offset, 6 * 65_536);
    assert_eq!(layout.slice_size, 22 * 65_536);

    // Vérifie les décalages de la queue 64 Kio sur 12 niveaux (128×256 texels -> 32×64 blocs, 12 niveaux).
    let mut tail_desc = SurfaceDesc::bc7(128, 256, SwizzleMode::Sw64KS);
    tail_desc.mip_levels = 12;
    let tail_layout = SurfaceLayout::compute(tail_desc).unwrap();
    assert_eq!(tail_layout.first_tail_mip, 0);
    assert_eq!(tail_layout.total_size, 65_536);
    let expected_offsets = [
        0x8000, 0x4000, 0x2000, 0x1000, 0x0800, 0x0600, 0x0500, 0x0400, 0x0300, 0x0200, 0x0100, 0x0000,
    ];
    let mut occupied = HashSet::new();
    for (mip, &exp_off) in expected_offsets.iter().enumerate() {
        let m = tail_layout.mips[mip];
        let t = m.tail.expect("dans la queue 64 Kio");
        assert_eq!(t.byte_offset, exp_off, "mip {mip}");
        for y in 0..m.height_elements {
            for x in 0..m.width_elements {
                let off = tail_layout.element_offset(mip as u32, 0, x, y).unwrap();
                assert!(occupied.insert(off), "chevauchement dans la queue 64 Kio au mip {mip} ({x},{y})");
            }
        }
    }

    // Réciprocité complète sur tous les mips d'une chaîne 1024×1024 en SW_64K_S_X avec pipeBankXor.
    let mut tiled = vec![0u8; layout.total_size as usize];
    let mut expected_mips = Vec::new();
    for mip in 0..desc.mip_levels {
        let m = layout.mips[mip as usize];
        let count = (m.width_elements * m.height_elements) as usize;
        let data: Vec<u8> = (0..count)
            .flat_map(|i| {
                let tag = ((mip + 1) << 24) | (i as u32);
                [tag.to_le_bytes(), (!tag).to_le_bytes(), [0xA5; 4], [0x5A; 4]].concat()
            })
            .collect();
        layout.tile_subresource(&data, &mut tiled, mip, 0).unwrap();
        expected_mips.push(data);
    }
    for (mip, expected) in expected_mips.iter().enumerate() {
        let round_trip = layout.detile_subresource(&tiled, mip as u32, 0).unwrap();
        assert_eq!(&round_trip, expected, "échec réciprocité mip {mip}");
    }
}

#[test]
fn texture_arrays_with_xor_and_non_square_mips_round_trip() {
    // Tableau 2D (6 faces cubemap) non carré (300×180 texels, 8 mips) en SW_64K_S_X et SW_64K_R_X.
    for mode in [SwizzleMode::Sw64KSX, SwizzleMode::Sw64KRX, SwizzleMode::Sw4KSX] {
        let mut desc = SurfaceDesc::bc7(300, 180, mode);
        desc.mip_levels = 8;
        desc.array_size = 6;
        desc.pipe_bank_xor = 0x87;
        let layout = SurfaceLayout::compute(desc).unwrap();
        assert_eq!(layout.total_size, layout.slice_size * 6);

        // Vérifie que l'adresse intra-bloc varie bien selon la tranche Z en mode _X.
        let off_s0 = layout.element_offset(0, 0, 5, 3).unwrap();
        let off_s1 = layout.element_offset(0, 1, 5, 3).unwrap() - layout.slice_size as usize;
        assert_ne!(off_s0, off_s1, "{mode:?} doit appliquer le XOR de tranche Z");

        let mut tiled = vec![0u8; layout.total_size as usize];
        for slice in 0..desc.array_size {
            for mip in 0..desc.mip_levels {
                let m = layout.mips[mip as usize];
                let count = (m.width_elements * m.height_elements) as usize;
                let linear: Vec<u8> = (0..count)
                    .flat_map(|i| {
                        let v = (slice << 20) | (mip << 16) | (i as u32);
                        [v.to_le_bytes(), [0x11; 4], [0x22; 4], [0x33; 4]].concat()
                    })
                    .collect();
                layout.tile_subresource(&linear, &mut tiled, mip, slice).unwrap();
            }
        }
        for slice in 0..desc.array_size {
            for mip in 0..desc.mip_levels {
                let m = layout.mips[mip as usize];
                let count = (m.width_elements * m.height_elements) as usize;
                let expected: Vec<u8> = (0..count)
                    .flat_map(|i| {
                        let v = (slice << 20) | (mip << 16) | (i as u32);
                        [v.to_le_bytes(), [0x11; 4], [0x22; 4], [0x33; 4]].concat()
                    })
                    .collect();
                let got = layout.detile_subresource(&tiled, mip, slice).unwrap();
                assert_eq!(got, expected, "{mode:?} slice={slice} mip={mip}");
            }
        }
    }
}

#[test]
fn bc7_decode_round_trip_through_64k_xor_array() {
    // Bloc BC7 mode 1 partition 13 (même bloc que tests/bc7.rs), répété sur 16×16 texels (4×4 blocs)
    // placé dans la tranche 2, mip 1 d'un tableau 32×32 en SW_64K_S_X.
    let block: [u8; 16] = [(1u128, 1), (13, 2), (63, 8), (63, 26), (1, 80), (7, 105)]
        .iter()
        .map(|&(v, o)| v << o)
        .sum::<u128>()
        .to_le_bytes();
    let linear_mip1: Vec<u8> = std::iter::repeat_n(block, 16).flatten().collect();

    let mut desc = SurfaceDesc::bc7(32, 32, SwizzleMode::Sw64KSX);
    desc.mip_levels = 4;
    desc.array_size = 3;
    desc.pipe_bank_xor = 0x45;
    let layout = SurfaceLayout::compute(desc).unwrap();
    let mut tiled = vec![0u8; layout.total_size as usize];
    layout.tile_subresource(&linear_mip1, &mut tiled, 1, 2).unwrap();

    let detiled = layout.detile_subresource(&tiled, 1, 2).unwrap();
    let pixels = decode_surface(&detiled, 16, 16).unwrap();
    let reference = decode_surface(&linear_mip1, 16, 16).unwrap();
    assert_eq!(pixels, reference);
    assert_eq!(bc7::decode(&detiled[..16]).unwrap(), bc7::decode(&block).unwrap());
}

#[test]
fn validates_pipe_configs_and_error_paths() {
    assert!(PipeConfig::new(3, 1).is_err());
    assert!(PipeConfig::new(8, 16).is_err());
    assert!(Equation::for_mode(SwizzleMode::Sw64KRX, 16, PipeConfig::SINGLE_PIPE).is_err());
    assert!(Equation::for_mode(SwizzleMode::Sw64KZX, 16, PipeConfig::new(8, 4).unwrap()).is_err());
    assert!(SwizzleMode::parse_name("invalide").is_err());
    assert_eq!(SwizzleMode::parse_name("SW_64KB_S_X").unwrap(), SwizzleMode::Sw64KSX);
    assert_eq!(SwizzleMode::from_hw(25), Some(SwizzleMode::Sw64KSX));
    assert_eq!(SwizzleMode::from_hw(99), None);

    let layout = SurfaceLayout::compute(SurfaceDesc::bc7(64, 64, SwizzleMode::Sw64KS)).unwrap();
    assert!(layout.element_offset(1, 0, 0, 0).is_err());
    assert!(layout.element_offset(0, 1, 0, 0).is_err());
    assert!(layout.element_offset(0, 0, 16, 0).is_err());
    assert!(layout.detile_subresource(&vec![0u8; 1024], 0, 0).is_err());
}
