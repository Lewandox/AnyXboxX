//! anyxboxx_texture bc7 <blocs.bin> <largeur> <hauteur> <sortie.png> [options]
//!
//! Décode une surface de blocs BC7 (ligne par ligne ou tuilée GFX10 : `--4k-s`, `--64k-s`,
//! `--mode <mode>`, `--xor <val>`, `--pipes <n>`, `--pkrs <n>`, `--mips <n>`, `--mip <i>`,
//! `--array <n>`, `--slice <i>`) et l'écrit en PNG.
use std::process::ExitCode;

use anyxboxx_texture::gfx10::{PipeConfig, SurfaceDesc, SurfaceLayout, SwizzleMode};
use anyxboxx_texture::{decode_surface, png};

const USAGE: &str = "Usage : anyxboxx_texture bc7 <blocs.bin> <largeur> <hauteur> <sortie.png> \
[--4k-s | --64k-s | --mode <mode>] [--xor <val>] [--pipes <n>] [--pkrs <n>] \
[--mips <n>] [--mip <i>] [--array <n>] [--slice <i>]";

fn parse_u32_radix(value: &str, label: &str) -> Result<u32, String> {
    if let Some(hex) = value.strip_prefix("0x").or_else(|| value.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).map_err(|_| format!("{label} invalide : {value}"))
    } else {
        value.parse::<u32>().map_err(|_| format!("{label} invalide : {value}"))
    }
}

struct CliOptions {
    mode: Option<SwizzleMode>,
    pipe_bank_xor: u32,
    pipes: u32,
    pkrs: u32,
    mip_levels: u32,
    mip: u32,
    array_size: u32,
    slice: u32,
}

fn parse_options(options: &[String]) -> Result<CliOptions, String> {
    let mut parsed = CliOptions {
        mode: None,
        pipe_bank_xor: 0,
        pipes: 16,
        pkrs: 8,
        mip_levels: 1,
        mip: 0,
        array_size: 1,
        slice: 0,
    };
    let mut i = 0;
    while i < options.len() {
        let next = |idx: usize, flag: &str| -> Result<&str, String> {
            options
                .get(idx + 1)
                .map(String::as_str)
                .ok_or_else(|| format!("valeur manquante pour {flag}"))
        };
        match options[i].as_str() {
            "--4k-s" => parsed.mode = Some(SwizzleMode::Sw4KS),
            "--64k-s" => parsed.mode = Some(SwizzleMode::Sw64KS),
            "--mode" => {
                parsed.mode = Some(SwizzleMode::parse_name(next(i, "--mode")?)?);
                i += 1;
            }
            "--xor" => {
                parsed.pipe_bank_xor = parse_u32_radix(next(i, "--xor")?, "xor")?;
                i += 1;
            }
            "--pipes" => {
                parsed.pipes = parse_u32_radix(next(i, "--pipes")?, "pipes")?;
                i += 1;
            }
            "--pkrs" => {
                parsed.pkrs = parse_u32_radix(next(i, "--pkrs")?, "pkrs")?;
                i += 1;
            }
            "--mips" => {
                parsed.mip_levels = parse_u32_radix(next(i, "--mips")?, "mips")?;
                i += 1;
            }
            "--mip" => {
                parsed.mip = parse_u32_radix(next(i, "--mip")?, "mip")?;
                i += 1;
            }
            "--array" => {
                parsed.array_size = parse_u32_radix(next(i, "--array")?, "array")?;
                i += 1;
            }
            "--slice" => {
                parsed.slice = parse_u32_radix(next(i, "--slice")?, "slice")?;
                i += 1;
            }
            other => return Err(format!("option inconnue : {other}\n{USAGE}")),
        }
        i += 1;
    }
    Ok(parsed)
}

fn run(args: &[String]) -> Result<String, String> {
    let [command, input, width, height, output, options @ ..] = args else {
        return Err(USAGE.into());
    };
    if command != "bc7" {
        return Err(USAGE.into());
    }
    let width: u32 = width.parse().map_err(|_| "largeur invalide")?;
    let height: u32 = height.parse().map_err(|_| "hauteur invalide")?;
    let opts = parse_options(options)?;
    let raw = std::fs::read(input).map_err(|error| format!("lecture de {input} : {error}"))?;

    let (blocks, out_w, out_h) = match opts.mode {
        None => {
            if opts.mip != 0 || opts.slice != 0 || opts.mip_levels != 1 || opts.array_size != 1 {
                return Err("--mips/--mip/--array/--slice exigent un mode GFX10 (--mode, --4k-s ou --64k-s)".into());
            }
            (raw, width as usize, height as usize)
        }
        Some(mode) => {
            let pipes = PipeConfig::new(opts.pipes, opts.pkrs)?;
            let mut desc = SurfaceDesc::bc7(width, height, mode);
            desc.mip_levels = opts.mip_levels;
            desc.array_size = opts.array_size;
            desc.pipes = pipes;
            desc.pipe_bank_xor = opts.pipe_bank_xor;
            let layout = SurfaceLayout::compute(desc)?;
            let blocks = layout.detile_subresource(&raw, opts.mip, opts.slice)?;
            let mw = (width >> opts.mip).max(1) as usize;
            let mh = (height >> opts.mip).max(1) as usize;
            (blocks, mw, mh)
        }
    };

    let image = decode_surface(&blocks, out_w, out_h)?;
    let file = png::encode(out_w, out_h, &image)?;
    std::fs::write(output, file).map_err(|error| format!("écriture de {output} : {error}"))?;
    Ok(format!("{output} : {out_w}×{out_h}"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(message) => {
            println!("{message}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
