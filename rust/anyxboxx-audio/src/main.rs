//! anyxboxx_xma <entrée.xma> <sortie.wav> [--extensible]
//! anyxboxx_xma --info <entrée.xma>
//!
//! Décode un fichier RIFF XMA2 (mono, stéréo ou multiflux 5.1/7.1) en WAV PCM 16 bits avec le
//! décodeur de TomBox (portage Rust de FFmpeg).
use std::process::ExitCode;

const USAGE: &str = "Usage : anyxboxx_xma <entrée.xma> <sortie.wav> [--extensible] | anyxboxx_xma --info <entrée.xma>";

fn run(args: &[String]) -> Result<String, String> {
    if let [flag, input] = args {
        if flag == "--info" {
            let file = std::fs::read(input).map_err(|error| format!("lecture de {input} : {error}"))?;
            let parsed = anyxboxx_audio::parse(&file)?;
            let f = parsed.format;
            let packets = parsed.data.len().div_ceil(anyxboxx_audio::PACKET_BYTES);
            return Ok(format!(
                "{input} : {} canal(aux), {} Hz, {} flux, masque 0x{:X}, bloc {} o, {} échantillons encodés, {packets} paquet(s)",
                f.channels, f.sample_rate, f.streams, f.channel_mask, f.bytes_per_block, f.samples_encoded
            ));
        }
    }
    let (input, output, extensible) = match args {
        [input, output] => (input.as_str(), output.as_str(), false),
        [input, output, flag] if flag == "--extensible" => (input.as_str(), output.as_str(), true),
        _ => return Err(USAGE.into()),
    };
    let file = std::fs::read(input).map_err(|error| format!("lecture de {input} : {error}"))?;
    let parsed = anyxboxx_audio::parse(&file)?;
    let samples = anyxboxx_audio::decode(parsed.format, parsed.data)?;
    let wav = if extensible {
        anyxboxx_audio::wav_extensible(
            parsed.format.channels,
            parsed.format.sample_rate,
            parsed.format.channel_mask,
            &samples,
        )
    } else {
        anyxboxx_audio::wav(parsed.format.channels, parsed.format.sample_rate, &samples)
    };
    std::fs::write(output, wav).map_err(|error| format!("écriture de {output} : {error}"))?;
    let frames = samples.len() / parsed.format.channels.max(1) as usize;
    Ok(format!(
        "{output} : {frames} échantillons par canal, {} canal(aux), {} flux, {} Hz",
        parsed.format.channels, parsed.format.streams, parsed.format.sample_rate
    ))
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
