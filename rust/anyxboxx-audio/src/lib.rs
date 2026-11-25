//! Son de la Xbox Series X : fichiers RIFF XMA2 (mono, stéréo et multiflux 5.1/7.1) décodés en
//! PCM 16 bits, voix de lecture incrémentale et export WAV / WAVEFORMATEXTENSIBLE.
//!
//! Le décodage s'appuie sur la crate `xma` de TomBox (omonkeyz/xbox-emulator, LGPL-2.1+), un
//! portage en Rust du décodeur WMA Pro / XMA de FFmpeg (`libavcodec/wmaprodec.c`).
pub mod multistream;
pub mod voice;

pub use multistream::{stream_channel_layout, MultiStreamDecoder, END_OF_BLOCK_SKIP};
pub use voice::Xma2Voice;

/// Format WAVE de XMA2 (XMA2WAVEFORMATEX).
pub const XMA2_FORMAT_TAG: u16 = 0x0166;
pub const PACKET_BYTES: usize = 2048;
const PCM_FORMAT_TAG: u16 = 1;
const EXTENSIBLE_FORMAT_TAG: u16 = 0xFFFE;
const BITS_PER_SAMPLE: u16 = 16;
const KSDATAFORMAT_SUBTYPE_PCM: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// Paramètres complets d'un flux XMA2 lus dans le bloc `fmt ` (`XMA2WAVEFORMATEX`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Xma2Format {
    pub channels: u16,
    pub sample_rate: u32,
    pub streams: u16,
    pub channel_mask: u32,
    pub samples_encoded: u32,
    pub bytes_per_block: u32,
    pub play_begin: u32,
    pub play_length: u32,
    pub loop_begin: u32,
    pub loop_length: u32,
    pub loop_count: u8,
    pub encoder_version: u8,
    pub block_count: u16,
}

impl Xma2Format {
    pub const fn simple(channels: u16, sample_rate: u32, streams: u16) -> Self {
        Self {
            channels,
            sample_rate,
            streams,
            channel_mask: 0,
            samples_encoded: 0,
            bytes_per_block: 0,
            play_begin: 0,
            play_length: 0,
            loop_begin: 0,
            loop_length: 0,
            loop_count: 0,
            encoder_version: 4,
            block_count: 0,
        }
    }

    pub fn validate(self) -> Result<(), String> {
        if self.sample_rate == 0 {
            return Err("fréquence d'échantillonnage nulle".into());
        }
        stream_channel_layout(self.channels, self.streams)?;
        if self.channel_mask != 0 && self.channel_mask.count_ones() != u32::from(self.channels) {
            return Err(format!(
                "masque de canaux 0x{:X} incompatible avec {} canal(aux)",
                self.channel_mask, self.channels
            ));
        }
        if self.bytes_per_block != 0 && !(self.bytes_per_block as usize).is_multiple_of(PACKET_BYTES) {
            return Err(format!(
                "BytesPerBlock ({}) n'est pas un multiple de {PACKET_BYTES}",
                self.bytes_per_block
            ));
        }
        if self.samples_encoded > 0
            && u64::from(self.play_begin) + u64::from(self.play_length) > u64::from(self.samples_encoded)
        {
            return Err("PlayBegin + PlayLength dépasse SamplesEncoded".into());
        }
        Ok(())
    }
}

/// Contenu utile d'un fichier RIFF XMA2.
#[derive(Debug)]
pub struct Xma2File<'a> {
    pub format: Xma2Format,
    pub data: &'a [u8],
}

fn u16_at(b: &[u8], offset: usize) -> Result<u16, String> {
    b.get(offset..offset + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or_else(|| "fichier tronqué".into())
}

fn u32_at(b: &[u8], offset: usize) -> Result<u32, String> {
    b.get(offset..offset + 4)
        .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
        .ok_or_else(|| "fichier tronqué".into())
}

fn parse_fmt_body(body: &[u8]) -> Result<Xma2Format, String> {
    let tag = u16_at(body, 0)?;
    if tag != XMA2_FORMAT_TAG {
        return Err(format!("format 0x{tag:04X}, XMA2 (0x0166) attendu"));
    }
    let mut format = Xma2Format::simple(u16_at(body, 2)?, u32_at(body, 4)?, u16_at(body, 18)?);
    if body.len() >= 52 {
        format.channel_mask = u32_at(body, 20)?;
        format.samples_encoded = u32_at(body, 24)?;
        format.bytes_per_block = u32_at(body, 28)?;
        format.play_begin = u32_at(body, 32)?;
        format.play_length = u32_at(body, 36)?;
        format.loop_begin = u32_at(body, 40)?;
        format.loop_length = u32_at(body, 44)?;
        format.loop_count = body[48];
        format.encoder_version = body[49];
        format.block_count = u16_at(body, 50)?;
    }
    format.validate()?;
    Ok(format)
}

/// Lit un fichier RIFF/WAVE dont le bloc `fmt ` est un XMA2WAVEFORMATEX.
pub fn parse(file: &[u8]) -> Result<Xma2File<'_>, String> {
    if file.get(..4) != Some(b"RIFF") || file.get(8..12) != Some(b"WAVE") {
        return Err("pas un fichier RIFF/WAVE".into());
    }
    let (mut format, mut data, mut offset) = (None, None, 12);
    while offset + 8 <= file.len() {
        let size = u32_at(file, offset + 4)? as usize;
        let body = file.get(offset + 8..offset + 8 + size).ok_or("bloc RIFF tronqué")?;
        match &file[offset..offset + 4] {
            b"fmt " => format = Some(parse_fmt_body(body)?),
            b"data" => data = Some(body),
            _ => {}
        }
        offset += 8 + size + (size & 1);
    }
    Ok(Xma2File {
        format: format.ok_or("bloc fmt absent")?,
        data: data.ok_or("bloc data absent")?,
    })
}

/// Décode des paquets XMA2 de 2 Kio (mono, stéréo ou multiflux) en échantillons PCM 16 bits entrelacés.
pub fn decode(format: Xma2Format, data: &[u8]) -> Result<Vec<i16>, String> {
    let mut decoder = MultiStreamDecoder::new(format)?;
    let mut samples = Vec::new();
    decoder.feed(data, &mut samples);
    decoder.flush(&mut samples);
    multistream::trim_to_format(format, &mut samples);
    Ok(samples)
}

/// Fichier WAV PCM 16 bits standard (en-tête 44 octets).
pub fn wav(channels: u16, sample_rate: u32, samples: &[i16]) -> Vec<u8> {
    let data_bytes = (samples.len() * 2) as u32;
    let block_align = channels * BITS_PER_SAMPLE / 8;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&PCM_FORMAT_TAG.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

/// Fichier WAV `WAVEFORMATEXTENSIBLE` (0xFFFE) avec masque de canaux explicite (5.1 / 7.1).
pub fn wav_extensible(channels: u16, sample_rate: u32, channel_mask: u32, samples: &[i16]) -> Vec<u8> {
    let data_bytes = (samples.len() * 2) as u32;
    let block_align = channels * BITS_PER_SAMPLE / 8;
    let mut out = Vec::with_capacity(68 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(60 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&EXTENSIBLE_FORMAT_TAG.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    out.extend_from_slice(&22u16.to_le_bytes()); // cbSize
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes()); // wValidBitsPerSample
    out.extend_from_slice(&channel_mask.to_le_bytes());
    out.extend_from_slice(&KSDATAFORMAT_SUBTYPE_PCM);
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xma2_riff(channels: u16, rate: u32, streams: u16, data: &[u8]) -> Vec<u8> {
        let mut fmt = Vec::new();
        for value in [XMA2_FORMAT_TAG, channels] {
            fmt.extend_from_slice(&value.to_le_bytes());
        }
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&[0; 8]);
        fmt.extend_from_slice(&34u16.to_le_bytes());
        fmt.extend_from_slice(&streams.to_le_bytes());
        fmt.extend_from_slice(&[0; 32]);
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        for (kind, body) in [(b"fmt ", &fmt[..]), (b"data", data)] {
            out.extend_from_slice(kind);
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(body);
        }
        out
    }

    #[test]
    fn parses_the_xma2_header() {
        let file = xma2_riff(2, 48_000, 1, &[0; PACKET_BYTES]);
        let parsed = parse(&file).unwrap();
        assert_eq!(parsed.format.channels, 2);
        assert_eq!(parsed.format.sample_rate, 48_000);
        assert_eq!(parsed.format.streams, 1);
        assert_eq!(parsed.data.len(), PACKET_BYTES);
    }

    #[test]
    fn rejects_other_formats_and_invalid_channel_layouts() {
        let mut pcm = xma2_riff(2, 48_000, 1, &[]);
        pcm[20..22].copy_from_slice(&1u16.to_le_bytes());
        assert!(parse(&pcm).unwrap_err().contains("0x0001"));
        assert!(parse(b"nope").is_err());
        let bad = Xma2Format::simple(6, 48_000, 2);
        assert!(decode(bad, &[]).unwrap_err().contains("impossible"));
    }

    #[test]
    fn garbage_packets_decode_without_crash() {
        let format = Xma2Format::simple(1, 44_100, 1);
        let samples = decode(format, &[0xA5; PACKET_BYTES * 2]).unwrap();
        assert_eq!(samples.len() % format.channels as usize, 0);
    }

    #[test]
    fn writes_a_pcm_wav_header() {
        let file = wav(2, 48_000, &[1, -1, 2, -2]);
        assert_eq!(&file[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(file[40..44].try_into().unwrap()), 8);
        assert_eq!(u16::from_le_bytes([file[22], file[23]]), 2);
        assert_eq!(file.len(), 44 + 8);
    }
}
