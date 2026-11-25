//! Décodage XMA2 multiflux (1 à 4 flux, 1 à 8 canaux) avec routage par `packet_skip_count`
//! et réinitialisation aux frontières de blocs XMA2 (`BytesPerBlock`).
//!
//! Suit le format public XMA2WAVEFORMATEX et l'algorithme de routage de paquets XMA2 de FFmpeg
//! (`libavcodec/wmaprodec.c`, LGPL-2.1+) en s'appuyant sur la crate `xma` de TomBox.
use std::collections::VecDeque;

use crate::{Xma2Format, PACKET_BYTES};

/// Sentinelle XMA2 dans l'octet `packet_skip_count` indiquant la fin d'un flux dans le bloc courant.
pub const END_OF_BLOCK_SKIP: u32 = 0xFF;

/// Répartition des canaux par flux XMA2 (1 ou 2 canaux par flux, ex. 6 canaux / 3 flux → `[2, 2, 2]`).
pub fn stream_channel_layout(channels: u16, streams: u16) -> Result<Vec<usize>, String> {
    if channels == 0 || streams == 0 {
        return Err("nombre de canaux ou de flux nul".into());
    }
    if channels < streams || channels > streams.saturating_mul(2) {
        return Err(format!(
            "répartition impossible : {channels} canal(aux) pour {streams} flux XMA2"
        ));
    }
    let mut layout = Vec::with_capacity(streams as usize);
    let mut remaining = channels as usize;
    for s in 0..streams as usize {
        let streams_left = streams as usize - s;
        let ch = if remaining > streams_left { 2 } else { 1 };
        layout.push(ch);
        remaining -= ch;
    }
    Ok(layout)
}

struct StreamState {
    decoder: xma::Xma2Stream,
    fifo: VecDeque<i16>,
    channels: usize,
    skip: u32,
    done_in_block: bool,
}

/// Décodeur XMA2 multiflux qui achemine les paquets de 2 Kio vers chaque flux et entrelace le PCM.
pub struct MultiStreamDecoder {
    format: Xma2Format,
    streams: Vec<StreamState>,
    current: usize,
    packets_per_block: usize,
    packets_in_block: usize,
    packets_decoded: u64,
    scratch: Vec<i16>,
}

impl MultiStreamDecoder {
    pub fn new(format: Xma2Format) -> Result<Self, String> {
        format.validate()?;
        let layout = stream_channel_layout(format.channels, format.streams)?;
        let mut streams = Vec::with_capacity(layout.len());
        for ch in layout {
            let decoder = xma::Xma2Stream::new(format.sample_rate, ch)
                .map_err(|error| format!("configuration XMA2 refusée : {error:?}"))?;
            streams.push(StreamState {
                decoder,
                fifo: VecDeque::new(),
                channels: ch,
                skip: 0,
                done_in_block: false,
            });
        }
        let packets_per_block = if format.bytes_per_block > 0 {
            format.bytes_per_block as usize / PACKET_BYTES
        } else {
            0
        };
        Ok(Self {
            format,
            streams,
            current: 0,
            packets_per_block,
            packets_in_block: 0,
            packets_decoded: 0,
            scratch: Vec::new(),
        })
    }

    pub fn format(&self) -> Xma2Format {
        self.format
    }

    /// Indice du flux qui recevra le prochain paquet XMA2 de 2 Kio.
    pub fn next_stream_index(&self) -> usize {
        self.current
    }

    pub fn packets_decoded(&self) -> u64 {
        self.packets_decoded
    }

    fn reset_block_routing(&mut self) {
        self.current = 0;
        self.packets_in_block = 0;
        for st in &mut self.streams {
            st.skip = 0;
            st.done_in_block = false;
        }
    }

    /// Décode un paquet XMA2 (jusqu'à 2 Kio) et met à jour le routage multiflux.
    pub fn feed_packet(&mut self, packet: &[u8]) {
        if packet.is_empty() {
            return;
        }
        if self.streams.iter().any(|s| !s.done_in_block) {
            let idx = self.current;
            self.scratch.clear();
            self.streams[idx].decoder.decode_packet(packet, &mut self.scratch);
            self.streams[idx].fifo.extend(self.scratch.iter().copied());
            self.packets_decoded += 1;

            let skip = xma::packet_skip_count(packet);
            if skip == END_OF_BLOCK_SKIP && self.streams.len() > 1 {
                self.streams[idx].done_in_block = true;
                self.streams[idx].skip = u32::MAX;
            } else {
                self.streams[idx].skip = skip;
            }

            if self.streams[idx].done_in_block || self.streams[idx].skip != 0 {
                if let Some((best, _)) = self
                    .streams
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| !s.done_in_block)
                    .min_by_key(|(_, s)| s.skip)
                {
                    self.current = best;
                }
            }
            for st in &mut self.streams {
                if !st.done_in_block {
                    st.skip = st.skip.saturating_sub(1);
                }
            }
        }
        if self.packets_per_block > 0 {
            self.packets_in_block += 1;
            if self.packets_in_block >= self.packets_per_block {
                self.reset_block_routing();
            }
        }
    }

    /// Soumet des paquets XMA2 consécutifs et vide les trames complètes entrelacées dans `out`.
    pub fn feed(&mut self, data: &[u8], out: &mut Vec<i16>) -> usize {
        for packet in data.chunks(PACKET_BYTES) {
            self.feed_packet(packet);
        }
        self.drain(out)
    }

    /// Vide toutes les trames prêtes dans tous les flux (entrelacées canal par canal).
    pub fn drain(&mut self, out: &mut Vec<i16>) -> usize {
        let frames = self
            .streams
            .iter()
            .map(|s| s.fifo.len() / s.channels)
            .min()
            .unwrap_or(0);
        out.reserve(frames * self.format.channels as usize);
        for _ in 0..frames {
            for st in &mut self.streams {
                for _ in 0..st.channels {
                    out.push(st.fifo.pop_front().unwrap_or(0));
                }
            }
        }
        frames
    }

    /// En fin de fichier, vide les trames restantes en complétant par du silence les flux plus courts.
    pub fn flush(&mut self, out: &mut Vec<i16>) -> usize {
        let frames = self
            .streams
            .iter()
            .map(|s| s.fifo.len() / s.channels)
            .max()
            .unwrap_or(0);
        out.reserve(frames * self.format.channels as usize);
        for _ in 0..frames {
            for st in &mut self.streams {
                for _ in 0..st.channels {
                    out.push(st.fifo.pop_front().unwrap_or(0));
                }
            }
        }
        frames
    }
}

/// Applique `SamplesEncoded` et la fenêtre `PlayBegin`/`PlayLength` si l'en-tête les renseigne.
pub fn trim_to_format(format: Xma2Format, samples: &mut Vec<i16>) {
    let ch = format.channels.max(1) as usize;
    if format.samples_encoded > 0 {
        let max_samples = (format.samples_encoded as usize).saturating_mul(ch);
        if samples.len() > max_samples {
            samples.truncate(max_samples);
        }
    }
    if format.play_begin > 0 || format.play_length > 0 {
        let total_frames = samples.len() / ch;
        let begin = (format.play_begin as usize).min(total_frames);
        let len = if format.play_length > 0 {
            (format.play_length as usize).min(total_frames - begin)
        } else {
            total_frames - begin
        };
        let start = begin * ch;
        let end = (begin + len) * ch;
        samples.copy_within(start..end, 0);
        samples.truncate(end - start);
    }
}
