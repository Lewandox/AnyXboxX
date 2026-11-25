//! Voix de lecture XMA2 en continu (file FIFO d'échantillons PCM 16 bits, activation/désactivation
//! et remplissage incrémental borné) pour le contrôle synthétique de sortie audio (#26).
use std::collections::VecDeque;

use crate::multistream::MultiStreamDecoder;
use crate::Xma2Format;

/// Voix XMA2 incrémentale : décode des paquets dans une file interne puis rend le nombre exact
/// d'échantillons demandé sans jamais écrire hors du tampon ni modifier le tampon si désactivée.
pub struct Xma2Voice {
    decoder: MultiStreamDecoder,
    pending: VecDeque<i16>,
    enabled: bool,
    frames_rendered: u64,
    scratch: Vec<i16>,
}

impl Xma2Voice {
    pub fn new(format: Xma2Format) -> Result<Self, String> {
        let decoder = MultiStreamDecoder::new(format)?;
        Ok(Self {
            decoder,
            pending: VecDeque::new(),
            enabled: true,
            frames_rendered: 0,
            scratch: Vec::new(),
        })
    }

    pub fn format(&self) -> Xma2Format {
        self.decoder.format()
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Nombre de trames PCM complètes actuellement en attente dans la file interne.
    pub fn pending_frames(&self) -> usize {
        self.pending.len() / self.decoder.format().channels.max(1) as usize
    }

    /// Nombre total de trames déjà livrées via `render_pcm` ou `render_pcm_bytes`.
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered
    }

    /// Soumet des paquets XMA2 (multiples de 2 Kio) et met en file les trames PCM décodées.
    pub fn submit_packets(&mut self, data: &[u8]) -> usize {
        self.scratch.clear();
        let frames = self.decoder.feed(data, &mut self.scratch);
        self.pending.extend(self.scratch.iter().copied());
        frames
    }

    /// Injecte directement des échantillons PCM dans la file (utile pour les tests de bordure).
    pub fn enqueue_pcm(&mut self, samples: &[i16]) -> Result<usize, String> {
        let ch = self.decoder.format().channels.max(1) as usize;
        if !samples.len().is_multiple_of(ch) {
            return Err(format!("{} échantillons non multiple de {ch} canal(aux)", samples.len()));
        }
        self.pending.extend(samples.iter().copied());
        Ok(samples.len() / ch)
    }

    /// Rend jusqu'à `out.len() / channels` trames dans `out`.
    /// Si la voix est désactivée, renvoie `0` et laisse `out` strictement inchangé.
    pub fn render_pcm(&mut self, out: &mut [i16]) -> usize {
        if !self.enabled {
            return 0;
        }
        let ch = self.decoder.format().channels.max(1) as usize;
        let requested = out.len() / ch;
        let count = requested.min(self.pending_frames());
        for sample in &mut out[..count * ch] {
            *sample = self.pending.pop_front().unwrap_or(0);
        }
        self.frames_rendered += count as u64;
        count
    }

    /// Rend des échantillons PCM 16 bits petit-boutiste directement dans un tampon d'octets.
    /// Les octets au-delà des trames rendues (et les octets de garde adjacents) restent intacts.
    pub fn render_pcm_bytes(&mut self, out_bytes: &mut [u8]) -> usize {
        if !self.enabled {
            return 0;
        }
        let ch = self.decoder.format().channels.max(1) as usize;
        let frame_bytes = ch * 2;
        let requested = out_bytes.len() / frame_bytes;
        let count = requested.min(self.pending_frames());
        for frame in 0..count {
            let base = frame * frame_bytes;
            for c in 0..ch {
                let sample = self.pending.pop_front().unwrap_or(0);
                out_bytes[base + c * 2..base + c * 2 + 2].copy_from_slice(&sample.to_le_bytes());
            }
        }
        self.frames_rendered += count as u64;
        count
    }
}
