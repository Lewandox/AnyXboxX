//! Tests d'intégration XMA2 multiflux, en-tête XMA2WAVEFORMATEX complet, export WAVEFORMATEXTENSIBLE
//! et contrôle synthétique de voix incrémentale (2 048 / 512 échantillons, 1 536 en attente,
//! 256 octets de garde intacts et voix désactivée, issue #26).
use anyxboxx_audio::{
    decode, parse, stream_channel_layout, wav_extensible, MultiStreamDecoder, Xma2Format, Xma2Voice,
    END_OF_BLOCK_SKIP, PACKET_BYTES, XMA2_FORMAT_TAG,
};
use xma::bits::BitWriter;

/// Construit une trame XMA2 synthétique valide de 512 échantillons (1 ou 2 canaux).
fn write_valid_frame(w: &mut BitWriter, channels: usize, more_frames: bool) {
    let bit_len = if channels == 1 { 31 } else { 36 };
    w.put(15, bit_len);
    w.put(1, 1);
    w.put(1, 0);
    if channels == 2 {
        w.put(1, 0);
    }
    w.put(8, 0);
    w.put(1, 0);
    w.put(1, 0);
    w.put(1, 0);
    if channels == 2 {
        w.put(1, 0);
        w.put(1, 1);
        w.put(1, 0);
    }
    for _ in 0..channels {
        w.put(1, 0);
    }
    w.put(1, 0);
    w.put(1, u32::from(more_frames));
}

/// Construit un paquet XMA2 de 2 Kio contenant `frames` trames valides et l'octet `skip`.
fn make_xma2_packet(channels: usize, frames: u32, skip: u32) -> Vec<u8> {
    let mut w = BitWriter::default();
    w.put(6, frames);
    w.put(15, 0);
    w.put(3, 1);
    w.put(8, skip);
    for i in 0..frames {
        write_valid_frame(&mut w, channels, i + 1 < frames);
    }
    let mut packet = w.buf;
    packet.resize(PACKET_BYTES, 0);
    packet
}

/// Construit un fichier RIFF/WAVE XMA2 complet avec en-tête `XMA2WAVEFORMATEX` de 52 octets.
fn build_xma2_riff(f: Xma2Format, data: &[u8]) -> Vec<u8> {
    let mut fmt = Vec::with_capacity(52);
    let block_align = f.channels * 2;
    let avg_bytes = f.sample_rate * u32::from(block_align);
    fmt.extend_from_slice(&XMA2_FORMAT_TAG.to_le_bytes());
    fmt.extend_from_slice(&f.channels.to_le_bytes());
    fmt.extend_from_slice(&f.sample_rate.to_le_bytes());
    fmt.extend_from_slice(&avg_bytes.to_le_bytes());
    fmt.extend_from_slice(&block_align.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());
    fmt.extend_from_slice(&34u16.to_le_bytes());
    fmt.extend_from_slice(&f.streams.to_le_bytes());
    fmt.extend_from_slice(&f.channel_mask.to_le_bytes());
    fmt.extend_from_slice(&f.samples_encoded.to_le_bytes());
    fmt.extend_from_slice(&f.bytes_per_block.to_le_bytes());
    fmt.extend_from_slice(&f.play_begin.to_le_bytes());
    fmt.extend_from_slice(&f.play_length.to_le_bytes());
    fmt.extend_from_slice(&f.loop_begin.to_le_bytes());
    fmt.extend_from_slice(&f.loop_length.to_le_bytes());
    fmt.push(f.loop_count);
    fmt.push(f.encoder_version);
    fmt.extend_from_slice(&f.block_count.to_le_bytes());
    assert_eq!(fmt.len(), 52);

    let body_len = 4 + 8 + fmt.len() + 8 + data.len();
    let mut out = Vec::with_capacity(8 + body_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(body_len as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    out.extend_from_slice(&fmt);
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(data);
    out
}

#[test]
fn parses_full_xma2waveformatex_and_rejects_invalid_headers() {
    let expected = Xma2Format {
        channels: 6,
        sample_rate: 48_000,
        streams: 3,
        channel_mask: 0x3F, // 5.1 : FL FR FC LFE BL BR
        samples_encoded: 4096,
        bytes_per_block: 65_536,
        play_begin: 128,
        play_length: 2048,
        loop_begin: 256,
        loop_length: 1024,
        loop_count: 2,
        encoder_version: 4,
        block_count: 1,
    };
    let riff = build_xma2_riff(expected, &[0u8; PACKET_BYTES]);
    let parsed = parse(&riff).unwrap();
    assert_eq!(parsed.format, expected);

    // Masque incompatible (2 bits pour 6 canaux)
    let mut bad_mask = expected;
    bad_mask.channel_mask = 0x03;
    assert!(parse(&build_xma2_riff(bad_mask, &[])).is_err());

    // BytesPerBlock non multiple de 2048
    let mut bad_block = expected;
    bad_block.bytes_per_block = 3000;
    assert!(parse(&build_xma2_riff(bad_block, &[])).is_err());

    // PlayBegin + PlayLength > SamplesEncoded
    let mut bad_play = expected;
    bad_play.play_begin = 3000;
    bad_play.play_length = 2000;
    assert!(parse(&build_xma2_riff(bad_play, &[])).is_err());
}

#[test]
fn stream_channel_layouts_cover_mono_to_7_1() {
    assert_eq!(stream_channel_layout(1, 1).unwrap(), vec![1]);
    assert_eq!(stream_channel_layout(2, 1).unwrap(), vec![2]);
    assert_eq!(stream_channel_layout(2, 2).unwrap(), vec![1, 1]);
    assert_eq!(stream_channel_layout(3, 2).unwrap(), vec![2, 1]);
    assert_eq!(stream_channel_layout(4, 2).unwrap(), vec![2, 2]);
    assert_eq!(stream_channel_layout(5, 3).unwrap(), vec![2, 2, 1]);
    assert_eq!(stream_channel_layout(6, 3).unwrap(), vec![2, 2, 2]);
    assert_eq!(stream_channel_layout(7, 4).unwrap(), vec![2, 2, 2, 1]);
    assert_eq!(stream_channel_layout(8, 4).unwrap(), vec![2, 2, 2, 2]);
    assert!(stream_channel_layout(6, 2).is_err());
    assert!(stream_channel_layout(2, 3).is_err());
    assert!(stream_channel_layout(0, 1).is_err());
}

#[test]
fn multistream_5_1_routes_round_robin_bursts_and_block_boundaries() {
    // 5.1 (6 canaux, 3 flux stéréo), bloc XMA2 de 4 paquets (8 Kio) pour tester la sentinelle 0xFF
    // et la réinitialisation à la frontière du bloc.
    let mut format = Xma2Format::simple(6, 48_000, 3);
    format.channel_mask = 0x3F;
    format.bytes_per_block = (4 * PACKET_BYTES) as u32;
    let mut dec = MultiStreamDecoder::new(format).unwrap();

    // Bloc 1 :
    // - Paquet 0 → flux 0 (skip=0 : rafale, le flux 0 garde la main)
    // - Paquet 1 → flux 0 (skip=0xFF : flux 0 terminé dans ce bloc, passe au flux 1)
    // - Paquet 2 → flux 1 (skip=0xFF : passe au flux 2)
    // - Paquet 3 → flux 2 (skip=0xFF : fin de bloc → réinitialisation sur flux 0)
    assert_eq!(dec.next_stream_index(), 0);
    dec.feed_packet(&make_xma2_packet(2, 2, 0));
    assert_eq!(dec.next_stream_index(), 0);
    dec.feed_packet(&make_xma2_packet(2, 2, END_OF_BLOCK_SKIP));
    assert_eq!(dec.next_stream_index(), 1);
    dec.feed_packet(&make_xma2_packet(2, 4, END_OF_BLOCK_SKIP));
    assert_eq!(dec.next_stream_index(), 2);
    dec.feed_packet(&make_xma2_packet(2, 4, END_OF_BLOCK_SKIP));
    assert_eq!(dec.next_stream_index(), 0);
    assert_eq!(dec.packets_decoded(), 4);

    // Chaque flux a reçu 4 trames (1re trame d'amorce ignorée + 3 trames × 512 - 64 start_skip = 1 472 trames).
    let mut out = Vec::new();
    let frames = dec.drain(&mut out);
    assert_eq!(frames, 3 * 512 - 64);
    assert_eq!(out.len(), frames * 6);

    // Bloc 2 : routage round-robin classique (skip = 2 sur chaque flux) → 0, 1, 2, 0.
    let mut block2 = Vec::new();
    for _ in 0..3 {
        block2.extend_from_slice(&make_xma2_packet(2, 2, 2));
    }
    let frames2 = dec.feed(&block2, &mut out);
    assert_eq!(frames2, 2 * 512);
    assert_eq!(dec.next_stream_index(), 0);
}

#[test]
fn synthetic_voice_control_2048_512_pending_guard_and_disabled() {
    // Contrôle synthétique (issue #26, NinosTech) :
    // signal mono 48 kHz, lecture des 2 048 premiers échantillons puis de 512 échantillons dans
    // un tampon encadré par 256 octets de garde ; vérification des 1 536 échantillons restants
    // en attente, de l'intégrité des octets de garde et du comportement voix désactivée.
    let format = Xma2Format::simple(1, 48_000, 1);

    // 1er paquet : 6 trames → (6 - 1) * 512 - 64 = 2 496 échantillons.
    // 2e paquet : 4 trames → +2 048 échantillons (total 4 544 échantillons).
    let pkt1 = make_xma2_packet(1, 6, 0);
    let pkt2 = make_xma2_packet(1, 4, 0);
    let mut all_packets = Vec::with_capacity(PACKET_BYTES * 2);
    all_packets.extend_from_slice(&pkt1);
    all_packets.extend_from_slice(&pkt2);
    let reference = decode(format, &all_packets).unwrap();
    assert_eq!(reference.len(), 4_544);

    let mut voice = Xma2Voice::new(format).unwrap();
    let queued1 = voice.submit_packets(&pkt1);
    assert_eq!(queued1, 2_496);

    // 1er remplissage : 2 048 échantillons (reste 448 en file).
    let mut first = vec![0i16; 2_048];
    assert_eq!(voice.render_pcm(&mut first), 2_048);
    assert_eq!(first, reference[..2_048]);
    assert_eq!(voice.pending_frames(), 448);

    // On soumet des paquets supplémentaires pour obtenir exactement 2 048 échantillons en file
    // (1 600 échantillons ajoutés sur les 448 déjà présents → 2 048 en attente).
    let extra_ref: Vec<i16> = (0..1_600i16).map(|i| i.wrapping_mul(17).wrapping_sub(3000)).collect();
    assert_eq!(voice.enqueue_pcm(&extra_ref).unwrap(), 1_600);
    assert_eq!(voice.pending_frames(), 2_048);

    // 2e remplissage : 512 échantillons (1 024 octets) encadrés par 256 octets de garde avant et après.
    const GUARD_BYTES: usize = 256;
    const GUARD_VAL: u8 = 0xCD;
    let mut guarded = vec![GUARD_VAL; GUARD_BYTES + 512 * 2 + GUARD_BYTES];
    let rendered = voice.render_pcm_bytes(&mut guarded[GUARD_BYTES..GUARD_BYTES + 512 * 2]);
    assert_eq!(rendered, 512);

    // Les 256 octets de garde avant et après restent strictement intacts.
    assert!(guarded[..GUARD_BYTES].iter().all(|&b| b == GUARD_VAL));
    assert!(guarded[GUARD_BYTES + 512 * 2..].iter().all(|&b| b == GUARD_VAL));

    // Il reste exactement 1 536 échantillons en attente (2 048 - 512).
    assert_eq!(voice.pending_frames(), 1_536);

    // Les 512 échantillons rendus correspondent aux 448 restants de pkt1 puis aux 64 premiers de extra_ref.
    let mut expected_512 = Vec::with_capacity(512);
    expected_512.extend_from_slice(&reference[2_048..2_496]);
    expected_512.extend_from_slice(&extra_ref[..64]);
    let got_512: Vec<i16> = (0..512)
        .map(|i| {
            let p = GUARD_BYTES + i * 2;
            i16::from_le_bytes([guarded[p], guarded[p + 1]])
        })
        .collect();
    assert_eq!(got_512, expected_512);

    // Voix désactivée : le tampon de destination reste 100 % inchangé et la file conserve ses 1 536 trames.
    voice.set_enabled(false);
    assert!(!voice.is_enabled());
    let mut untouched = vec![0x5Au8; 512 * 2];
    assert_eq!(voice.render_pcm_bytes(&mut untouched), 0);
    assert!(untouched.iter().all(|&b| b == 0x5A));
    assert_eq!(voice.pending_frames(), 1_536);
}

#[test]
fn trims_samples_encoded_and_writes_extensible_5_1_wav() {
    let mut format = Xma2Format::simple(6, 48_000, 3);
    format.channel_mask = 0x3F;
    format.samples_encoded = 1000;
    format.play_begin = 128;
    format.play_length = 512;

    let mut payload = Vec::new();
    for _ in 0..3 {
        payload.extend_from_slice(&make_xma2_packet(2, 4, 2));
    }
    let pcm = decode(format, &payload).unwrap();
    assert_eq!(pcm.len(), 512 * 6);

    let wav = wav_extensible(6, 48_000, 0x3F, &pcm);
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 0xFFFE); // WAVE_FORMAT_EXTENSIBLE
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 6); // 6 canaux
    assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 0x3F); // dwChannelMask 5.1
    assert_eq!(&wav[60..64], b"data");
    assert_eq!(u32::from_le_bytes(wav[64..68].try_into().unwrap()) as usize, pcm.len() * 2);
}
