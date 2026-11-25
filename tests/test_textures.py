"""Textures console en Rust (rust/anyxboxx-texture, portage du décodeur BC7 de Xbox2Windows et
de l'agencement GFX10 d'AMD addrlib, MIT) : la ligne de commande écrit un PNG que Python relit
indépendamment (zlib, CRC)."""
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

BINARY = Path(__file__).resolve().parents[1] / "rust" / "target" / "release" / "anyxboxx_texture.exe"
pytestmark = pytest.mark.skipif(not BINARY.is_file(), reason="crate Rust non compilée (cargo build --release)")


def hand_block():
    """Mode 1, partition 13 (calcul à la main dans les tests de Xbox2Windows)."""
    fields = ((1, 1), (13, 2), (63, 8), (63, 26), (1, 80), (7, 105))
    return sum(value << offset for value, offset in fields).to_bytes(16, "little")


def read_png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    position, chunks = 8, {}
    while position < len(data):
        length, kind = struct.unpack_from(">I4s", data, position)
        body = data[position + 8:position + 8 + length]
        crc = struct.unpack_from(">I", data, position + 8 + length)[0]
        assert crc == zlib.crc32(kind + body), kind
        chunks[kind] = chunks.get(kind, b"") + body
        position += 12 + length
    width, height = struct.unpack_from(">II", chunks[b"IHDR"])
    raw = zlib.decompress(chunks[b"IDAT"])
    stride = width * 4 + 1
    assert all(raw[row * stride] == 0 for row in range(height))
    return width, height, b"".join(raw[row * stride + 1:(row + 1) * stride] for row in range(height))


def test_cli_decodes_a_block_into_a_valid_png(tmp_path):
    source, image = tmp_path / "bloc.bin", tmp_path / "bloc.png"
    source.write_bytes(hand_block())
    result = subprocess.run([str(BINARY), "bc7", str(source), "4", "4", str(image)], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    width, height, pixels = read_png(image)
    expected = bytes([255, 2, 2, 255]) * 8 + bytes([253, 0, 0, 255]) + bytes([0, 0, 0, 255]) * 7
    assert (width, height, pixels) == (4, 4, expected)


def test_cli_rejects_a_truncated_surface(tmp_path):
    source = tmp_path / "court.bin"
    source.write_bytes(bytes(15))
    result = subprocess.run([str(BINARY), "bc7", str(source), "4", "4", str(tmp_path / "x.png")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "attendus" in result.stderr


def test_cli_decodes_64k_s_and_64k_s_x_array_tail_mip(tmp_path):
    # Surface 8×8 BC7 (2×2 blocs), 2 mips dans la queue 64 Kio, tableau de 2 tranches,
    # mode SW_64K_S_X avec pipeBankXor = 0x45 : le mip 1 (4×4, 1 bloc) vit au niveau 1
    # de la queue (origine (0, 32) -> Y5=1 -> bit 14 | bit 9 = 0x4200),
    # XOR tranche Z0=1 (bit 11 = 0x0800) et pipeBankXor effectif ((0x45 & 0xCF) << 8 = 0x4500)
    # -> offset intra-tranche = 0x4200 ^ 0x0800 ^ 0x4500 = 0x0F00.
    block = hand_block()
    # 1) --64k-s mono-mip 4×4 : bloc (0, 0) à l'offset 0 d'une tuile de 64 Kio.
    tile_64k = bytearray(65_536)
    tile_64k[0:16] = block
    src_64k, out_64k = tmp_path / "s64k.bin", tmp_path / "s64k.png"
    src_64k.write_bytes(tile_64k)
    res1 = subprocess.run(
        [str(BINARY), "bc7", str(src_64k), "4", "4", str(out_64k), "--64k-s"],
        capture_output=True, text=True, timeout=60,
    )
    assert res1.returncode == 0, res1.stderr
    expected = bytes([255, 2, 2, 255]) * 8 + bytes([253, 0, 0, 255]) + bytes([0, 0, 0, 255]) * 7
    assert read_png(out_64k) == (4, 4, expected)

    # 2) --mode 64k-s-x --xor 0x45 --mips 2 --mip 1 --array 2 --slice 1
    tiled_array = bytearray(2 * 65_536)
    offset_slice1_mip1 = 65_536 + (0x4200 ^ 0x0800 ^ 0x4500)
    tiled_array[offset_slice1_mip1:offset_slice1_mip1 + 16] = block
    src_xor, out_xor = tmp_path / "xor_arr.bin", tmp_path / "xor_arr.png"
    src_xor.write_bytes(tiled_array)
    res2 = subprocess.run(
        [
            str(BINARY), "bc7", str(src_xor), "8", "8", str(out_xor),
            "--mode", "64k-s-x", "--xor", "0x45", "--mips", "2", "--mip", "1", "--array", "2", "--slice", "1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert res2.returncode == 0, res2.stderr
    assert read_png(out_xor) == (4, 4, expected)


def test_cli_rejects_unknown_option(tmp_path):
    source = tmp_path / "bloc.bin"
    source.write_bytes(hand_block())
    result = subprocess.run(
        [str(BINARY), "bc7", str(source), "4", "4", str(tmp_path / "bad.png"), "--inconnu"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1 and "inconnue" in result.stderr
