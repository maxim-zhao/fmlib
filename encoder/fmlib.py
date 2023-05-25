import gzip
import sys
import time
import unittest
from dataclasses import dataclass


# We define some classes to represent the data in a VGM file

@dataclass
class Wait:
    """This represents a wait command"""
    length: int


@dataclass
class Ym2413Command:
    """This represents a YM2413 command"""
    register: int
    data: int


@dataclass
class LoopPoint:
    """This represents the loop point"""


# Next we make a class that parses a VGM file

class VgmFile:
    def __init__(self, filename: str):
        with gzip.open(filename, "rb") as f:
            self.data = f.read()

        # Check VGM header
        if self.data[0:4].decode("ASCII") != "Vgm ":
            raise Exception("Not a valid VGM file")

        # Read some of the header
        # We could use the struct library if this becomes more complex
        self.eof_offset = self.read32_relative(0x4)
        self.version = self.read32(0x8)  # Leave it as 32-bit rather than convert BCD...
        self.loop_offset = self.read32_relative(0x1c)
        if self.version >= 0x150:
            self.data_offset = self.read32_relative(0x34)
        else:
            self.data_offset = 0x40

        self.position = self.data_offset

    def read32(self, offset: int):
        return int.from_bytes(self.data[offset:offset + 4], byteorder='little')

    def read32_relative(self, offset: int):
        value = self.read32(offset)
        return value + offset if value > 0 else 0

    """This function yields the VGM commands as a stream"""

    def commands(self):
        while self.position < self.eof_offset:
            if self.position == self.loop_offset:
                yield LoopPoint()
            b = self.data[self.position]
            self.position += 1
            if b == 0x4f:
                pass  # GG stereo
            elif b == 0x50:
                self.position += 1
                pass  # PSG
            elif b == 0x51:  # YM2413
                yield Ym2413Command(self.data[self.position], self.data[self.position + 1])
                self.position += 2
                pass
            elif b == 0x61:  # Wait nnnn
                yield Wait(int.from_bytes(self.data[self.position: self.position + 2], byteorder='little'))
                self.position += 2
                pass
            elif b == 0x62:  # Wait 1/60
                yield Wait(735)
                pass
            elif b == 0x62:  # Wait 1/50
                yield Wait(882)
                pass
            elif b == 0x66:  # End
                return
        # TODO: handle other VGM commands (by skipping them)


# Next we make classes which represent single channels in the YM2413.
# These process the write commands into the FMLib binary format.

class ChannelBase:
    """A channel with some data"""

    def __init__(self, index: int, frame_length: int = 735):
        self.data = bytearray()
        self.index = index
        self.wait_samples = 0
        self.frame_length = frame_length

    def add_loop(self):
        self.emit_wait_if_needed()
        self.data.append(0b10100001)

    def wait(self, length: int):
        # We get waits in "samples" (1/44100 s).
        # We want to convert them to frames.
        # We do that by accumulating and writing them later.
        self.wait_samples += length

    def emit_wait_if_needed(self):
        while self.wait_samples > self.frame_length:
            frames = min(self.wait_samples // self.frame_length, 32)
            # Waits: %100xxxxx for x+1 frames (min 1, max 32)
            self.data.append((0b100 << 5) | (frames - 1))
            self.wait_samples -= frames * self.frame_length

    def end(self):
        self.emit_wait_if_needed()
        self.data.append(0b10100000)

    def __str__(self):
        return f"ch{self.index}[{len(self.data)}]"

    def append_if_changed(self, prefix: int, shift: int, mask: int, old_value: int, new_value: int):
        """Append prefix | ((new_value >> shift) & mask) if the relevant bits are different to old_value"""
        new_value >>= shift
        new_value &= mask
        if old_value != -1:
            old_value >>= shift
            old_value &= mask
            if old_value == new_value:
                return
        self.data.append(prefix | new_value)


class ToneChannel(ChannelBase):
    def __init__(self, index: int):
        super().__init__(index)
        self.reg1 = -1
        self.reg2 = -1
        self.reg3 = -1

    def reg1x(self, b: int):
        self.emit_wait_if_needed()
        # Only emit changes
        self.append_if_changed(0b00000000, 0, 0b1111, self.reg1, b)
        self.append_if_changed(0b00010000, 4, 0b1111, self.reg1, b)
        self.reg1 = b

    def reg2x(self, b: int):
        self.emit_wait_if_needed()
        self.append_if_changed(0b01000000, 0, 0b111111, self.reg2, b)
        self.reg2 = b

    def reg3x(self, b: int):
        self.emit_wait_if_needed()
        self.append_if_changed(0b00100000, 0, 0b1111, self.reg3, b)
        self.append_if_changed(0b00110000, 4, 0b1111, self.reg3, b)
        self.reg3 = b


class RhythmChannel(ChannelBase):
    def __init__(self, index: int):
        super().__init__(index)
        self.value = -1
        self.custom_instruments = [-1 for x in range(8)]

    def write(self, b: int):
        self.emit_wait_if_needed()
        self.append_if_changed(0b00000000, 0, 0b111111, self.value, b)
        self.value = b

    def custom_instrument(self, index: int, value: int):
        self.emit_wait_if_needed()
        self.data.append(0b01000000 | index)
        size = len(self.data)
        self.append_if_changed(0b01010000, 0, 0b00001111, self.custom_instruments[index], value)
        self.append_if_changed(0b01110000, 4, 0b00001111, self.custom_instruments[index], value)
        if len(self.data) == size:
            # Remove the index selector as it wasn't needed
            del self.data[-1]


# FMLib format speculation:
#
# Each tone channel is managed as a separate stream, with an implicit channel number.
# We also have a stream that holds rhythm and custom instrument writes.
# We translate the YM2413 register writes as follows:
# Tone channels:
# Register 1x: F-num low bits (8)
# Register 2x: F-num high bit (1); block (2); key (1); sustain (1)
# Register 3x: volume (4); instrument (4)
# Encoding:
# %0000xxxx: 1x low nibble (if changed)
# %0001xxxx: 1x high nibble (if changed)
# %01xxxxxx: 2x all bits
# %0010xxxx: 3x instrument (if changed)
# %0011xxxx: 3x volume (if changed)
# %100xxxxx: wait x+1 frames (range 1..32)
# %101xxxxx: compression: x+3 bytes length (range 3..34), followed by offset relative to start of file (2 bytes)
# %10100000: end of data
# %10100001: loop point
# %11sxxyyy: small f-num change + wait:
#   s = sign 
#   xx = abs(change) - 1
#   yyy = wait duration in frames
# And for rhythm/custom instrument:
# %00xxxxxx = set register $0e to x
# %01000xxx = select custom instrument register x
# %0101xxxx = set low nibble of custom register
# %0111xxxx = set high nibble of custom register
# %100xxxxx = wait up to 32 frames
# %101xxxxx = compression, up to 34 bytes, followed by offset (2 bytes)
#             x = 0 => end of data
#             x = 1 => loop point
# This means the wait and compression parts are the same for both.


class FMLibFile:
    def __init__(self):
        self.channels = [ToneChannel(index) for index in range(9)]
        self.channels.append(RhythmChannel(9))

    def save(self, filename: str):
        # The end format is 20 bytes of pointers to per-channel data.
        # A zero pointer will be used to indicate a channel is not used. TODO
        with open(filename, "wb") as f:
            lengths = [len(channel.data) for channel in self.channels]
            offset = len(self.channels) * 2
            for length in lengths:
                f.write(offset.to_bytes(2, byteorder='little'))
                offset += length
            for channel in self.channels:
                f.write(channel.data)


def longest_match(needle, haystack, min_length, max_length):
    if len(haystack) < min_length or len(needle) < min_length:
        return 0, 0

    for length in range(min(len(needle), len(haystack), max_length), min_length, -1):
        offset = haystack.find(needle[:length])
        if offset != -1:
            return offset, length
    return 0, 0


def compress2(data: bytes, min_length: int) -> bytes:
    max_length = 31 + min_length - 2  # 5 bits length, 2 reserved
    print(f"compressing with min length {min_length} -> max length {max_length}")

    # We look for the "best" runs in the data to compress first. 
    # "Best" means the one that saves the most bytes, for example:
    # A run of length 4 that is found 50 times will cost 4 bytes for the first run, 
    #   then 3 bytes for each repeat, so it will save 49 bytes to use it.
    # A run of length 50 that is found 3 times will cost 50 bytes for the first run, 
    #   then 3 bytes for each repeat, so it will save 94 bytes to use it.
    # If a run of data is chosen as "best", it is still valid to use it as a source for
    # other matching sub-runs later on, but not to substitute data within it. We do not
    # count these "submatches" towards the decision to pick that run, though.
    # Thus:
    # - We need to split the data into sets of:
    #   1. Runs referenced elsewhere, which may not be altered
    #   2. Runs of bytes that have not yet been processed, which may be candidates for substitution
    #   3. References to sequences in #1
    # - New "best" runs might be inside existing runs, or cross their boundaries
    # This gets complicated fast...

    runs = []  # will be tuples of start, end
    pointers = set(range(0, int.from_bytes(data[0:2], byteorder='little'), 2))
    # will be offsets of the pointers themselves (value can be read back)
    # we populate with the existing header

    print(f"Before compression: {len(data)} bytes")
    with open("before.bin", "wb") as f:
        f.write(data)

    # First, find the "best" run to substitute.
    # We define "best" as "the most bytes will be saved with this one substitution". This is not optimal: some of those
    # substitutions may de-optimise future match opportunities.
    next_print_time = time.time() + 0.1
    round = 1
    while True:
        best_savings = 0
        max_position = len(data) - min_length
        tried_needles = set()

        # For each position in the data...
        for position in range(0, max_position):
            if time.time() > next_print_time:
                print(f"{position}/{max_position}, best: {best_savings}", end="\r")
                next_print_time += 0.1
            if position - 1 in pointers or position in pointers or position + 1 in pointers:
                # not a valid source
                continue
            # For each possible run length...
            for length in range(min_length, min(max_length, len(data) - position) + 1):
                # Extract the slice of data
                run = data[position:position + length]
                # Skip if already done
                s = str(run)
                if s in tried_needles:
                    # Note that this can miss opportunities. If the needle is in a replaceable area but the match is
                    # in an existing run, then it cannot be used. However, it may be used the other wat round.
                    continue
                tried_needles.add(s)
                # Find matches after the run itself
                # print(f"Looking for matches of length {length} at offset {position}")
                matches = [x for x in find_matches(run, data, pointers, runs) if x != position]
                # If none, then don't try longer lengths
                if len(matches) == 0:
                    break
                # Then compute the savings
                savings = (length - 3) * len(matches)
                # if savings > 100:
                #    print(f"{length}@{position:x}->{savings}")
                # And decide if it's our favourite
                if savings > best_savings:
                    best_savings = savings
                    best_match = (position, length, matches)

        # Next, try to substitute it.
        if best_savings > 0:
            (position, length, matches) = best_match
            print(
                f"Round {round}: offset 0x{position:x} length {length} matches {len(matches)} places, saving {best_savings} bytes")
            # Calculate the savings per match
            bytes_saved_per_match = length - 3
            # The matches refer to positions before the data is spliced - we adjust them in advance
            for n in range(len(matches)):
                matches[n] -= bytes_saved_per_match * n
            # Now replace them
            for match in matches:
                # If the match is before the data position, we need to adjust the position to account for the
                # shift we are about to make, before we write it
                if match < position:
                    position -= bytes_saved_per_match
                # Replace the bytes in the data
                splice(data, match, position, length)
                # We need to adjust the pointers and runs after it as their locations have changed.
                for pointer in [x for x in pointers]:  # We have to iterate a copy as we modify it
                    if pointer > match:
                        # Adjust pointer position
                        pointers.remove(pointer)
                        pointer -= bytes_saved_per_match
                        pointers.add(pointer)
                    # Read existing value
                    value = int.from_bytes(data[pointer:pointer + 2], byteorder='little')
                    if value > match:
                        # Adjust
                        value -= bytes_saved_per_match
                        # Write back
                        data[pointer:pointer + 2] = value.to_bytes(2, byteorder='little')
                # Remember the new pointer location. It is stored one byte after the match location.
                pointers.add(match + 1)
                # We also need to adjust the stored runs
                for (start, end) in runs:
                    if start <= match:
                        continue
                    # Could make a mutable object here?
                    runs.remove((start, end))
                    runs.append((start - bytes_saved_per_match, end - bytes_saved_per_match))

            # Record the run so we don't mess with its data in future
            runs.append((position, position + length))

            with open(f"round{round}.bin", "wb") as f:
                f.write(data)
            round += 1
        else:
            print("No more matches found")
            break

    return data


def splice(data, offset, pointer, length):
    replacement = bytearray()
    replacement.append(0b10100000 | (length - 2))
    replacement.extend(pointer.to_bytes(2, byteorder='little'))
    data[offset:offset + length] = replacement


def find_matches(needle, haystack, pointers, runs):
    # Yields offsets of needle in haystack, where there's no overlap to pointers and existing run sources
    offset = 0
    while True:
        offset = haystack.find(needle, offset)
        if offset == -1:
            return
        # If the match overlaps any pointers or runs, it's invalid, and we try the next byte
        invalid = False
        for x in range(offset, offset + len(needle)):
            # Pointers are generally 3 bytes, we have the offset of the middle byte here
            if x - 1 in pointers or x in pointers or x + 1 in pointers:
                # Matched a pointer (or the byte before it, which is the marker)
                invalid = True
                break
        if not invalid:
            for (start, end) in runs:
                if start < offset + len(needle) and end > offset:
                    # Existing run overlaps
                    # Logic:
                    #   [ () ]   ( is before ] and ) is after [
                    #   [ (  ] ) ( is before ] and ) is after [
                    # ( [    ] ) ( is before ] and ) is after [
                    # ( [  ) ]   ( is before ] and ) is after [
                    # ()[    ]   ( is before ] but ) is before [
                    #   [    ]() ( is after ] and ) is after [
                    # As our ends are exclusive, we use < and > rather than <= and >=
                    invalid = True
                    break
        if invalid:
            # Could be cleverer to skip further ahead here?
            offset += 1
        else:
            yield offset
            offset += len(needle)


def save(data: bytes, filename: str):
    # The end format is 20 bytes of pointers to per-channel data.
    # A zero pointer will be used to indicate a channel is not used. TODO
    with open(filename, "wb") as f:
        chunks = [x for x in data]
        lengths = [len(x) for x in chunks]
        offset = len(data) * 2
        for length in lengths:
            f.write(offset.to_bytes(2, byteorder='little'))
            offset += length
        for chunk in chunks:
            f.write(chunk)


def convert(filename: str) -> FMLibFile:
    file = VgmFile(filename)
    print(f"Loaded \"{filename}\", VGM data has {len(file.data)} commands")
    print(f"Data starts at {file.data_offset:#x}")
    print(f"Data loops at {file.loop_offset:#x}")
    print(f"Data ends at {file.eof_offset:#x}")

    print(f"Converting to FMLib (unoptimised)...")
    result = FMLibFile()

    for command in file.commands():
        # print(line)
        if type(command) is Wait:
            for channel in result.channels:
                channel.wait(command.length)
        elif type(command) is LoopPoint:
            for i in result.channels:
                channel.add_loop()
        elif type(command) is Ym2413Command:
            if command.register >= 0x10:
                # tone register
                channel = command.register & 0xf
                if channel > 8:
                    print(f"Unhandled command for register {command.register}")
                    continue  # shouldn't happen

                if command.register & 0xf0 == 0x10:
                    result.channels[channel].reg1x(command.data)
                elif command.register & 0xf0 == 0x20:
                    result.channels[channel].reg2x(command.data)
                elif command.register & 0xf0 == 0x30:
                    result.channels[channel].reg3x(command.data)
            elif command.register == 0x0e:
                # rhythm
                result.channels[9].write(command.data)
            elif command.register <= 0x08:
                result.channels[9].custom_instrument(command.register, command.data)
            else:
                # Something else
                print(f"Unhandled command for register {command.register}")

    # Print some stuff
    for channel in result.channels:
        channel.end()
        print(f"ch{channel.index} data = {len(channel.data)} bytes")

    print(f"Total data = {sum([len(x.data) for x in result.channels])} bytes")

    # Save uncompressed
    result.save(filename + ".pass1.fmlib")


def foo():
    # Compression time...

    # First we assemble the data into the uncompressed final form
    data = bytearray()
    chunks = [channels[index].data for index in channels]
    header_size = len(chunks) * 2
    # Fill in pointers
    offset = header_size
    for chunk in chunks:
        data.extend(offset.to_bytes(2, byteorder='little'))
        offset += len(chunk)
    # Then append the data itself
    for chunk in chunks:
        data.extend(chunk)

    for i in range(4, 5):
        compressed = compress2(data, i)
        print(f"Compressed to {len(compressed)} with min length {i}")
        with open(f"{filename}.compressed.{i}.fm2", "wb") as f:
            f.write(compressed)


def testcompress2():
    data = bytearray()
    data.extend(int(2).to_bytes(length=2, byteorder='little'))
    data.extend("ABCDExABCDEFGHIJyABCDEFGHIJzABCDEFGHIJaABCDEFGHIJbABCDEFGHIJ".encode(encoding='ascii'))
    compressed = compress2(data, 4)
    with open("testcompressed.bin", "wb") as f:
        f.write(compressed)
    print(f"{len(data) + 2} -> {len(compressed)}")


def main():
    verb = sys.argv[1]
    if verb == "print":
        # Test the VGM parsing
        f = VgmFile(sys.argv[2])
        for c in f.commands():
            print(c)
    elif verb == "convert":
        f = convert(sys.argv[2])
    elif verb == 'convert':
        save(convert2(sys.argv[2]), sys.argv[2] + ".fm")
    elif verb == 'testcompress2':
        testcompress2()
    elif verb == 'compress2':
        with open(sys.argv[2], "rb") as f:
            compress2(f.read(), 4)
    else:
        raise Exception(f"Unknown verb \"{verb}\"")


class Test(unittest.TestCase):
    def test_compress(self):
        data = bytearray()
        data.extend(int(2).to_bytes(length=2, byteorder='little'))
        data.extend("ABCDExABCDEFGHIJyABCDEFGHIJzABCDEFGHIJaABCDEFGHIJbABCDEFGHIJ".encode(encoding='ascii'))
        # ABCDExABCDEFGHIJyABCDEFGHIJzABCDEFGHIJaABCDEFGHIJbABCDEFGHIJ
        #       ^^^^^^^^^^-^^^^^^^^^^-^^^^^^^^^^-^^^^^^^^^^-^^^^^^^^^^
        # ABCDExABCDEFGHIJy[]z[]a[]b[]
        #       ^^^^^
        # []xABCDEFGHIJy[]z[]a[]b[]
        compressed = compress2(data, 4)
        assert compressed == bytearray.fromhex("00")


if __name__ == "__main__":
    main()
