import gzip
import sys
import time
import unittest
from dataclasses import dataclass
from typing import BinaryIO


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

    def optimise(self):
        # We want to inspect the data. In each chunk of data between pauses, if all it does is change the frequency and
        # then do a short pause, we can optimise that.
        result = bytearray()
        frequency = 0
        last_frequency = 0
        reg2 = 0
        last_reg2 = 0
        # First chunk up the data...
        for chunk in self.get_chunks():
            # Process the chunk
            has_other_data = False
            pause_length = 0
            for b in chunk:
                if b & 0b11110000 == 0b00000000:
                    # Freq low nibble
                    frequency &= 0b111110000
                    frequency |= b & 0b1111
                elif b & 0b11110000 == 0b00010000:
                    # Freq high nibble
                    frequency &= 0b100001111
                    frequency |= (b & 0b1111) << 4
                elif b & 0b11000000 == 0b01000000:
                    # Freq high bit + other stuff
                    frequency &= 0b011111111
                    frequency |= (b & 1) << 8
                    reg2 = b & 0b11111110
                    if reg2 != last_reg2:
                        has_other_data = True
                elif b & 0b11100000 == 0b10000000:
                    # Pause
                    pause_length = (b & 0b11111) + 1
                else:
                    has_other_data = True

            # Decide if it's optimisable
            frequency_change = frequency - last_frequency
            if abs(frequency_change) < 4 and 0 < pause_length < 9 and not has_other_data:
                # Yes: alter the chunk to just that
                sign = 1 if frequency_change < 0 else 0
                chunk = [0b11000000 | (sign << 5) | (abs(frequency_change) << 3) | pause_length]

            # Then pass the chunk through
            result.extend(chunk)

            # And remember the state
            last_frequency = frequency
            last_reg2 = reg2

        # Finally replace our data
        self.data = result

    def get_chunks(self):
        chunk = []
        for b in self.data:
            chunk.append(b)
            if b & 0b10000000 != 0:
                # All register writes start with a 0, everything else starts with a 1
                yield chunk
                chunk = []


class RhythmChannel(ChannelBase):
    def __init__(self, index: int):
        super().__init__(index)
        self.value = -1
        self.custom_instruments = [-1 for _ in range(8)]

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

    def optimise(self):
        # Nothing to do here
        pass


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
# %101xxxxx: compression: x+2 bytes length (range 4..33), followed by offset relative to start of file (2 bytes)
# %10100000: end of data
# %10100001: loop point
# %11sxxyyy: small f-num change + wait:
#   s = sign 
#   xx = abs(change) - 1, change in range 1..4 inclusive
#   yyy = wait duration in frames, minus 1, range 1..8 inclusive
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

    def __str__(self):
        result = ""
        for channel in self.channels:
            result += f"{channel} "

        result += f"Total {sum([len(x.data) for x in self.channels])} bytes"
        return result

    def save(self, filename: str):
        # The end format is 20 bytes of pointers to per-channel data.
        # A zero pointer will be used to indicate a channel is not used. TODO
        with open(filename, "wb") as f:
            self.save_to(f)

    def save_to(self, f: BinaryIO):
        lengths = [len(channel.data) for channel in self.channels]
        offset = len(self.channels) * 2
        for length in lengths:
            f.write(offset.to_bytes(2, byteorder='little'))
            offset += length
        for channel in self.channels:
            f.write(channel.data)

    def print(self):
        for channel in self.channels:
            print(f"Channel {channel.index}")
            for b in channel.data:
                if (b & 0b11110000) == 0b00000000:
                    print(f"{b:x} Register 1x low nibble -> {b & 0b1111:x}")
                elif (b & 0b11110000) == 0b00010000:
                    print(f"{b:x} Register 1x high nibble -> {b & 0b1111:x}")
                elif (b & 0b11000000) == 0b01000000:
                    print(f"{b:x} Register 2x -> {b & 0b00111111:x}")
                elif (b & 0b11110000) == 0b00100000:
                    print(f"{b:x} Register 3x instrument -> {b & 0b1111:x}")
                elif (b & 0b11110000) == 0b00110000:
                    print(f"{b:x} Register 3x volume -> {b & 0b1111:x}")
                elif (b & 0b11100000) == 0b10000000:
                    print(f"{b:x} Wait {(b & 0b11111) + 1} frames")
                elif b == 0b10100000:
                    print(f"{b:x} End of data")
                elif b == 0b10100001:
                    print(f"{b:x} Loop point")
                elif (b & 0b11100000) == 0b10100000:
                    print(f"{b:x} compression")
                elif (b & 0b11000000) == 0b11000000:
                    print(f"{b:x} Vibrato: frequency {'-' if b & 0b100000 > 0 else '+'}{((b & 0b11000) >> 3) + 1}"
                          f" then wait {(b & 0b111) + 1} frames")
                else:
                    print(f"{b:x} Unexpected!")

    @dataclass
    class Run:
        data: bytearray
        offset: int
        is_source: bool

    @dataclass
    class Pointer:
        offset: int

    def compress(self) -> None:
        print(f"Compressing...")
        start_time = time.time()

        # We maintain a set of pointer offsets in each channel, initially empty
        pointers: list[set[int]] = [set[int]() for _ in self.channels]
        # And ranges of "fixed" bytes. These are referenced elsewhere so may not be changed.
        runs: list[set[int, int]] = [set[int, int]() for _ in self.channels]

        # Find the best candidate for compression.
        # This is the run of bytes in the current data which will save the most bytes if we convert other data to
        # point to it.
        best_savings = 0
        best_length = 0
        best_matches = []
        next_print_time = time.time() + 0.1
        # One optimisation is to not try the same sequence of bytes more than once.
        tried_runs = set[str]()
        for channel in self.channels:
            # Find candidate runs in this channel
            max_run_length = 33
            min_run_length = 4  # TODO is it better to skew this range?
            # We consider all run lengths that don't overlap existing pointers
            for offset in range(len(channel.data) - min_run_length):
                # If it starts in a pointer, skip it
                if offset in pointers or offset + 1 in pointers:
                    continue
                # We do consider sources that overlap existing runs. However, targets must be replaceable.

                if time.time() > next_print_time:
                    print(f"ch{channel.index} {offset}/{len(channel.data)}, best: {best_savings} "
                          f"with run length {best_length} matching {len(best_matches)} places", end="\r")
                    next_print_time += 0.1

                # Then try all the lengths
                for run_length in range(min_run_length, max_run_length + 1):
                    # Check if we have hit a pointer
                    if offset + run_length in pointers:
                        # If so, don't try this or any longer length
                        break
                    # Else this is a candidate run
                    candidate_run = channel.data[offset:offset + run_length]

                    # Don't consider runs we've seen before
                    s = str(candidate_run)
                    if s in tried_runs:
                        continue
                    tried_runs.add(s)

                    # Measure its effectiveness
                    matches, savings = self.find_matches(candidate_run, offset, channel)

                    if savings == 0:
                        # If we don't find this run then we won't find any longer ones with the same prefix
                        break

                    if savings > best_savings:
                        best_savings = savings
                        best_matches = matches
                        best_channel = channel
                        best_offset = offset
                        best_length = run_length

        print(f"Best saving is channel {best_channel.index} offset {best_offset} length {best_length}, "
              f"saving {best_savings} bytes with {len(best_matches)} matches")

        for b in best_channel.data[best_offset:best_offset + best_length]:
            print(f"{b:02x} ", end="")

        print("")
        print(f"Finished in {time.time() - start_time} seconds")

    def optimise(self) -> None:
        print("Optimising vibrato...")
        for channel in self.channels:
            channel.optimise()

    @staticmethod
    def get_candidate_runs(channel: ChannelBase, pointers: set[int]) -> (bytearray, int):
        max_run_length = 33
        min_run_length = 4  # TODO is it better to skew this range?
        # We consider all run lengths that don't overlap existing pointers
        for offset in range(len(channel.data) - min_run_length):
            # If it starts in a pointer, skip it
            if offset in pointers or offset + 1 in pointers:
                continue
            # We do consider sources that overlap existing runs. However, targets must be replaceable.

            # Then try all the lengths
            for run_length in range(min_run_length, max_run_length + 1):
                # Check if we have hit a pointer
                if offset + run_length in pointers:
                    # If so, don't try this or any longer length
                    break
                # Else this is a candidate run
                yield channel.data[offset:offset + run_length], offset

    def find_matches(self, candidate_run: bytearray, source_offset: int, source_channel: ChannelBase) \
            -> (set[(int, int)], int):
        savings = 0
        matches = set[(int, int)]()
        for channel in self.channels:
            # Look for the candidate run in its data
            # If it's the source channel, exclude the source run itself
            search_start = 0
            search_end = len(channel.data)
            while search_start <= search_end - len(candidate_run):
                # TODO this will match on pointer values!?
                offset = channel.data.find(candidate_run, search_start, search_end)
                if offset == -1:
                    # Not found
                    search_start = search_end
                    continue
                # We found it!
                # Was it the original, or something overlapping it?
                if channel == source_channel:
                    if source_offset <= offset < source_offset + len(candidate_run):
                        # Search after it
                        search_start = source_offset + len(candidate_run)
                        continue
                # A pointer costs 3 bytes
                savings += len(candidate_run) - 3
                search_start = offset + len(candidate_run)
                matches.add((channel.index, offset))

        return matches, savings


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
            print(f"Round {round}: offset 0x{position:x} length {length} matches {len(matches)} places, "
                  f"saving {best_savings} bytes")
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


def save(data: bytes, filename: str) -> None:
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

    print(f"Converting to FMLib...")
    result = FMLibFile()

    for command in file.commands():
        if type(command) is Wait:
            for channel in result.channels:
                channel.wait(command.length)
        elif type(command) is LoopPoint:
            for channel in result.channels:
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

    for channel in result.channels:
        channel.end()

    return result


def main():
    verb = sys.argv[1]
    if verb == "print":
        # Test the VGM parsing
        f = VgmFile(sys.argv[2])
        for c in f.commands():
            print(c)
    elif verb == "convert":
        f = convert(sys.argv[2])
        f.save(f"{sys.argv[2]}.pass1.fmlib")
        print(f)
    elif verb == "convert2":
        f = convert(sys.argv[2])
        f.save(f"{sys.argv[2]}.pass1.fmlib")
        print(f)
        f.optimise()
        f.save(f"{sys.argv[2]}.pass2.fmlib")
        print(f)
    elif verb == "compress":
        f = convert(sys.argv[2])
        f.save(f"{sys.argv[2]}.pass1.fmlib")
        print(f)
        f.optimise()
        f.save(f"{sys.argv[2]}.pass2.fmlib")
        print(f)
        f.compress()
        f.save(f"{sys.argv[2]}.pass3.fmlib")
        print(f)
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
