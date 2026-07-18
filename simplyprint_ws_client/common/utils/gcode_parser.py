from typing import Generator, List, Iterable, Tuple, NamedTuple, Optional, Union

#: A parsed argument value: ``F0``/``F1`` become booleans, numerics become
#: ``int``/``float``, anything else stays the raw string.
GcodeArgValue = Union[str, int, float, bool]


class GcodeCommand(NamedTuple):
    cmd: str
    args: Optional[List[Tuple[str, GcodeArgValue]]] = None

    def arg(
        self, name: str, default: Optional[GcodeArgValue] = None
    ) -> Optional[GcodeArgValue]:
        """The value of argument ``name`` (e.g. ``"S"``), or ``default``."""
        for key, value in self.args or ():
            if key == name:
                return value

        return default

    def __str__(self):
        s = f"{self.cmd}"

        for c, v in self.args:
            if isinstance(v, bool):
                s += f" {c}{int(v)}"
            else:
                s += f" {c}{v}"

        return s

    @staticmethod
    def arg_cast(c: str, s: str) -> GcodeArgValue:
        try:
            if c == "F":
                if s == "0":
                    return False
                if s == "1":
                    return True

            if "." in s:
                return float(s)

            return int(s)
        except ValueError:
            return s

    @classmethod
    def from_line(cls, s: str) -> "GcodeCommand":
        wc = 0
        p = ""
        command = ""
        args = []

        for c in s:
            if c.isspace() and not p.isspace():
                wc += 1

            p = c

            if c.isspace():
                continue

            if wc == 0:
                command += c
            elif c:
                if len(args) < wc:
                    args.append((c, ""))
                    continue

                idx = wc - 1
                item = args[idx]
                args[idx] = (item[0], item[1] + c)

        args = [(arg[0], cls.arg_cast(*arg)) for arg in args]

        return GcodeCommand(command, args)


class GcodeParser:
    @staticmethod
    def _cleanup_gcode_line(s: str) -> str:
        if (comment_idx := s.find(";")) != -1:
            s = s[:comment_idx]

        return s.strip()

    def _clean_gcode_lines(self, lines: Iterable[str]) -> Iterable[str]:
        for line in lines:
            if line := self._cleanup_gcode_line(line):
                yield line

    def parse_gcode(self, lines: List[str]) -> Generator[GcodeCommand, None, None]:
        for line in self._clean_gcode_lines(lines):
            yield GcodeCommand.from_line(line)
