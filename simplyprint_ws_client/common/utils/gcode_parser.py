from typing import List, Iterable, Tuple, Any, NamedTuple, Optional


class GcodeCommand(NamedTuple):
    cmd: str
    args: Optional[List[Tuple[str, Any]]] = None

    def __getattr__(self, item):
        for arg in self.args:
            if arg[0] == item:
                return arg[1]

        return None

    def __str__(self):
        s = f"{self.cmd}"

        for c, v in self.args:
            if isinstance(v, bool):
                s += f" {c}{int(v)}"
            else:
                s += f" {c}{v}"

        return s

    @staticmethod
    def arg_cast(c: str, s: str) -> Any:
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
    def from_line(cls, s: str):
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

    def parse_gcode(self, lines: List[str]):
        for line in self._clean_gcode_lines(lines):
            yield GcodeCommand.from_line(line)
