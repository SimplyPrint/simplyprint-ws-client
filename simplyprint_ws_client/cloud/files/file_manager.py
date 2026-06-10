import base64
import string
import time
from typing import List, NamedTuple, Generator

File = NamedTuple("File", [("name", str), ("size", int), ("last_modified", int)])


class FileManager:
    """Virtual file manager logic"""

    max_age: int
    max_count: int
    max_size: int
    least_remaining_space_percentage: float

    def __init__(
        self,
        max_age: int = 0,
        max_count=0,
        max_size=0,
        least_remaining_space_percentage=0.1,
    ) -> None:
        self.max_age = max_age
        self.max_count = max_count
        self.max_size = max_size
        self.least_remaining_space_percentage = least_remaining_space_percentage

    @staticmethod
    def get_smaller_file_id(file_id: str):
        # Remove all non-hex characters
        padded_str = "".join([char for char in file_id if char in string.hexdigits])
        padded_str = padded_str.ljust(len(padded_str) + (len(padded_str) % 2), "0")
        return (
            base64.urlsafe_b64encode(bytes.fromhex(padded_str))
            .decode()
            .replace("=", "")
        )

    def get_files_to_remove(
        self, files: List[File], total_disk_space: int, total_disk_usage: int
    ) -> Generator[File, None, None]:
        # Sort files by last modified
        files.sort(key=lambda f: f.last_modified)
        time_now = time.time()

        for file in reversed(files):
            if 0 < self.max_age < time_now - file.last_modified:
                files.remove(file)
                total_disk_usage -= file.size
                yield file

        for file in reversed(files):
            if 0 < self.max_size < file.size:
                files.remove(file)
                total_disk_usage -= file.size
                yield file

        if self.least_remaining_space_percentage > 0:
            space_left = total_disk_space - total_disk_usage
            space_required = total_disk_space * self.least_remaining_space_percentage

            while len(files) > 0 and space_required > space_left:
                file = files.pop(0)
                total_disk_usage -= file.size
                space_left = total_disk_space - total_disk_usage
                yield file

        if self.max_count > 0:
            while len(files) > self.max_count:
                file = files.pop(0)
                total_disk_usage -= file.size
                yield file
