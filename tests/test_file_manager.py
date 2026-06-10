import time
from typing import List

import pytest

from simplyprint_ws_client.cloud.files.file_manager import File, FileManager


@pytest.fixture
def test_time() -> int:
    return int(time.time())


@pytest.fixture
def test_files(test_time: int) -> List[File]:
    return [
        File("test1.gcode", 100, last_modified=test_time - 1000),
        File("test2.gcode", 200, last_modified=test_time - 2000),
        File("test3.gcode", 300, last_modified=test_time - 3000),
        File("test4.gcode", 400, last_modified=test_time - 4000),
        File("test5.gcode", 500, last_modified=test_time - 5000),
    ]


def test_file_manager(test_files: List[File], test_time: int):
    fm = FileManager(
        max_age=1001,
        max_count=2,
        max_size=250,
        least_remaining_space_percentage=0.1,
    )

    files_to_remove = list(fm.get_files_to_remove(test_files, 2000, 1500))

    assert files_to_remove == [
        File("test2.gcode", 200, last_modified=test_time - 2000),
        File("test3.gcode", 300, last_modified=test_time - 3000),
        File("test4.gcode", 400, last_modified=test_time - 4000),
        File("test5.gcode", 500, last_modified=test_time - 5000),
    ]


def test_file_manager_max_age(test_files: List[File], test_time: int):
    fm = FileManager(
        max_age=1001, max_count=0, max_size=0, least_remaining_space_percentage=0
    )

    files_to_remove = list(fm.get_files_to_remove(test_files, 2000, 1500))
    assert files_to_remove == [
        File("test2.gcode", 200, last_modified=test_time - 2000),
        File("test3.gcode", 300, last_modified=test_time - 3000),
        File("test4.gcode", 400, last_modified=test_time - 4000),
        File("test5.gcode", 500, last_modified=test_time - 5000),
    ]


def test_file_manager_max_count(test_files: List[File], test_time: int):
    fm = FileManager(
        max_age=0, max_count=2, max_size=0, least_remaining_space_percentage=0
    )

    files_to_remove = list(fm.get_files_to_remove(test_files, 2000, 1500))

    assert files_to_remove == [
        File("test5.gcode", 500, last_modified=test_time - 5000),
        File("test4.gcode", 400, last_modified=test_time - 4000),
        File("test3.gcode", 300, last_modified=test_time - 3000),
    ]


def test_file_manager_max_size(test_files: List[File], test_time: int):
    fm = FileManager(
        max_age=0, max_count=0, max_size=250, least_remaining_space_percentage=0
    )

    files_to_remove = list(fm.get_files_to_remove(test_files, 2000, 1500))

    assert files_to_remove == [
        File("test3.gcode", 300, last_modified=test_time - 3000),
        File("test4.gcode", 400, last_modified=test_time - 4000),
        File("test5.gcode", 500, last_modified=test_time - 5000),
    ]


def test_file_manager_least_remaining_space_percentage(
    test_files: List[File], test_time: int
):
    fm = FileManager(
        max_age=0, max_count=0, max_size=0, least_remaining_space_percentage=1
    )

    files_to_remove = list(fm.get_files_to_remove(test_files, 2000, 1500))

    # Create the expected result manually to avoid reversed() issues with fixtures
    expected = [
        File("test5.gcode", 500, last_modified=test_time - 5000),
        File("test4.gcode", 400, last_modified=test_time - 4000),
        File("test3.gcode", 300, last_modified=test_time - 3000),
        File("test2.gcode", 200, last_modified=test_time - 2000),
        File("test1.gcode", 100, last_modified=test_time - 1000),
    ]

    assert files_to_remove == expected
