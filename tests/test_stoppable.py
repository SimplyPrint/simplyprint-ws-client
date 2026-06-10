from typing import List

from simplyprint_ws_client.common.utils.stoppable import StoppableThread


class StoppableTaskExample(StoppableThread):
    state: List[int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = []

    def run(self):
        self.state.append(0)

        while not self.is_stopped():
            self.wait()
            self.state.append(1)

        self.state.append(2)


def test_parent_stoppable_stop():
    parent = StoppableTaskExample()
    child = StoppableTaskExample(parent_stoppable=parent)

    assert not parent.is_stopped()
    assert not child.is_stopped()

    parent.start()
    child.start()

    assert parent.state == [0]
    assert child.state == [0]

    parent.stop()
    parent.join()
    child.join()

    assert parent.state == [0, 1, 2]
    assert child.state == [0, 1, 2]


def test_child_stoppable_stop():
    parent = StoppableTaskExample()
    child = StoppableTaskExample(parent_stoppable=parent)

    assert not parent.is_stopped()
    assert not child.is_stopped()
    assert parent.state == []
    assert child.state == []

    parent.start()
    child.start()

    assert parent.state == [0]
    assert child.state == [0]

    child.stop()
    child.join()

    assert child.state == [0, 1, 2]

    parent.stop()
    parent.join()

    assert parent.state.count(0) == 1
    # Not guaranteed to hit 1 twice, but at least once.
    assert 1 <= parent.state.count(1) <= 2
    assert parent.state.count(2) == 1


def test_nested_stoppable():
    task1 = StoppableTaskExample()
    task2 = StoppableTaskExample(nested_stoppable=task1)

    assert not task1.is_stopped()
    assert not task2.is_stopped()

    task1.start()
    task2.start()
    task2.stop()

    assert task2.is_stopped()
    assert task1.is_stopped()

    task1.join()
    task2.join()

    assert task1.state == [0, 1, 2]
    assert task2.state == [0, 1, 2]
