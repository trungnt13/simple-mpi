import random
import sys
import time
import types
from multiprocessing import Lock, Process, Queue, Value, cpu_count
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

__all__ = ["MPI"]

_SLEEP_TIME = 0.01


def segment_list(
    jobs: List[Any],
    size: Optional[int] = None,
    n_seg: Optional[int] = None,
) -> List[List[Any]]:
    """
    Example
    -------
    >>> segment_list([1,2,3,4,5],2)
    >>> # [[1, 2, 3], [4, 5]]
    >>> segment_list([1,2,3,4,5],4)
    >>> # [[1], [2], [3], [4, 5]]
    """
    # by floor, make sure and process has it own job
    if n_seg is None:
        n_seg = int(np.ceil(len(jobs) / float(size)))
    # start segmenting
    segments = []
    start = 0
    remain_data = len(jobs)
    remain_seg = n_seg
    while remain_data > 0:
        # ====== adaptive size ====== #
        size = remain_data // remain_seg
        segments.append(jobs[start : (start + size)])
        # ====== update remain ====== #
        start += size
        remain_data -= size
        remain_seg -= 1
    return segments


class SharedCounter(object):
    """A multiprocessing syncrhonized counter"""

    def __init__(self, initial_value: int = 0):
        self.val = Value("i", initial_value)
        self.lock = Lock()

    def __iadd__(self, value: int):
        with self.lock:
            self.val.value += value
        return self

    def __isub__(self, value: int):
        with self.lock:
            self.val.value -= value
        return self

    def add(self, value: int = 1):
        with self.lock:
            self.val.value += value

    @property
    def value(self):
        with self.lock:
            return self.val.value

    def __del__(self):
        del self.lock
        del self.val


class MPI:
    r"""MPI - Simple multi-processing interface

    This class use round robin to schedule the tasks to each processes.

    Args:
      jobs: list, tuple, numpy.ndarray
          list of works.
      func: call-able
          take a `list of jobs` as input (i.e. map_func([job1, job2, ...])),
          the length of this list is determined by `buffer_size`
          NOTE: the argument of map_func is always a list.
      ncpu: int
          number of processes.
      batch: int
          the amount of samples grouped into list, and feed to each
          process each iteration. (i.e. func([job0, job1, ...]))
          if `batch=1`, each input is feed individually to `func`
          (i.e. func(job0); func(job1]); ...)
      hwm: int
          "high water mark" for SEND socket, is a hard limit on the
          maximum number of outstanding messages ØMQ shall queue
          in memory for any single peer that the specified socket
          is communicating with.

    Example:

        >>> jobs = list(range(10))
        >>> fn
    """

    def __init__(
        self,
        jobs: Sequence[Any],
        func: Callable,
        ncpu: int = 1,
        batch: int = 1,
        hwm: int = 144,
    ):
        self._ID = random.randint(0, 10e8)
        # ====== check map_func ====== #
        if not hasattr(func, "__call__"):
            raise Exception('"func" must be call-able')
        self._func = func
        # ====== MPI parameters ====== #
        # never use all available CPU
        if ncpu is None:
            ncpu = cpu_count() - 1
        self._ncpu = min(np.clip(int(ncpu), 1, cpu_count() - 1), len(jobs))
        self._batch = max(1, int(batch))
        self._hwm = max(0, int(hwm))
        # ====== internal states ====== #
        self._nb_working_cpu = self._ncpu
        # processes manager
        self._is_init = False
        self._is_running = False
        self._is_finished = False
        self._terminate_now = False
        # ====== other queue ====== #
        if not isinstance(jobs, (tuple, list, np.ndarray)):
            raise ValueError("`jobs` must be instance of tuple or list.")
        self._jobs = jobs
        self._remain_jobs = SharedCounter(len(self._jobs))
        # Equally split for all processes
        self._tasks = Queue(maxsize=0)
        for i in segment_list(np.arange(len(self._jobs), dtype="int32"), size=self._batch):
            self._tasks.put_nowait(i)
        for i in range(self._ncpu):  # ending signal
            self._tasks.put_nowait(None)
        # ====== only 1 iteration is created ====== #
        self._current_iter = None

    # ==================== properties ==================== #
    def __len__(self):
        """Return the number of remain jobs"""
        return max(self._remain_jobs.value, 0)

    @property
    def nb_working_cpu(self):
        return self._nb_working_cpu

    @property
    def is_initialized(self):
        return self._is_init

    @property
    def is_finished(self):
        return self._is_finished

    @property
    def is_running(self):
        return self._is_running

    def terminate(self):
        self._terminate_now = True
        # force everything finished
        if self._current_iter is not None and not self.is_finished:
            try:
                next(self._current_iter)
            except StopIteration:
                pass

    # ==================== helper ==================== #
    def __iter(self):
        # Initialize
        if not self._is_init:
            self._init()
            self._is_init = True
        yield None  # yeild not thing for init
        # Select run function
        self._is_running = True
        for i in self._run():
            if self._terminate_now:
                break
            yield i
        # Finalize
        self._is_running = False
        self._finalize()
        self._is_finished = True

    def __iter__(self):
        if self._current_iter is None:
            self._current_iter = self.__iter()
            next(self._current_iter)
        return self._current_iter

    # ==================== python queue ==================== #
    def _init(self):
        def worker_func(tasks, queue, counter, remain_jobs):
            hwm = self._hwm
            minimum_update_size = max(hwm // self._ncpu, 1)
            # ====== Doing the jobs ====== #
            t = tasks.get()
            while t is not None:
                # `t` is just list of indices
                t = [self._jobs[i] for i in t]
                remain_jobs.add(-len(t))  # monitor current number of remain jobs
                if self._batch == 1:  # batch=1, NO need for list of inputs
                    ret = self._func(t[0])
                else:  # we have input is list of inputs here
                    ret = self._func(t)
                # if a generator is return, traverse through the
                # iterator and return each result
                if not isinstance(ret, types.GeneratorType):
                    ret = (ret,)
                nb_returned = 0
                for r in ret:
                    if r is not None:  # ignore None values
                        queue.put(r)
                        nb_returned += 1
                        # sometime 1 batch get too big, and we need to stop
                        # putting too many data into the queue
                        if nb_returned >= minimum_update_size:
                            counter.add(nb_returned)
                            nb_returned = 0
                            while counter.value > hwm:
                                time.sleep(_SLEEP_TIME)
                del ret  # delete old data (this work, checked)
                # increase shared counter (this number must perfectly
                # counted, only 1 mismatch and deadlock will happen)
                if nb_returned > 0:
                    counter.add(nb_returned)
                # check if we need to wait for the consumer here
                while counter.value > hwm:
                    time.sleep(_SLEEP_TIME)
                # get new tasks
                t = tasks.get()
            # ending signal
            queue.put(None)
            sys.exit(0)

        # ====== multiprocessing variables ====== #
        self._queue = Queue(maxsize=0)
        self._counter = SharedCounter(initial_value=0)
        self._processes = [
            Process(
                target=worker_func,
                args=(self._tasks, self._queue, self._counter, self._remain_jobs),
            )
            for i in range(self._ncpu)
        ]
        [p.start() for p in self._processes]

    def _run(self):
        while self._nb_working_cpu > 0:
            r = self._queue.get()
            while r is None:
                self._nb_working_cpu -= 1
                if self._nb_working_cpu <= 0:
                    break
                r = self._queue.get()
            if r is not None:
                self._counter.add(-1)
                yield r

    # ==================== finalize ==================== #
    def _finalize(self):
        # terminate or join all processes
        if self._terminate_now:
            [p.terminate() for p in self._processes if p._popen is not None and p.is_alive()]
        # only join started process which has _popen is not None
        else:
            [p.join() for p in self._processes if p._popen is not None]

        self._tasks.close()
        del self._remain_jobs

        self._queue.close()
        del self._counter
