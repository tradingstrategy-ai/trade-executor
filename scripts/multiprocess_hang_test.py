"""Multiprocess manager hang test.

Good output (no multiprocessing.Manager):

.. code-block:: text

    Doing subrouting launch_and_read_process
    poll() is None
    Terminating
    Got exit code -15
    Got output Doing subrouting run_unkillable
    This is an example output

The hang with Manager:

.. code-block:: text

    Doing subrouting launch_and_read_process
    poll() is None
    Terminating
    Got exit code -15

"""
import multiprocessing
import subprocess
import sys
import time


def launch_and_read_process():
    proc = subprocess.Popen(
        [
            "python",
            sys.argv[0],
            "run_unkillable"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Give time for the process to run and print()
    time.sleep(3)

    status = proc.poll()
    print("poll() is", status)

    print("Terminating")
    assert proc.returncode is None
    proc.terminate()
    exit_code = proc.wait()
    print("Got exit code", exit_code)
    stdout, stderr = proc.communicate()
    print("Got output", stdout.decode("utf-8"))


def run_unkillable():
    # Disable manager creation to make the code run correctly
    manager = multiprocessing.Manager()
    d = manager.dict()
    d["foo"] = "bar"
    print("This is an example output", flush=True)
    time.sleep(999)


def main():
    mode = sys.argv[1]
    print("Doing subrouting", mode)
    func = globals().get(mode)
    func()


if __name__ == "__main__":
    main()