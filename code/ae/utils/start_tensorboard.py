"""
For starting TensorBoard visualizer.
"""
import os
import signal
import subprocess

from tensorflow import tensorboard as tb

from .flags import FLAGS, home_out

# pylint: disable=invalid-name
_image_dir = FLAGS.data_dir
_summary_dir = FLAGS.summary_dir

_tb_pid_file = home_out(".tbpid")
_tb_path = os.path.join(os.path.dirname(tb.__file__), 'tensorboard.py')
_tb_port = "6006"


def start():
    """ Starts TensorBoard. """
    if not os.path.exists(_tb_path):
        raise EnvironmentError("tensorboard.py not found!")

    try:
        if os.path.exists(_tb_pid_file):
            tb_pid = int(open(_tb_pid_file, 'r').readline().strip())
            try:
                os.kill(tb_pid, signal.SIGKILL)
            except OSError:
                pass

            os.remove(_tb_pid_file)

        devnull = open(os.devnull, 'wb')
        proc = subprocess.Popen(['nohup', FLAGS.python,
                                 '-u', _tb_path, '--logdir={0}'.format(_summary_dir),
                                 '--port=' + _tb_port], stdout=devnull, stderr=devnull)
        with open(_tb_pid_file, 'w') as fwrite:
            fwrite.write(str(proc.pid))

        if not FLAGS.no_browser:
            subprocess.Popen(['open', 'http://localhost:{0}'.format(_tb_port)])

    except FileNotFoundError:
        # Probably Windows.
        print("Automatically launching tensorboard is not supported on Windows.")
        print("Lauch tensorboard with {} --logdir {} --port {}"
              .format(_tb_path, _summary_dir, _tb_port))


if __name__ == '__main__':
    start()
